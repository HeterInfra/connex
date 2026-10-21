//! Threaded data-plane engine on top of [`Endpoint`].
//!
//! One dedicated progress thread owns the UCX worker (single-threaded mode);
//! every other thread talks to it through a command queue and waits on a
//! completion, so callers (including Python with the GIL released) never
//! touch UCX directly and never have to call `progress` themselves. Peer
//! failures, heartbeats and rejoins are therefore handled even while the
//! application thread is busy.
//!
//! Data path: registered memory and one-sided RMA. A peer's region is
//! addressed by `(peer url, remote address)`, like Mooncake's
//! `transfer_sync_read(url, dst, src, len)`. The first access to a remote
//! region fetches its packed rkey over an active message and caches it per
//! peer incarnation. Unregistering a region that peers have fetched first
//! invalidates their cached rkeys and waits for their acknowledgements (and
//! for their in-flight RMA on it), so a recycled address can never be read
//! with a stale key.
//!
//! Signals: small active messages ([`Engine::send_msg`]) that are delivered
//! in order with the data the same sender put before them
//! ([`Engine::put_signal`] flushes the put first).
//!
//! Peer URL: `ip:port#model|stage|shard|replica`. The logical id in the URL
//! lets the reader bind the connection to the sender's identity, so a
//! restarted sender that comes back under the same id and address is the
//! same peer, and failure events name logical ids.

use crate::{Endpoint, Error, Event, Heartbeat, LogicalId, MemRegion, Req, Result, Rkey};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const K_RKEY_REQ: u8 = 1;
const K_RKEY_REP: u8 = 2;
const K_INVAL: u8 = 3;
const K_INVAL_ACK: u8 = 4;
const K_MSG: u8 = 5;

/// Default deadline for one RMA when the caller gives none.
pub const DEFAULT_OP_TIMEOUT: Duration = Duration::from_secs(3);
/// How long an unregister waits for readers to drop their cached rkeys.
const INVAL_TIMEOUT: Duration = Duration::from_secs(1);

/// Parsed peer URL.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeerUrl {
    pub ip: String,
    pub port: u16,
    pub id: LogicalId,
}

impl PeerUrl {
    pub fn parse(url: &str) -> Result<Self> {
        let bad = || Error::Invalid(format!("bad connex-rs url {url:?} (want ip:port#model|stage|shard|replica)"));
        let (addr, id) = url.split_once('#').ok_or_else(bad)?;
        let (ip, port) = addr.rsplit_once(':').ok_or_else(bad)?;
        let port = port.parse().map_err(|_| bad())?;
        let id = LogicalId::decode(id).ok_or_else(bad)?;
        Ok(PeerUrl { ip: ip.trim_matches(|c| c == '[' || c == ']').to_string(), port, id })
    }

    pub fn format(ip: &str, port: u16, id: &LogicalId) -> String {
        format!("{ip}:{port}#{}", id.encode())
    }
}

/// Where a message goes: a URL (connect on demand) or an already bound peer
/// (for example a reader that connected to us).
#[derive(Clone, Debug)]
pub enum Target {
    Url(PeerUrl),
    Id(LogicalId),
}

impl Target {
    fn id(&self) -> &LogicalId {
        match self {
            Target::Url(u) => &u.id,
            Target::Id(i) => i,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub id: LogicalId,
    /// Address to listen on ("" or "0.0.0.0" = any).
    pub listen_ip: String,
    /// 0 = ephemeral.
    pub port: u16,
    /// Address advertised in [`Engine::local_url`] (defaults to listen_ip).
    pub advertise_ip: Option<String>,
    pub heartbeat: Option<Heartbeat>,
    /// Busy-poll this long after the last activity before sleeping on the
    /// worker's event fd. Lower = less CPU, higher = lower wake latency.
    pub spin: Duration,
}

impl EngineConfig {
    pub fn new(id: LogicalId, listen_ip: &str, port: u16) -> Self {
        EngineConfig { id, listen_ip: listen_ip.into(), port, advertise_ip: None, heartbeat: None, spin: Duration::from_micros(500) }
    }
}

/// Engine counters (leak checks, tests).
#[derive(Clone, Debug, Default)]
pub struct EngineStats {
    pub ep: crate::Stats,
    pub regions: usize,
    pub region_bytes: u64,
    pub remote_rkeys: usize,
    pub inflight: usize,
    pub rma_ok: u64,
    pub rma_err: u64,
    pub rkey_fetches: u64,
    pub invalidations: u64,
    pub op_timeouts: u64,
    pub wakeups: u64,
}

// ------------------------------------------------------------- completions

/// A completion a caller can wait on (set once by the progress thread).
pub struct Completion {
    m: Mutex<Option<Result<()>>>,
    cv: Condvar,
    flag: AtomicBool,
}

/// A waiter spins this long before sleeping on the condvar: most RMA on
/// RDMA completes within it, and a futex wake costs several µs.
const WAIT_SPIN: Duration = Duration::from_micros(30);

impl Completion {
    fn new() -> Arc<Self> {
        Arc::new(Completion { m: Mutex::new(None), cv: Condvar::new(), flag: AtomicBool::new(false) })
    }

    fn set(&self, r: Result<()>) {
        let mut g = self.m.lock().unwrap();
        if g.is_none() {
            *g = Some(r);
            self.flag.store(true, Ordering::Release);
            self.cv.notify_all();
        }
    }

    pub fn is_done(&self) -> bool {
        self.m.lock().unwrap().is_some()
    }

    /// Waits for the result. `None` timeout waits forever. A wait that
    /// times out leaves the operation running (it has its own deadline).
    pub fn wait(&self, timeout: Option<Duration>) -> Option<Result<()>> {
        let t0 = Instant::now();
        while !self.flag.load(Ordering::Acquire) && t0.elapsed() < WAIT_SPIN {
            std::hint::spin_loop();
        }
        let mut g = self.m.lock().unwrap();
        let deadline = timeout.map(|t| Instant::now() + t);
        while g.is_none() {
            match deadline {
                None => g = self.cv.wait(g).unwrap(),
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        return None;
                    }
                    g = self.cv.wait_timeout(g, d - now).unwrap().0;
                }
            }
        }
        g.clone()
    }
}

enum Cmd {
    Register { addr: u64, len: u64, cuda_ctx: usize, done: Arc<Completion> },
    Unregister { addr: u64, done: Arc<Completion> },
    Rma { cuda_ctx: usize, url: PeerUrl, is_put: bool, local: u64, remote: u64, len: u64, deadline: Instant, signal: Option<Vec<u8>>, done: Arc<Completion> },
    Msg { to: Target, data: Vec<u8>, done: Arc<Completion> },
    Connect { url: PeerUrl, done: Arc<Completion> },
    Disconnect { peer: LogicalId, done: Arc<Completion> },
    Stats { out: mpsc::Sender<EngineStats> },
}

struct Shared {
    cmds: Mutex<VecDeque<Cmd>>,
    cmd_fd: i32,
    sleeping: AtomicBool,
    stop: AtomicBool,
    /// Set (under the cmds lock) when the thread no longer takes commands.
    closed: AtomicBool,
    events: Mutex<VecDeque<Event>>,
    events_cv: Condvar,
    msgs: Mutex<VecDeque<(LogicalId, Vec<u8>)>>,
    msgs_cv: Condvar,
    wakeups: AtomicU64,
}

impl Shared {
    fn push(&self, c: Cmd) {
        {
            let mut q = self.cmds.lock().unwrap();
            if self.closed.load(Ordering::SeqCst) {
                drop(q);
                fail_cmd(c, Error::Invalid("engine is shut down".into()));
                return;
            }
            q.push_back(c);
        }
        if self.sleeping.load(Ordering::SeqCst) {
            self.kick();
        }
    }

    fn kick(&self) {
        let one: u64 = 1;
        unsafe { libc::write(self.cmd_fd, &one as *const u64 as *const libc::c_void, 8) };
    }
}

/// The engine handle. Cheap to share between threads (`&self` methods).
pub struct Engine {
    shared: Arc<Shared>,
    thread: Mutex<Option<JoinHandle<()>>>,
    url: String,
    id: LogicalId,
    port: u16,
    managed: Mutex<HashMap<u64, (u64, usize)>>,
}

impl Engine {
    /// Starts the progress thread, creates the UCX context and listens.
    pub fn start(cfg: EngineConfig) -> Result<Engine> {
        let cmd_fd = unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) };
        if cmd_fd < 0 {
            return Err(Error::Invalid("eventfd failed".into()));
        }
        let shared = Arc::new(Shared {
            cmds: Mutex::new(VecDeque::new()),
            cmd_fd,
            sleeping: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            events: Mutex::new(VecDeque::new()),
            events_cv: Condvar::new(),
            msgs: Mutex::new(VecDeque::new()),
            msgs_cv: Condvar::new(),
            wakeups: AtomicU64::new(0),
        });
        let (tx, rx) = mpsc::channel();
        let sh = shared.clone();
        let c2 = cfg.clone();
        let cuda_ctx = crate::cuda_ctx_current();
        let thread = std::thread::Builder::new()
            .name(format!("connex-rs-{}", cfg.id.stage))
            .spawn(move || {
                // UCX resolves CUDA memory support (lanes for device memory)
                // when the context and endpoints are created, so the thread
                // that creates them needs the caller's CUDA context.
                crate::cuda_ctx_set(cuda_ctx);
                run(c2, sh, tx, cuda_ctx)
            })
            .map_err(|e| Error::Invalid(format!("spawn: {e}")))?;
        let port = match rx.recv() {
            Ok(Ok(p)) => p,
            Ok(Err(e)) => {
                let _ = thread.join();
                unsafe { libc::close(cmd_fd) };
                return Err(e);
            }
            Err(_) => return Err(Error::Invalid("progress thread died".into())),
        };
        let ip = cfg.advertise_ip.clone().unwrap_or_else(|| cfg.listen_ip.clone());
        let url = PeerUrl::format(&ip, port, &cfg.id);
        Ok(Engine { shared, thread: Mutex::new(Some(thread)), url, id: cfg.id, port, managed: Mutex::new(HashMap::new()) })
    }

    pub fn local_url(&self) -> &str {
        &self.url
    }

    pub fn id(&self) -> &LogicalId {
        &self.id
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn is_running(&self) -> bool {
        self.thread.lock().unwrap().is_some()
    }

    fn call(&self, make: impl FnOnce(Arc<Completion>) -> Cmd, timeout: Option<Duration>) -> Result<()> {
        let done = Completion::new();
        self.shared.push(make(done.clone()));
        done.wait(timeout).unwrap_or(Err(Error::Timeout))
    }

    /// Registers host or GPU memory for RMA (local or remote side).
    pub fn register(&self, addr: u64, len: u64) -> Result<()> {
        let cuda_ctx = crate::cuda_ctx_current();
        self.call(|done| Cmd::Register { addr, len, cuda_ctx, done }, None)
    }

    /// Unregisters a region previously registered at exactly `addr`. Readers
    /// that fetched its rkey drop it first (bounded wait).
    pub fn unregister(&self, addr: u64) -> Result<()> {
        self.call(|done| Cmd::Unregister { addr, done }, None)
    }

    /// Allocates and registers page-aligned host memory.
    pub fn alloc_managed(&self, size: usize) -> Result<u64> {
        let mut p: *mut libc::c_void = std::ptr::null_mut();
        if unsafe { libc::posix_memalign(&mut p, 4096, size.max(1)) } != 0 {
            return Err(Error::Invalid("posix_memalign failed".into()));
        }
        let addr = p as u64;
        if let Err(e) = self.register(addr, size as u64) {
            unsafe { libc::free(p) };
            return Err(e);
        }
        self.managed.lock().unwrap().insert(addr, (addr, size));
        Ok(addr)
    }

    pub fn free_managed(&self, addr: u64) -> Result<()> {
        let Some(_) = self.managed.lock().unwrap().remove(&addr) else {
            if !self.is_running() {
                return Ok(()); // freed at shutdown
            }
            return Err(Error::Invalid(format!("{addr:#x} is not a managed buffer")));
        };
        let r = self.unregister(addr);
        unsafe { libc::free(addr as *mut libc::c_void) };
        r
    }

    /// Starts a one-sided read (`is_put` false: remote `src` -> local `dst`)
    /// or write. The op fails with a peer error when the peer dies, and with
    /// `Timeout` after `timeout` (the connection to the peer is then closed,
    /// which also stops any late DMA into `local`).
    pub fn rma_async(&self, url: &str, is_put: bool, local: u64, remote: u64, len: u64, timeout: Option<Duration>, signal: Option<Vec<u8>>) -> Result<Arc<Completion>> {
        let url = PeerUrl::parse(url)?;
        let done = Completion::new();
        let deadline = Instant::now() + timeout.unwrap_or(DEFAULT_OP_TIMEOUT);
        self.shared.push(Cmd::Rma { cuda_ctx: crate::cuda_ctx_current(), url, is_put, local, remote, len, deadline, signal, done: done.clone() });
        Ok(done)
    }

    /// Blocking one-sided read of `len` bytes at `src` in the peer at `url`
    /// into local `dst`.
    pub fn get(&self, url: &str, dst: u64, src: u64, len: u64, timeout: Option<Duration>) -> Result<()> {
        let c = self.rma_async(url, false, dst, src, len, timeout, None)?;
        c.wait(None).unwrap()
    }

    /// Blocking one-sided write of local `src` to `dst` in the peer.
    pub fn put(&self, url: &str, src: u64, dst: u64, len: u64, timeout: Option<Duration>) -> Result<()> {
        let c = self.rma_async(url, true, src, dst, len, timeout, None)?;
        c.wait(None).unwrap()
    }

    /// Put, then (once the put is remotely complete) a signal message to the
    /// same peer. The peer sees the message only after the data is in place.
    pub fn put_signal(&self, url: &str, src: u64, dst: u64, len: u64, msg: Vec<u8>, timeout: Option<Duration>) -> Result<Arc<Completion>> {
        self.rma_async(url, true, src, dst, len, timeout, Some(msg))
    }

    /// Sends a message (signal) to a peer. Completes once handed to UCX.
    pub fn send_msg(&self, to: Target, data: Vec<u8>) -> Result<()> {
        self.call(|done| Cmd::Msg { to, data, done }, None)
    }

    /// Next received message, waiting up to `timeout` (None = forever).
    pub fn recv_msg(&self, timeout: Option<Duration>) -> Option<(LogicalId, Vec<u8>)> {
        let mut g = self.shared.msgs.lock().unwrap();
        let deadline = timeout.map(|t| Instant::now() + t);
        loop {
            if let Some(m) = g.pop_front() {
                return Some(m);
            }
            match deadline {
                None => g = self.shared.msgs_cv.wait(g).unwrap(),
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        return None;
                    }
                    g = self.shared.msgs_cv.wait_timeout(g, d - now).unwrap().0;
                }
            }
        }
    }

    /// Connects (wires up) ahead of the first transfer.
    pub fn connect(&self, url: &str) -> Result<()> {
        let url = PeerUrl::parse(url)?;
        self.call(|done| Cmd::Connect { url, done }, None)
    }

    /// Closes the connection to `peer` (no event); in-flight ops fail.
    pub fn disconnect(&self, peer: &LogicalId) -> Result<()> {
        let peer = peer.clone();
        self.call(|done| Cmd::Disconnect { peer, done }, None)
    }

    /// Next peer event (joined / failed), waiting up to `timeout`.
    pub fn wait_event(&self, timeout: Option<Duration>) -> Option<Event> {
        let mut g = self.shared.events.lock().unwrap();
        let deadline = timeout.map(|t| Instant::now() + t);
        loop {
            if let Some(e) = g.pop_front() {
                return Some(e);
            }
            if self.shared.stop.load(Ordering::SeqCst) {
                return None;
            }
            match deadline {
                None => g = self.shared.events_cv.wait(g).unwrap(),
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        return None;
                    }
                    g = self.shared.events_cv.wait_timeout(g, d - now).unwrap().0;
                }
            }
        }
    }

    pub fn stats(&self) -> EngineStats {
        let (tx, rx) = mpsc::channel();
        self.shared.push(Cmd::Stats { out: tx });
        let mut s = rx.recv_timeout(Duration::from_secs(5)).unwrap_or_default();
        s.wakeups = self.shared.wakeups.load(Ordering::Relaxed);
        s
    }

    /// Stops the progress thread (closes all endpoints). Idempotent.
    pub fn shutdown(&self) {
        let t = self.thread.lock().unwrap().take();
        if let Some(t) = t {
            self.shared.stop.store(true, Ordering::SeqCst);
            self.shared.kick();
            let _ = t.join();
            self.shared.events_cv.notify_all();
            for (_, (p, _)) in self.managed.lock().unwrap().drain() {
                unsafe { libc::free(p as *mut libc::c_void) };
            }
            unsafe { libc::close(self.shared.cmd_fd) };
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ------------------------------------------------------------ progress thread

struct LocalRegion {
    region: MemRegion,
    readers: HashSet<LogicalId>,
    dying: bool,
}

struct RemoteRegion {
    base: u64,
    len: u64,
    rkey: Rc<Rkey>,
}

struct WaitingRma {
    is_put: bool,
    local: u64,
    remote: u64,
    len: u64,
    deadline: Instant,
    signal: Option<Vec<u8>>,
    done: Arc<Completion>,
}

#[derive(Default)]
struct RemotePeer {
    regions: Vec<RemoteRegion>,
    /// Ops waiting for an rkey reply, by requested address.
    waiting: Vec<WaitingRma>,
    /// Addresses with an outstanding rkey request.
    requested: HashSet<u64>,
    /// Address the peer was last connected at.
    addr: Option<(String, u16)>,
    /// Invalidations whose ack waits for in-flight ops on the region.
    deferred_acks: Vec<u64>,
}

struct Inflight {
    peer: LogicalId,
    req: Req,
    base: u64,
    _rkey: Rc<Rkey>,
    deadline: Instant,
    /// Puts complete only once remotely complete (flush after the put).
    is_put: bool,
    /// Put + signal: after the flush, send this message.
    signal: Option<Vec<u8>>,
    flushing: bool,
    done: Arc<Completion>,
}

struct PendingUnreg {
    addr: u64,
    acks: HashSet<LogicalId>,
    deadline: Instant,
    done: Arc<Completion>,
}

struct State {
    ep: Endpoint,
    sh: Arc<Shared>,
    local: BTreeMap<u64, LocalRegion>,
    remote: HashMap<LogicalId, RemotePeer>,
    inflight: Vec<Inflight>,
    unreg: Vec<PendingUnreg>,
    /// Requests whose callers were already answered (peer closed); UCX
    /// completes them with an error later and they are freed then.
    orphans: Vec<Req>,
    /// Our connects waiting for their first flush (-> PeerUp).
    pending_up: Vec<(LogicalId, Req)>,
    cuda_ctx: usize,
    stats: EngineStats,
}

fn rd64(b: &[u8], off: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(a)
}

fn hdr(kind: u8, words: &[u64]) -> Vec<u8> {
    let mut v = Vec::with_capacity(1 + 8 * words.len());
    v.push(kind);
    for w in words {
        v.extend_from_slice(&w.to_le_bytes());
    }
    v
}

fn run(cfg: EngineConfig, sh: Arc<Shared>, ready: mpsc::Sender<Result<u16>>, cuda_ctx: usize) {
    let mut ep = match Endpoint::new(cfg.id.clone()) {
        Ok(e) => e,
        Err(e) => {
            sh.closed.store(true, Ordering::SeqCst);
            let _ = ready.send(Err(e));
            return;
        }
    };
    let port = match ep.listen(&cfg.listen_ip, cfg.port) {
        Ok(p) => p,
        Err(e) => {
            sh.closed.store(true, Ordering::SeqCst);
            let _ = ready.send(Err(e));
            return;
        }
    };
    ep.set_heartbeat(cfg.heartbeat);
    let efd = ep.efd();
    let epfd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
    for fd in [efd, sh.cmd_fd] {
        if fd >= 0 {
            let mut ev = libc::epoll_event { events: libc::EPOLLIN as u32, u64: fd as u64 };
            unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fd, &mut ev) };
        }
    }
    let _ = ready.send(Ok(port));
    let mut st = State { ep, sh: sh.clone(), local: BTreeMap::new(), remote: HashMap::new(), inflight: Vec::new(), unreg: Vec::new(), orphans: Vec::new(), pending_up: Vec::new(), cuda_ctx, stats: EngineStats::default() };
    let spin = cfg.spin;
    let hb_tick = cfg.heartbeat.map(|h| h.interval.min(h.timeout / 4)).unwrap_or(Duration::from_millis(100));
    let mut last_work = Instant::now();
    while !sh.stop.load(Ordering::SeqCst) {
        let mut work = st.ep.progress() > 0;
        work |= st.drain_cmds();
        work |= st.drain_events();
        work |= st.check_inflight();
        st.check_unreg();
        if work {
            last_work = Instant::now();
            continue;
        }
        if last_work.elapsed() < spin {
            std::hint::spin_loop();
            continue;
        }
        // Sleep on the worker fd + command fd until something happens, a
        // heartbeat is due or the nearest op deadline passes.
        sh.sleeping.store(true, Ordering::SeqCst);
        if !sh.cmds.lock().unwrap().is_empty() || sh.stop.load(Ordering::SeqCst) {
            sh.sleeping.store(false, Ordering::SeqCst);
            continue;
        }
        if efd >= 0 && !st.ep.arm() {
            sh.sleeping.store(false, Ordering::SeqCst);
            continue;
        }
        let mut wait = hb_tick;
        let now = Instant::now();
        for d in st.inflight.iter().map(|o| o.deadline).chain(st.unreg.iter().map(|u| u.deadline)) {
            wait = wait.min(d.saturating_duration_since(now));
        }
        for p in st.remote.values() {
            for w in &p.waiting {
                wait = wait.min(w.deadline.saturating_duration_since(now));
            }
        }
        let ms = if efd < 0 { 1 } else { (wait.as_millis() as i32).clamp(1, 1000) };
        let mut evs = [libc::epoll_event { events: 0, u64: 0 }; 2];
        let n = unsafe { libc::epoll_wait(epfd, evs.as_mut_ptr(), 2, ms) };
        sh.sleeping.store(false, Ordering::SeqCst);
        sh.wakeups.fetch_add(1, Ordering::Relaxed);
        for e in evs.iter().take(n.max(0) as usize) {
            if e.u64 == sh.cmd_fd as u64 {
                let mut v: u64 = 0;
                unsafe { libc::read(sh.cmd_fd, &mut v as *mut u64 as *mut libc::c_void, 8) };
            }
        }
        last_work = Instant::now();
    }
    // Shutdown: fail everything outstanding, unmap, close (Endpoint drop).
    for o in st.inflight.drain(..) {
        o.done.set(Err(Error::Canceled));
    }
    for (_, p) in st.remote.drain() {
        for w in p.waiting {
            w.done.set(Err(Error::Canceled));
        }
    }
    for u in st.unreg.drain(..) {
        u.done.set(Err(Error::Canceled));
    }
    let regions: Vec<_> = std::mem::take(&mut st.local).into_values().collect();
    for r in regions {
        let _ = st.ep.mem_unmap(r.region);
    }
    let left = {
        let mut q = sh.cmds.lock().unwrap();
        sh.closed.store(true, Ordering::SeqCst);
        std::mem::take(&mut *q)
    };
    for c in left {
        fail_cmd(c, Error::Canceled);
    }
    unsafe { libc::close(epfd) };
    drop(st);
    sh.events_cv.notify_all();
}

fn fail_cmd(c: Cmd, e: Error) {
    match c {
        Cmd::Register { done, .. } | Cmd::Unregister { done, .. } | Cmd::Rma { done, .. } | Cmd::Msg { done, .. } | Cmd::Connect { done, .. } | Cmd::Disconnect { done, .. } => done.set(Err(e)),
        Cmd::Stats { .. } => {}
    }
}

impl State {
    fn drain_cmds(&mut self) -> bool {
        let cmds: Vec<Cmd> = {
            let mut q = self.sh.cmds.lock().unwrap();
            if q.is_empty() {
                return false;
            }
            q.drain(..).collect()
        };
        for c in cmds {
            self.cmd(c);
        }
        true
    }

    fn cmd(&mut self, c: Cmd) {
        match c {
            Cmd::Register { addr, len, cuda_ctx, done } => {
                self.use_cuda_ctx(cuda_ctx);
                if self.local.contains_key(&addr) {
                    done.set(Err(Error::Invalid(format!("{addr:#x} already registered"))));
                    return;
                }
                match self.ep.mem_map(addr, len) {
                    Ok(region) => {
                        if std::env::var_os("CONNEX_RS_DEBUG").is_some() {
                            eprintln!("connex-rs: registered {addr:#x}+{len} mem_type={} rkey={}B cuda_ctx={cuda_ctx:#x}", region.mem_type, region.rkey.len());
                        }
                        self.local.insert(addr, LocalRegion { region, readers: HashSet::new(), dying: false });
                        done.set(Ok(()));
                    }
                    Err(e) => done.set(Err(e)),
                }
            }
            Cmd::Unregister { addr, done } => {
                let Some(r) = self.local.get_mut(&addr) else {
                    done.set(Err(Error::Invalid(format!("{addr:#x} not registered"))));
                    return;
                };
                if r.dying {
                    done.set(Err(Error::Invalid(format!("{addr:#x} already being unregistered"))));
                    return;
                }
                r.dying = true;
                let readers: Vec<LogicalId> = r.readers.drain().collect();
                let mut acks = HashSet::new();
                for p in readers {
                    if self.ep.is_connected(&p) && self.ep.am_send(&p, hdr(K_INVAL, &[addr]), Vec::new()).is_ok() {
                        acks.insert(p);
                    }
                }
                self.stats.invalidations += acks.len() as u64;
                self.unreg.push(PendingUnreg { addr, acks, deadline: Instant::now() + INVAL_TIMEOUT, done });
                self.check_unreg();
            }
            Cmd::Rma { cuda_ctx, url, is_put, local, remote, len, deadline, signal, done } => {
                self.use_cuda_ctx(cuda_ctx);
                if let Err(e) = self.ensure_connected(&url) {
                    done.set(Err(e));
                    return;
                }
                let w = WaitingRma { is_put, local, remote, len, deadline, signal, done };
                self.start_rma(&url.id, w);
            }
            Cmd::Msg { to, data, done } => {
                if let Target::Url(u) = &to {
                    if let Err(e) = self.ensure_connected(u) {
                        done.set(Err(e));
                        return;
                    }
                }
                done.set(self.ep.am_send(to.id(), vec![K_MSG], data));
            }
            Cmd::Connect { url, done } => done.set(self.ensure_connected(&url)),
            Cmd::Disconnect { peer, done } => {
                self.ep.drop_peer(&peer);
                self.peer_gone(&peer, Error::PeerFailed(peer.clone()));
                done.set(Ok(()));
            }
            Cmd::Stats { out } => {
                let mut s = self.stats.clone();
                s.ep = self.ep.stats();
                s.regions = self.local.len();
                s.region_bytes = self.local.values().map(|r| r.region.len).sum();
                s.remote_rkeys = self.remote.values().map(|p| p.regions.len()).sum();
                s.inflight = self.inflight.len();
                let _ = out.send(s);
            }
        }
    }

    /// Adopts the caller's CUDA context on the progress thread (UCX calls
    /// the CUDA driver for device memory from this thread).
    fn use_cuda_ctx(&mut self, ctx: usize) {
        if ctx != 0 && ctx != self.cuda_ctx && crate::cuda_ctx_set(ctx) {
            self.cuda_ctx = ctx;
        }
    }

    fn ensure_connected(&mut self, url: &PeerUrl) -> Result<()> {
        let addr = (url.ip.clone(), url.port);
        let rp = self.remote.entry(url.id.clone()).or_default();
        if self.ep.is_connected(&url.id) && rp.addr.as_ref().map_or(true, |a| *a == addr) {
            // Already linked, in either direction (an inbound link from the
            // peer serves our reads too: one endpoint per peer).
            rp.addr = Some(addr);
            return Ok(());
        }
        // New peer, or the id moved to another address (migration/rejoin
        // learned from the URL): drop whatever we had and connect fresh.
        rp.regions.clear();
        rp.requested.clear();
        rp.addr = Some(addr);
        self.ep.connect(&url.id, &url.ip, url.port)?;
        // The link is up once a flush completes (wireup done, peer reached).
        match self.ep.flush(&url.id) {
            Ok(req) => self.pending_up.push((url.id.clone(), req)),
            Err(_) => {} // the endpoint error event follows
        }
        Ok(())
    }

    fn find_remote(&self, peer: &LogicalId, addr: u64, len: u64) -> Option<(u64, Rc<Rkey>)> {
        let p = self.remote.get(peer)?;
        p.regions.iter().find(|r| addr >= r.base && addr + len <= r.base + r.len).map(|r| (r.base, r.rkey.clone()))
    }

    fn local_memh(&self, addr: u64, len: u64) -> *mut std::ffi::c_void {
        if let Some((_, r)) = self.local.range(..=addr).next_back() {
            if addr + len <= r.region.addr + r.region.len {
                return r.region.memh;
            }
        }
        std::ptr::null_mut()
    }

    fn start_rma(&mut self, peer: &LogicalId, w: WaitingRma) {
        match self.find_remote(peer, w.remote, w.len) {
            Some((base, rkey)) => self.post_rma(peer, base, rkey, w),
            None => {
                let rp = self.remote.entry(peer.clone()).or_default();
                let addr = w.remote;
                rp.waiting.push(w);
                if rp.requested.insert(addr) {
                    self.stats.rkey_fetches += 1;
                    if let Err(e) = self.ep.am_send(peer, hdr(K_RKEY_REQ, &[addr]), Vec::new()) {
                        self.fail_waiting(peer, e);
                    }
                }
            }
        }
    }

    fn post_rma(&mut self, peer: &LogicalId, base: u64, rkey: Rc<Rkey>, w: WaitingRma) {
        let memh = self.local_memh(w.local, w.len);
        let r = unsafe { self.ep.post_rma(peer, w.is_put, w.local, w.len as usize, w.remote, &rkey, memh) };
        match r {
            Ok(req) => self.inflight.push(Inflight { peer: peer.clone(), req, base, _rkey: rkey, deadline: w.deadline, is_put: w.is_put, signal: w.signal, flushing: false, done: w.done }),
            Err(e) => {
                self.stats.rma_err += 1;
                w.done.set(Err(e));
            }
        }
    }

    fn fail_waiting(&mut self, peer: &LogicalId, e: Error) {
        if let Some(rp) = self.remote.get_mut(peer) {
            rp.requested.clear();
            for w in rp.waiting.drain(..) {
                self.stats.rma_err += 1;
                w.done.set(Err(e.clone()));
            }
        }
    }

    fn check_inflight(&mut self) -> bool {
        self.orphans.retain_mut(|r| r.test().is_none());
        if !self.pending_up.is_empty() {
            let mut up = Vec::new();
            self.pending_up.retain_mut(|(p, r)| match r.test() {
                None => true,
                Some(Ok(())) => {
                    up.push(p.clone());
                    false
                }
                Some(Err(_)) => false,
            });
            for p in up {
                if self.ep.is_connected(&p) {
                    self.publish(Event::PeerUp { peer: p, ts_ns: crate::now_ns() });
                }
            }
        }
        if self.inflight.is_empty() && self.remote.values().all(|p| p.waiting.is_empty()) {
            return false;
        }
        let now = Instant::now();
        let mut progressed = false;
        let mut timed_out: Vec<LogicalId> = Vec::new();
        let mut i = 0;
        while i < self.inflight.len() {
            let o = &mut self.inflight[i];
            match o.req.test() {
                None => {
                    if now >= o.deadline {
                        timed_out.push(o.peer.clone());
                    }
                    i += 1;
                }
                Some(Ok(())) if o.is_put && !o.flushing => {
                    // Put done locally: flush for remote completion (then signal).
                    let peer = o.peer.clone();
                    match self.ep.flush(&peer) {
                        Ok(req) => {
                            let o = &mut self.inflight[i];
                            o.req = req;
                            o.flushing = true;
                            i += 1;
                        }
                        Err(e) => {
                            let o = self.inflight.swap_remove(i);
                            self.stats.rma_err += 1;
                            o.done.set(Err(e));
                        }
                    }
                    progressed = true;
                }
                Some(r) => {
                    let o = self.inflight.swap_remove(i);
                    let r = match (r, o.signal) {
                        (Ok(()), Some(msg)) => self.ep.am_send(&o.peer, vec![K_MSG], msg),
                        (Ok(()), None) => Ok(()),
                        (Err(_), _) => Err(Error::PeerFailed(o.peer.clone())),
                    };
                    if r.is_ok() {
                        self.stats.rma_ok += 1;
                    } else {
                        self.stats.rma_err += 1;
                    }
                    o.done.set(r);
                    self.maybe_ack(&o.peer, o.base);
                    progressed = true;
                }
            }
        }
        // rkey replies that never came
        let mut expired: Vec<(LogicalId, Arc<Completion>)> = Vec::new();
        for (id, p) in self.remote.iter_mut() {
            p.waiting.retain(|w| {
                if now >= w.deadline {
                    expired.push((id.clone(), w.done.clone()));
                    false
                } else {
                    true
                }
            });
        }
        for (id, done) in expired {
            self.stats.op_timeouts += 1;
            done.set(Err(Error::Timeout));
            timed_out.push(id);
        }
        // A timed-out op means the peer is hung or unreachable: close the
        // connection so that no late DMA lands in the caller's buffer, fail
        // everything else on it and report the peer.
        timed_out.sort();
        timed_out.dedup();
        for peer in timed_out {
            if !self.ep.is_connected(&peer) {
                continue;
            }
            self.stats.op_timeouts += 1;
            self.ep.drop_peer(&peer);
            self.peer_gone(&peer, Error::Timeout);
            self.publish(Event::PeerFailed { peer: Some(peer), status: 0, reason: "operation timeout".into(), ts_ns: crate::now_ns() });
            progressed = true;
        }
        progressed
    }

    fn maybe_ack(&mut self, peer: &LogicalId, base: u64) {
        let busy = self.inflight.iter().any(|o| &o.peer == peer && o.base == base);
        if busy {
            return;
        }
        if let Some(rp) = self.remote.get_mut(peer) {
            if let Some(pos) = rp.deferred_acks.iter().position(|&b| b == base) {
                rp.deferred_acks.swap_remove(pos);
                let _ = self.ep.am_send(peer, hdr(K_INVAL_ACK, &[base]), Vec::new());
            }
        }
    }

    fn check_unreg(&mut self) {
        if self.unreg.is_empty() {
            return;
        }
        let now = Instant::now();
        let mut i = 0;
        while i < self.unreg.len() {
            let u = &mut self.unreg[i];
            u.acks.retain(|p| self.ep.is_connected(p));
            if u.acks.is_empty() || now >= u.deadline {
                let u = self.unreg.swap_remove(i);
                if let Some(r) = self.local.remove(&u.addr) {
                    let r = self.ep.mem_unmap(r.region);
                    u.done.set(r);
                } else {
                    u.done.set(Ok(()));
                }
            } else {
                i += 1;
            }
        }
    }

    /// Fails all ops on `peer` and forgets its rkeys (its endpoint is gone).
    fn peer_gone(&mut self, peer: &LogicalId, e: Error) {
        let mut i = 0;
        while i < self.inflight.len() {
            if &self.inflight[i].peer == peer {
                // The request completes with an error once UCX purges the
                // closed endpoint; we answer the caller now. The buffer is
                // safe: a force-closed endpoint does no further DMA.
                let o = self.inflight.swap_remove(i);
                self.stats.rma_err += 1;
                o.done.set(Err(e.clone()));
                self.orphans.push(o.req);
            } else {
                i += 1;
            }
        }
        self.fail_waiting(peer, e);
        if let Some(rp) = self.remote.get_mut(peer) {
            rp.regions.clear();
            rp.deferred_acks.clear();
        }
        for r in self.local.values_mut() {
            r.readers.remove(peer);
        }
    }

    fn publish(&self, ev: Event) {
        self.sh.events.lock().unwrap().push_back(ev);
        self.sh.events_cv.notify_all();
    }

    fn drain_events(&mut self) -> bool {
        let mut any = false;
        while let Some(ev) = self.ep.next_event() {
            any = true;
            match ev {
                Event::Am { peer, hdr: h, data, .. } => self.on_am(peer, h, data),
                Event::PeerFailed { ref peer, .. } => {
                    if let Some(p) = peer {
                        self.peer_gone(p, Error::PeerFailed(p.clone()));
                    }
                    self.publish(ev);
                }
                Event::PeerUp { .. } => self.publish(ev),
                Event::PeerJoined { ref peer, rejoin, ts_ns } => {
                    // Inbound (the peer's hello arrived): the link works. Our
                    // own connects report PeerUp after their flush instead.
                    let ours = self.pending_up.iter().any(|(p, _)| p == peer);
                    if !ours {
                        self.publish(Event::PeerUp { peer: peer.clone(), ts_ns });
                    }
                    if rejoin {
                        // A new incarnation replaced the old endpoint: its
                        // ops are dead and its rkeys belong to the old one.
                        // (Our own reconnect also lands here.)
                        let p = peer.clone();
                        let waiting: Vec<WaitingRma> = self.remote.get_mut(&p).map(|rp| rp.waiting.drain(..).collect()).unwrap_or_default();
                        self.peer_gone(&p, Error::PeerFailed(p.clone()));
                        // Ops queued for an rkey were not posted yet: retry
                        // them on the new endpoint.
                        for w in waiting {
                            self.start_rma(&p, w);
                        }
                    }
                    self.publish(ev);
                }
            }
        }
        any
    }

    fn on_am(&mut self, peer: LogicalId, h: Vec<u8>, data: Vec<u8>) {
        let Some(&kind) = h.first() else { return };
        match kind {
            K_RKEY_REQ if h.len() >= 9 => {
                let addr = rd64(&h, 1);
                let found = self.local.range_mut(..=addr).next_back().and_then(|(_, r)| {
                    if !r.dying && addr < r.region.addr + r.region.len {
                        r.readers.insert(peer.clone());
                        Some((r.region.addr, r.region.len, r.region.rkey.clone()))
                    } else {
                        None
                    }
                });
                let (base, len, rkey, ok) = match found {
                    Some((b, l, k)) => (b, l, k, 1),
                    None => (0, 0, Vec::new(), 0),
                };
                let _ = self.ep.am_send(&peer, hdr(K_RKEY_REP, &[addr, base, len, ok]), rkey);
            }
            K_RKEY_REP if h.len() >= 33 => {
                let (addr, base, len, ok) = (rd64(&h, 1), rd64(&h, 9), rd64(&h, 17), rd64(&h, 25));
                let rp = self.remote.entry(peer.clone()).or_default();
                rp.requested.remove(&addr);
                if ok == 1 {
                    match self.ep.rkey_unpack(&peer, &data) {
                        Ok(rk) => {
                            let rp = self.remote.get_mut(&peer).unwrap();
                            rp.regions.retain(|r| r.base + r.len <= base || r.base >= base + len);
                            rp.regions.push(RemoteRegion { base, len, rkey: Rc::new(rk) });
                        }
                        Err(e) => {
                            self.fail_waiting(&peer, e);
                            return;
                        }
                    }
                }
                // Post every waiting op the new key covers. An op on the
                // requested address that it does not cover fails (the peer
                // has no such region); others keep waiting for their reply.
                let waiting: Vec<WaitingRma> = self.remote.get_mut(&peer).map(|rp| rp.waiting.drain(..).collect()).unwrap_or_default();
                for w in waiting {
                    if let Some((b, rk)) = self.find_remote(&peer, w.remote, w.len) {
                        self.post_rma(&peer, b, rk, w);
                    } else if w.remote == addr {
                        self.stats.rma_err += 1;
                        w.done.set(Err(Error::Invalid(format!("peer {peer} has no registered region at {:#x}+{}", w.remote, w.len))));
                    } else {
                        let rp = self.remote.get_mut(&peer).unwrap();
                        let again = rp.requested.insert(w.remote);
                        let a = w.remote;
                        rp.waiting.push(w);
                        if again {
                            let _ = self.ep.am_send(&peer, hdr(K_RKEY_REQ, &[a]), Vec::new());
                        }
                    }
                }
            }
            K_INVAL if h.len() >= 9 => {
                let base = rd64(&h, 1);
                let rp = self.remote.entry(peer.clone()).or_default();
                rp.regions.retain(|r| r.base != base);
                rp.deferred_acks.push(base);
                self.maybe_ack(&peer, base);
            }
            K_INVAL_ACK if h.len() >= 9 => {
                let base = rd64(&h, 1);
                for u in self.unreg.iter_mut().filter(|u| u.addr == base) {
                    u.acks.remove(&peer);
                }
                self.check_unreg();
            }
            K_MSG => {
                self.sh.msgs.lock().unwrap().push_back((peer, data));
                self.sh.msgs_cv.notify_all();
            }
            _ => {}
        }
    }
}

