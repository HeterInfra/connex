//! connex-rs: a Connex-native transport prototype over UCX.
//!
//! An [`Endpoint`] is owned by one process and named by a [`LogicalId`]
//! (model, stage, shard, replica). Peers are addressed by their logical id,
//! never by a UCX endpoint handle, so a peer that dies and comes back under
//! the same id is the same peer to everyone else:
//!
//! * every UCX endpoint is created with `UCP_ERR_HANDLING_MODE_PEER`; its
//!   error callback becomes an [`Event::PeerFailed`] with a monotonic
//!   timestamp taken inside the callback;
//! * the connecting side sends a `hello` active message carrying its logical
//!   id; the accepting side binds the new UCX endpoint to that id. If the id
//!   is already bound (the old incarnation was not yet detected as dead, or
//!   was), the old endpoint is force-closed and replaced
//!   ([`Event::PeerJoined`] with `rejoin = true`);
//! * [`Endpoint::reconnect`] does the same from the connecting side (drop the
//!   old endpoint, connect fresh) for a rejoin/migrate event learned from the
//!   control plane.
//!
//! Data moves with UCP tagged messages. The 64-bit tag is
//! `hash16(sender id) << 48 | correlation id (48 bits)`, so a receive names
//! (peer, correlation id) and matches whatever incarnation of that peer sends
//! it. Single-threaded: all calls must come from the thread that created the
//! endpoint (UCS_THREAD_MODE_SINGLE); progress happens inside the blocking
//! calls or [`Endpoint::progress`].

mod ffi;
pub mod engine;
#[cfg(feature = "python")]
mod python;

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::{c_void, CString};
use std::fmt;
use std::time::{Duration, Instant};

pub const CORR_BITS: u32 = 48;
pub const CORR_MASK: u64 = (1u64 << CORR_BITS) - 1;

/// Heartbeat timeout before a peer has been heard from once.
pub const CONNECT_GRACE_NS: u64 = 5_000_000_000;

/// The calling thread's current CUDA context (null without CUDA).
pub fn cuda_ctx_current() -> usize {
    unsafe { ffi::cx_cuda_ctx_current() as usize }
}

/// Makes `ctx` current on the calling thread; false if CUDA is unavailable.
pub fn cuda_ctx_set(ctx: usize) -> bool {
    ctx != 0 && unsafe { ffi::cx_cuda_ctx_set(ctx as *mut c_void) } == 0
}

/// Monotonic clock (CLOCK_MONOTONIC, ns), the same clock event timestamps use.
pub fn now_ns() -> u64 {
    unsafe { ffi::cx_now_ns() }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LogicalId {
    pub model: String,
    pub stage: u32,
    pub shard: u32,
    pub replica: u32,
}

impl LogicalId {
    pub fn new(model: &str, stage: u32, shard: u32, replica: u32) -> Self {
        LogicalId { model: model.to_string(), stage, shard, replica }
    }

    /// Wire form: `model|stage|shard|replica`.
    pub fn encode(&self) -> String {
        format!("{}|{}|{}|{}", self.model, self.stage, self.shard, self.replica)
    }

    pub fn decode(s: &str) -> Option<Self> {
        let mut it = s.rsplitn(4, '|');
        let replica = it.next()?.parse().ok()?;
        let shard = it.next()?.parse().ok()?;
        let stage = it.next()?.parse().ok()?;
        let model = it.next()?.to_string();
        Some(LogicalId { model, stage, shard, replica })
    }

    /// 16-bit sender prefix of the tag (FNV-1a folded). Collisions between
    /// two peers of one endpoint are detected at hello time.
    pub fn tag_prefix(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in self.encode().bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        (h ^ (h >> 16) ^ (h >> 32) ^ (h >> 48)) & 0xffff
    }

    pub fn tag(&self, corr: u64) -> u64 {
        (self.tag_prefix() << CORR_BITS) | (corr & CORR_MASK)
    }
}

impl fmt::Display for LogicalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/s{}/sh{}/r{}", self.model, self.stage, self.shard, self.replica)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// A peer bound its logical id to a new UCX endpoint. `rejoin` is true
    /// when the id was already bound (old endpoint replaced) or had failed
    /// before.
    PeerJoined { peer: LogicalId, rejoin: bool, ts_ns: u64 },
    /// The UCX endpoint of a peer reported an error (peer failure). `peer`
    /// is None for an accepted endpoint that never said hello.
    PeerFailed { peer: Option<LogicalId>, status: i32, reason: String, ts_ns: u64 },
    /// The link to a peer is known to work ([`engine::Engine`] only): our
    /// connect completed a round trip, or the peer's hello arrived.
    PeerUp { peer: LogicalId, ts_ns: u64 },
    /// A user active message ([`Endpoint::am_send`]) from a bound peer.
    Am { peer: LogicalId, hdr: Vec<u8>, data: Vec<u8>, ts_ns: u64 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    Ucx(i32, String),
    UnknownPeer(LogicalId),
    PeerFailed(LogicalId),
    Timeout,
    Canceled,
    Invalid(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Ucx(st, s) => write!(f, "UCX error {st}: {s}"),
            Error::UnknownPeer(p) => write!(f, "unknown peer {p}"),
            Error::PeerFailed(p) => write!(f, "peer {p} failed"),
            Error::Timeout => write!(f, "timeout"),
            Error::Canceled => write!(f, "canceled"),
            Error::Invalid(s) => write!(f, "invalid: {s}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn ucx_err(st: i32) -> Error {
    Error::Ucx(st, ffi::status_string(st))
}

struct Peer {
    ep: *mut c_void,
    /// Incarnation counter of this id as seen by this endpoint.
    generation: u64,
    /// Heartbeat bookkeeping (monotonic ns).
    last_heard: u64,
    /// False until the first heartbeat arrives; until then the (longer)
    /// connect grace applies, because wireup (rdmacm, first registration) can
    /// take hundreds of ms.
    heard: bool,
    last_ping: u64,
}

/// Optional application-level heartbeat. UCX's own error handling reports
/// a peer whose process died (its kernel tears the connection down), but not
/// a peer that is hung or unreachable while its kernel/HCA still ACKs, nor (in
/// time) a lost node; this covers those cases. Pongs are sent from
/// `progress`, so the peer must call progress at least every `timeout`.
#[derive(Clone, Copy, Debug)]
pub struct Heartbeat {
    pub interval: Duration,
    pub timeout: Duration,
}

/// Counters for leak checks.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub eps_created: u64,
    pub eps_closed: u64,
    pub closes_pending: usize,
    pub live_peers: usize,
    pub anon_eps: usize,
    pub failures: u64,
    pub rejoins: u64,
    pub dropped_events: u64,
    pub heartbeat_failures: u64,
}

/// A pending send or receive that owns its buffer.
#[allow(dead_code)]
pub struct Op {
    req: *mut c_void,
    is_recv: bool,
    peer: LogicalId,
    /// UCX endpoint the op was posted on (sends) / peer generation (recvs).
    ep: *mut c_void,
    generation: u64,
    buf: Option<Vec<u8>>,
    len: usize,
    done: Option<Result<usize>>,
}

impl Op {
    pub fn peer(&self) -> &LogicalId {
        &self.peer
    }
    pub fn is_done(&self) -> bool {
        self.done.is_some()
    }
}

type Callback = Box<dyn FnMut(&Event)>;

pub struct Endpoint {
    ctx: *mut ffi::CxCtx,
    me: LogicalId,
    hello: Box<[u8]>,
    peers: HashMap<LogicalId, Peer>,
    owner: HashMap<usize, LogicalId>,
    anon: HashSet<usize>,
    closing: Vec<*mut c_void>,
    hellos: Vec<*mut c_void>,
    /// generation per id survives failures so a rejoin is recognised.
    generations: HashMap<LogicalId, u64>,
    prefixes: HashMap<u64, LogicalId>,
    events: VecDeque<Event>,
    callback: Option<Callback>,
    stats: Stats,
    heartbeat: Option<Heartbeat>,
    /// Fire-and-forget AM sends with the buffers they must keep alive.
    am_pending: Vec<(*mut c_void, Vec<u8>, Vec<u8>)>,
}

/// A registered memory region ([`Endpoint::mem_map`]).
pub struct MemRegion {
    pub memh: *mut c_void,
    pub addr: u64,
    pub len: u64,
    /// ucs_memory_type_t (0 host, 1 cuda, ...).
    pub mem_type: i32,
    /// Packed remote key, to be sent to peers.
    pub rkey: Vec<u8>,
}

/// An unpacked remote key, valid for the UCX endpoint it was unpacked on.
pub struct Rkey(pub(crate) *mut c_void);

impl Drop for Rkey {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { ffi::cx_rkey_destroy(self.0) };
        }
    }
}

/// A raw UCX request (RMA, flush). `None` inside = completed at post time.
pub struct Req(*mut c_void);

impl Req {
    /// Non-blocking test (does not progress): None while pending.
    pub fn test(&mut self) -> Option<Result<()>> {
        if self.0.is_null() {
            return Some(Ok(()));
        }
        let st = unsafe { ffi::cx_req_test(self.0, 0, std::ptr::null_mut()) };
        if st == unsafe { ffi::cx_inprogress() } {
            return None;
        }
        unsafe { ffi::cx_req_free(self.0) };
        self.0 = std::ptr::null_mut();
        Some(if st == 0 { Ok(()) } else { Err(ucx_err(st)) })
    }
}

fn req_from(r: *mut c_void) -> Result<Req> {
    let st = unsafe { ffi::cx_is_err(r) };
    if st != 0 {
        return Err(ucx_err(st));
    }
    Ok(Req(r))
}

impl Endpoint {
    /// Creates the UCP context + worker. UCX is configured through the usual
    /// environment (UCX_TLS, UCX_NET_DEVICES, UCX_KEEPALIVE_INTERVAL,
    /// UCX_RC_TIMEOUT, UCX_RC_RETRY_COUNT, UCX_TCP_KEEPIDLE ...).
    pub fn new(me: LogicalId) -> Result<Self> {
        let mut ctx = std::ptr::null_mut();
        let st = unsafe { ffi::cx_create(&mut ctx) };
        if st != 0 {
            return Err(ucx_err(st));
        }
        let hello = me.encode().into_bytes().into_boxed_slice();
        if hello.len() > ffi::HELLO_MAX {
            unsafe { ffi::cx_destroy(ctx) };
            return Err(Error::Invalid("logical id too long".into()));
        }
        Ok(Endpoint {
            ctx,
            me,
            hello,
            peers: HashMap::new(),
            owner: HashMap::new(),
            anon: HashSet::new(),
            closing: Vec::new(),
            hellos: Vec::new(),
            generations: HashMap::new(),
            prefixes: HashMap::new(),
            events: VecDeque::new(),
            callback: None,
            stats: Stats::default(),
            heartbeat: None,
            am_pending: Vec::new(),
        })
    }

    pub fn id(&self) -> &LogicalId {
        &self.me
    }

    /// Called synchronously (from inside progress) for every event, in
    /// addition to queueing it for [`Endpoint::poll_event`].
    pub fn set_callback(&mut self, cb: Option<Callback>) {
        self.callback = cb;
    }

    /// Enables (Some) or disables (None) the application-level heartbeat.
    pub fn set_heartbeat(&mut self, hb: Option<Heartbeat>) {
        self.heartbeat = hb;
        let now = now_ns();
        for p in self.peers.values_mut() {
            p.last_heard = now;
        }
    }

    /// Listens on `ip:port` (port 0 = ephemeral). Returns the bound port.
    pub fn listen(&mut self, ip: &str, port: u16) -> Result<u16> {
        let cip = CString::new(ip).map_err(|_| Error::Invalid(ip.into()))?;
        let r = unsafe { ffi::cx_listen(self.ctx, cip.as_ptr(), port) };
        if r <= 0 {
            return Err(ucx_err(r));
        }
        Ok(r as u16)
    }

    /// Connects to the endpoint listening at `ip:port`, which the caller
    /// knows (from the control plane) to be `peer`. Any existing endpoint to
    /// `peer` is dropped first, which makes this also the identity-aware
    /// reconnect.
    pub fn connect(&mut self, peer: &LogicalId, ip: &str, port: u16) -> Result<()> {
        self.drop_peer(peer);
        let cip = CString::new(ip).map_err(|_| Error::Invalid(ip.into()))?;
        let mut ep = std::ptr::null_mut();
        let st = unsafe { ffi::cx_connect(self.ctx, cip.as_ptr(), port, &mut ep) };
        if st != 0 {
            return Err(ucx_err(st));
        }
        self.stats.eps_created += 1;
        let r = unsafe { ffi::cx_hello(ep, self.hello.as_ptr() as *const c_void, self.hello.len()) };
        let st = unsafe { ffi::cx_is_err(r) };
        if st != 0 {
            self.close_ep(ep);
            return Err(ucx_err(st));
        }
        if !r.is_null() {
            self.hellos.push(r);
        }
        self.bind(peer.clone(), ep, ffi_now());
        Ok(())
    }

    /// Rejoin/migrate notification from the control plane: drop the old
    /// endpoint to `peer` and connect to its new address.
    pub fn reconnect(&mut self, peer: &LogicalId, ip: &str, port: u16) -> Result<()> {
        self.connect(peer, ip, port)
    }

    /// Forgets `peer` and force-closes its endpoint (no event).
    pub fn drop_peer(&mut self, peer: &LogicalId) {
        if let Some(p) = self.peers.remove(peer) {
            self.owner.remove(&(p.ep as usize));
            self.close_ep(p.ep);
        }
    }

    pub fn is_connected(&self, peer: &LogicalId) -> bool {
        self.peers.contains_key(peer)
    }

    pub fn peers(&self) -> Vec<LogicalId> {
        let mut v: Vec<_> = self.peers.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn generation(&self, peer: &LogicalId) -> Option<u64> {
        self.peers.get(peer).map(|p| p.generation)
    }

    pub fn stats(&self) -> Stats {
        let mut s = self.stats.clone();
        s.closes_pending = self.closing.len();
        s.live_peers = self.peers.len();
        s.anon_eps = self.anon.len();
        s.dropped_events = unsafe { ffi::cx_dropped_events(self.ctx) };
        s
    }

    fn bind(&mut self, peer: LogicalId, ep: *mut c_void, ts_ns: u64) {
        let prefix = peer.tag_prefix();
        if let Some(other) = self.prefixes.get(&prefix) {
            if *other != peer {
                eprintln!("connex-rs: tag prefix collision between {other} and {peer}");
            }
        }
        self.prefixes.insert(prefix, peer.clone());
        let gen = self.generations.entry(peer.clone()).or_insert(0);
        let rejoin = *gen > 0;
        *gen += 1;
        let generation = *gen;
        let now = now_ns();
        if let Some(old) = self.peers.insert(peer.clone(), Peer { ep, generation, last_heard: now, last_ping: 0, heard: false }) {
            self.owner.remove(&(old.ep as usize));
            self.close_ep(old.ep);
        }
        self.owner.insert(ep as usize, peer.clone());
        if rejoin {
            self.stats.rejoins += 1;
        }
        self.emit(Event::PeerJoined { peer, rejoin, ts_ns });
    }

    fn close_ep(&mut self, ep: *mut c_void) {
        let r = unsafe { ffi::cx_ep_close(ep, 1) };
        self.stats.eps_closed += 1;
        if !r.is_null() {
            self.closing.push(r);
        }
    }

    fn emit(&mut self, ev: Event) {
        if std::env::var_os("CONNEX_RS_DEBUG").is_some() {
            eprintln!("connex-rs[{}]: {:?}", self.me, ev);
        }
        if let Some(cb) = self.callback.as_mut() {
            cb(&ev);
        }
        self.events.push_back(ev);
    }

    /// Drives UCX and turns callback records into events. Returns the number
    /// of UCX progress steps that did work.
    pub fn progress(&mut self) -> u32 {
        let mut n = 0;
        loop {
            let k = unsafe { ffi::cx_progress(self.ctx) };
            n += k;
            if k == 0 {
                break;
            }
        }
        let mut ev = ffi::CxEvent::default();
        while unsafe { ffi::cx_poll_event(self.ctx, &mut ev) } == 1 {
            self.handle(&ev);
        }
        if self.heartbeat.is_some() {
            self.heartbeat_tick();
        }
        self.reap();
        n
    }

    fn heartbeat_tick(&mut self) {
        let hb = self.heartbeat.unwrap();
        let now = now_ns();
        let (iv, to) = (hb.interval.as_nanos() as u64, hb.timeout.as_nanos() as u64);
        let mut dead = Vec::new();
        let mut reqs = Vec::new();
        for (id, p) in self.peers.iter_mut() {
            let limit = if p.heard { to } else { to.max(CONNECT_GRACE_NS) };
            if now.saturating_sub(p.last_heard) > limit {
                dead.push(id.clone());
                continue;
            }
            if now.saturating_sub(p.last_ping) >= iv {
                p.last_ping = now;
                let r = unsafe { ffi::cx_hb(p.ep, 0) };
                if !r.is_null() && unsafe { ffi::cx_is_err(r) } == 0 {
                    reqs.push(r);
                }
            }
        }
        self.hellos.extend(reqs);
        for id in dead {
            if let Some(p) = self.peers.remove(&id) {
                self.owner.remove(&(p.ep as usize));
                self.close_ep(p.ep);
                self.stats.failures += 1;
                self.stats.heartbeat_failures += 1;
                self.emit(Event::PeerFailed {
                    peer: Some(id),
                    status: 0,
                    reason: "heartbeat timeout".into(),
                    ts_ns: now,
                });
            }
        }
    }

    fn handle(&mut self, ev: &ffi::CxEvent) {
        match ev.kind {
            ffi::EV_CONN_REQUEST => {
                let mut ep = std::ptr::null_mut();
                let st = unsafe { ffi::cx_accept(self.ctx, ev.ep, &mut ep) };
                if st == 0 {
                    self.stats.eps_created += 1;
                    self.anon.insert(ep as usize);
                }
            }
            ffi::EV_HELLO => {
                let s = String::from_utf8_lossy(&ev.data[..ev.len as usize]).into_owned();
                let Some(peer) = LogicalId::decode(&s) else { return };
                let key = ev.ep as usize;
                if self.anon.remove(&key) {
                    self.bind(peer, ev.ep, ev.ts_ns);
                }
            }
            ffi::EV_HB => {
                let key = ev.ep as usize;
                if let Some(id) = self.owner.get(&key) {
                    if let Some(p) = self.peers.get_mut(id) {
                        p.last_heard = p.last_heard.max(ev.ts_ns);
                        p.heard = true;
                    }
                    if ev.data[0] == b'P' {
                        let r = unsafe { ffi::cx_hb(ev.ep, 1) };
                        if !r.is_null() && unsafe { ffi::cx_is_err(r) } == 0 {
                            self.hellos.push(r);
                        }
                    }
                }
            }
            ffi::EV_AM => {
                let data = if ev.payload.is_null() {
                    Vec::new()
                } else {
                    let v = unsafe { std::slice::from_raw_parts(ev.payload as *const u8, ev.plen) }.to_vec();
                    unsafe { ffi::cx_free(ev.payload) };
                    v
                };
                if let Some(peer) = self.owner.get(&(ev.ep as usize)).cloned() {
                    let hdr = ev.data[..ev.len as usize].to_vec();
                    self.emit(Event::Am { peer, hdr, data, ts_ns: ev.ts_ns });
                }
            }
            ffi::EV_ERROR => {
                let key = ev.ep as usize;
                let reason = ffi::status_string(ev.status);
                if let Some(peer) = self.owner.remove(&key) {
                    self.peers.remove(&peer);
                    self.close_ep(ev.ep);
                    self.stats.failures += 1;
                    self.emit(Event::PeerFailed { peer: Some(peer), status: ev.status, reason, ts_ns: ev.ts_ns });
                } else if self.anon.remove(&key) {
                    self.close_ep(ev.ep);
                    self.emit(Event::PeerFailed { peer: None, status: ev.status, reason, ts_ns: ev.ts_ns });
                }
                // else: endpoint already replaced/closed by us; nothing to do.
            }
            _ => {}
        }
    }

    fn reap(&mut self) {
        let inprog = unsafe { ffi::cx_inprogress() };
        self.closing.retain(|&r| {
            let st = unsafe { ffi::cx_req_test(r, 0, std::ptr::null_mut()) };
            if st == inprog {
                true
            } else {
                unsafe { ffi::cx_req_free(r) };
                false
            }
        });
        self.am_pending.retain(|(r, _, _)| {
            let st = unsafe { ffi::cx_req_test(*r, 0, std::ptr::null_mut()) };
            if st == inprog {
                true
            } else {
                unsafe { ffi::cx_req_free(*r) };
                false
            }
        });
        self.hellos.retain(|&r| {
            let st = unsafe { ffi::cx_req_test(r, 0, std::ptr::null_mut()) };
            if st == inprog {
                true
            } else {
                unsafe { ffi::cx_req_free(r) };
                false
            }
        });
    }

    pub fn poll_event(&mut self) -> Option<Event> {
        if self.events.is_empty() {
            self.progress();
        }
        self.events.pop_front()
    }

    /// Progresses until an event is available or `timeout` passes.
    pub fn wait_event(&mut self, timeout: Duration) -> Option<Event> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(e) = self.poll_event() {
                return Some(e);
            }
            if Instant::now() >= deadline {
                return None;
            }
        }
    }

    /// Pops a queued event without progressing.
    pub fn next_event(&mut self) -> Option<Event> {
        self.events.pop_front()
    }

    /// The worker's wakeup fd (see [`Endpoint::arm`]).
    pub fn efd(&self) -> i32 {
        unsafe { ffi::cx_efd(self.ctx) }
    }

    /// Arms the wakeup fd. Returns false when there is pending work, in
    /// which case the caller must progress instead of sleeping.
    pub fn arm(&mut self) -> bool {
        unsafe { ffi::cx_arm(self.ctx) == 0 }
    }

    // ------------------------------------------------------------ AM / RMA

    /// Sends a user active message (eager) to `peer`. Completion is not
    /// reported: a failed peer shows up as [`Event::PeerFailed`].
    pub fn am_send(&mut self, peer: &LogicalId, hdr: Vec<u8>, data: Vec<u8>) -> Result<()> {
        if hdr.len() > ffi::HELLO_MAX {
            return Err(Error::Invalid("AM header too long".into()));
        }
        let ep = self.peers.get(peer).ok_or_else(|| Error::UnknownPeer(peer.clone()))?.ep;
        let r = unsafe {
            ffi::cx_am_send(ep, hdr.as_ptr() as *const c_void, hdr.len(), data.as_ptr() as *const c_void, data.len())
        };
        let st = unsafe { ffi::cx_is_err(r) };
        if st != 0 {
            return Err(ucx_err(st));
        }
        if !r.is_null() {
            self.am_pending.push((r, hdr, data));
        }
        Ok(())
    }

    /// Registers `addr..addr+len` (host or GPU memory) and packs its rkey.
    pub fn mem_map(&mut self, addr: u64, len: u64) -> Result<MemRegion> {
        let mut memh = std::ptr::null_mut();
        let st = unsafe { ffi::cx_mem_map(self.ctx, addr as *mut c_void, len as usize, &mut memh) };
        if st != 0 {
            return Err(ucx_err(st));
        }
        let mut buf = std::ptr::null_mut();
        let mut blen: usize = 0;
        let st = unsafe { ffi::cx_rkey_pack(self.ctx, memh, &mut buf, &mut blen) };
        if st != 0 {
            unsafe { ffi::cx_mem_unmap(self.ctx, memh) };
            return Err(ucx_err(st));
        }
        let rkey = unsafe { std::slice::from_raw_parts(buf as *const u8, blen) }.to_vec();
        unsafe { ffi::cx_rkey_buffer_release(buf) };
        let mem_type = unsafe { ffi::cx_mem_type(memh) };
        Ok(MemRegion { memh, addr, len, mem_type, rkey })
    }

    pub fn mem_unmap(&mut self, region: MemRegion) -> Result<()> {
        let st = unsafe { ffi::cx_mem_unmap(self.ctx, region.memh) };
        if st != 0 {
            return Err(ucx_err(st));
        }
        Ok(())
    }

    /// Unpacks a peer's packed rkey on the current endpoint to `peer`.
    pub fn rkey_unpack(&mut self, peer: &LogicalId, packed: &[u8]) -> Result<Rkey> {
        let ep = self.peers.get(peer).ok_or_else(|| Error::UnknownPeer(peer.clone()))?.ep;
        let mut rk = std::ptr::null_mut();
        let st = unsafe { ffi::cx_rkey_unpack(ep, packed.as_ptr() as *const c_void, &mut rk) };
        if st != 0 {
            return Err(ucx_err(st));
        }
        Ok(Rkey(rk))
    }

    /// One-sided get (remote -> `local`) or put (`local` -> remote).
    ///
    /// # Safety
    /// `local..local+len` must stay valid until the request completes, and
    /// `rkey` must have been unpacked on the current endpoint to `peer`.
    pub unsafe fn post_rma(&mut self, peer: &LogicalId, is_put: bool, local: u64, len: usize, raddr: u64, rkey: &Rkey, memh: *mut c_void) -> Result<Req> {
        let ep = self.peers.get(peer).ok_or_else(|| Error::UnknownPeer(peer.clone()))?.ep;
        req_from(ffi::cx_rma(ep, is_put as i32, local as *mut c_void, len, raddr, rkey.0, memh))
    }

    /// Flushes `peer`'s endpoint: completes when all RMA posted before it
    /// is remotely complete.
    pub fn flush(&mut self, peer: &LogicalId) -> Result<Req> {
        let ep = self.peers.get(peer).ok_or_else(|| Error::UnknownPeer(peer.clone()))?.ep;
        req_from(unsafe { ffi::cx_ep_flush(ep) })
    }

    // ------------------------------------------------------------ data path


    fn post(&mut self, is_recv: bool, peer: &LogicalId, corr: u64, ptr: *mut u8, len: usize, buf: Option<Vec<u8>>) -> Result<Op> {
        let mut imm_len = len;
        let (r, ep, generation) = if is_recv {
            let generation = self.peers.get(peer).map(|p| p.generation).unwrap_or(0);
            let r = unsafe { ffi::cx_tag_recv(self.ctx, ptr as *mut c_void, len, peer.tag(corr), u64::MAX, &mut imm_len) };
            (r, std::ptr::null_mut(), generation)
        } else {
            let p = self.peers.get(peer).ok_or_else(|| Error::UnknownPeer(peer.clone()))?;
            let (ep, generation) = (p.ep, p.generation);
            let r = unsafe { ffi::cx_tag_send(ep, ptr as *const c_void, len, self.me.tag(corr)) };
            (r, ep, generation)
        };
        let mut op = Op { req: r, is_recv, peer: peer.clone(), ep, generation, buf, len, done: None };
        let st = unsafe { ffi::cx_is_err(r) };
        if st != 0 {
            op.req = std::ptr::null_mut();
            op.done = Some(Err(ucx_err(st)));
        } else if r.is_null() {
            op.done = Some(Ok(imm_len));
        }
        Ok(op)
    }

    /// Posts a send of `buf` to `peer` under correlation id `corr`.
    pub fn post_send(&mut self, peer: &LogicalId, corr: u64, mut buf: Vec<u8>) -> Result<Op> {
        let (ptr, len) = (buf.as_mut_ptr(), buf.len());
        self.post(false, peer, corr, ptr, len, Some(buf))
    }

    /// Posts a receive of up to `buf.len()` bytes from `peer` (any
    /// incarnation) with correlation id `corr`.
    pub fn post_recv(&mut self, peer: &LogicalId, corr: u64, mut buf: Vec<u8>) -> Result<Op> {
        let (ptr, len) = (buf.as_mut_ptr(), buf.len());
        self.post(true, peer, corr, ptr, len, Some(buf))
    }

    /// Send from caller-owned memory (host, or GPU memory when UCX was built
    /// with CUDA: UCX detects the memory type of the pointer).
    ///
    /// # Safety
    /// `ptr..ptr+len` must stay valid and unmodified until the op completes.
    pub unsafe fn post_send_ptr(&mut self, peer: &LogicalId, corr: u64, ptr: *const u8, len: usize) -> Result<Op> {
        self.post(false, peer, corr, ptr as *mut u8, len, None)
    }

    /// Receive into caller-owned memory.
    ///
    /// # Safety
    /// `ptr..ptr+len` must stay valid and untouched until the op completes.
    pub unsafe fn post_recv_ptr(&mut self, peer: &LogicalId, corr: u64, ptr: *mut u8, len: usize) -> Result<Op> {
        self.post(true, peer, corr, ptr, len, None)
    }

    /// Non-blocking completion check (does not progress).
    pub fn test(&mut self, op: &mut Op) -> bool {
        if op.done.is_some() {
            return true;
        }
        let mut len: usize = 0;
        let st = unsafe { ffi::cx_req_test(op.req, op.is_recv as i32, &mut len) };
        if st == unsafe { ffi::cx_inprogress() } {
            // A send on an endpoint that has since failed / been replaced is
            // completed by UCX with an error; a receive is worker-wide, so we
            // fail it ourselves when its peer is gone and has not come back.
            if op.is_recv && !self.peers.contains_key(&op.peer) && self.generations.get(&op.peer).copied().unwrap_or(0) > 0 {
                self.cancel(op);
                op.done = Some(Err(Error::PeerFailed(op.peer.clone())));
                return true;
            }
            return false;
        }
        unsafe { ffi::cx_req_free(op.req) };
        op.req = std::ptr::null_mut();
        op.done = Some(if st == 0 {
            Ok(if op.is_recv { len } else { op.len })
        } else if st == unsafe { ffi::cx_err_canceled() } {
            Err(Error::Canceled)
        } else if !op.is_recv {
            Err(Error::PeerFailed(op.peer.clone()))
        } else {
            Err(ucx_err(st))
        });
        true
    }

    fn cancel(&mut self, op: &mut Op) {
        if op.req.is_null() {
            return;
        }
        unsafe { ffi::cx_req_cancel(self.ctx, op.req) };
        // A canceled request completes on the next progress.
        let inprog = unsafe { ffi::cx_inprogress() };
        for _ in 0..1_000_000 {
            unsafe { ffi::cx_progress(self.ctx) };
            let st = unsafe { ffi::cx_req_test(op.req, op.is_recv as i32, std::ptr::null_mut()) };
            if st != inprog {
                break;
            }
        }
        unsafe { ffi::cx_req_free(op.req) };
        op.req = std::ptr::null_mut();
    }

    /// Progresses until `op` completes or `timeout` passes (then cancels it).
    /// Returns the buffer and the transferred length.
    pub fn wait(&mut self, mut op: Op, timeout: Option<Duration>) -> (Vec<u8>, Result<usize>) {
        let deadline = timeout.map(|t| Instant::now() + t);
        loop {
            self.progress();
            if self.test(&mut op) {
                break;
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                self.cancel(&mut op);
                op.done = Some(Err(Error::Timeout));
                break;
            }
        }
        let buf = op.buf.take().unwrap_or_default();
        (buf, op.done.take().unwrap())
    }

    /// Blocking send. The buffer is returned for reuse.
    pub fn send(&mut self, peer: &LogicalId, corr: u64, buf: Vec<u8>, timeout: Option<Duration>) -> (Vec<u8>, Result<usize>) {
        match self.post_send(peer, corr, buf) {
            Ok(op) => self.wait(op, timeout),
            Err(e) => (Vec::new(), Err(e)),
        }
    }

    /// Blocking receive into `buf`. Returns the buffer and received length.
    pub fn recv(&mut self, peer: &LogicalId, corr: u64, buf: Vec<u8>, timeout: Option<Duration>) -> (Vec<u8>, Result<usize>) {
        match self.post_recv(peer, corr, buf) {
            Ok(op) => self.wait(op, timeout),
            Err(e) => (Vec::new(), Err(e)),
        }
    }

    /// Waits until `peer` is bound (hello received / connect issued).
    pub fn wait_peer(&mut self, peer: &LogicalId, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while !self.peers.contains_key(peer) {
            self.progress();
            if Instant::now() >= deadline {
                return false;
            }
        }
        true
    }
}

fn ffi_now() -> u64 {
    now_ns()
}

impl Drop for Endpoint {
    fn drop(&mut self) {
        let eps: Vec<_> = self.peers.drain().map(|(_, p)| p.ep).collect();
        for ep in eps {
            self.close_ep(ep);
        }
        let anon: Vec<_> = self.anon.drain().collect();
        for ep in anon {
            self.close_ep(ep as *mut c_void);
        }
        let deadline = Instant::now() + Duration::from_secs(2);
        while (!self.closing.is_empty() || !self.hellos.is_empty() || !self.am_pending.is_empty()) && Instant::now() < deadline {
            unsafe { ffi::cx_progress(self.ctx) };
            self.reap();
        }
        unsafe { ffi::cx_destroy(self.ctx) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_id_roundtrip() {
        let id = LogicalId::new("llama|8b", 2, 1, 3);
        assert_eq!(LogicalId::decode(&id.encode()), Some(id.clone()));
        assert!(LogicalId::decode("bad").is_none());
    }

    #[test]
    fn tag_layout() {
        let id = LogicalId::new("m", 0, 0, 0);
        let t = id.tag(0x1234);
        assert_eq!(t & CORR_MASK, 0x1234);
        assert_eq!(t >> CORR_BITS, id.tag_prefix());
        assert_eq!(id.tag(CORR_MASK + 5) & CORR_MASK, 4);
    }

    #[test]
    fn replicas_get_distinct_prefixes() {
        let mut seen = HashSet::new();
        for stage in 0..4 {
            for replica in 0..8 {
                seen.insert(LogicalId::new("llama-8b", stage, 0, replica).tag_prefix());
            }
        }
        assert_eq!(seen.len(), 32);
    }
}
