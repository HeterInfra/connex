//! cxbench: failure-detection, rejoin and steady-state benchmarks for connex-rs.
//!
//! Two roles, each a process:
//!
//! * `survivor` listens (UCX + a plain TCP control socket), streams steps to
//!   the victim and measures. It injects faults by writing one byte to the
//!   victim's control socket (`K` = SIGKILL itself, `S` = SIGSTOP itself), so
//!   the fault timestamp and the detection timestamp come from the same
//!   monotonic clock even when the victim is on another host. The one-way
//!   control-message latency (tens of us) is included in the numbers.
//! * `victim` connects to the survivor under a fixed logical id, receives
//!   steps, verifies them and answers each with a checksum (or echoes them in
//!   latency mode). A shell loop restarts it after it dies, which is the
//!   same-identity rejoin.
//!
//! Output: one JSON object per line on stdout.

use connex_rs::{now_ns, Endpoint, Event, LogicalId, Op};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

const MODE_CRC: u64 = 0;
const MODE_ECHO: u64 = 1;

fn usage() -> ! {
    eprintln!(
        "usage:\n  cxbench survivor [--listen IP] [--port P] [--ctl-port P] [--peer m,s,sh,r] [--reps N] [--size B]\n\
         \x20   [--fault kill|stop|none] [--idle] [--depth D] [--warmup-ms MS] [--detect-timeout-ms MS] [--duration-s S]\n\
         \x20   [--lat-sizes 16384,1048576,...] [--lat-iters N]\n\
         \x20 cxbench victim --connect IP:PORT --ctl IP:PORT [--id m,s,sh,r] [--burn N]\n\
         \x20 cxbench burn --threads N --seconds S"
    );
    std::process::exit(2)
}

struct Args(HashMap<String, String>);

impl Args {
    fn parse(v: &[String]) -> Args {
        let mut m = HashMap::new();
        let mut i = 0;
        while i < v.len() {
            let k = v[i].trim_start_matches("--").to_string();
            if i + 1 < v.len() && !v[i + 1].starts_with("--") {
                m.insert(k, v[i + 1].clone());
                i += 2;
            } else {
                m.insert(k, "1".into());
                i += 1;
            }
        }
        Args(m)
    }
    fn get(&self, k: &str, d: &str) -> String {
        self.0.get(k).cloned().unwrap_or_else(|| d.to_string())
    }
    fn num(&self, k: &str, d: u64) -> u64 {
        self.0.get(k).map(|s| s.parse().unwrap_or_else(|_| usage())).unwrap_or(d)
    }
    fn has(&self, k: &str) -> bool {
        self.0.contains_key(k)
    }
}

fn parse_id(s: &str) -> LogicalId {
    let p: Vec<&str> = s.split(',').collect();
    if p.len() != 4 {
        usage()
    }
    LogicalId::new(p[0], p[1].parse().unwrap(), p[2].parse().unwrap(), p[3].parse().unwrap())
}

fn split_hp(s: &str) -> (String, u16) {
    let (h, p) = s.rsplit_once(':').unwrap_or_else(|| usage());
    (h.to_string(), p.parse().unwrap_or_else(|_| usage()))
}

// ------------------------------------------------------------------ payload

fn fill(buf: &mut [u8], step: u64) {
    let seed = step.wrapping_mul(0x9e3779b97f4a7c15);
    for (i, c) in buf.chunks_mut(8).enumerate() {
        let v = (seed ^ (i as u64)).to_le_bytes();
        c.copy_from_slice(&v[..c.len()]);
    }
}

fn checksum(buf: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for c in buf.chunks(8) {
        let mut w = [0u8; 8];
        w[..c.len()].copy_from_slice(c);
        h = (h ^ u64::from_le_bytes(w)).wrapping_mul(0x100000001b3);
    }
    h ^ buf.len() as u64
}

fn proc_stat() -> (usize, u64) {
    let fds = std::fs::read_dir("/proc/self/fd").map(|d| d.count()).unwrap_or(0);
    let rss_kb = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|x| x.parse().ok()))
        })
        .unwrap_or(0);
    (fds, rss_kb)
}

fn ucx_tls() -> String {
    std::env::var("UCX_TLS").unwrap_or_else(|_| "default".into())
}

/// --hb-ms I --hb-timeout-ms T enables the app-level heartbeat.
fn apply_hb(ep: &mut Endpoint, a: &Args) {
    let iv = a.num("hb-ms", 0);
    if iv > 0 {
        ep.set_heartbeat(Some(connex_rs::Heartbeat {
            interval: Duration::from_millis(iv),
            timeout: Duration::from_millis(a.num("hb-timeout-ms", iv * 6)),
        }));
    }
}

// ------------------------------------------------------------------ victim

fn victim(a: &Args) {
    let me = parse_id(&a.get("id", "bench,1,0,0"));
    let survivor = parse_id(&a.get("peer", "bench,0,0,0"));
    let (uh, up) = split_hp(&a.get("connect", ""));
    let (ch, cp) = split_hp(&a.get("ctl", ""));
    let burn = a.num("burn", 0);
    for _ in 0..burn {
        std::thread::spawn(|| loop {
            std::hint::black_box(0u64);
        });
    }
    // The survivor may still be initialising UCX: retry for up to 30 s.
    let t0 = Instant::now();
    let mut ctl = loop {
        match TcpStream::connect((ch.as_str(), cp)) {
            Ok(s) => break s,
            Err(e) if t0.elapsed() > Duration::from_secs(30) => panic!("ctl connect: {e}"),
            Err(_) => std::thread::sleep(Duration::from_millis(20)),
        }
    };
    ctl.set_nodelay(true).ok();
    writeln!(ctl, "{}", std::process::id()).ok();
    // Control thread: acts on a fault command the moment it arrives, so the
    // injection delay does not depend on what the main loop is doing (a
    // 64 MB verify takes ~100 ms). Local runs bypass this: the survivor
    // signals the pid directly.
    std::thread::spawn(move || {
        let mut b = [0u8; 1];
        loop {
            match ctl.read(&mut b) {
                Ok(1) => match b[0] {
                    b'K' => unsafe {
                        libc::kill(libc::getpid(), libc::SIGKILL);
                    },
                    b'S' => unsafe {
                        libc::kill(libc::getpid(), libc::SIGSTOP);
                    },
                    b'Q' => std::process::exit(0),
                    _ => {}
                },
                _ => std::process::exit(0), // survivor closed the control socket
            }
        }
    });

    let mut ep = Endpoint::new(me).expect("endpoint");
    apply_hb(&mut ep, a);
    ep.connect(&survivor, &uh, up).expect("connect");
    // Start message: (first step, mode, size).
    let (hdr, r) = ep.recv(&survivor, 0, vec![0u8; 24], Some(Duration::from_secs(30)));
    if r.is_err() {
        eprintln!("victim: no start message: {r:?}");
        std::process::exit(1);
    }
    let w = |i: usize| u64::from_le_bytes(hdr[i * 8..i * 8 + 8].try_into().unwrap());
    let (mut step, mode, size) = (w(0), w(1), w(2) as usize);
    let mut rbuf = vec![0u8; size];
    let mut expect = vec![0u8; size];
    loop {
        let (b, r) = ep.recv(&survivor, step, rbuf, None);
        rbuf = b;
        match r {
            Ok(n) if mode == MODE_ECHO => {
                let out = rbuf[..n].to_vec();
                let (_, r) = ep.send(&survivor, step, out, Some(Duration::from_secs(10)));
                if r.is_err() {
                    std::process::exit(3);
                }
            }
            Ok(n) => {
                fill(&mut expect[..n], step);
                let ok = expect[..n] == rbuf[..n];
                let cs = if ok { checksum(&rbuf[..n]) } else { 0 };
                let (_, r) = ep.send(&survivor, step, cs.to_le_bytes().to_vec(), Some(Duration::from_secs(10)));
                if r.is_err() {
                    std::process::exit(3);
                }
            }
            Err(e) => {
                eprintln!("victim: recv failed: {e}");
                std::process::exit(3);
            }
        }
        step += 1;
    }
}

// ------------------------------------------------------------------ survivor

struct Victim {
    ctl: TcpStream,
    pid: u32,
}

fn accept_ctl(l: &TcpListener) -> Option<Victim> {
    match l.accept() {
        Ok((mut s, _)) => {
            s.set_nodelay(true).ok();
            s.set_nonblocking(false).ok();
            s.set_read_timeout(Some(Duration::from_secs(5))).ok();
            let mut line = Vec::new();
            let mut b = [0u8; 1];
            while s.read(&mut b).map(|n| n == 1).unwrap_or(false) && b[0] != b'\n' {
                line.push(b[0]);
            }
            let pid = String::from_utf8_lossy(&line).trim().parse().unwrap_or(0);
            Some(Victim { ctl: s, pid })
        }
        Err(_) => None,
    }
}

/// Waits for `peer` to (re)join and returns the join timestamp.
fn wait_join(ep: &mut Endpoint, peer: &LogicalId, timeout: Duration) -> Option<u64> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Some(ev) = ep.wait_event(Duration::from_millis(50)) {
            if let Event::PeerJoined { peer: p, ts_ns, .. } = &ev {
                if p == peer {
                    return Some(*ts_ns);
                }
            }
        }
    }
    None
}

fn start_msg(ep: &mut Endpoint, peer: &LogicalId, first: u64, mode: u64, size: u64) -> bool {
    let mut h = Vec::with_capacity(24);
    for v in [first, mode, size] {
        h.extend_from_slice(&v.to_le_bytes());
    }
    ep.send(peer, 0, h, Some(Duration::from_secs(10))).1.is_ok()
}

/// One step: send pattern(step), receive its checksum. Returns Ok(true) if
/// the checksum matched.
struct StepIo {
    sbuf: Vec<u8>,
}

fn survivor(a: &Args) {
    let me = parse_id(&a.get("id", "bench,0,0,0"));
    let peer = parse_id(&a.get("peer", "bench,1,0,0"));
    let reps = a.num("reps", 20);
    let size = a.num("size", 1 << 20) as usize;
    let fault = a.get("fault", "kill");
    let idle = a.has("idle");
    // --local: victim runs on this host; inject faults with kill(2) directly.
    let local = a.has("local");
    let depth = a.num("depth", 2).max(1) as usize;
    let warmup = Duration::from_millis(a.num("warmup-ms", 300));
    let detect_timeout = Duration::from_millis(a.num("detect-timeout-ms", 10_000));
    let join_timeout = Duration::from_millis(a.num("join-timeout-ms", 60_000));
    let duration = a.num("duration-s", 0);
    let lat_sizes: Vec<usize> = a
        .get("lat-sizes", "")
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    let lat_iters = a.num("lat-iters", 1000);

    let mut ep = Endpoint::new(me).expect("endpoint");
    apply_hb(&mut ep, a);
    let port = ep.listen(&a.get("listen", "0.0.0.0"), a.num("port", 13337) as u16).expect("listen");
    let ctl_l = TcpListener::bind(("0.0.0.0", a.num("ctl-port", 13338) as u16)).expect("ctl bind");
    ctl_l.set_nonblocking(true).unwrap();
    println!("{{\"type\":\"ready\",\"port\":{port},\"tls\":\"{}\"}}", ucx_tls());

    // ---------------- steady-state latency / bandwidth (echo mode)
    if !lat_sizes.is_empty() {
        if wait_join(&mut ep, &peer, join_timeout).is_none() {
            println!("{{\"type\":\"error\",\"what\":\"no victim\"}}");
            return;
        }
        let maxs = *lat_sizes.iter().max().unwrap();
        let _victim = loop {
            if let Some(v) = accept_ctl(&ctl_l) {
                break v;
            }
            ep.progress();
        };
        let mut corr = 1u64;
        assert!(start_msg(&mut ep, &peer, corr, MODE_ECHO, maxs as u64));
        for &s in &lat_sizes {
            let iters = if s >= (8 << 20) { lat_iters.min(200) } else { lat_iters };
            let mut buf = vec![0u8; s];
            fill(&mut buf, 7);
            let mut rbuf = vec![0u8; s];
            let mut samples = Vec::with_capacity(iters as usize);
            for i in 0..iters + 20 {
                let t0 = Instant::now();
                let rop = ep.post_recv(&peer, corr, rbuf).unwrap();
                let (b, r) = ep.send(&peer, corr, buf, None);
                buf = b;
                r.expect("send");
                let (b, r) = ep.wait(rop, Some(Duration::from_secs(10)));
                rbuf = b;
                r.expect("recv");
                if i >= 20 {
                    samples.push(t0.elapsed().as_nanos() as f64 / 2000.0);
                }
                corr += 1;
            }
            assert!(rbuf == buf, "echo mismatch");
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = samples[samples.len() / 2];
            let p99 = samples[samples.len() * 99 / 100];
            let gbps = s as f64 / (p50 * 1e-6) / 1e9;
            println!(
                "{{\"type\":\"lat\",\"tls\":\"{}\",\"size\":{s},\"iters\":{iters},\"half_rtt_p50_us\":{p50:.2},\"half_rtt_p99_us\":{p99:.2},\"GBps_p50\":{gbps:.2}}}",
                ucx_tls()
            );
        }
        return;
    }

    // ---------------- failure detection + rejoin cycles
    let mut step: u64 = 1;
    let mut io = StepIo { sbuf: vec![0u8; size] };
    let mut cycle = 0u64;
    let mut false_positives = 0u64;
    let t_begin = Instant::now();
    let mut join_ts = match wait_join(&mut ep, &peer, join_timeout) {
        Some(t) => t,
        None => {
            println!("{{\"type\":\"error\",\"what\":\"no victim\"}}");
            return;
        }
    };
    let mut detect_ts: Option<u64> = None;
    loop {
        // Bind the control socket of this incarnation.
        let mut victim = loop {
            if let Some(v) = accept_ctl(&ctl_l) {
                break v;
            }
            ep.progress();
        };
        if !start_msg(&mut ep, &peer, step, MODE_CRC, size as u64) {
            println!("{{\"type\":\"error\",\"what\":\"start message failed\"}}");
            return;
        }
        // Stream until warmup passes, then inject the fault.
        let cycle_start = Instant::now();
        let run_for = if duration > 0 && reps == 0 { Duration::from_secs(duration) } else { warmup };
        let mut first_ok: Option<u64> = None;
        let mut steps_ok = 0u64;
        let mut bad = 0u64;
        let mut inflight: std::collections::VecDeque<(u64, Op, Op)> = Default::default();
        let mut kill_ts: Option<u64> = None;
        let mut data_err_ts: Option<u64> = None;
        let mut fail_ev: Option<(u64, i32, String)> = None;
        let mut faulted = false;
        loop {
            // Inject the fault once.
            if !faulted && cycle_start.elapsed() >= run_for {
                if reps == 0 {
                    break; // pure streaming run (false-positive test) finished
                }
                if idle {
                    // drain in-flight steps, then stop traffic before the fault
                    while let Some((_, s, r)) = inflight.pop_front() {
                        let _ = ep.wait(s, Some(Duration::from_secs(10)));
                        let _ = ep.wait(r, Some(Duration::from_secs(10)));
                    }
                }
                let c = if fault == "stop" { b'S' } else { b'K' };
                if local && victim.pid > 0 {
                    // same host: signal the pid directly (exact fault time)
                    let sig = if fault == "stop" { libc::SIGSTOP } else { libc::SIGKILL };
                    kill_ts = Some(now_ns());
                    unsafe {
                        libc::kill(victim.pid as i32, sig);
                    }
                } else {
                    kill_ts = Some(now_ns());
                    let _ = victim.ctl.write_all(&[c]);
                }
                faulted = true;
            }
            // Keep the pipeline full unless idle-after-fault.
            let streaming = !(faulted && idle) && data_err_ts.is_none();
            while streaming && inflight.len() < depth {
                fill(&mut io.sbuf, step);
                let expect = checksum(&io.sbuf);
                let r = ep.post_recv(&peer, step, vec![0u8; 8]);
                let s = ep.post_send(&peer, step, io.sbuf.clone());
                match (s, r) {
                    (Ok(s), Ok(r)) => inflight.push_back((expect, s, r)),
                    _ => {
                        data_err_ts.get_or_insert(now_ns());
                        break;
                    }
                }
                step += 1;
            }
            ep.progress();
            while let Some(ev) = ep.poll_event() {
                if let Event::PeerFailed { peer: Some(p), ts_ns, status, reason } = &ev {
                    if *p == peer {
                        if !faulted {
                            false_positives += 1;
                        }
                        fail_ev.get_or_insert((*ts_ns, *status, reason.clone()));
                    }
                }
            }
            // Complete steps in order.
            if let Some((expect, s, r)) = inflight.front_mut() {
                let (sd, rd) = (ep.test(s), ep.test(r));
                if sd && rd || fail_ev.is_some() {
                    let (expect, s, r) = inflight.pop_front().unwrap();
                    let (_, sr) = ep.wait(s, Some(Duration::from_millis(1)));
                    let (rb, rr) = ep.wait(r, Some(Duration::from_millis(1)));
                    match (sr, rr) {
                        (Ok(_), Ok(8)) => {
                            let got = u64::from_le_bytes(rb[..8].try_into().unwrap());
                            if got != expect {
                                bad += 1;
                            }
                            steps_ok += 1;
                            first_ok.get_or_insert(now_ns());
                        }
                        _ => {
                            data_err_ts.get_or_insert(now_ns());
                        }
                    }
                } else {
                    let _ = expect;
                }
            }
            if faulted && fail_ev.is_some() && inflight.is_empty() {
                break;
            }
            if faulted && kill_ts.is_some_and(|k| now_ns() - k > detect_timeout.as_nanos() as u64) {
                break;
            }
            if !faulted && fail_ev.is_some() {
                break; // false positive (or victim died on its own)
            }
        }
        // Drain leftovers (cancel).
        while let Some((_, s, r)) = inflight.pop_front() {
            let _ = ep.wait(s, Some(Duration::from_millis(1)));
            let _ = ep.wait(r, Some(Duration::from_millis(1)));
        }
        let (fds, rss) = proc_stat();
        let st = ep.stats();
        let resume_us = match (detect_ts, first_ok) {
            (Some(d), Some(f)) if f > d => Some((f - d) / 1000),
            _ => None,
        };
        let join_to_resume_us = first_ok.map(|f| f.saturating_sub(join_ts) / 1000);
        let detect_us = match (kill_ts, &fail_ev) {
            (Some(k), Some((t, _, _))) => Some(t.saturating_sub(k) / 1000),
            _ => None,
        };
        let data_err_us = match (kill_ts, data_err_ts) {
            (Some(k), Some(t)) => Some(t.saturating_sub(k) / 1000),
            _ => None,
        };
        let opt = |v: Option<u64>| v.map(|x| x.to_string()).unwrap_or("null".into());
        println!(
            "{{\"type\":\"cycle\",\"cycle\":{cycle},\"tls\":\"{}\",\"fault\":\"{fault}\",\"idle\":{idle},\"size\":{size},\"victim_pid\":{},\
\"detect_us\":{},\"data_err_us\":{},\"reason\":\"{}\",\"steps_ok\":{steps_ok},\"bad_checksums\":{bad},\
\"detect_to_resume_us\":{},\"join_to_resume_us\":{},\"false_positives\":{false_positives},\
\"fds\":{fds},\"rss_kb\":{rss},\"eps_created\":{},\"eps_closed\":{},\"closes_pending\":{},\"live_peers\":{},\"anon_eps\":{},\"dropped_events\":{}}}",
            ucx_tls(),
            victim.pid,
            opt(detect_us),
            opt(data_err_us),
            fail_ev.as_ref().map(|f| f.2.clone()).unwrap_or_default(),
            opt(resume_us),
            opt(join_to_resume_us),
            st.eps_created,
            st.eps_closed,
            st.closes_pending,
            st.live_peers,
            st.anon_eps,
            st.dropped_events
        );
        let _ = std::io::stdout().flush();
        if fault == "stop" && victim.pid > 0 && local {
            // Hung peer (SIGSTOP) is left to us to clean up when on this host.
            unsafe {
                libc::kill(victim.pid as i32, libc::SIGKILL);
            }
        }
        cycle += 1;
        if reps == 0 || cycle >= reps {
            let _ = victim.ctl.write_all(b"Q");
            println!(
                "{{\"type\":\"done\",\"cycles\":{cycle},\"false_positives\":{false_positives},\"elapsed_s\":{:.1}}}",
                t_begin.elapsed().as_secs_f64()
            );
            // let the victim see Q before we tear down
            let t = Instant::now();
            while t.elapsed() < Duration::from_millis(300) {
                ep.progress();
            }
            return;
        }
        detect_ts = fail_ev.as_ref().map(|f| f.0).or(Some(now_ns()));
        drop(victim);
        join_ts = match wait_join(&mut ep, &peer, join_timeout) {
            Some(t) => t,
            None => {
                println!("{{\"type\":\"error\",\"what\":\"victim did not rejoin\",\"cycle\":{cycle}}}");
                return;
            }
        };
        if !ep.is_connected(&peer) {
            // joined then failed again before we looked; wait once more
            join_ts = wait_join(&mut ep, &peer, join_timeout).unwrap_or(join_ts);
        }
    }
}

fn burn(a: &Args) {
    let n = a.num("threads", 1);
    let secs = a.num("seconds", 60);
    let hs: Vec<_> = (0..n)
        .map(|_| {
            std::thread::spawn(move || {
                let t = Instant::now();
                let mut x = 0u64;
                while t.elapsed().as_secs() < secs {
                    for _ in 0..100_000 {
                        x = std::hint::black_box(x.wrapping_mul(31).wrapping_add(7));
                    }
                }
            })
        })
        .collect();
    for h in hs {
        h.join().ok();
    }
}

fn main() {
    let v: Vec<String> = std::env::args().collect();
    if v.len() < 2 {
        usage()
    }
    let a = Args::parse(&v[2..]);
    match v[1].as_str() {
        "survivor" => survivor(&a),
        "victim" => victim(&a),
        "burn" => burn(&a),
        _ => usage(),
    }
}
