//! Integration tests: real UCX endpoints in threads / processes on this host.
//! Transport: $CONNEX_RS_TEST_TLS (default "tcp"; e.g. "rc_mlx5" with
//! CONNEX_RS_TEST_IP set to the IPoIB address).
use connex_rs::{Endpoint, Event, Heartbeat, LogicalId};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

fn setup() -> String {
    let tls = std::env::var("CONNEX_RS_TEST_TLS").unwrap_or_else(|_| "tcp".into());
    std::env::set_var("UCX_TLS", tls);
    std::env::set_var("UCX_WARN_UNUSED_ENV_VARS", "n");
    std::env::var("CONNEX_RS_TEST_IP").unwrap_or_else(|_| "127.0.0.1".into())
}

fn id(stage: u32, replica: u32) -> LogicalId {
    LogicalId::new("test", stage, 0, replica)
}

fn pattern(n: usize, seed: u8) -> Vec<u8> {
    (0..n).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed)).collect()
}

const T: Option<Duration> = Some(Duration::from_secs(10));

/// Server thread: listens, reports its port, then serves `rounds` echo
/// rounds per incarnation of `client`, for `incarnations` incarnations.
/// Returns the events it saw.
fn echo_server(ip: String, client: LogicalId, incarnations: usize, rounds: u64, tx: mpsc::Sender<u16>) -> Vec<Event> {
    let mut ep = Endpoint::new(id(0, 0)).unwrap();
    tx.send(ep.listen(&ip, 0).unwrap()).unwrap();
    let mut events = Vec::new();
    for _ in 0..incarnations {
        // wait for (re)join
        loop {
            let ev = ep.wait_event(Duration::from_secs(20)).expect("no join");
            events.push(ev.clone());
            if matches!(ev, Event::PeerJoined { .. }) {
                break;
            }
        }
        for corr in 1..=rounds {
            let (buf, r) = ep.recv(&client, corr, vec![0u8; 1 << 20], T);
            let n = r.expect("server recv");
            let (_, r) = ep.send(&client, corr, buf[..n].to_vec(), T);
            r.expect("server send");
        }
    }
    // drain remaining events briefly
    let t = Instant::now();
    while t.elapsed() < Duration::from_millis(300) {
        if let Some(ev) = ep.poll_event() {
            events.push(ev);
        }
    }
    events
}

fn echo_rounds(ep: &mut Endpoint, server: &LogicalId, first: u64, rounds: u64, size: usize) {
    for corr in first..first + rounds {
        let data = pattern(size, corr as u8);
        let rop = ep.post_recv(server, corr, vec![0u8; size]).unwrap();
        let (_, r) = ep.send(server, corr, data.clone(), T);
        r.expect("client send");
        let (got, r) = ep.wait(rop, T);
        assert_eq!(r.expect("client recv"), size);
        assert_eq!(got, data, "payload mismatch at corr {corr}");
    }
}

#[test]
fn send_recv_integrity() {
    let ip = setup();
    let (tx, rx) = mpsc::channel();
    let client = id(1, 0);
    let c2 = client.clone();
    let ip2 = ip.clone();
    let srv = thread::spawn(move || echo_server(ip2, c2, 1, 20, tx));
    let port = rx.recv().unwrap();
    let mut ep = Endpoint::new(client).unwrap();
    ep.connect(&id(0, 0), &ip, port).unwrap();
    for (i, size) in [1usize, 4096, 65536, 1 << 20].iter().enumerate() {
        echo_rounds(&mut ep, &id(0, 0), 1 + 5 * i as u64, 5, *size);
    }
    drop(ep);
    let events = srv.join().unwrap();
    assert!(matches!(&events[0], Event::PeerJoined { rejoin: false, .. }));
}

/// A client endpoint goes away and a new one comes back under the same
/// logical id: the server keeps running, sees a rejoin and serves it.
#[test]
fn same_identity_rejoin_in_process() {
    let ip = setup();
    let (tx, rx) = mpsc::channel();
    let client = id(1, 1);
    let c2 = client.clone();
    let ip2 = ip.clone();
    let srv = thread::spawn(move || echo_server(ip2, c2, 3, 3, tx));
    let port = rx.recv().unwrap();
    for _ in 0..3 {
        let mut ep = Endpoint::new(client.clone()).unwrap();
        ep.connect(&id(0, 0), &ip, port).unwrap();
        echo_rounds(&mut ep, &id(0, 0), 1, 3, 4096);
        drop(ep);
    }
    let events = srv.join().unwrap();
    let joins: Vec<bool> = events
        .iter()
        .filter_map(|e| match e {
            Event::PeerJoined { rejoin, .. } => Some(*rejoin),
            _ => None,
        })
        .collect();
    assert_eq!(joins, vec![false, true, true]);
}

/// Client-side identity-aware reconnect: the server is replaced by a new
/// endpoint (new port, same logical id); the client reconnects and resumes.
#[test]
fn client_reconnect_to_replaced_server() {
    let ip = setup();
    let client = id(1, 2);
    let mut ep = Endpoint::new(client.clone()).unwrap();
    for gen in 0..2u64 {
        let (tx, rx) = mpsc::channel();
        let c2 = client.clone();
        let ip2 = ip.clone();
        let srv = thread::spawn(move || echo_server(ip2, c2, 1, 2, tx));
        let port = rx.recv().unwrap();
        ep.reconnect(&id(0, 0), &ip, port).unwrap();
        assert_eq!(ep.generation(&id(0, 0)), Some(gen + 1));
        echo_rounds(&mut ep, &id(0, 0), 1, 2, 65536);
        srv.join().unwrap();
    }
}

/// A peer that stops progressing (hung) is reported by the heartbeat, and a
/// healthy idle peer is not.
#[test]
fn heartbeat_detects_hung_peer_only() {
    let ip = setup();
    let (tx, rx) = mpsc::channel::<u16>();
    let (go_tx, go_rx) = mpsc::channel::<()>();
    let ip2 = ip.clone();
    let hb = Heartbeat { interval: Duration::from_millis(20), timeout: Duration::from_millis(150) };
    let peer = thread::spawn(move || {
        let mut ep = Endpoint::new(id(1, 3)).unwrap();
        ep.set_heartbeat(Some(hb));
        let port = rx.recv().unwrap();
        ep.connect(&id(0, 0), &ip2, port).unwrap();
        // healthy for 1 s
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(1) {
            ep.progress();
        }
        go_tx.send(()).unwrap();
        // hang: stop progressing for 1 s
        thread::sleep(Duration::from_secs(1));
    });
    let mut ep = Endpoint::new(id(0, 0)).unwrap();
    ep.set_heartbeat(Some(hb));
    tx.send(ep.listen(&ip, 0).unwrap()).unwrap();
    let mut hang_at = None;
    let mut failed = None;
    let t = Instant::now();
    while t.elapsed() < Duration::from_secs(5) && failed.is_none() {
        if hang_at.is_none() && go_rx.try_recv().is_ok() {
            hang_at = Some(Instant::now());
        }
        if let Some(Event::PeerFailed { reason, .. }) = ep.poll_event() {
            failed = Some((Instant::now(), reason));
        }
    }
    let (when, reason) = failed.expect("hung peer not detected");
    let hang_at = hang_at.expect("failure reported before the peer hung (false positive)");
    assert_eq!(reason, "heartbeat timeout");
    let dt = when - hang_at;
    assert!(dt < Duration::from_millis(600), "detection took {dt:?}");
    peer.join().unwrap();
}

/// End-to-end with real processes: cxbench survivor + a restarted victim,
/// SIGKILL faults, same-identity rejoin, payload checksums.
#[test]
fn sigkill_detect_and_rejoin_processes() {
    let ip = setup();
    let bin = env!("CARGO_BIN_EXE_cxbench");
    let port = 30000 + (std::process::id() % 10000) as u16;
    let mut surv = Command::new(bin)
        .args(["survivor", "--listen", &ip, "--port", &port.to_string(), "--ctl-port", &(port + 1).to_string()])
        .args(["--reps", "3", "--size", "65536", "--warmup-ms", "100", "--detect-timeout-ms", "3000"])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    thread::sleep(Duration::from_millis(300));
    let connect = format!("{ip}:{port}");
    let ctl = format!("{ip}:{}", port + 1);
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop2 = stop.clone();
    let looper = thread::spawn(move || {
        use std::sync::atomic::Ordering;
        for _ in 0..10 {
            let mut child = Command::new(bin)
                .args(["victim", "--connect", &connect, "--ctl", &ctl])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap();
            loop {
                if let Some(st) = child.try_wait().unwrap() {
                    if st.success() {
                        return;
                    }
                    break;
                }
                if stop2.load(Ordering::Relaxed) {
                    let _ = child.kill();
                    let _ = child.wait();
                    return;
                }
                thread::sleep(Duration::from_millis(10));
            }
        }
    });
    let out = surv.wait_with_output().unwrap();
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    looper.join().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    let cycles: Vec<&str> = text.lines().filter(|l| l.contains("\"type\":\"cycle\"")).collect();
    assert_eq!(cycles.len(), 3, "output: {text}");
    for c in &cycles {
        assert!(!c.contains("\"detect_us\":null"), "undetected kill: {c}");
        assert!(c.contains("\"bad_checksums\":0"), "{c}");
        assert!(c.contains("\"false_positives\":0"), "{c}");
    }
    assert!(text.contains("\"type\":\"done\""));
}
