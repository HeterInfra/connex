//! Engine tests: real UCX on loopback ($CONNEX_RS_TEST_TLS, default tcp).
use connex_rs::engine::{Engine, EngineConfig, Target};
use connex_rs::{Error, Event, LogicalId};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn setup() -> String {
    let tls = std::env::var("CONNEX_RS_TEST_TLS").unwrap_or_else(|_| "tcp".into());
    std::env::set_var("UCX_TLS", tls);
    std::env::set_var("UCX_WARN_UNUSED_ENV_VARS", "n");
    std::env::set_var("UCX_TCP_CM_REUSEADDR", "y");
    std::env::var("CONNEX_RS_TEST_IP").unwrap_or_else(|_| "127.0.0.1".into())
}

fn engine(ip: &str, stage: u32, port: u16) -> Engine {
    let mut c = EngineConfig::new(LogicalId::new("t", stage, 0, 0), ip, port);
    c.spin = Duration::from_micros(200);
    Engine::start(c).expect("engine")
}

fn fill(e: &Engine, ptr: u64, n: usize, seed: u8) -> Vec<u8> {
    let _ = e;
    let v: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_mul(13).wrapping_add(seed)).collect();
    unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), ptr as *mut u8, n) };
    v
}

fn read(ptr: u64, n: usize) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(ptr as *const u8, n) }.to_vec()
}

const T: Option<Duration> = Some(Duration::from_secs(5));

#[test]
fn get_put_integrity() {
    let ip = setup();
    let a = engine(&ip, 0, 0);
    let b = engine(&ip, 1, 0);
    let n = 4 << 20;
    let src = a.alloc_managed(n).unwrap();
    let dst = b.alloc_managed(n).unwrap();
    for (i, &sz) in [64usize, 4096, 16 << 10, 1 << 20, n].iter().enumerate() {
        let want = fill(&a, src, sz, i as u8);
        b.get(a.local_url(), dst, src, sz as u64, T).unwrap();
        assert_eq!(read(dst, sz), want, "get size {sz}");
        // offset inside the region
        b.get(a.local_url(), dst + 8, src + 16, (sz / 2) as u64, T).unwrap();
        assert_eq!(read(dst + 8, sz / 2), want[16..16 + sz / 2].to_vec());
    }
    let want = fill(&b, dst, n, 99);
    b.put(a.local_url(), dst, src, n as u64, T).unwrap();
    assert_eq!(read(src, n), want, "put");
    let s = b.stats();
    assert_eq!(s.rkey_fetches, 1, "one rkey fetch for the whole region");
    a.free_managed(src).unwrap();
    b.free_managed(dst).unwrap();
}

#[test]
fn concurrent_readers() {
    let ip = setup();
    let a = engine(&ip, 0, 0);
    let b = Arc::new(engine(&ip, 1, 0));
    let n = 64 << 10;
    let src = a.alloc_managed(n * 8).unwrap();
    let want = fill(&a, src, n * 8, 7);
    let url = a.local_url().to_string();
    let hs: Vec<_> = (0..8)
        .map(|t| {
            let b = b.clone();
            let url = url.clone();
            let want = want.clone();
            std::thread::spawn(move || {
                let dst = b.alloc_managed(n).unwrap();
                for _ in 0..200 {
                    b.get(&url, dst, src + (t * n) as u64, n as u64, T).unwrap();
                    assert_eq!(read(dst, n), want[t * n..(t + 1) * n].to_vec());
                }
                b.free_managed(dst).unwrap();
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
}

#[test]
fn unregister_invalidates_cached_rkey() {
    let ip = setup();
    let a = engine(&ip, 0, 0);
    let b = engine(&ip, 1, 0);
    let n = 1 << 20;
    let dst = b.alloc_managed(n).unwrap();
    // Same address registered, read, unregistered, re-registered: the reader
    // must fetch the new key (a stale key would kill the connection).
    let mut buf = vec![0u8; n];
    let p = buf.as_mut_ptr() as u64;
    for round in 0..20u8 {
        a.register(p, n as u64).unwrap();
        let want = fill(&a, p, n, round);
        b.get(a.local_url(), dst, p, n as u64, T).unwrap();
        assert_eq!(read(dst, n), want);
        a.unregister(p).unwrap();
    }
    assert_eq!(b.stats().rkey_fetches, 20);
    assert_eq!(a.stats().invalidations, 20);
    assert_eq!(a.stats().regions, 0);
    // No failure events along the way.
    while let Some(ev) = b.wait_event(Some(Duration::from_millis(10))) {
        assert!(!matches!(ev, Event::PeerFailed { .. }), "{ev:?}");
    }
}

#[test]
fn unknown_region_fails_cleanly() {
    let ip = setup();
    let a = engine(&ip, 0, 0);
    let b = engine(&ip, 1, 0);
    let dst = b.alloc_managed(4096).unwrap();
    let r = b.get(a.local_url(), dst, 0x1000, 4096, T);
    assert!(matches!(r, Err(Error::Invalid(_))), "{r:?}");
    // The connection is still good.
    let src = a.alloc_managed(4096).unwrap();
    let want = fill(&a, src, 4096, 3);
    b.get(a.local_url(), dst, src, 4096, T).unwrap();
    assert_eq!(read(dst, 4096), want);
}

#[test]
fn put_signal_orders_data_before_message() {
    let ip = setup();
    let a = engine(&ip, 0, 0);
    let b = engine(&ip, 1, 0);
    let n = 1 << 20;
    let src = a.alloc_managed(n).unwrap();
    let dst = b.alloc_managed(n).unwrap();
    for i in 0..50u8 {
        let want = fill(&a, src, n, i);
        let c = a.put_signal(b.local_url(), src, dst, n as u64, vec![i], T).unwrap();
        c.wait(None).unwrap().unwrap();
        let (from, msg) = b.recv_msg(Some(Duration::from_secs(5))).expect("signal");
        assert_eq!(from, *a.id());
        assert_eq!(msg, vec![i]);
        assert_eq!(read(dst, n), want, "data visible when the signal arrives");
    }
    // plain message back to a peer known by id
    b.send_msg(Target::Id(a.id().clone()), b"pong".to_vec()).unwrap();
    assert_eq!(a.recv_msg(Some(Duration::from_secs(5))).unwrap().1, b"pong".to_vec());
}

#[test]
fn peer_failure_and_same_identity_rejoin() {
    let ip = setup();
    let b = engine(&ip, 1, 0);
    let a = engine(&ip, 0, 0);
    let port = a.port();
    let url = a.local_url().to_string();
    let n = 1 << 16;
    let src = a.alloc_managed(n).unwrap();
    let dst = b.alloc_managed(n).unwrap();
    fill(&a, src, n, 1);
    b.get(&url, dst, src, n as u64, T).unwrap();
    // "kill" a: shutdown closes its endpoints; b must see a failure event.
    let t0 = Instant::now();
    a.shutdown();
    let mut failed = false;
    while t0.elapsed() < Duration::from_secs(5) {
        if let Some(Event::PeerFailed { peer: Some(p), .. }) = b.wait_event(Some(Duration::from_millis(100))) {
            assert_eq!(p, LogicalId::new("t", 0, 0, 0));
            failed = true;
            break;
        }
    }
    assert!(failed, "no PeerFailed after peer shutdown");
    let detect = t0.elapsed();
    assert!(detect < Duration::from_secs(2), "slow detection {detect:?}");
    // Reads of the dead peer fail fast.
    let r = b.get(&url, dst, src, n as u64, Some(Duration::from_secs(2)));
    assert!(r.is_err());
    // Same identity, same port: b's reads work again without restarting b.
    let a2 = engine(&ip, 0, port);
    assert_eq!(a2.local_url(), url);
    let src2 = a2.alloc_managed(n).unwrap();
    let want = fill(&a2, src2, n, 42);
    let mut ok = false;
    for _ in 0..50 {
        if b.get(&url, dst, src2, n as u64, T).is_ok() {
            ok = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(ok, "read after rejoin");
    assert_eq!(read(dst, n), want);
}

#[test]
fn leak_check_many_reconnects() {
    let ip = setup();
    let b = engine(&ip, 1, 0);
    let dst = b.alloc_managed(4096).unwrap();
    let mut port = 0;
    for i in 0..30 {
        let a = engine(&ip, 0, port);
        port = a.port();
        let src = a.alloc_managed(4096).unwrap();
        let want = fill(&a, src, 4096, i as u8);
        let mut ok = false;
        for _ in 0..100 {
            if b.get(a.local_url(), dst, src, 4096, T).is_ok() {
                ok = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(ok, "cycle {i}");
        assert_eq!(read(dst, 4096), want);
        drop(a);
    }
    std::thread::sleep(Duration::from_millis(300));
    let s = b.stats();
    assert!(s.ep.live_peers <= 1, "{s:?}");
    assert!(s.remote_rkeys <= 1, "{s:?}");
    assert_eq!(s.inflight, 0);
    assert!(s.ep.eps_created - s.ep.eps_closed <= 1, "{s:?}");
}

#[test]
fn links_both_ways_are_one_endpoint_and_report_up() {
    let ip = setup();
    let a = engine(&ip, 0, 0);
    let b = engine(&ip, 1, 0);
    // b reads from a (b -> a), then a links to b: reuses the inbound link.
    let src = a.alloc_managed(4096).unwrap();
    let dst = b.alloc_managed(4096).unwrap();
    b.get(a.local_url(), dst, src, 4096, T).unwrap();
    a.connect(b.local_url()).unwrap();
    b.connect(a.local_url()).unwrap();
    b.get(a.local_url(), dst, src, 4096, T).unwrap();
    let mut up_a = false;
    let mut up_b = false;
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_millis(500) {
        if let Some(ev) = a.wait_event(Some(Duration::from_millis(20))) {
            assert!(!matches!(ev, Event::PeerFailed { .. }), "a: {ev:?}");
            up_a |= matches!(ev, Event::PeerUp { .. });
        }
        if let Some(ev) = b.wait_event(Some(Duration::from_millis(20))) {
            assert!(!matches!(ev, Event::PeerFailed { .. }), "b: {ev:?}");
            up_b |= matches!(ev, Event::PeerUp { .. });
        }
    }
    assert!(up_a && up_b);
    assert_eq!(a.stats().ep.eps_created, 1);
    assert_eq!(b.stats().ep.eps_created, 1);
    // A connect to a port nobody listens on fails without a PeerUp.
    let c = engine(&ip, 2, 0);
    let dead = connex_rs::engine::PeerUrl::format(&ip, 1, &LogicalId::new("t", 9, 0, 0));
    let _ = c.connect(&dead);
    let mut saw_up = false;
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_millis(500) {
        if let Some(Event::PeerUp { .. }) = c.wait_event(Some(Duration::from_millis(50))) {
            saw_up = true;
        }
    }
    assert!(!saw_up);
}
