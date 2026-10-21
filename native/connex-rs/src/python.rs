//! Python binding (feature "python", built with maturin).
//!
//! `connex_rs.Engine` is the data plane used by connex.pp (TransferEngine
//! surface, one-sided RMA, signals, peer events; GIL released while
//! waiting). `connex_rs.Endpoint` is the single-threaded step-1 endpoint
//! (tagged messages), kept for the failure-detection benchmarks.
//!
//! ```python
//! import connex_rs
//! ep = connex_rs.Endpoint("llama-8b", 0, 0, 0)
//! port = ep.listen("0.0.0.0", 0)
//! ep.connect(("llama-8b", 1, 0, 0), "10.0.0.2", 13337)
//! ep.send(("llama-8b", 1, 0, 0), corr=7, data=b"...")
//! data = ep.recv(("llama-8b", 1, 0, 0), corr=7, nbytes=1 << 20, timeout_ms=1000)
//! for ev in ep.poll_events(): ...   # {"type": "failed"|"joined", "peer": (...), "ts_ns": ...}
//! ```
//!
//! Peers are tuples (model, stage, shard, replica). Blocking calls hold the
//! GIL (the endpoint is single-threaded and not Send); use timeouts.
use crate::engine::{Completion, Engine, EngineConfig, PeerUrl, Target};
use crate::{now_ns, Endpoint, Error, Event, Heartbeat, LogicalId};
use std::sync::Arc;
use pyo3::exceptions::{PyConnectionError, PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::time::Duration;

type PeerT = (String, u32, u32, u32);

fn lid(p: &PeerT) -> LogicalId {
    LogicalId::new(&p.0, p.1, p.2, p.3)
}

fn tup(p: &LogicalId) -> PeerT {
    (p.model.clone(), p.stage, p.shard, p.replica)
}

fn to_py(e: Error) -> PyErr {
    match e {
        Error::Timeout => PyTimeoutError::new_err("timeout"),
        Error::PeerFailed(p) => PyConnectionError::new_err(format!("peer {p} failed")),
        Error::UnknownPeer(p) => PyConnectionError::new_err(format!("unknown peer {p}")),
        Error::Invalid(s) => PyValueError::new_err(s),
        e => PyRuntimeError::new_err(e.to_string()),
    }
}

fn ms(t: Option<u64>) -> Option<Duration> {
    t.map(Duration::from_millis)
}

#[pyclass(unsendable, name = "Endpoint")]
struct PyEndpoint {
    ep: Endpoint,
}

#[pymethods]
impl PyEndpoint {
    #[new]
    fn new(model: &str, stage: u32, shard: u32, replica: u32) -> PyResult<Self> {
        Ok(PyEndpoint { ep: Endpoint::new(LogicalId::new(model, stage, shard, replica)).map_err(to_py)? })
    }

    #[getter]
    fn id(&self) -> PeerT {
        tup(self.ep.id())
    }

    /// interval_ms = 0 disables the heartbeat.
    #[pyo3(signature = (interval_ms, timeout_ms=None))]
    fn set_heartbeat(&mut self, interval_ms: u64, timeout_ms: Option<u64>) {
        self.ep.set_heartbeat((interval_ms > 0).then(|| Heartbeat {
            interval: Duration::from_millis(interval_ms),
            timeout: Duration::from_millis(timeout_ms.unwrap_or(interval_ms * 6)),
        }));
    }

    fn listen(&mut self, ip: &str, port: u16) -> PyResult<u16> {
        self.ep.listen(ip, port).map_err(to_py)
    }

    fn connect(&mut self, peer: PeerT, ip: &str, port: u16) -> PyResult<()> {
        self.ep.connect(&lid(&peer), ip, port).map_err(to_py)
    }

    /// Rejoin/migrate: drop the old endpoint of `peer`, connect to its new address.
    fn reconnect(&mut self, peer: PeerT, ip: &str, port: u16) -> PyResult<()> {
        self.ep.reconnect(&lid(&peer), ip, port).map_err(to_py)
    }

    fn drop_peer(&mut self, peer: PeerT) {
        self.ep.drop_peer(&lid(&peer))
    }

    fn wait_peer(&mut self, peer: PeerT, timeout_ms: u64) -> bool {
        self.ep.wait_peer(&lid(&peer), Duration::from_millis(timeout_ms))
    }

    fn is_connected(&self, peer: PeerT) -> bool {
        self.ep.is_connected(&lid(&peer))
    }

    fn peers(&self) -> Vec<PeerT> {
        self.ep.peers().iter().map(tup).collect()
    }

    #[pyo3(signature = (peer, corr, data, timeout_ms=None))]
    fn send(&mut self, peer: PeerT, corr: u64, data: &[u8], timeout_ms: Option<u64>) -> PyResult<usize> {
        self.ep.send(&lid(&peer), corr, data.to_vec(), ms(timeout_ms)).1.map_err(to_py)
    }

    #[pyo3(signature = (peer, corr, nbytes, timeout_ms=None))]
    fn recv<'py>(&mut self, py: Python<'py>, peer: PeerT, corr: u64, nbytes: usize, timeout_ms: Option<u64>) -> PyResult<Bound<'py, PyBytes>> {
        let (buf, r) = self.ep.recv(&lid(&peer), corr, vec![0u8; nbytes], ms(timeout_ms));
        let n = r.map_err(to_py)?;
        Ok(PyBytes::new_bound(py, &buf[..n]))
    }

    /// Send `length` bytes at address `ptr` (host or, with CUDA-enabled UCX,
    /// device memory, e.g. tensor.data_ptr()).
    #[pyo3(signature = (peer, corr, ptr, length, timeout_ms=None))]
    fn send_ptr(&mut self, peer: PeerT, corr: u64, ptr: usize, length: usize, timeout_ms: Option<u64>) -> PyResult<usize> {
        let op = unsafe { self.ep.post_send_ptr(&lid(&peer), corr, ptr as *const u8, length) }.map_err(to_py)?;
        self.ep.wait(op, ms(timeout_ms)).1.map_err(to_py)
    }

    #[pyo3(signature = (peer, corr, ptr, length, timeout_ms=None))]
    fn recv_ptr(&mut self, peer: PeerT, corr: u64, ptr: usize, length: usize, timeout_ms: Option<u64>) -> PyResult<usize> {
        let op = unsafe { self.ep.post_recv_ptr(&lid(&peer), corr, ptr as *mut u8, length) }.map_err(to_py)?;
        self.ep.wait(op, ms(timeout_ms)).1.map_err(to_py)
    }

    fn progress(&mut self) -> u32 {
        self.ep.progress()
    }

    /// Drains queued events (progressing once).
    fn poll_events<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let mut out = Vec::new();
        while let Some(ev) = self.ep.poll_event() {
            out.push(event_dict(py, &ev)?);
        }
        Ok(out)
    }

    fn wait_event<'py>(&mut self, py: Python<'py>, timeout_ms: u64) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.ep.wait_event(Duration::from_millis(timeout_ms)) {
            Some(ev) => Ok(Some(event_dict(py, &ev)?)),
            None => Ok(None),
        }
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.ep.stats();
        let d = PyDict::new_bound(py);
        d.set_item("eps_created", s.eps_created)?;
        d.set_item("eps_closed", s.eps_closed)?;
        d.set_item("closes_pending", s.closes_pending)?;
        d.set_item("live_peers", s.live_peers)?;
        d.set_item("anon_eps", s.anon_eps)?;
        d.set_item("failures", s.failures)?;
        d.set_item("rejoins", s.rejoins)?;
        d.set_item("heartbeat_failures", s.heartbeat_failures)?;
        d.set_item("dropped_events", s.dropped_events)?;
        Ok(d)
    }
}

fn event_dict<'py>(py: Python<'py>, ev: &Event) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    match ev {
        Event::PeerJoined { peer, rejoin, ts_ns } => {
            d.set_item("type", "joined")?;
            d.set_item("peer", tup(peer))?;
            d.set_item("rejoin", *rejoin)?;
            d.set_item("ts_ns", *ts_ns)?;
        }
        Event::PeerFailed { peer, status, reason, ts_ns } => {
            d.set_item("type", "failed")?;
            d.set_item("peer", peer.as_ref().map(tup))?;
            d.set_item("status", *status)?;
            d.set_item("reason", reason)?;
            d.set_item("ts_ns", *ts_ns)?;
        }
        Event::PeerUp { peer, ts_ns } => {
            d.set_item("type", "up")?;
            d.set_item("peer", tup(peer))?;
            d.set_item("ts_ns", *ts_ns)?;
        }
        Event::Am { peer, ts_ns, .. } => {
            d.set_item("type", "am")?;
            d.set_item("peer", tup(peer))?;
            d.set_item("ts_ns", *ts_ns)?;
        }
    }
    Ok(d)
}

// ------------------------------------------------------------------ Engine

/// Return codes of the TransferEngine-style calls (Mooncake returns < 0 on
/// failure; the endpoint only checks the sign).
fn code(e: &Error) -> i64 {
    match e {
        Error::PeerFailed(_) | Error::UnknownPeer(_) => -2,
        Error::Timeout => -3,
        Error::Invalid(_) => -4,
        Error::Canceled => -5,
        Error::Ucx(..) => -1,
    }
}

/// Completion handle of an async RMA (`get_async`, `put_signal`).
#[pyclass(frozen, name = "Transfer")]
struct PyTransfer {
    c: Arc<Completion>,
}

#[pymethods]
impl PyTransfer {
    fn done(&self) -> bool {
        self.c.is_done()
    }

    /// Waits (GIL released). Returns True when done; raises on failure;
    /// False when `timeout_ms` passed first.
    #[pyo3(signature = (timeout_ms=None))]
    fn wait(&self, py: Python<'_>, timeout_ms: Option<u64>) -> PyResult<bool> {
        let c = self.c.clone();
        match py.allow_threads(move || c.wait(ms(timeout_ms))) {
            None => Ok(false),
            Some(Ok(())) => Ok(true),
            Some(Err(e)) => Err(to_py(e)),
        }
    }
}

/// Threaded data-plane engine: one progress thread owns UCX; every method
/// releases the GIL while it waits. Implements the TransferEngine surface
/// used by connex.pp (allocate/free/write/read managed buffers,
/// register/unregister memory, transfer_sync_read) plus one-sided writes,
/// signals and peer events.
#[pyclass(frozen, name = "Engine")]
struct PyEngine {
    e: Arc<Engine>,
}

fn target(peer: &Bound<'_, PyAny>) -> PyResult<Target> {
    if let Ok(url) = peer.extract::<String>() {
        return Ok(Target::Url(PeerUrl::parse(&url).map_err(to_py)?));
    }
    let p: PeerT = peer.extract()?;
    Ok(Target::Id(lid(&p)))
}

#[pymethods]
impl PyEngine {
    #[new]
    #[pyo3(signature = (model, stage, shard=0, replica=0, listen_ip="0.0.0.0", port=0, advertise_ip=None, heartbeat_ms=0, heartbeat_timeout_ms=None, spin_us=500))]
    #[allow(clippy::too_many_arguments)]
    fn new(py: Python<'_>, model: &str, stage: u32, shard: u32, replica: u32, listen_ip: &str, port: u16, advertise_ip: Option<String>, heartbeat_ms: u64, heartbeat_timeout_ms: Option<u64>, spin_us: u64) -> PyResult<Self> {
        let mut cfg = EngineConfig::new(LogicalId::new(model, stage, shard, replica), listen_ip, port);
        cfg.advertise_ip = advertise_ip;
        cfg.spin = Duration::from_micros(spin_us);
        cfg.heartbeat = (heartbeat_ms > 0).then(|| Heartbeat {
            interval: Duration::from_millis(heartbeat_ms),
            timeout: Duration::from_millis(heartbeat_timeout_ms.unwrap_or(heartbeat_ms * 3)),
        });
        let e = py.allow_threads(|| Engine::start(cfg)).map_err(to_py)?;
        Ok(PyEngine { e: Arc::new(e) })
    }

    #[getter]
    fn local_url(&self) -> String {
        self.e.local_url().to_string()
    }

    #[getter]
    fn id(&self) -> PeerT {
        tup(self.e.id())
    }

    #[getter]
    fn port(&self) -> u16 {
        self.e.port()
    }

    // ---- TransferEngine protocol

    fn allocate_managed_buffer(&self, py: Python<'_>, size: usize) -> PyResult<u64> {
        let e = self.e.clone();
        py.allow_threads(move || e.alloc_managed(size)).map_err(to_py)
    }

    #[pyo3(signature = (ptr, size=0))]
    fn free_managed_buffer(&self, py: Python<'_>, ptr: u64, size: usize) -> PyResult<()> {
        let _ = size;
        let e = self.e.clone();
        py.allow_threads(move || e.free_managed(ptr)).map_err(to_py)
    }

    /// Copies `data[:length]` to host address `ptr` (a managed buffer).
    fn write_bytes_to_buffer(&self, ptr: u64, data: &[u8], length: usize) -> PyResult<()> {
        if length > data.len() {
            return Err(PyValueError::new_err("length > len(data)"));
        }
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, length) };
        Ok(())
    }

    fn read_bytes_from_buffer<'py>(&self, py: Python<'py>, ptr: u64, length: usize) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, unsafe { std::slice::from_raw_parts(ptr as *const u8, length) })
    }

    /// One-sided read of `length` bytes at `src_ptr` in the peer at
    /// `remote_url` into local `dst_ptr`. Returns 0, or < 0 on failure
    /// (-2 peer failed, -3 timeout, -4 invalid, -1 other).
    #[pyo3(signature = (remote_url, dst_ptr, src_ptr, length, timeout_ms=None))]
    fn transfer_sync_read(&self, py: Python<'_>, remote_url: &str, dst_ptr: u64, src_ptr: u64, length: u64, timeout_ms: Option<u64>) -> i64 {
        let e = self.e.clone();
        match py.allow_threads(move || e.get(remote_url, dst_ptr, src_ptr, length, ms(timeout_ms))) {
            Ok(()) => 0,
            Err(err) => {
                if std::env::var_os("CONNEX_RS_DEBUG").is_some() {
                    eprintln!("connex-rs: read from {remote_url} failed: {err}");
                }
                code(&err)
            }
        }
    }

    /// One-sided write of local `src_ptr` to `dst_ptr` in the peer.
    #[pyo3(signature = (remote_url, src_ptr, dst_ptr, length, timeout_ms=None))]
    fn transfer_sync_write(&self, py: Python<'_>, remote_url: &str, src_ptr: u64, dst_ptr: u64, length: u64, timeout_ms: Option<u64>) -> i64 {
        let e = self.e.clone();
        match py.allow_threads(move || e.put(remote_url, src_ptr, dst_ptr, length, ms(timeout_ms))) {
            Ok(()) => 0,
            Err(err) => code(&err),
        }
    }

    fn register_memory(&self, py: Python<'_>, ptr: u64, length: u64) -> i64 {
        let e = self.e.clone();
        match py.allow_threads(move || e.register(ptr, length)) {
            Ok(()) => 0,
            Err(err) => {
                eprintln!("connex-rs: register_memory({ptr:#x}, {length}) failed: {err}");
                code(&err)
            }
        }
    }

    fn unregister_memory(&self, py: Python<'_>, ptr: u64) -> i64 {
        let e = self.e.clone();
        match py.allow_threads(move || e.unregister(ptr)) {
            Ok(()) => 0,
            Err(err) => code(&err),
        }
    }

    // ---- async / signals / events

    /// Starts a one-sided read; returns a Transfer handle.
    #[pyo3(signature = (remote_url, dst_ptr, src_ptr, length, timeout_ms=None))]
    fn get_async(&self, remote_url: &str, dst_ptr: u64, src_ptr: u64, length: u64, timeout_ms: Option<u64>) -> PyResult<PyTransfer> {
        let c = self.e.rma_async(remote_url, false, dst_ptr, src_ptr, length, ms(timeout_ms), None).map_err(to_py)?;
        Ok(PyTransfer { c })
    }

    /// One-sided write followed (after remote completion) by a signal
    /// message carrying `msg`; the peer receives it with recv_msg().
    #[pyo3(signature = (remote_url, src_ptr, dst_ptr, length, msg, timeout_ms=None))]
    fn put_signal(&self, remote_url: &str, src_ptr: u64, dst_ptr: u64, length: u64, msg: Vec<u8>, timeout_ms: Option<u64>) -> PyResult<PyTransfer> {
        let c = self.e.put_signal(remote_url, src_ptr, dst_ptr, length, msg, ms(timeout_ms)).map_err(to_py)?;
        Ok(PyTransfer { c })
    }

    /// Sends a message to a peer given by URL (connects on demand) or by
    /// logical-id tuple (an already connected peer).
    fn send_msg(&self, py: Python<'_>, peer: &Bound<'_, PyAny>, data: Vec<u8>) -> PyResult<()> {
        let t = target(peer)?;
        let e = self.e.clone();
        py.allow_threads(move || e.send_msg(t, data)).map_err(to_py)
    }

    /// Next message as (peer tuple, bytes), or None after `timeout_ms`.
    #[pyo3(signature = (timeout_ms=None))]
    fn recv_msg<'py>(&self, py: Python<'py>, timeout_ms: Option<u64>) -> Option<(PeerT, Bound<'py, PyBytes>)> {
        let e = self.e.clone();
        let m = py.allow_threads(move || e.recv_msg(ms(timeout_ms)));
        m.map(|(p, d)| (tup(&p), PyBytes::new_bound(py, &d)))
    }

    fn connect(&self, py: Python<'_>, remote_url: &str) -> PyResult<()> {
        let e = self.e.clone();
        py.allow_threads(move || e.connect(remote_url)).map_err(to_py)
    }

    fn disconnect(&self, py: Python<'_>, peer: PeerT) -> PyResult<()> {
        let e = self.e.clone();
        py.allow_threads(move || e.disconnect(&lid(&peer))).map_err(to_py)
    }

    /// Next peer event dict ({"type": "joined"|"failed", "peer", ...}) or
    /// None after `timeout_ms` (or once the engine is closed).
    #[pyo3(signature = (timeout_ms=None))]
    fn wait_event<'py>(&self, py: Python<'py>, timeout_ms: Option<u64>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let e = self.e.clone();
        match py.allow_threads(move || e.wait_event(ms(timeout_ms))) {
            Some(ev) => Ok(Some(event_dict(py, &ev)?)),
            None => Ok(None),
        }
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let e = self.e.clone();
        let s = py.allow_threads(move || e.stats());
        let d = PyDict::new_bound(py);
        d.set_item("eps_created", s.ep.eps_created)?;
        d.set_item("eps_closed", s.ep.eps_closed)?;
        d.set_item("live_peers", s.ep.live_peers)?;
        d.set_item("failures", s.ep.failures)?;
        d.set_item("rejoins", s.ep.rejoins)?;
        d.set_item("heartbeat_failures", s.ep.heartbeat_failures)?;
        d.set_item("dropped_events", s.ep.dropped_events)?;
        d.set_item("regions", s.regions)?;
        d.set_item("region_bytes", s.region_bytes)?;
        d.set_item("remote_rkeys", s.remote_rkeys)?;
        d.set_item("inflight", s.inflight)?;
        d.set_item("rma_ok", s.rma_ok)?;
        d.set_item("rma_err", s.rma_err)?;
        d.set_item("rkey_fetches", s.rkey_fetches)?;
        d.set_item("invalidations", s.invalidations)?;
        d.set_item("op_timeouts", s.op_timeouts)?;
        d.set_item("wakeups", s.wakeups)?;
        Ok(d)
    }

    fn close(&self, py: Python<'_>) {
        let e = self.e.clone();
        py.allow_threads(move || e.shutdown());
    }

    #[getter]
    fn closed(&self) -> bool {
        !self.e.is_running()
    }
}

#[pyfunction(name = "now_ns")]
fn py_now_ns() -> u64 {
    now_ns()
}

#[pymodule]
fn connex_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEndpoint>()?;
    m.add_class::<PyEngine>()?;
    m.add_class::<PyTransfer>()?;
    m.add_function(wrap_pyfunction!(py_now_ns, m)?)?;
    Ok(())
}
