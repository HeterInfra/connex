//! Raw bindings to csrc/cxshim.c (which wraps UCP). Only opaque pointers and
//! plain integers cross this boundary, so no UCX struct layout is mirrored.
use libc::{c_char, c_int, c_uint, c_void, size_t};

#[repr(C)]
pub struct CxCtx {
    _private: [u8; 0],
}

pub const EV_ERROR: c_int = 1;
pub const EV_CONN_REQUEST: c_int = 2;
pub const EV_HELLO: c_int = 3;
pub const EV_HB: c_int = 4;
pub const EV_AM: c_int = 5;
pub const HELLO_MAX: usize = 256;

#[repr(C)]
pub struct CxEvent {
    pub kind: c_int,
    pub status: c_int,
    pub ts_ns: u64,
    pub ep: *mut c_void,
    pub len: u32,
    pub data: [u8; HELLO_MAX],
    pub payload: *mut c_void,
    pub plen: size_t,
}

impl Default for CxEvent {
    fn default() -> Self {
        CxEvent { kind: 0, status: 0, ts_ns: 0, ep: std::ptr::null_mut(), len: 0, data: [0; HELLO_MAX], payload: std::ptr::null_mut(), plen: 0 }
    }
}

extern "C" {
    pub fn cx_now_ns() -> u64;
    pub fn cx_create(out: *mut *mut CxCtx) -> c_int;
    pub fn cx_destroy(c: *mut CxCtx);
    pub fn cx_progress(c: *mut CxCtx) -> c_uint;
    pub fn cx_poll_event(c: *mut CxCtx, out: *mut CxEvent) -> c_int;
    pub fn cx_dropped_events(c: *mut CxCtx) -> u64;
    pub fn cx_status_string(st: c_int) -> *const c_char;
    pub fn cx_listen(c: *mut CxCtx, ip: *const c_char, port: u16) -> c_int;
    pub fn cx_accept(c: *mut CxCtx, conn_request: *mut c_void, ep_out: *mut *mut c_void) -> c_int;
    pub fn cx_connect(c: *mut CxCtx, ip: *const c_char, port: u16, ep_out: *mut *mut c_void) -> c_int;
    pub fn cx_ep_close(ep: *mut c_void, force: c_int) -> *mut c_void;
    pub fn cx_is_err(r: *mut c_void) -> c_int;
    pub fn cx_hello(ep: *mut c_void, buf: *const c_void, len: size_t) -> *mut c_void;
    pub fn cx_hb(ep: *mut c_void, pong: c_int) -> *mut c_void;
    pub fn cx_tag_send(ep: *mut c_void, buf: *const c_void, len: size_t, tag: u64) -> *mut c_void;
    pub fn cx_tag_recv(c: *mut CxCtx, buf: *mut c_void, len: size_t, tag: u64, mask: u64, len_out: *mut size_t) -> *mut c_void;
    pub fn cx_req_test(req: *mut c_void, is_recv: c_int, len_out: *mut size_t) -> c_int;
    pub fn cx_req_cancel(c: *mut CxCtx, req: *mut c_void);
    pub fn cx_req_free(req: *mut c_void);
    pub fn cx_inprogress() -> c_int;
    pub fn cx_free(p: *mut c_void);
    pub fn cx_cuda_ctx_current() -> *mut c_void;
    pub fn cx_cuda_ctx_set(ctx: *mut c_void) -> c_int;
    pub fn cx_efd(c: *mut CxCtx) -> c_int;
    pub fn cx_arm(c: *mut CxCtx) -> c_int;
    pub fn cx_am_send(ep: *mut c_void, hdr: *const c_void, hlen: size_t, buf: *const c_void, len: size_t) -> *mut c_void;
    pub fn cx_mem_map(c: *mut CxCtx, addr: *mut c_void, len: size_t, memh_out: *mut *mut c_void) -> c_int;
    pub fn cx_mem_unmap(c: *mut CxCtx, memh: *mut c_void) -> c_int;
    pub fn cx_mem_type(memh: *mut c_void) -> c_int;
    pub fn cx_rkey_pack(c: *mut CxCtx, memh: *mut c_void, buf_out: *mut *mut c_void, len_out: *mut size_t) -> c_int;
    pub fn cx_rkey_buffer_release(buf: *mut c_void);
    pub fn cx_rkey_unpack(ep: *mut c_void, buf: *const c_void, rkey_out: *mut *mut c_void) -> c_int;
    pub fn cx_rkey_destroy(rkey: *mut c_void);
    pub fn cx_rma(ep: *mut c_void, is_put: c_int, local: *mut c_void, len: size_t, raddr: u64, rkey: *mut c_void, memh: *mut c_void) -> *mut c_void;
    pub fn cx_ep_flush(ep: *mut c_void) -> *mut c_void;
    pub fn cx_err_canceled() -> c_int;
}

pub fn status_string(st: i32) -> String {
    unsafe {
        let p = cx_status_string(st);
        if p.is_null() {
            return format!("status {st}");
        }
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}
