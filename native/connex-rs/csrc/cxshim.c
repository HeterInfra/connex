// SPDX-License-Identifier: Apache-2.0
//
// Thin C shim over UCP. It exists so that the Rust side never has to mirror
// UCX struct layouts (ucp_params_t, ucp_ep_params_t, ucp_request_param_t ...):
// this file is compiled against the system UCX headers, and Rust only sees
// opaque pointers and plain integers.
//
// Threading: one worker, UCS_THREAD_MODE_SINGLE. Every call must come from the
// thread that owns the context. Callbacks (endpoint errors, connection
// requests, active messages) only append to a fixed-size event ring; all
// endpoint creation / destruction happens outside callbacks.

#include <ucp/api/ucp.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <dlfcn.h>
#include <dirent.h>

#define CX_AM_HELLO 1
#define CX_AM_HB 2
#define CX_AM_USER 3
#define CX_EVQ 4096
#define CX_HELLO_MAX 256

enum { CX_EV_ERROR = 1, CX_EV_CONN_REQUEST = 2, CX_EV_HELLO = 3, CX_EV_HB = 4, CX_EV_AM = 5 };

typedef struct {
    int type;
    int status;          // ucs_status_t for CX_EV_ERROR
    uint64_t ts_ns;      // CLOCK_MONOTONIC when the callback ran
    void *ep;            // ucp_ep_h (error, hello) or ucp_conn_request_h
    uint32_t len;        // inline payload length (hello id, hb byte, AM header)
    char data[CX_HELLO_MAX];
    void *payload;       // CX_EV_AM: malloc'd copy of the AM data (free with cx_free)
    size_t plen;
} cx_event_t;

typedef struct {
    ucp_context_h context;
    ucp_worker_h worker;
    ucp_listener_h listener;
    cx_event_t *ring;
    uint32_t head, tail;  // ring indices (tail = next write)
    uint64_t dropped;
} cx_ctx_t;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

uint64_t cx_now_ns(void) { return now_ns(); }

static cx_event_t *push_event(cx_ctx_t *c) {
    if (c->tail - c->head >= CX_EVQ) {
        c->dropped++;
        return NULL;
    }
    cx_event_t *e = &c->ring[c->tail % CX_EVQ];
    memset(e, 0, offsetof(cx_event_t, data));
    e->payload = NULL;
    e->plen = 0;
    c->tail++;
    return e;
}

static void err_cb(void *arg, ucp_ep_h ep, ucs_status_t status) {
    uint64_t ts = now_ns();
    cx_event_t *e = push_event((cx_ctx_t *)arg);
    if (!e) return;
    e->type = CX_EV_ERROR;
    e->status = (int)status;
    e->ts_ns = ts;
    e->ep = ep;
}

static void conn_cb(ucp_conn_request_h req, void *arg) {
    cx_event_t *e = push_event((cx_ctx_t *)arg);
    if (!e) {
        ucp_listener_reject(((cx_ctx_t *)arg)->listener, req);
        return;
    }
    e->type = CX_EV_CONN_REQUEST;
    e->ts_ns = now_ns();
    e->ep = req;
}

static ucs_status_t hello_cb(void *arg, const void *header, size_t header_length,
                             void *data, size_t length,
                             const ucp_am_recv_param_t *param) {
    (void)header;
    (void)header_length;
    cx_ctx_t *c = (cx_ctx_t *)arg;
    if (!(param->recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP)) return UCS_OK;
    if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) return UCS_OK;  // hello is tiny
    cx_event_t *e = push_event(c);
    if (!e) return UCS_OK;
    e->type = CX_EV_HELLO;
    e->ts_ns = now_ns();
    e->ep = param->reply_ep;
    e->len = length > CX_HELLO_MAX ? CX_HELLO_MAX : (uint32_t)length;
    memcpy(e->data, data, e->len);
    return UCS_OK;
}

// Heartbeat: 1-byte payload, 'P' = ping, 'R' = pong.
static ucs_status_t hb_cb(void *arg, const void *header, size_t header_length,
                          void *data, size_t length, const ucp_am_recv_param_t *param) {
    (void)header;
    (void)header_length;
    cx_ctx_t *c = (cx_ctx_t *)arg;
    if (!(param->recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP) || length < 1) return UCS_OK;
    cx_event_t *e = push_event(c);
    if (!e) return UCS_OK;
    e->type = CX_EV_HB;
    e->ts_ns = now_ns();
    e->ep = param->reply_ep;
    e->len = 1;
    e->data[0] = ((const char *)data)[0];
    return UCS_OK;
}

// User active messages (engine control: rkey request/reply, invalidation,
// signals). The header (<= CX_HELLO_MAX bytes) is kept inline, the data is
// copied to the heap. Senders always use the eager protocol, so a whole
// message is delivered here in one call.
static ucs_status_t user_cb(void *arg, const void *header, size_t header_length,
                            void *data, size_t length, const ucp_am_recv_param_t *param) {
    cx_ctx_t *c = (cx_ctx_t *)arg;
    if (!(param->recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP)) return UCS_OK;
    if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) return UCS_OK;  // never sent
    if (header_length > CX_HELLO_MAX) return UCS_OK;
    void *copy = NULL;
    if (length) {
        copy = malloc(length);
        if (!copy) return UCS_OK;
        memcpy(copy, data, length);
    }
    cx_event_t *e = push_event(c);
    if (!e) {
        free(copy);
        return UCS_OK;
    }
    e->type = CX_EV_AM;
    e->ts_ns = now_ns();
    e->ep = param->reply_ep;
    e->len = (uint32_t)header_length;
    if (header_length) memcpy(e->data, header, header_length);
    e->payload = copy;
    e->plen = length;
    return UCS_OK;
}

void cx_free(void *p) { free(p); }

// ---------------------------------------------------------------- CUDA context
// UCX calls into the CUDA driver (memory type detection, registration of
// device memory) from whatever thread calls it. The progress thread has no
// CUDA context of its own, so the caller's current context is captured and
// made current there. libcuda is loaded lazily so that CPU-only hosts work.

typedef int (*cu_ctx_get_t)(void **);
typedef int (*cu_ctx_set_t)(void *);
typedef int (*cu_ptr_attr_t)(void *, int, unsigned long long);
static cu_ctx_get_t cu_get;
static cu_ctx_set_t cu_set;
static cu_ptr_attr_t cu_ptr_attr;
static int cu_tried;

static void cu_load(void) {
    if (cu_tried) return;
    cu_tried = 1;
    void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!h) return;  // CUDA not in use by this process
    cu_get = (cu_ctx_get_t)dlsym(h, "cuCtxGetCurrent");
    cu_set = (cu_ctx_set_t)dlsym(h, "cuCtxSetCurrent");
    cu_ptr_attr = (cu_ptr_attr_t)dlsym(h, "cuPointerGetAttribute");
}

// 1 when addr is CUDA device memory (CU_POINTER_ATTRIBUTE_MEMORY_TYPE ==
// CU_MEMORYTYPE_DEVICE), else 0.
static int cu_is_device(void *addr) {
    cu_load();
    if (!cu_ptr_attr) return 0;
    unsigned int t = 0;
    if (cu_ptr_attr(&t, 2 /* MEMORY_TYPE */, (unsigned long long)(uintptr_t)addr) != 0) return 0;
    return t == 2 /* DEVICE */;
}

void *cx_cuda_ctx_current(void) {
    cu_load();
    void *c = NULL;
    if (cu_get && cu_get(&c) != 0) c = NULL;
    return c;
}

int cx_cuda_ctx_set(void *ctx) {
    cu_load();
    if (!cu_set || !ctx) return -1;
    return cu_set(ctx);
}

// RDMA usable: an IB device exists and neither UCX_TLS nor UCX_NET_DEVICES
// rules it out (homogeneous clusters assumed: the peer has RDMA too).
static int cx_rdma_usable(void) {
    DIR *d = opendir("/sys/class/infiniband");
    int any = 0;
    if (d) {
        struct dirent *e;
        while ((e = readdir(d)) != NULL)
            if (e->d_name[0] != '.') any = 1;
        closedir(d);
    }
    if (!any) return 0;
    const char *tls = getenv("UCX_TLS");
    if (tls && tls[0] != '^' && !strstr(tls, "all") && !strstr(tls, "rc") &&
        !strstr(tls, "dc") && !strstr(tls, "ib"))
        return 0;
    const char *nd = getenv("UCX_NET_DEVICES");  // IB ports are "dev:port"
    if (nd && !strstr(nd, "all") && !strchr(nd, ':')) return 0;
    return 1;
}

// ---------------------------------------------------------------- lifecycle

int cx_create(cx_ctx_t **out) {
    cx_ctx_t *c = calloc(1, sizeof(*c));
    if (!c) return UCS_ERR_NO_MEMORY;
    c->ring = calloc(CX_EVQ, sizeof(cx_event_t));
    ucp_config_t *cfg;
    ucs_status_t st = ucp_config_read(NULL, NULL, &cfg);
    if (st != UCS_OK) goto fail;
    // Protocol choice (unless UCX_PROTO_ENABLE is set): v2 in HPC-X 1.16
    // selects software emulation (host staging, < 1 GB/s) for get/put on
    // CUDA memory over client/server endpoints, v1 does GPUDirect zero-copy.
    // Without RDMA, though, v1 emulates RMA on CUDA memory by touching the
    // device pointer from the CPU (the owner segfaults), while v2 stages it
    // through cuda_copy. So: v1 when RDMA can be used, else v2.
    if (!getenv("UCX_PROTO_ENABLE") && cx_rdma_usable())
        ucp_config_modify(cfg, "PROTO_ENABLE", "n");
    ucp_params_t p;
    memset(&p, 0, sizeof(p));
    p.field_mask = UCP_PARAM_FIELD_FEATURES;
    p.features = UCP_FEATURE_TAG | UCP_FEATURE_AM | UCP_FEATURE_RMA | UCP_FEATURE_WAKEUP;
    st = ucp_init(&p, cfg, &c->context);
    ucp_config_release(cfg);
    if (st != UCS_OK) goto fail;
    ucp_worker_params_t wp;
    memset(&wp, 0, sizeof(wp));
    wp.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE | UCP_WORKER_PARAM_FIELD_EVENTS;
    wp.thread_mode = UCS_THREAD_MODE_SINGLE;
    wp.events = UCP_WAKEUP_RMA | UCP_WAKEUP_AMO | UCP_WAKEUP_TAG_SEND | UCP_WAKEUP_TAG_RECV |
                UCP_WAKEUP_TX | UCP_WAKEUP_RX;
    st = ucp_worker_create(c->context, &wp, &c->worker);
    if (st != UCS_OK) goto fail_ctx;
    ucp_am_handler_param_t ap;
    memset(&ap, 0, sizeof(ap));
    ap.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID | UCP_AM_HANDLER_PARAM_FIELD_FLAGS |
                    UCP_AM_HANDLER_PARAM_FIELD_CB | UCP_AM_HANDLER_PARAM_FIELD_ARG;
    ap.id = CX_AM_HELLO;
    ap.flags = UCP_AM_FLAG_WHOLE_MSG;
    ap.cb = hello_cb;
    ap.arg = c;
    st = ucp_worker_set_am_recv_handler(c->worker, &ap);
    if (st != UCS_OK) goto fail_worker;
    ap.id = CX_AM_HB;
    ap.cb = hb_cb;
    st = ucp_worker_set_am_recv_handler(c->worker, &ap);
    if (st != UCS_OK) goto fail_worker;
    ap.id = CX_AM_USER;
    ap.cb = user_cb;
    st = ucp_worker_set_am_recv_handler(c->worker, &ap);
    if (st != UCS_OK) goto fail_worker;
    *out = c;
    return UCS_OK;
fail_worker:
    ucp_worker_destroy(c->worker);
fail_ctx:
    ucp_cleanup(c->context);
fail:
    free(c->ring);
    free(c);
    return st;
}

void cx_destroy(cx_ctx_t *c) {
    if (!c) return;
    if (c->listener) ucp_listener_destroy(c->listener);
    for (; c->head != c->tail; c->head++) free(c->ring[c->head % CX_EVQ].payload);
    ucp_worker_destroy(c->worker);
    ucp_cleanup(c->context);
    free(c->ring);
    free(c);
}

unsigned cx_progress(cx_ctx_t *c) { return ucp_worker_progress(c->worker); }

// Wakeup: the worker's event fd, to be polled after cx_arm() returned 0.
// cx_arm returns 0 when armed, 1 when there is pending work (call progress).
int cx_efd(cx_ctx_t *c) {
    int fd = -1;
    if (ucp_worker_get_efd(c->worker, &fd) != UCS_OK) return -1;
    return fd;
}

int cx_arm(cx_ctx_t *c) {
    ucs_status_t st = ucp_worker_arm(c->worker);
    if (st == UCS_OK) return 0;
    if (st == UCS_ERR_BUSY) return 1;
    return (int)st;
}

int cx_poll_event(cx_ctx_t *c, cx_event_t *out) {
    if (c->head == c->tail) return 0;
    *out = c->ring[c->head % CX_EVQ];
    c->head++;
    return 1;
}

uint64_t cx_dropped_events(cx_ctx_t *c) { return c->dropped; }

const char *cx_status_string(int st) { return ucs_status_string((ucs_status_t)st); }

// ---------------------------------------------------------------- endpoints

static int fill_sockaddr(const char *ip, uint16_t port, struct sockaddr_in *sa) {
    memset(sa, 0, sizeof(*sa));
    sa->sin_family = AF_INET;
    sa->sin_port = htons(port);
    if (ip == NULL || ip[0] == 0) {
        sa->sin_addr.s_addr = INADDR_ANY;
        return 0;
    }
    return inet_pton(AF_INET, ip, &sa->sin_addr) == 1 ? 0 : -1;
}

// Returns the bound port (>0) or a negative ucs_status_t.
int cx_listen(cx_ctx_t *c, const char *ip, uint16_t port) {
    struct sockaddr_in sa;
    if (fill_sockaddr(ip, port, &sa)) return UCS_ERR_INVALID_ADDR;
    ucp_listener_params_t lp;
    memset(&lp, 0, sizeof(lp));
    lp.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    lp.sockaddr.addr = (const struct sockaddr *)&sa;
    lp.sockaddr.addrlen = sizeof(sa);
    lp.conn_handler.cb = conn_cb;
    lp.conn_handler.arg = c;
    ucs_status_t st = ucp_listener_create(c->worker, &lp, &c->listener);
    if (st != UCS_OK) return st;
    ucp_listener_attr_t la;
    la.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;
    st = ucp_listener_query(c->listener, &la);
    if (st != UCS_OK) return st;
    return ntohs(((struct sockaddr_in *)&la.sockaddr)->sin_port);
}

static void set_err(cx_ctx_t *c, ucp_ep_params_t *ep) {
    ep->field_mask |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE | UCP_EP_PARAM_FIELD_ERR_HANDLER;
    ep->err_mode = UCP_ERR_HANDLING_MODE_PEER;
    ep->err_handler.cb = err_cb;
    ep->err_handler.arg = c;
}

int cx_accept(cx_ctx_t *c, void *conn_request, void **ep_out) {
    ucp_ep_params_t p;
    memset(&p, 0, sizeof(p));
    p.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST;
    p.conn_request = (ucp_conn_request_h)conn_request;
    set_err(c, &p);
    return ucp_ep_create(c->worker, &p, (ucp_ep_h *)ep_out);
}

int cx_connect(cx_ctx_t *c, const char *ip, uint16_t port, void **ep_out) {
    struct sockaddr_in sa;
    if (fill_sockaddr(ip, port, &sa)) return UCS_ERR_INVALID_ADDR;
    ucp_ep_params_t p;
    memset(&p, 0, sizeof(p));
    p.field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR;
    p.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    p.sockaddr.addr = (const struct sockaddr *)&sa;
    p.sockaddr.addrlen = sizeof(sa);
    set_err(c, &p);
    return ucp_ep_create(c->worker, &p, (ucp_ep_h *)ep_out);
}

// Starts a close; returns a request (to be tested with cx_req_test) or NULL.
void *cx_ep_close(void *ep, int force) {
    ucp_request_param_t p;
    memset(&p, 0, sizeof(p));
    p.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    p.flags = force ? UCP_EP_CLOSE_FLAG_FORCE : 0;
    ucs_status_ptr_t r = ucp_ep_close_nbx((ucp_ep_h)ep, &p);
    if (UCS_PTR_IS_ERR(r)) return NULL;
    return r;
}

// ---------------------------------------------------------------- messages
// All send/recv functions return: NULL = completed immediately, an error
// pointer (cx_is_err) or a request handle.

int cx_is_err(void *r) { return UCS_PTR_IS_ERR(r) ? (int)UCS_PTR_STATUS(r) : 0; }

void *cx_hello(void *ep, const void *buf, size_t len) {
    ucp_request_param_t p;
    memset(&p, 0, sizeof(p));
    p.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    p.flags = UCP_AM_SEND_FLAG_REPLY | UCP_AM_SEND_FLAG_EAGER;
    return ucp_am_send_nbx((ucp_ep_h)ep, CX_AM_HELLO, NULL, 0, buf, len, &p);
}

static const char hb_bytes[2] = {'P', 'R'};

// Heartbeat ping (pong=0) or pong (pong=1). The payload is static.
void *cx_hb(void *ep, int pong) {
    ucp_request_param_t p;
    memset(&p, 0, sizeof(p));
    p.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    p.flags = UCP_AM_SEND_FLAG_REPLY | UCP_AM_SEND_FLAG_EAGER;
    return ucp_am_send_nbx((ucp_ep_h)ep, CX_AM_HB, NULL, 0, &hb_bytes[pong ? 1 : 0], 1, &p);
}

// User AM: header (<= CX_HELLO_MAX) + data, eager, reply ep attached.
// buf/hdr must stay valid until the returned request completes.
void *cx_am_send(void *ep, const void *hdr, size_t hlen, const void *buf, size_t len) {
    ucp_request_param_t p;
    memset(&p, 0, sizeof(p));
    p.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    p.flags = UCP_AM_SEND_FLAG_REPLY | UCP_AM_SEND_FLAG_EAGER;
    return ucp_am_send_nbx((ucp_ep_h)ep, CX_AM_USER, hdr, hlen, buf, len, &p);
}

// ---------------------------------------------------------------- memory / RMA

static int cu_is_device(void *addr);

int cx_mem_map(cx_ctx_t *c, void *addr, size_t len, void **memh_out) {
    ucp_mem_map_params_t mp;
    memset(&mp, 0, sizeof(mp));
    mp.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    // Name the memory type: with detection alone, HPC-X 1.16 registered GPU
    // memory only with the cuda_copy domain (no IB key, so RMA fell back to
    // software emulation through host staging).
    if (cu_is_device(addr)) {
        mp.field_mask |= UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
        mp.memory_type = UCS_MEMORY_TYPE_CUDA;
    }
    mp.address = addr;
    mp.length = len;
    return ucp_mem_map(c->context, &mp, (ucp_mem_h *)memh_out);
}

int cx_mem_unmap(cx_ctx_t *c, void *memh) { return ucp_mem_unmap(c->context, (ucp_mem_h)memh); }

// Memory type of a mapped region (ucs_memory_type_t: 0 host, 1 cuda, ...).
int cx_mem_type(void *memh) {
    ucp_mem_attr_t a;
    a.field_mask = UCP_MEM_ATTR_FIELD_MEM_TYPE;
    if (ucp_mem_query((ucp_mem_h)memh, &a) != UCS_OK) return -1;
    return (int)a.mem_type;
}

int cx_rkey_pack(cx_ctx_t *c, void *memh, void **buf_out, size_t *len_out) {
    return ucp_rkey_pack(c->context, (ucp_mem_h)memh, buf_out, len_out);
}

void cx_rkey_buffer_release(void *buf) { ucp_rkey_buffer_release(buf); }

int cx_rkey_unpack(void *ep, const void *buf, void **rkey_out) {
    return ucp_ep_rkey_unpack((ucp_ep_h)ep, buf, (ucp_rkey_h *)rkey_out);
}

void cx_rkey_destroy(void *rkey) { ucp_rkey_destroy((ucp_rkey_h)rkey); }

// One-sided get (is_put = 0: remote -> local) or put (local -> remote).
// memh: local memory handle of `local` or NULL (UCX then registers on demand).
void *cx_rma(void *ep, int is_put, void *local, size_t len, uint64_t raddr, void *rkey,
             void *memh) {
    ucp_request_param_t p;
    memset(&p, 0, sizeof(p));
    // MULTI_SEND selects the "(multi)" protocol variants. In HPC-X 1.16 the
    // single-op variant of get/put between CUDA buffers is software
    // emulation (host staging, < 1 GB/s); the multi variant is RDMA
    // zero-copy via GPUDirect (as ucx_perftest uses).
    p.op_attr_mask = UCP_OP_ATTR_FLAG_MULTI_SEND;
    if (memh) {
        p.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
        p.memh = (ucp_mem_h)memh;
    }
    if (is_put) return ucp_put_nbx((ucp_ep_h)ep, local, len, raddr, (ucp_rkey_h)rkey, &p);
    return ucp_get_nbx((ucp_ep_h)ep, local, len, raddr, (ucp_rkey_h)rkey, &p);
}

// Remote completion of all RMA posted on ep so far.
void *cx_ep_flush(void *ep) {
    ucp_request_param_t p;
    memset(&p, 0, sizeof(p));
    return ucp_ep_flush_nbx((ucp_ep_h)ep, &p);
}

void *cx_tag_send(void *ep, const void *buf, size_t len, uint64_t tag) {
    ucp_request_param_t p;
    memset(&p, 0, sizeof(p));
    return ucp_tag_send_nbx((ucp_ep_h)ep, buf, len, tag, &p);
}

// When the message is already in the unexpected queue the receive can
// complete immediately (NULL return); *len_out then holds the real length.
void *cx_tag_recv(cx_ctx_t *c, void *buf, size_t len, uint64_t tag, uint64_t mask,
                  size_t *len_out) {
    ucp_request_param_t p;
    ucp_tag_recv_info_t info;
    memset(&p, 0, sizeof(p));
    memset(&info, 0, sizeof(info));
    p.op_attr_mask = UCP_OP_ATTR_FIELD_RECV_INFO;
    p.recv_info.tag_info = &info;
    void *r = ucp_tag_recv_nbx(c->worker, buf, len, tag, mask, &p);
    if (r == NULL && len_out) *len_out = info.length;
    return r;
}

// Returns UCS_INPROGRESS (1? no: negative) while pending, UCS_OK or an error
// once done. For receives, *len_out gets the received length.
int cx_req_test(void *req, int is_recv, size_t *len_out) {
    if (is_recv) {
        ucp_tag_recv_info_t info;
        ucs_status_t st = ucp_tag_recv_request_test(req, &info);
        if (st == UCS_OK && len_out) *len_out = info.length;
        return st;
    }
    return ucp_request_check_status(req);
}

void cx_req_cancel(cx_ctx_t *c, void *req) { ucp_request_cancel(c->worker, req); }
void cx_req_free(void *req) { ucp_request_free(req); }

int cx_inprogress(void) { return UCS_INPROGRESS; }
int cx_err_canceled(void) { return UCS_ERR_CANCELED; }
