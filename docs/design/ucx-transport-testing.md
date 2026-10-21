# UCXTransport testing notes

## One UCX transport per process
`ucp.init()` is process-global. A second `UCXTransport` in the same process now
reuses the existing context (warning: its own options are ignored); before, it hit
"UCX is already initiated" and failed. Tests and examples still run the receiver in
a `spawn` child process (`tests/test_transport.py::_ucx_receiver`), which matches
deployment, where each pipeline stage is its own process.

## Listener side receives via queue
For accepted connections, `UCXTransport` starts a background `_ucx_recv_loop`
per endpoint that pushes `(peer_id, data)` into a thread queue. Read it with
`recv_from_queue(timeout=...)`. `recv(endpoint=None)` does not work in UCX
mode. ZeroMQ mode uses `recv_sync(timeout_ms=...)` on the ROUTER socket.

## Shutdown noise (fixed)
About 2 in 5 runs of `examples/two_stage_pipeline.py` used to print
`Task exception was never retrieved ... BlockingMode._arm_worker ... Bad file descriptor`
at exit. Why: `stop_sync` never touched the UCX loop thread. The daemon loop was
still running at interpreter exit with ucx-py's per-loop progress task
(`BlockingMode._arm_worker`) parked in `sock_recv` on a socketpair; ucx-py's
finalizer closed that socket, and the dying task was reported. Fix
(`UCXTransport._shutdown_ucx` / `_stop_ucx_loop`): on the loop itself, close the
listener, abort endpoints, cancel our recv-loop tasks, then remove the epoll reader,
cancel the progress task and drop it from `ucp.core._ctx.progress_tasks`; only then
stop and join the loop, cancelling any leftover task before `loop.close()`.
Result: 0/20 runs show the message (was 3/6). Scripts that never call
`stop_sync` (e.g. `benchmarks/ucx_bandwidth.py`) can still print it.
Regression test: `tests/test_ucx_fixes.py::test_stop_sync_no_task_exception_at_exit`.

## RDMA rendezvous segfault on peer death (fixed: rcache or eager)
Symptom: SIGKILL of a receiver with a rendezvous send in flight kills the
*sender* with SIGSEGV at NULL+0xb8 in
`ucp_ep_reqs_purge -> ucp_proto_rndv_rts_abort -> ucp_request_memh_invalidate`
(also via `ucp_ep_req_purge_send` with `UCX_PROTO_ENABLE=n`).

Root cause (UCX 1.16.0 from hpcx 2.18, `/opt/hpcx/ucx`, commit a73e3f3): on an
endpoint with peer error handling (ucx-py default) whose send buffer's rkey was
handed to the peer, `ucp_request_memh_invalidate` calls
`ucs_rcache_region_invalidate(context->rcache, ...)` without checking that the
UCP registration cache exists (`src/ucp/core/ucp_request.c:437`,
`ucp_mm.c:384`). The cache is often *not* created in our processes: with
`UCX_RCACHE_ENABLE=try` (the default) UCX silently continues when UCM's memory
hooks fail their self-test ("rcache failed to install UCM event handler:
Unsupported operation", visible with `UCX_LOG_LEVEL=diag`). The self-test is
intermittent (with the default bistro hooks: a plain RDMA transport got the cache
in 5/6 starts; the elastic_bench sender in 36/40 runs even with 5 retries; one
batch of 6 bare `import ucp; ucp.init()` runs under load failed 6/6).
`UCX_MEM_MMAP_HOOK_MODE=reloc` is worse once torch is imported (0/5 starts with the
cache), so it is *not* set by default.

Fix (`UCXTransport._build_ucp_options` / `_init_ucp`): in UCX_RDMA mode, unless the
user set `UCX_RNDV_THRESH` or `UCX_RCACHE_ENABLE`, init with `RCACHE_ENABLE=y`
(retried `CONNEX_UCX_RCACHE_RETRIES`=5 times); if the cache still cannot be
created, re-init eager-only (`RNDV_THRESH=inf`) with a warning naming the pid.
UCX_TCP mode stays eager as before.

Measured with RDMA loopback on one A40 host (mlx5_0, its IB address), a scratch copy of
`benchmarks/elastic_bench.py` in UCX_RDMA mode, `UCX_TLS=rc,sm,self,cuda_copy
UCX_NET_DEVICES=mlx5_0:1`, `--duration 8 --rate 300 --size 4194304`, 4 runs in
parallel (the bench binds 127.0.0.1, which rdmacm/rc cannot use, so the copy uses
the IB address):

| config | sender crashes / 20 |
|---|---|
| old default (rndv, rcache=try) | 3 SIGSEGV + 1 hang in the SEGV handler |
| `UCX_PROTO_ENABLE=n` | 4 |
| `UCX_RNDV_SCHEME=get_zcopy` | 4 (hangs in the SEGV handler) |
| `UCX_RNDV_SCHEME=put_zcopy` | 2 |
| `UCX_MEM_MMAP_HOOK_MODE=reloc` | 10 |
| new default, run 1 | 0 (sender had rcache + rendezvous in 17/20) |
| new default, run 2 | 0 (rendezvous in 19/20) |

Turning peer error handling off (`endpoint_error_handling=False`) also avoids the
code path but needs the same setting on both sides ("asymmetric endpoint
configuration is not supported") and a send to the dead peer then stalled 10 s.

Bandwidth (`benchmarks/ucx_bandwidth.py` copy in UCX_RDMA mode, `UCX_TLS=rc,self,cuda_copy`,
host buffers through `send_obj`, GB/s, 2-3 runs):

| size | rendezvous + rcache (new default) | eager (`RNDV_THRESH=inf`) |
|---|---|---|
| 1 MB | 1.13-1.25 | 2.39-3.58 |
| 8 MB | 3.26-3.48 | 0.93-5.62 (unstable) |
| 32 MB | 1.43-1.56 | 0.76-1.66 |

So on this host the fallback to eager costs nothing on average for host buffers
(eager is faster at 1 MB, rendezvous is steadier at 8-32 MB). How to reproduce:
copy the bench with `sed 's/TransportMode.UCX_TCP/TransportMode.UCX_RDMA/; s/127.0.0.1/<ib-address>/'`,
loop it, and count `rc=139` exits and `Caught signal 11` backtraces.

## UCX_SOCKADDR_TLS_PRIORITY and the silent RDMA->TCP downgrade (fixed)
Some ucx-py builds choke on `UCX_SOCKADDR_TLS_PRIORITY` in the environment
(`ucx_config_to_dict`: "too many values to unpack" inside `ucp.init`). The old
code caught *any* init error, re-initialized with `TLS=tcp,cuda_copy` and switched
the mode to UCX_TCP with only an INFO line. Now `_init_ucp` hides that variable
from the environment during `ucp.init` (its value is passed as the
`SOCKADDR_TLS_PRIORITY` option and the env is restored afterwards), and an init
failure in RDMA mode raises unless `CONNEX_UCX_ALLOW_TCP_FALLBACK=1`, which logs
"DOWNGRADING" at ERROR. The vLLM communicator logs its own ZeroMQ fallback at ERROR.
The parse error did not reproduce with the ucx-py 0.36 / hpcx UCX on that host, so the
test emulates it (`tests/test_ucx_fixes.py::test_sockaddr_env_passed_as_option_not_env`).

## GPU-direct (CONNEX_GPU_DIRECT=1) never delivered (fixed)
Why: GPU-direct sends and receives are posted from the UCX event loop thread,
which had never touched CUDA. UCX's cuda_copy memory detection then fails
(`cuMemGetAddressRange ... invalid device context`), the TCP endpoint errors out
("send(...) failed: Input/output error", "Endpoint timeout"), and with
`CONNEX_ASYNC_SEND=1` the error is only logged, so stage 1 waits forever. The recv
side also allocated receive tensors on the loop thread's default device (cuda:0),
not the stage's GPU. Fix: `_bind_cuda_device` makes the tensor's device current in
the calling thread (cached per thread) before every CUDA send, and receives
allocate on the device captured at `start_sync` (`_recv_cuda_device`). The 'G' ZMQ
signal now carries the header timestamp and correlation frame like the 'UCX'
signal, and the receiver pairs it with its header via `_recv_ucx_matching`.
Repro/test: `tests/test_ucx_fixes.py::test_gpu_direct_two_process` (GPU 0 -> GPU 1,
skipped without 2 GPUs or ucx-py). Not re-tested on a full vLLM pipeline.

## No guessed UCX port
A peer that advertises no UCX address in its ASSIGN reply is now reached over
ZeroMQ (`resolve_peer_ucx_address` returns None; cached 5 s per connection
version). The old code guessed `router_port + 1000`, which is wrong whenever the
peer bound a fallback port or has no UCX. Note: a receiver running with
`CONNEX_UCX_DIRECT_RECV=1` only reads UCX and will not see such ZeroMQ messages.

## History
The first open-source migration copied the stale `submodule/connex` tree
(7.9k LOC). That tree had broken UCX accept/recv and lacked `recv_from_queue`,
`send_async`, `send_batch_gpu_direct` and `get_ucx_config`, which the vLLM
communicator needs. The source of truth is the vLLM fork's `vllm/connex`
(13.2k LOC). A UCX loopback test plus an API cross-check between the communicator
and the transport caught the mismatch.
