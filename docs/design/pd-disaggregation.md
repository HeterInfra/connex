# PD disaggregation on upstream engines: design notes and what we learned

User-facing guide: [`docs/pd-disaggregation.md`](../pd-disaggregation.md).
This note records why the PD integration looks the way it does and the
problems found while validating it, with how to reproduce them.

## Why pull with a renewable hold

vLLM's built-in P/D connectors (NIXL, Mooncake) free the prefill instance's
blocks as soon as the decode instance has read them. That is the right
default for throughput, but it means the prompt KV of a request exists in
exactly one place once decoding starts: if the decode instance dies, the
request must be prefilled again from scratch.

Connex keeps a request's KV on the prefill instance as a *hold*: a lease
named by a transfer id, renewed by every acquire, released by the decode
instance when the request finishes, and dropped when it expires
(`retain_s`). A decode instance that takes over a resumed request acquires
the same transfer id and pulls the prompt KV again; it computes only the k
tokens generated before the failure. The cost is prefill memory held for
the lifetime of the decode request (bounded by the lease); `release:
after_pull` restores the conventional behaviour.

This is the endpoint mobility contract applied to PD: the transfer id is a
logical name for "the prompt KV of request R", and which decode endpoint
consumes it can change while the request is in flight.

Pull (decode reads) rather than push (prefill writes) because the consumer
of the KV is the endpoint that changes: a pull needs no knowledge of the new
decode instance on the prefill side, only that the hold still exists.

## Division of labour

| piece | where | engine specific? |
|---|---|---|
| KV layout of a paged cache, block runs to one-sided reads | `connex.pd.kv_layout` | no |
| data planes: connex-rs, Mooncake, loopback | `connex.pd.dataplane` | no |
| holds, acquire/release control channel, async pulls, failure reporting | `connex.pd.transfer` | no |
| PD pools in the front-router | `connex.pd.membership` | no |
| vLLM KV connector (scheduler and worker sides) | `connex.adapters.vllm.pd_connector` | vLLM |
| SGLang transfer backend (Mooncake protocol, connex-rs engine) | `connex.adapters.sglang.pd_backend` | SGLang |
| router: prefill then decode, elastic pools, resume | `connex.serving.pd_router` | protocol only |

The router extends the resume proxy through two small hooks
(`Proxy.resumer_cls`, `_wait_first(..., abort)`); everything else is a
subclass.

## Failure handling, end to end

* **Decode dies after k tokens.** The router's stream breaks; the resume
  proxy logic re-sends prompt + k token ids with `max_tokens - k` and the same
  `kv_transfer_params` to another decode instance; that instance acquires the
  hold and pulls. If the prefill instance is gone or the hold expired the
  acquire fails, vLLM reports a KV load failure, and the router prefills the
  extended prompt again (`kv_reprefills`). Chat resumes from text and always
  re-prefills, because the re-templated prefix need not tokenise identically.
* **Prefill dies during the transfer.** connex-rs reports the dead peer within
  milliseconds; the pull fails, the decode instance fails the request before
  its first token (`kv_load_failure_policy="fail"`), and the router prefills
  it again on another prefill instance without blaming the decode instance.
* **Prefill dies before answering.** Plain retry on another prefill instance.
* **Scale up / down.** New members take traffic after one successful probe;
  a member removed from the pool (file, front-router deregistration, or
  `POST /pd/drain`) takes no new requests and finishes its in-flight ones.

## Problems found during validation (and how to reproduce)

**1. UCX picks a CPU copy into GPU memory for reads below its zero-copy
threshold.** UCX 1.16 (HPC-X 2.18) v1 RMA chooses `get_bcopy` purely by
length (`get_zcopy_thresh`), regardless of memory type; bcopy completes with
a CPU `memcpy` into the destination, which segfaults on device memory
(`Segmentation fault: invalid permissions for mapped object` at a KV block
address, backtrace in `uct_rc_ep_get_bcopy_handler`). The threshold is
normally small, but it is derived from the registration cost, and without
UCX's registration cache it grows past a KV block. The cache needs UCX's
memory hooks, and **importing OpenCV (`cv2`) before UCX initialises disables
them** ("rcache failed to install UCM event handler: Unsupported operation").
vLLM imports `cv2`, so every pull into a vLLM KV cache crashed while the
same pull in a plain Python process worked. Reproduce:

```bash
UCX_LOG_LEVEL=info python -c "import cv2, connex_rs, torch; torch.zeros(1, device='cuda'); \
  connex_rs.Engine('t', 1, 0, 0, listen_ip='<IPoIB ip>', port=0)" 2>&1 | grep 'registration cache'
```

Fix: `ConnexRsPlane` sets `UCX_ZCOPY_THRESH=4096` (unless set) before the
engine starts, and the consumer refuses reads shorter than that with a clear
error instead of crashing. A threshold of 0 is wrong: connex-rs sends
heartbeats from read-only memory, and zero-copy sends try to register it
(`ibv_reg_mr ... Bad address` once per second, heartbeats lost).

**2. Prefill and decode on the same GPU.** With both processes on one GPU and
no `cuda_ipc` lane, UCX also serves small gets with bcopy over `rc_mlx5`; the
consumer refuses such pulls (GPU UUIDs are exchanged in the control
channel). Use different GPUs, which is the deployment shape anyway.

**3. Two UCX copies in one process.** The `nixl` wheel bundles its own UCX,
and vLLM imports NIXL when it is installed, so connex-rs (system HPC-X UCX)
and NIXL's UCX ended up in the same process (two `libucp`/`libucs` in
`/proc/<pid>/maps`). We did not isolate whether that alone breaks transfers
(the same run also hit problem 2); keep NIXL in a separate environment.

**4. The decode request can overtake the hold.** The prefill instance
answers the HTTP request when the request finishes; its worker learns the
hold one scheduler step later. The consumer retries `unknown` for up to
`acquire_wait_s` (2 s) before treating the hold as gone.

**5. `logprobs: 0` was dropped by the resume proxy.** `0 in (None, False)`
is true in Python, so the proxy treated a request for sampled-token logprobs
as no logprobs request and stripped them. Fixed in `connex.serving.proxy`
(test `test_logprobs_zero_is_passed_through`).

**6. Corrupt local weights.** The local copy of Qwen2.5-1.5B-Instruct
produces garbage even on CPU with transformers; use another model for smoke
tests.

## Correctness methodology

Exact greedy match against a non-disaggregated instance is reported but is
not the pass criterion: batching changes floating-point results, and the same
non-disaggregated instance run at concurrency 1 and 8 already differs on
about a third of 128-token Llama-3-8B completions. The criterion is the
teacher-forced check (`pd_bench.py score`): every generated token must be
the argmax, within 0.5 nats, given its own prefix, scored on a
non-disaggregated instance. Wrong or missing KV produces tokens far from the
argmax; the PD runs, including runs with killed instances, produced none.
