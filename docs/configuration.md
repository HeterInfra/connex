# Configuration reference

Environment variables and flags of the Connex components. The overview is in the [README](../README.md); deployment commands are in [`deploy.md`](deploy.md).

## Request retry/resume proxy (optional)

Connex keeps the data path free of a request router: clients talk to a
stage-0 replica's OpenAI API directly. A downstream failure never loses a
request, but if a stage-0 replica itself dies (its API server goes with it),
the requests it was serving are lost. `connex/serving/proxy.py` is a thin,
optional aiohttp proxy for `/v1/completions` and `/v1/chat/completions`
(streaming and non-streaming) that hides such deaths from the client:

- **before the first token** it retries a failed request on another replica;
- **after k tokens** it resumes on another replica: it re-sends the prompt
  token ids plus the k generated token ids, lowers `max_tokens` by k, and
  splices the new stream onto the old one, so the client sees one response
  with one id and correct usage. Chat resumes from text through
  `continue_final_message`;
- a stream that goes silent for `--idle-timeout` after its first token counts
  as dead (no timer runs before the first token, so slow prefill is never
  cut). Dead replicas are found by a `/v1/models` prober and by failures of
  other requests.

```bash
pip install -e ".[proxy]"
python -m connex.serving.proxy --port 5199 \
    --endpoints http://stage0-a:5100,http://stage0-b:5101
# or let it re-read the endpoint list:   --endpoints-file endpoints.txt
python benchmarks/proxy_overhead.py --concurrency 16    # steady-state cost, fake backend
```

| Flag | Default | Meaning |
|---|---|---|
| `--host` / `--port` | `0.0.0.0` / `5199` | listen address |
| `--endpoints` | empty | comma-separated stage-0 base URLs |
| `--endpoints-file` | unset | one URL per line, re-read every second |
| `--idle-timeout` | `2.5` | max gap between chunks after the first token (s) before resuming elsewhere |
| `--first-token-timeout` | none | optional hard cap on time to first token |
| `--connect-timeout` | `2.0` | backend connect timeout (s) |
| `--max-attempts` | `4` | attempts per request (first try plus retries/resumes) |
| `--blacklist-s` | `5.0` | how long a failed replica is avoided |
| `--probe-interval` | `1.0` | `/v1/models` probe period (s); `0` disables probing |
| `--no-token-ids` | off | resume completions from text instead of token ids (avoids requesting logprobs; re-tokenisation can drift at the splice) |

With greedy decoding the resumed output is identical to an uninterrupted one.
With sampling it is still a valid sample, but it follows a different random
stream. Each resume costs one extra prefill. Requests with `n>1`, `best_of`
or a batch of prompts are only retried before their first token. `GET
/proxy/stats` reports resumes, retries and failures. Design notes:
[`docs/design/request-resume-proxy.md`](design/request-resume-proxy.md).

## Variables by component

### Stage entrypoint (`docker/launch_stage.sh`)

| Variable | Default | Meaning |
|---|---|---|
| `STAGE`, `NUM_STAGES` | required | stage index and pipeline depth |
| `ROUTER_ADDR` | required | front-router API `HOST:PORT` |
| `BACKEND` | `connexte` | `connexte` (Mooncake TE) or `connex` (UCX) |
| `REPLICA` | `0` | replica index of this stage |
| `MODEL_PATH` / `MODEL_NAME` | `/model` / `connex-model` | model directory or Hugging Face id (`HF_TOKEN` for gated models), and served name |
| `NUM_LAYERS` / `LAYER_RANGE` | `32` / even split | layer assignment (`lo-hi`) |
| `API_PORT` | `5100+REPLICA` | stage 0: OpenAI API port; stages 1+: metrics port |
| `BASE_PORT` | `25500+10*REPLICA` | Connex pipeline port |
| `HOST_IP` | first `hostname -I` | address advertised to peers; also sets `VLLM_HOST_IP`, `MOONCAKE_LOCAL_IP`, `MC_TCP_BIND_ADDRESS` |
| `GPU_SELECT` | `none` if `CUDA_VISIBLE_DEVICES` set, else `auto` | `none`, `auto` (most free memory), `nic-local` (prefer GPUs near NIC0) |
| `MIN_FREE_MIB`, `PREFERRED_GPU`, `PREFERRED_GPU_WAIT_S` | `30000`, unset, `120` | GPU selection tuning |
| `GPU_MEM_UTIL` | `0.6` | `--gpu-memory-utilization` |
| `EXTRA_ARGS` | empty | extra vLLM server arguments |

The entrypoint also prepends the pip CUDA runtime (`nvidia/cuda_runtime/lib`)
to `LD_LIBRARY_PATH`; some base images do not have it on the search path and
the fork's extensions fail to load `libcudart`.

### Front-router and control plane

| Variable | Default | Meaning |
|---|---|---|
| `CONNEX_HEARTBEAT_TIMEOUT_S` | `30` | router: route lease; a stage without heartbeat is marked unhealthy |
| `CONNEX_UNHEALTHY_GRACE_S` | `60` | router: keep unhealthy routes this long before removal |
| `CONNEX_HEALTH_CHECK_INTERVAL_S` | `min(5, timeout/4)` | router: health sweep period |
| `CONNEX_HEARTBEAT_INTERVAL_S` | `10` | client heartbeat period (use 0.5 with a 2 s router lease) |
| `CONNEX_EPC_API_TIMEOUT_MS` | `5000` | client API request timeout |
| `CONNEX_EPC_HEARTBEAT_TIMEOUT_MS` | `0` (= API timeout) | client heartbeat request timeout |
| `CONNEXTE_FRONT_ROUTER_API`, `CONNEXTE_FRONT_ROUTER_EVENTS` | derived from `--pipeline-front-router-addr` | explicit router API / event addresses (`FRONT_ROUTER_API`/`FRONT_ROUTER_EVENTS` also accepted) |
| `PIPELINE_STAGE_IDX`, `PIPELINE_REPLICA_IDX`, `SERVED_MODEL_NAME` | set by the entrypoint | stage identity used for registration |

### ConnexTE data plane (`connex.pp`, used by `connexte_communicator.py`)

| Variable | Default | Meaning |
|---|---|---|
| `CONNEX_PP_BACKEND` | `auto` | data plane: `auto` (connex-rs if installed and an RDMA device exists, else Mooncake, else ZeroMQ), `connex-rs`, `mooncake`, `zmq` |
| `CONNEX_RS_LOCAL_IP` | `MOONCAKE_LOCAL_IP` / `CONNEX_HOST_IP` / `VLLM_HOST_IP` | address connex-rs listens on and advertises |
| `CONNEX_RS_PORT` / `CONNEX_RS_BASE_PORT` | unset / `MOONCAKE_BASE_PORT` + 1000 if set, else `51000 + 100*REPLICA` | fixed listen port, or base + stage index (fixed so a restarted stage keeps its URL) |
| `CONNEX_RS_NET_DEVICES` | `auto` | sets `UCX_NET_DEVICES` when that is unset; `auto` = the RDMA device whose IPoIB interface carries the local IP, plus that interface |
| `CONNEX_RS_HEARTBEAT_MS` / `CONNEX_RS_HEARTBEAT_TIMEOUT_MS` | `1000` / `3000` | transport heartbeat on the progress thread (hung/unreachable peers become suspects); timeouts below 1 s are raised to 1 s |
| `CONNEX_RS_SPIN_US` | `500` | progress thread busy-polls this long after activity before sleeping |
| `CONNEX_RS_REMOTE_FAILOVER_DELAY_S` | `0.05` | stage-0 grace after a transport-reported death farther down the chain |
| `CONNEX_RS_STATS_S` | `60` | log engine counters (endpoints, regions, fds, RSS) this often; `0` = off |
| `CONNEX_RS_DEBUG` | unset | print every connex-rs event and failed transfer |
| `MOONCAKE_METADATA_SERVER` | unset (falls back to ZeroMQ) | `P2PHANDSHAKE` or the etcd/redis address |
| `MOONCAKE_METADATA_BACKEND` | unset | `etcd` or `redis` when not P2P |
| `MOONCAKE_PROTOCOL` | `tcp` | `tcp` or `rdma` |
| `MOONCAKE_DEVICE_NAME` | empty (auto) | RDMA device(s) |
| `MOONCAKE_BASE_PORT` | `50000` | local TE port base (+stage index) |
| `MOONCAKE_LOCAL_IP` | `VLLM_HOST_IP` | address of the local TE segment |
| `MC_TCP_BIND_ADDRESS` | Mooncake default (first NIC) | handshake RPC bind address; set when the host has several NICs |
| `MC_METADATA_CLUSTER_ID` | Mooncake default | key prefix to share one etcd between deployments |
| `CONNEXTE_SEND_TIMEOUT_MS` / `CONNEXTE_RECV_TIMEOUT_MS` | `5000` / `8000` | base timeouts (adaptive on top) |
| `CONNEXTE_SEND_RETRIES` / `CONNEXTE_RECV_RETRIES` | `3` / `3` | retry budgets; use ~15 recv retries so a K=1 loop outlasts lease-based failover |
| `CONNEXTE_FAST_HANDOVER` | `1` | on a missed lease or a receive timeout, probe the peer (TCP connect) and fail over at once if it refuses; stage 0 re-sends every in-flight step sent on the dead chain at once. `0` waits for router deregistration and per-step timeouts |
| `CONNEXTE_PROBE_TIMEOUT_S` | `0.5` | timeout of that TCP probe |
| `CONNEXTE_REMOTE_FAILOVER_DELAY_S` | `0.2` | grace before stage 0 re-sends after a death farther down the chain |
| `CONNEXTE_GDR` | `1` | GPUDirect path (register GPU buffers with the TE) |
| `CONNEXTE_GDR_HOST` | `0` | stage through pinned host buffers instead |
| `CONNEXTE_GDR_SEND_MB` / `CONNEXTE_GDR_RECV_MB` | `512` / `256` | registered buffer sizes |
| `CONNEXTE_GDR_RECV_SLOTS` | `0` (auto) | receive slots |
| `CONNEXTE_GDR_ACK` / `CONNEXTE_GDR_ACK_TIMEOUT_S` / `CONNEXTE_GDR_HOLD_S` | `1` / `10.0` / `1.0` | slot reuse acknowledgements (hold time in no-ack mode) |
| `CONNEXTE_GDR_WIRE` | `2` | wire format version |
| `CONNEXTE_ASYNC_SEND` | `1` | send on a background thread |
| `CONNEXTE_STALE_DROP` | `1` | drop replies to steps that were already retried elsewhere |
| `CONNEXTE_ROUTE_REFRESH_ALWAYS` | `0` | refresh routes before every send (debug) |
| `CONNEXTE_SWITCH_INTERVAL_S` | `0.0002` | GIL switch interval while the loop runs |
| `CONNEXTE_PROFILE` / `CONNEXTE_PROFILE_EVERY` | `0` / `64` | per-phase timing logs |
| `CONNEXTE_TRACE` | empty | directory for per-step send/recv traces |

### Connex UCX data plane (`connex_communicator.py`, `connex.ucx_transport`)

| Variable | Default | Meaning |
|---|---|---|
| `CONNEX_GPU_DIRECT` | `1` in the communicator, `0` in the transport | send GPU tensors without host staging |
| `CONNEX_ASYNC_SEND` | `0` | asynchronous sends |
| `CONNEX_UCX_DIRECT_RECV` | `0` | receive directly into the destination tensor |
| `CONNEX_RECV_TIMEOUT_MS` | `8000` | receive timeout |
| `CONNEX_UCX_PORT_FALLBACK` | `16` | ports to try when the base port is taken |
| `CONNEX_UCX_RCACHE_RETRIES` | `5` | RDMA init attempts with the registration cache before falling back to eager |
| `CONNEX_UCX_ALLOW_TCP_FALLBACK` | `0` | allow RDMA init failure to downgrade to TCP (logged at ERROR) |
| `UCX_*` | UCX defaults | passed through (e.g. `UCX_TLS`, `UCX_NET_DEVICES`, `UCX_RNDV_THRESH`) |

### Engine patches (`integrations/vllm/patches`, applied at image build)

| Variable | Default | Meaning |
|---|---|---|
| `CONNEX_PP_INFLIGHT` | `1` | K pipeline steps in flight (set the same K on every stage); K = number of stages fills the pipeline |
| `CONNEX_PP_OVERLAP` | `0` | overlap stage-0 scheduling with in-flight steps (K>1) |
| `CONNEX_PP_DOWNSTREAM_GRAPHS` | `0` | replay CUDA graphs for decode steps on stages 1+ |
| `CONNEX_PP_REPREFILL` | `1` | stage 0 re-prefills requests whose downstream KV was lost in a failover (reported as `cx_missing`), so their output stays correct |
| `CONNEX_PP_REPREFILL_MAX` | `8` | re-prefills allowed per request; past it the request is aborted with an error |
| `CONNEX_WARMUP_LOGPROBS` | `1` | compile vLLM's logprobs op during engine warm-up instead of on the first logprobs request (~8.7 s of startup instead of a ~9 s first-request stall) |
| `CONNEX_PP_STEP_BUDGET_S` | `60` | give up on a step after this long (stage 0) |
| `CONNEX_PP_REQ_TTL_S` | `120` | downstream block-allocator TTL for idle requests |
| `CONNEX_PP_SEND_FINISHED` / `CONNEX_PP_FINISHED_REPEAT` | `1` / `2` | piggyback finished request ids downstream, and how often |
| `CONNEX_PP_HOST_LOGITS_IDX` | `1` | compute logits indices on the host |
| `CONNEX_PP_FIX_MIXED_PREFILL` | `1` | mixed prefill/decode seq_lens fix |
| `CONNEX_PP_VERIFY_SEQLENS` | `0` | debug check of host-built metadata |
| `CONNEX_PP_PROFILE` | `0` | stage-0 phase profile and microbatch histogram |
| `CONNEX_PP_SKIP_PATCHES` | empty | comma-separated patch files for `apply_all.py` to skip |
| `VLLM_EXTERNAL_PP_DEBUG` | unset | verbose external-PP logging |

## Benchmarks

| Tool | Measures |
|---|---|
| `benchmarks/control_plane_bench.py` | route update / lookup / notification latency at N endpoints |
| `benchmarks/elastic_bench.py`, `run_elastic_matrix.sh` | scale-out readiness, leave lag, kill stall |
| `benchmarks/ucx_bandwidth.py` | UCX per-message time and bandwidth |
| `benchmarks/k8s/gen_trace.py` + `trace_load.py` + `run_trace.sh` | open-loop trace replay, goodput (all tokens within the timeout) |
| `benchmarks/k8s/churn.py`, `run_churn.sh`, `churn_matrix.sh` | trace replay while replica 1 goes down/up (`soft` deregistration or `hard` kill); `PROXY=local` routes the load through the resume proxy |
| `benchmarks/proxy_overhead.py` | per-chunk cost of the resume proxy (CPU, fake backend) |
| `benchmarks/k8s/closed_loop.py` | closed-loop throughput |
| `benchmarks/k8s/failover_correctness.py` | greedy-output correctness across a downstream kill |
| `benchmarks/k8s/hop_profile.py` | per-hop communication cost per decode step |

None of them has a built-in address; pass endpoints with flags or
`CONNEX_API_URL`, `CONNEX_R0_URL`/`CONNEX_R1_URL`, `CONNEX_STAGE_HOSTS`,
`CONNEX_NAMESPACE`. Example on the compose demo:

```bash
python benchmarks/k8s/gen_trace.py --qps 4 --duration 120 --out traces/gamma_qps4.0.csv
python benchmarks/k8s/trace_load.py --trace traces/gamma_qps4.0.csv \
    --endpoints http://127.0.0.1:8000 --model connex-demo --out run.json
python benchmarks/k8s/closed_loop.py --url http://127.0.0.1:8000 --model connex-demo --clients 32
python benchmarks/control_plane_bench.py --backend connex --endpoints 2000
```

