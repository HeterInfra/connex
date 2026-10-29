# Changelog

## Unreleased

**Full review: fixes**
- Security: `connex.ucx_transport` no longer unpickles received payloads (a
  strict GDIR header parser; dtypes without a GDIR code go the bytes path
  instead of being sent as float32). The PP link (`connex.adapters.vllm.pp`)
  unpickles plain data only, and the EP mover only CUDA IPC handles.
  `connex-directory` binds `127.0.0.1:5555` by default (pass `--bind
  0.0.0.0:5555` for other hosts). `connex-pd-router` admin routes
  (`/pd/drain`, `/pd/undrain`) need loopback or `--admin-token` /
  `CONNEX_PD_ADMIN_TOKEN`. The vLLM PD connector's KV control channel binds
  the advertised IPv4 host (`bind_ip` overrides). SECURITY.md lists every
  listener and what is still unauthenticated (PP stage control sockets
  still unpickle peer frames: keep them on a private network).
- Resume proxy: a stream that breaks after the last token (finish_reason or
  `max_tokens` reached) completes instead of resuming past the end or
  reporting a failure; a replica whose blacklist expired is usable again
  with probing off; `max_completion_tokens` is honoured when `max_tokens`
  is null.
- PD: a late release from a decode instance the request moved away from no
  longer frees the prefill hold the new reader is pulling from; the vLLM
  connector closes its data plane on shutdown.
- PP: a malformed message no longer kills the link receive thread; channel
  maps are guarded against the abort and event threads; a BUSY failover
  runs off the control thread; the watchdog step timer starts at SAMPLE;
  chains of an already aborted epoch start cancelled; `connex.pp`
  endpoints with GPUDirect acks close without hanging.
- RL: readers of an arena that the publisher overwrote after a timeout are
  told (`stale`) and re-read instead of loading torn weights; a reader that
  dies mid-pull gives its load back; `receive_weights` syncs only CUDA
  devices (failed on CPU workers).
- EP: a failed move releases its staged slots. Agent sessions honour
  `session_ttl_s`.
- Core: directory control calls drop late replies of timed-out requests and
  are serialised; the front router guards cutover/abort states, applies a
  cutover under the route lock and stops its threads before closing
  sockets; `ProtocolHeader` round-trips; `epc_client` closes only the
  exact replica's connections; `ElasticPeerSet.stop` closes draining
  endpoints; dual-path cutover sets its start time.
- Tests: vLLM-dependent PP tests skip when the local vLLM build does not
  load; the fast-hash timing test uses best-of-N; benchmark scripts only
  signal the caller's own processes.

- `connex-ai[rs]` requires `connex-rs>=0.1.5,<0.2` (`Engine.put_chain` with a
  cancel word and cyclic comparisons, GPU RMA below the zero-copy threshold
  rejected); submodule at connex-rs 0.1.5 (unreleased).

**PP on upstream vLLM: review fixes** ([`docs/results.md`](docs/results.md#13-pipeline-parallelism-on-upstream-vllm-no-fork))
- Cross-host pushes and reads from GPU memory are padded to 4 KB: smaller
  ones segfaulted in UCX (seen with Qwen2.5-0.5B at batch 1).
- A failed push chain is reported to the receiver (CHANFAIL; a hidden-state
  hop stops the receiving stage so the pipeline fails over, sampled tokens
  are resent the slow way); a failed read or a push on an unknown channel
  of a live epoch no longer runs on garbage; a same-host sampled-token
  channel timeout raises.
- Stage 0's watchdog also fails over a stage that answers pings but is
  stuck on a step. Ring frees no longer wait behind one stuck chain; ring
  allocation and chains have deadlines; channels to peers that leave the
  route are retired; aborted epochs release channel waits and cancel their
  pending pushes.
- Sampled-token push channels: one per sender (not per epoch), with flow
  control (gate) and no reuse of a staging slot still being read; a payload
  that arrives before its receive registers is no longer lost.
- Flag counters wrap safely (cyclic comparisons, releases written ahead of
  every awaited count and re-asserted while streams still use a retired
  channel; no cancel value inside the counter range).
- One port layout (`connex.adapters.vllm.pp.ports`): TP rank 1 of stage k no
  longer collides with stage k+1. Default ports changed (control
  47000 + 1000·stage + 16·replica, data + 1 + TP rank).
- Faster failover: the route's endpoints warm the hops to spares, ABORT is
  sent asynchronously, the last stage compiles the top-k/top-p kernels at
  start, a restarted replica announces itself to stage 0. Largest
  no-token window across hosts 0.4-0.5 s (TP=2: 0.9-1.1 s; was up to 2.6 s).
- Pruned leaking state (retired push channels, announcements, stale inbox
  items). `failover_load.py` reports `no_token_window_s`.
- Benchmarks re-run with the same KV size for NCCL and Connex: TTFT on par.

**Upstream vLLM is the default for pipeline parallelism** ([`docs/deploy.md`](docs/deploy.md))
- `docker/Dockerfile` (`PP_ENGINE=upstream`, default) builds on
  `vllm/vllm-openai:latest`, merges the branches of vllm#63897/#63898/#63899
  into vLLM main and installs it with the precompiled kernels
  (`VLLM_INSTALL=precompiled|source|none`); `PP_ENGINE=fork` keeps the
  external-PP fork build as a legacy option.
- `docker/launch_stage.sh` starts upstream stages (`PP_STAGES`,
  `CONTROL_PORT`); `docker/docker-compose.yml` is a 2-stage pipeline with two
  stage-1 replicas (the fork demo moved to `docker/docker-compose.fork.yml`);
  `deploy/k8s/manifests.py --engine upstream|fork`, `--stage-replicas`.

**Pipeline parallelism on upstream vLLM** ([`integrations/vllm/README.md`](integrations/vllm/README.md#upstream-vllm-no-fork), [`docs/results.md`](docs/results.md#13-pipeline-parallelism-on-upstream-vllm-no-fork))
- `connex.adapters.vllm.pp`: `ConnexPPGroup` (`--pp-group-cls`; hidden
  states and the V2 runner's sampled tokens over connex-rs, the receiving
  stream waits on a host flag instead of the CPU), `ConnexPPWorker`
  (`--worker-cls`), `ConnexPPExecutor` (`--distributed-executor-backend`:
  stage 0 drives stage instances with replicas, fails over on connex-rs
  peer-failure events, reports the lost KV so vLLM recomputes) and the stage
  server `python -m connex.adapters.vllm.pp.stage`.
- Needs vLLM main with the three changes of RFC vllm#63818
  (`--pp-group-cls`, `--pipeline-stage-rank`, `KVConnectorOutput.lost_kv_req_ids`
  / `lost_all_kv`); the fork path is unchanged.
- `benchmarks/pp_upstream/`: per-hop latency of vLLM's PP API (NCCL vs
  connex-rs), failover under load.
- Same-host stage pairs over CUDA IPC with stream wait/write on shared
  pinned flags (`connex.adapters.vllm.pp.ipcchan`, `CONNEX_PP_CUDA_IPC`):
  on par with NCCL end to end, 32 MB per hop 0.80 ms vs 1.54 ms; safe unmap
  of a dead peer's ring; failure events confirmed with a ping. Kill
  comparison with stock vLLM under a restart loop in `docs/results.md`.
- Fixes: steps wait until every stage has a live replica (a step launched
  with a hole in the route hung stage 0's TP ranks in the sampled-token
  broadcast; PP2 x TP2 failover now completes all requests, with and without
  a spare); the stage server exits within 50 ms when a TP rank dies; failure
  events right after a rejoin are confirmed with a ping; stale flag files of
  killed processes are removed.
- Between hosts, receives wait on the CPU instead of on a stream released by
  a Python thread (deadlock with GIL-holding device syncs seen in a
  cross-host failover); about 5 ms per token at PP=2, C=1;
  `CONNEX_PP_XHOST_STREAM_WAIT=1` to opt out.
- Between hosts, the sender's NIC now releases the receiver's stream
  (`connex.adapters.vllm.pp.rdmachan`, needs
  connex-rs>=0.1.4, `Engine.put_chain`): the receiver owns the slots and flags, the sender's
  stream rings a doorbell and connex-rs writes the data, reads it back and
  writes the flag, so no Python thread on either side is on the data path.
  Two hosts, PP=2: 28.32 ms TPOT at C=1 (NCCL 28.28, CPU waits 34.2), C=32
  within the run-to-run spread of NCCL, also for PP=2 x TP=2;
  `CONNEX_PP_RDMA_PUSH=0` falls back to reads with CPU waits. 20 kills (12
  across hosts, including a TP=2 replica taking over on another host): every
  request completed, 0 tokens off the argmax.
- One stage per instance with TP > 1: every TP rank is a pipeline endpoint
  and exchanges the hidden states with the same rank of the next stage (no
  broadcast of the received hidden states over the TP group;
  `CONNEX_PP_TP_ENDPOINTS=0` restores it). Mean TTFT at C=32 10% lower;
  throughput the same within the spread.

**Across hosts and mixed GPUs** ([`docs/results.md`](docs/results.md#12-across-hosts-and-mixed-gpus), [`docs/design/multi-host-validation.md`](docs/design/multi-host-validation.md))
- PD (vLLM, SGLang), TP PD and EP validated with instances on different
  nodes over 100 Gb InfiniBand: on par with NIXL / Mooncake, decode and
  prefill kills, pool scaling and TP rank kills complete every request with
  0 off-argmax tokens (one near-tie token in a request untouched by a kill).
- PD host staging (`connex.pd.host_staging`, `CONNEX_PD_HOST_STAGING`,
  default `auto`): on a host without GPUDirect RDMA the KV cache is mirrored
  in pinned host memory and read with one-sided RDMA between host buffers
  (before: UCX over TCP). RTX A6000 prefill -> A40 decode at QPS 4: TTFT p50
  5.8 s -> 0.53 s, 414 -> 488 tok/s.
- `connex.ep.groups.RankDeathWatcher`: fails an EP group when connex-rs
  reports one of its ranks dead, wherever the rank runs (0.29 s across
  nodes; the health probe alone never noticed); `EPGroup(on_rejoin=...)`.
- EP/RL connex-rs transport defaults `UCX_ZCOPY_THRESH` to 4 KB: the first
  cross-node expert move crashed the receiving rank in UCX wire-up.
- `benchmarks/pd/local_pd.py --hosts` / `tp_experiments.py --hosts`: run
  PD deployments across hosts through any shell prefix (`ssh`, `kubectl
  exec`); `tp_experiments.py scale` (pool add/drain/stop under load) and
  `kill --rank -1` (whole instance); `CONNEX_PD_PORT_OFFSET`; NIXL
  side-channel host set for cross-host runs; pool files written atomically.

**Expert parallelism (MoE) with recovery and elasticity: `connex.ep`**
- `PlacementTable`: versioned expert placement (expert -> physical slots on
  EP ranks, replicas, rank states up / draining / down, routable masks),
  epochs applied only at a step boundary, repair planning after a rank fails
  or drains (destination on a live rank, source a live replica or the
  checkpoint), warm-standby replicas (`active_replicas`).
- `ExpertStore`: a rank's expert weights registered with connex-rs; experts
  move between ranks with one-sided reads straight into the destination slot.
- `EPController`: two-epoch reconfiguration (route around, copy, route to the
  new copies) for fail / drain / join / migrate; a move whose source dies is
  re-planned.
- `EPGroupSupervisor`: replica-level recovery for engines whose EP group
  cannot survive a rank loss (health probes, drop from the resume proxy's
  routing set, rebuild, rejoin).
- vLLM 0.32 (upstream, no fork): worker extension
  `connex.ep.vllm_ext.ConnexEPWorker` drives vLLM's EPLB expert maps from the
  placement table through `collective_rpc`; EPLB support for OLMoE.
- SGLang 0.5.22 (upstream, no fork, no Mooncake kernels): the `connex_ep`
  `sglang.srt.plugins` entry point (`CONNEX_EP=1`) adds a control endpoint
  to TP rank 0's scheduler; commands ride SGLang's per-iteration request
  broadcast, so every TP rank installs a placement epoch into SGLang's
  `ExpertLocationMetadata` between the same two steps.
  `connex.ep.sglang_ext.SGLangEPEngine` is the controller's engine handle.
  The plugin also makes `TopK` apply the expert location dispatch on MoE
  paths that skip it (SGLang's Qwen2-MoE without an all-to-all backend sent
  every token to EP rank 0 when replicas existed). Epoch writes are ordered
  on the GPU behind the overlapped batch instead of a device sync (apply
  round trip 48 -> 23 ms, scheduler blocked 22 -> 1.7 ms), and the connex-rs
  endpoint is created at server start with registration in the background
  (attaching a controller under load no longer stalls serving for 0.3-2.3 s).
  The control endpoint binds a free port unless `CONNEX_EP_PORT` is set and
  advertises it in the log and in `$TMPDIR/connex-ep-ctl-<server port>.json`
  (`SGLangEPEngine.from_server`).
- `EPGroupSupervisor`: probe timeout and path configurable
  (`probe_timeout` / `probe_path`, `CONNEX_EP_PROBE_TIMEOUT` /
  `CONNEX_EP_PROBE_PATH`, `ep_failover.py --probe-timeout/--probe-path`);
  the default timeout is 3 s (was 1 s), long enough for SGLang's `/health`.
- `benchmarks/ep/`: load + events (`ep_bench.py`, `--engine sglang`), EP-group
  kill under load (`ep_failover.py`, `--worker-needle` for SGLang), link
  bandwidth reference (`link_bw.py`).
- `ConnexRsTransport.read_many` / `LocalTransport.read_many`: reads into
  several destination tensors in one wave.

**Tensor parallelism: a TP group is one mobile endpoint** ([`docs/tp.md`](docs/tp.md))
- PD with TP > 1 on vLLM 0.32 and SGLang 0.5.22. vLLM: decode TP rank r
  pulls its KV heads from every prefill rank that holds some of them
  (`connex.pd.kv_layout.tp_plan` / `head_read_ops`), so prefill and decode
  may use different TP sizes (KV head re-layout); the connector asks vLLM
  for the head-major `LBHNC` layout so a head range of a block is one read.
  A prefill rank read by several decode ranks frees its hold after all of
  them released it. SGLang: equal TP sizes need nothing new; different
  sizes use SGLang's staging buffer, which the `connex` backend is now
  allowed to use.
- Group-level failure: every TP rank registers as one shard of its
  instance or stage (`group_size`). The front-router fails the whole group
  when any shard misses its lease, leaves, or is reported dead
  (`report_failure`, fed by connex-rs peer-death events through
  `connex.pd.group.PeerDeathReporter`; SGLang endpoints resolved through
  their `rs=` address). Peers only see complete groups; a restarted group
  rejoins under the same identity.
- PD router: subscribes to front-router failure events and cuts a failed
  instance's in-flight streams and prefills at once (stall after killing a
  TP-2 decode rank: 4.1 s -> 0.4 s).
- Pipeline stages with TP > 1 on the external-PP fork (engine patch 16,
  `patch_external_pp_tp.py`): TP rank 0 is the stage's endpoint, steps are
  broadcast in the TP group, other ranks register liveness shards
  (`connex.pp.group.TPShardMember`), so one dead rank fails the stage over
  to another chain. `docker/launch_stage.sh` takes `TP`.
- connex-rs PD plane excludes UCX `cuda_ipc` by default
  (`CONNEX_RS_CUDA_IPC=1` restores it): a decode instance on the same host
  kept a dead prefill instance's KV memory mapped, so the prefill could not
  restart.
- `benchmarks/pd/tp_experiments.py`, `benchmarks/pp/local_pp.py`,
  `benchmarks/pp/tp_pp_experiments.py`; results in `docs/results.md`
  section 11.

## 0.1.1

**Prefill-decode disaggregation on upstream vLLM and SGLang (no fork)**
- `connex.pd`: engine-neutral KV transfer with renewable holds (the prefill
  instance keeps a request's KV under a transfer id until the decode instance
  releases it or a lease expires), pulls with one-sided reads over connex-rs
  or Mooncake, per-request failure reporting, PD pool membership in the
  front-router.
- vLLM: `ConnexConnector` (`connex.adapters.vllm.pd_connector`), a
  KVConnectorBase_V1 loaded with `kv_connector_module_path` or by name via the
  `vllm.general_plugins` entry point. A request whose decode instance dies is
  resumed on another decode instance that pulls the held prompt KV again.
- SGLang: `connex` disaggregation transfer backend (SGLang's Mooncake protocol
  over connex-rs), registered at import time by the `sglang.srt.plugins` entry
  point.
- `connex.serving.pd_router` (`connex-pd-router`): prefill then decode (vLLM,
  also with NixlConnector) or paired by bootstrap room (SGLang); elastic pools
  from URLs, files or the front-router; retries failed prefills, re-prefills
  failed hand-offs, resumes decode failures.
- Validated on vLLM 0.32.0 and SGLang 0.5.22 with Llama-3-8B on one 4-GPU A40
  host: on par with NIXL and Mooncake in steady state, no off-argmax tokens,
  every request completed under decode/prefill kills and pool scaling
  (`docs/results.md` section 7, `docs/pd-disaggregation.md`). Not validated
  across hosts, with TP > 1 or with MLA/hybrid KV.
- Resume proxy: `logprobs: 0` is passed through (it was stripped because
  `0 == False`); error events inside a 200 stream fail the attempt; token ids
  are requested with `return_token_ids` (SGLang); `Proxy.resumer_cls` and an
  abort hook for subclasses.
- Benchmarks: `benchmarks/pd/` (local deployment, load/check/score, tables).

**RL post-training: elastic weight sync (`connex.rl`)**
- `WeightPublisher` / `WeightSubscriber`: the trainer packs weights into a
  registered, double-buffered arena and publishes version `v`; engines pull
  it with one-sided connex-rs reads from the trainer or from engines that
  already hold it (tree fan-out), replace a source that dies mid-pull, and
  catch up to the latest version when they join.
- `WeightRegistry` (HTTP/JSON, soft state): members by logical id, their
  versions, holders and fan-out assignment, heartbeats and expiry.
- `RolloutPool`: a request runs only on an instance at the version it was
  admitted with; drain or abort-and-restart before an update, rolling
  updates, failed instances dropped.
- vLLM: `connex` backend for the pluggable weight-transfer API, registered
  through the `vllm.general_plugins` entry point (`connex.rl.vllm_backend`);
  SGLang: custom weight loader `connex.rl.sglang_ext.connex_load`.
- Examples `examples/rl/` (weight sync into vLLM and SGLang, a GRPO loop with
  engine churn); benchmarks `benchmarks/rl/` (vs NCCL broadcast, vs vLLM's
  NCCL backend, a receiver dying mid-broadcast); `docs/rl.md`.

**Agent session migration (`connex.agents`)**
- `SessionProxy`: session logical ids with placement and epoch, sticky
  routing, `migrate` (KV movers, re-prefill built in, then cut-over of
  in-flight generations through the resume proxy), `drain`, failover.
- Resume proxy hooks: `Proxy.pick`, `Proxy.move` (planned cut-over that is
  not a failure), `begin` / `end`.
- Benchmark `benchmarks/agents/migration_bench.py`; `docs/agent-migration.md`.

## 0.1.0

First public release of Connex, a communication library for distributed AI
infrastructure built on endpoint mobility primitives: logical endpoint ids,
leased membership, versioned routes and a step-level delivery contract that
let endpoints join, leave, fail over and migrate with traffic in flight,
so that in a deployment of many replicas (pipeline chains, prefill/decode
pools, attention/FFN workers) in-flight requests hand over across replicas on
failure or scale-down and new replicas take traffic as they join.
Multi-replica pipeline parallelism (vLLM) is the first fully integrated and
GPU-validated scenario; prefill-decode primitives are included, and engine integration for
PD and attention-FFN disaggregation is on the roadmap (see the README's
scope table and `docs/design/overview.md`).

**Documentation**
- README: logo (light/dark), badges, highlights, table of contents, and SVG
  figures in `docs/assets/` for the architecture, multi-replica handover and
  the connex-rs vs Mooncake results (chart generated from `docs/results.md`
  by `docs/assets/src/results_chart.py`).

**Packaging: PyPI distribution `connex-ai`**
- The package is distributed on PyPI as `connex-ai` (`pip install connex-ai`,
  extras `connex-ai[rs,proxy,mooncake,ucx,vllm,sglang,test,dev]`); the import
  name stays `connex`. Source checkouts install as before
  (`pip install -e ".[test]"`).

**connex-rs: its own repository, consumed as a component**
- connex-rs moved out of this tree (`native/connex-rs`) into
  https://github.com/HeterInfra/connex-rs, released as `connex-rs` on PyPI
  and crates.io. Connex depends on it through the `rs` extra
  (`connex-ai[rs]` -> `connex-rs>=0.1,<0.2`) and imports `connex_rs` only when
  that backend is used; `auto` falls back to Mooncake or ZeroMQ without it.
- Source builds and the image use the git submodule `third_party/connex-rs`
  (or `CONNEX_RS_REPO`/`CONNEX_RS_REF`); CI builds the wheel from the
  submodule and runs the backend tests. `scripts/check_release.py` skips
  submodules unless `--recurse-submodules`.

**connex-rs: Connex's own data plane, default on RDMA hosts**
- connex-rs: a Rust data plane over UCX with a Python binding
  (`connex_rs.Engine`). One-sided RMA (get/put) on registered host or GPU
  memory with GPUDirect RDMA, `put_signal` (write, then a message that
  arrives after the data), plain messages, on a dedicated progress thread
  (the GIL is released while waiting). Peers are addressed by logical id
  (`ip:port#model|stage|shard|replica`), so a stage restarted under the same
  identity rejoins without restarting its chain.
- `connex.pp` data plane selection: `CONNEX_PP_BACKEND=auto` (default:
  connex-rs when installed and an RDMA device is usable, else Mooncake,
  else ZeroMQ), `connex-rs`, `mooncake`, `zmq`. Mooncake Transfer Engine
  stays fully supported.
- Peer-failure events from the transport are the primary failure signal:
  a dead process is reported to every linked stage within milliseconds and
  fails the chain over at once; the router lease and the TCP probe remain
  for hung or partitioned peers (transport heartbeat 1 s / 3 s only raises a
  suspicion). Kill-to-resume after a stage dies drops from 2-3 s to
  0.01-0.83 s with K=4 (see `docs/results.md`); steady-state latency and
  throughput match Mooncake.
- Single-stage same-identity rejoin works (with Mooncake a re-added stage
  could break its peers).
- Image: a builder stage compiles UCX (verbs, rdmacm, CUDA) and the
  connex-rs wheel (`BUILD_CONNEX_RS`, `UCX_VERSION`, `RUST_TOOLCHAIN`);
  `pip install "connex-ai[rs]"`; CI builds the wheel and tests the Python
  backend over UCX/TCP.

**Single-box deployment**
- `docker/docker-compose.yml` runs two replicas of a 2-stage pipeline by
  default, so in-flight handover across replicas works out of the box; the
  resume proxy (`--profile proxy`) fronts both. Router ports are
  configurable (`ROUTER_PORT`, `ROUTER_EVENT_PORT`).

**Seamless in-flight handover**
- Every pipeline step carries a correlation id; sends and receives are
  idempotent and retried with adaptive timeouts, with a reply cache and a
  stale-reply filter.
- Sticky, replica-aligned chains; a request fails over only when its peer is
  confirmed dead.
- Fast handover (`CONNEXTE_FAST_HANDOVER=1`): a missed lease or a receive
  timeout triggers a TCP probe of the peer; a refused probe fails every
  affected in-flight step over at once (2-3 s from kill to next token with a
  2 s router lease).
- Re-prefill (`CONNEX_PP_REPREFILL`, `CONNEX_PP_REPREFILL_MAX`): requests
  whose KV cache was lost in a failover are recomputed by stage 0 on the new
  chain, so greedy outputs stay correct. Same-step shared prefixes are not
  mistaken for missing KV.
- Up to K pipeline steps in flight (`CONNEX_PP_INFLIGHT=K`), surviving churn;
  optional downstream CUDA graphs for decode steps.

**Engine-neutral core and adapters**
- `connex.pp`: endpoint (Mooncake TE data plane with a GPUDirect send ring,
  host buffers or ZeroMQ only), step protocol, membership, handover rules,
  `Stage0Link` / `StageWorker` drivers, `EngineHooks` contract, wire format,
  loopback test doubles.
- `connex.adapters.vllm`: the ConnexTE communicator for the vLLM fork's
  external pipeline mode; engine patches in `integrations/vllm/patches`
  (including the logprobs compile warm-up, `CONNEX_WARMUP_LOGPROBS`).
- `connex.adapters.sglang`: `pp_group` replacement (prototype, communication
  side only).
- Connex UCX data plane and communicator (`connex.ucx_transport`).

**Serving**
- Optional request retry/resume proxy (`connex.serving.proxy`) that retries
  before the first token and resumes mid-stream on another stage-0 replica.

**Control plane and mobility primitives**
- Directory / front-router with leases, health checks and membership events;
  `ElasticPeerSet`; cutover state machine; versioned route table with epochs
  and draining.
- Prefill-decode roles on logical endpoint ids (`PDRole`), role lookup and a
  KV block transfer manager (`connex.mooncake_integration`); primitives only,
  no engine integration yet.

**Deployment and tooling**
- Deployment guide (`docs/deploy.md`): CPU-only demo, one GPU box, several
  hosts with RDMA or TCP, Kubernetes.
- Container image built from public sources only (`docker/Dockerfile`,
  `make image`), models from Hugging Face or a mounted directory; stage
  entrypoint (`docker/launch_stage.sh`); single-host compose demo with
  failover and proxy profiles and a smoke test; CPU-only demo
  (`examples/cpu_demo.sh`).
- Kubernetes manifest generator (`deploy/k8s/manifests.py`) for the router,
  stages, stage-0 Services and the proxy; portable defaults (pod network,
  TCP, device-plugin GPUs, no node pinning), RDMA and host network opt-in.
- `--router-name` (several pipelines per namespace), proxy named
  `PREFIX-proxy`, per-stage port offsets on the host network, and GPU
  auto-selection that spreads stages started together on one node.
- Image build and k8s path validated end to end on GPUs (smoke, greedy
  equivalence, steady state, failover, churn with the proxy); see
  `docs/design/public-deploy-validation.md`.
- Sample synthetic traces in `benchmarks/traces/` (heavier than the traces
  behind `docs/results.md`).
- Benchmarks: control plane, elasticity, trace replay, churn, failover
  correctness, proxy overhead; synthetic trace generator.
- Release scanner (`scripts/check_release.py`) and CI workflow.
