# Changelog

## Unreleased

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
  every token to EP rank 0 when replicas existed).
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
