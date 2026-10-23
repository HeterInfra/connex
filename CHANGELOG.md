# Changelog

## Unreleased

**Documentation**
- README: logo (light/dark), badges, highlights, table of contents, and SVG
  figures in `docs/assets/` for the architecture, multi-replica handover and
  the connex-rs vs Mooncake results (chart generated from `docs/results.md`
  by `docs/assets/src/results_chart.py`).

**Packaging: PyPI distribution `connex-ai`**
- The package is published on PyPI as `connex-ai` (`pip install connex-ai`,
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
  0.01-0.9 s with K=4 (see `docs/results.md`); steady-state latency and
  throughput match Mooncake.
- Single-stage same-identity rejoin works (with Mooncake a re-added stage
  could break its peers).
- Image: a builder stage compiles UCX (verbs, rdmacm, CUDA) and the
  connex-rs wheel (`BUILD_CONNEX_RS`, `UCX_VERSION`, `RUST_TOOLCHAIN`);
  `pip install "connex-ai[rs]"`; CI builds the wheel and tests the Python
  backend over UCX/TCP.

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
