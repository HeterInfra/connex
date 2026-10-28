# vLLM integration

Connex pipeline parallelism runs on the latest vLLM main as a plugin,
`connex.adapters.vllm.pp`, selected by class name on the vLLM command line.
It needs three small changes to vLLM that are proposed upstream (below);
everything else (transport, replicas, failover) lives in Connex. This is the
default path, and the image in `docker/Dockerfile` builds it.

The original path, the external-PP vLLM fork with engine patches, is kept as
a legacy option ([below](#legacy-external-pp-fork)).

## Upstream vLLM (no fork)

### What vLLM needs

Part of the vLLM RFC [#63818](https://github.com/vllm-project/vllm/issues/63818)
(pluggable PP transport). Until these are merged, build vLLM from main with
the branches of `https://github.com/yanyinglin/vllm` merged in (each is one <!-- release-scan: allow -->
or two commits on vLLM main):

| vLLM change | PR | Branch | Needed for |
|---|---|---|---|
| `--pp-group-cls`: a plugin supplies the PP `GroupCoordinator`, including the V2 model runner's sampled-token broadcast (`sibling_broadcast`) | [vllm#63897](https://github.com/vllm-project/vllm/pull/63897) | `feat/pluggable-pp-group` | both modes |
| `--pipeline-stage-rank`: one stage per vLLM instance, torch world without PP; executors given by import path may declare `supports_pp` | [vllm#63898](https://github.com/vllm-project/vllm/pull/63898) | `feat/external-pp-stages` (on top of the first) | separate stages |
| `KVConnectorOutput.lost_kv_req_ids` / `lost_all_kv`: the scheduler preempts the requests, drops their in-flight outputs and evicts their blocks from the prefix cache, so they are recomputed | [vllm#63899](https://github.com/vllm-project/vllm/pull/63899) | `feat/recompute-lost-kv` | failover |

By hand (the changes are Python only, so the compiled kernels of a vLLM
main nightly wheel can be reused):

```bash
PP_BRANCHES=https://github.com/yanyinglin/vllm.git   # release-scan: allow
git clone https://github.com/vllm-project/vllm.git && cd vllm
git fetch $PP_BRANCHES feat/external-pp-stages:pp-stages feat/recompute-lost-kv:lost-kv
git merge --no-edit pp-stages && git merge --no-edit lost-kv
VLLM_USE_PRECOMPILED=1 pip install -e .
pip install "connex-ai[vllm,rs]"
```

`make image` (`docker/Dockerfile`, default `PP_ENGINE=upstream`) does the
same on top of `vllm/vllm-openai:latest`; `docker/launch_stage.sh` starts a
stage of either mode from environment variables, and
`docker/docker-compose.yml` and `deploy/k8s/manifests.py` deploy a pipeline
with downstream replicas (see `docs/deploy.md`). The results in
`docs/results.md` section 13 were measured on vLLM main as of 2026-10-26
with the V2 model runner (the default).

### Mode 1: stock vLLM PP, inter-stage traffic over connex-rs instead of NCCL

All stages are ranks of one vLLM instance, as with plain
`--pipeline-parallel-size`. Only the PP group changes: hidden states and
the sampled tokens go over connex-rs (UCX; GPUDirect RDMA where the NIC
supports it).

```bash
vllm serve meta-llama/Meta-Llama-3-8B -pp 2 \
  --pp-group-cls connex.adapters.vllm.pp.group.ConnexPPGroup \
  --worker-cls connex.adapters.vllm.pp.worker.ConnexPPWorker
```

### Mode 2: each stage its own instance, replicas, in-flight failover

Stage 0 serves the API and schedules; it sends each step (the
`SchedulerOutput`, unchanged) to one replica of every downstream stage and
gets the step's output from the last stage. Hidden states go stage to stage
over connex-rs. When a replica on the route dies (connex-rs peer-failure
event, or a step timeout confirmed by a ping), stage 0 moves the route to
another replica of that stage, completes the steps in flight as lost and
reports the lost KV, so vLLM's scheduler recomputes the running requests on
the new route. Clients see a delay, not an error.

```bash
MODEL=meta-llama/Meta-Llama-3-8B
ARGS="--pipeline-parallel-size 2 --num-gpu-blocks-override 8192 \
  --pp-group-cls connex.adapters.vllm.pp.group.ConnexPPGroup \
  --worker-cls connex.adapters.vllm.pp.worker.ConnexPPWorker \
  --distributed-executor-backend connex.adapters.vllm.pp.executor.ConnexPPExecutor"

# stage 1, two replicas (any hosts); fixed ports so a restarted replica rejoins
CONNEX_PP_DATA_PORT=47021 python -m connex.adapters.vllm.pp.stage --model $MODEL $ARGS \
  --pipeline-stage-rank 1 --connex-control-port 47020 --connex-replica 0
CONNEX_PP_DATA_PORT=47023 python -m connex.adapters.vllm.pp.stage --model $MODEL $ARGS \
  --pipeline-stage-rank 1 --connex-control-port 47022 --connex-replica 1

# stage 0 (API server)
CONNEX_PP_STAGES='[["10.0.0.2:47020","10.0.0.3:47022"]]' \
  vllm serve $MODEL $ARGS --pipeline-stage-rank 0
```

All stages take the same engine arguments (the block ids in the steps are
stage 0's, so `--num-gpu-blocks-override` must be the same everywhere);
tensor parallelism per stage works (`-tp 2`; every TP rank is an endpoint
and exchanges the hidden states with the same rank of the next stage, TP
rank r listens on `CONNEX_PP_DATA_PORT` + 100·r). Restarting a killed replica with the same command makes it a
spare again.

| Variable | Default | Meaning |
|---|---|---|
| `CONNEX_PP_STAGES` | (stage 0, required) | JSON list, one entry per downstream stage, of the stage servers' control addresses; list position = replica index |
| `CONNEX_PP_REPLICA` | 0 | this instance's replica index (stage 0: preferred replica of each stage) |
| `CONNEX_PP_DATA_PORT` | ephemeral | port of the stage's data endpoint |
| `CONNEX_RS_LOCAL_IP` | vLLM's host IP | address to listen on (use the RDMA interface's) |
| `CONNEX_PP_RING_MB` | from the config | registered GPU buffer per direction for RDMA ((PP+1) steps of hidden states) |
| `CONNEX_PP_CUDA_IPC` | 1 | same-host stage pairs over CUDA IPC (0: RDMA loopback) |
| `CONNEX_PP_RDMA_PUSH` | 1 | between hosts, the sender's NIC writes the receiver's ring and flag (needs connex-rs>=0.1.4, `Engine.put_chain`); 0: one-sided reads, the receiver waits on the CPU |
| `CONNEX_PP_PUSH_MARGIN` | 8 | sampled tokens from a new (epoch, sender) use the push channel from this many steps on (the channel is announced on the first) |
| `CONNEX_PP_TP_ENDPOINTS` | 1 | one stage per instance with TP > 1: one endpoint per TP rank (0: TP rank 0 receives and broadcasts over the TP group) |
| `CONNEX_PP_TP_PORT_STRIDE` | 100 | data port of TP rank r = `CONNEX_PP_DATA_PORT` + stride·r |
| `CONNEX_PP_XHOST_STREAM_WAIT` | 0 | between hosts, for messages that do not come over a push channel: let streams wait for data a Python thread releases (can deadlock against GIL-holding device syncs) |
| `CONNEX_PP_STEP_TIMEOUT_S` | 60 | a step without reply after this long triggers a ping; no pong = failover |
| `CONNEX_PP_REPLACE_TIMEOUT_S` | 120 | how long stage 0 waits for a live replacement |
| `CONNEX_RS_HEARTBEAT_MS` / `_TIMEOUT_MS` | 1000 / 3000 | connex-rs heartbeat for hung peers |

### Compared with the fork path

| | upstream path | fork + patches |
|---|---|---|
| vLLM | main + 3 proposed changes | fork `380186d57` + 16 patches |
| model runner | V2 (default), unchanged | V1, patched |
| steps in flight | vLLM's own PP batch queue (PP+1) | `CONNEX_PP_INFLIGHT=K` |
| failover | recompute all running requests of the pipeline | recompute only the requests the new replica lacks |
| a replica shared by several pipelines | no: one pipeline at a time (spares are taken over) | yes |
| membership | static list in `CONNEX_PP_STAGES` | front-router, leases |
| data plane | connex-rs | Mooncake TE, connex-rs, UCX, ZeroMQ |
| stage-0 TP > 1 | yes, with failover | with the resume proxy |

Stage pairs on one host exchange the hidden states and the sampled tokens
over CUDA IPC: the sender's GPU copies into a ring the receiver has mapped
and sets a flag in shared pinned memory with `cuStreamWriteValue32`; the
receiver's stream waits on it with `cuStreamWaitValue32`. Nothing on that
path runs in Python, so it is as asynchronous as NCCL, and on these PCIe A40s
it is faster than vLLM's NCCL path for large steps. `CONNEX_PP_CUDA_IPC=0`
uses RDMA loopback instead. When a peer dies its flags are released at once
and its memory is unmapped once the local streams have passed that point,
so a replica restarted on the same GPU finds its memory free.

Between hosts the same contract holds over RDMA
(`connex.adapters.vllm.pp.rdmachan`, needs connex-rs>=0.1.4): the receiver owns a ring (GPU memory
for hidden states, pinned host memory for sampled tokens) and flags in
pinned host memory, registered with connex-rs, and announces them to the
sender. The sender's stream copies into its registered buffer and writes a
doorbell word with `cuStreamWriteValue32`; connex-rs's progress thread sees
it (`Engine.put_chain`), writes the bytes into the receiver's slot, reads
the last word back (so the data is in GPU memory before the flag, also on
GPUs that do not order inbound RDMA writes), then writes the receiver's
flag; the receiving stream waits on the flag. No Python thread on either
side releases a stream wait, which matters: a stream wait that only a
Python thread of the same process can release deadlocks against a CUDA call
that holds the GIL and waits for the device (a first Triton kernel load).
Across two hosts this is on par with NCCL (`docs/results.md` section 13).
The first messages from a new sender (before the receiver's announcement
reached it) go the old way, a one-sided read waited for on the CPU.

## Legacy: external-PP fork

The original integration: each stage runs as its own process on the
external-PP vLLM fork (commit `380186d57`, `--pipeline-stage-mode
external`) with 16 engine patches. It still works and has features the
upstream path does not have yet (see the comparison above). Build it with
`make image PP_ENGINE=fork VLLM_FORK_REPO=<git URL of the fork>
BASE_IMAGE=<vLLM image matching the fork's torch>`; deployment:
`docs/deploy.md`, section "Legacy: external-PP fork".

| File | Purpose |
|---|---|
| `connexte_communicator.py` | ConnexTE: Mooncake TE data plane + Connex control plane (recommended). A re-export shim of `connex.adapters.vllm.connexte`, built on the engine-neutral core `connex.pp` (see `docs/design/engine-contract.md`); needs the Connex package importable |
| `connex_communicator.py` | Connex over UCX (RDMA/TCP) |
| `zeromq_communicator.py` | ZeroMQ base communicator the others build on (same as the fork's copy) |
| `pipeline_comm_metrics.py` | Prometheus metrics for pipeline communication (same as the fork's copy) |
| `patches/` | engine patches for the external-PP fork; `apply_all.py` applies them in order |

### Status

This is a reference integration, not an installable plugin. It targets the
**external-PP vLLM fork at commit `380186d57`** (stages run as separate
processes with `--pipeline-stage-mode external`) and imports vLLM internals
(`vllm.distributed.parallel_state`, `vllm.utils.network_utils`, the v1
worker/scheduler), which change between vLLM releases.

### Installing

The easiest way is the image in `docker/Dockerfile` built with
`PP_ENGINE=fork`, which does all of this at build time. By hand:

1. `pip install connex-ai` (or `pip install -e /path/to/connex` from a
   checkout; the communicators import `connex.*`;
   remove any vendored `vllm/connex` package from the fork so the standalone
   package is used).
2. Copy `connex_communicator.py` and `connexte_communicator.py` into
   `vllm/distributed/device_communicators/`.
3. `python patches/apply_all.py <vllm>/v1/worker/gpu_model_runner.py`
   (patches the runner and its siblings; idempotent).
4. Start stages with `--pipeline-comm-backend connexte` (or `connex`); see
   `docker/launch_stage.sh` for the full command line.

Runtime knobs are listed in the configuration tables of the top-level
`README.md`.
