# vLLM integration

There are two ways to run Connex pipeline parallelism on vLLM:

* **Upstream vLLM, no fork** (`connex.adapters.vllm.pp`, this section): a
  plugin selected by class name on the vLLM command line. It needs three
  small changes to vLLM that are proposed upstream (below); everything else
  (transport, replicas, failover) lives in Connex.
* **The external-PP fork with engine patches** (the files listed further
  down): the original path, still supported, with features the upstream
  path does not have yet (see the comparison table).

## Upstream vLLM (no fork)

### What vLLM needs

Part of the vLLM RFC [#63818](https://github.com/vllm-project/vllm/issues/63818)
(pluggable PP transport). Until these are merged, build vLLM from the
branches (each is one or two commits on vLLM main):

| vLLM change | Branch | Needed for |
|---|---|---|
| `--pp-group-cls`: a plugin supplies the PP `GroupCoordinator`, including the V2 model runner's sampled-token broadcast (`sibling_broadcast`) | `feat/pluggable-pp-group` | both modes |
| `--pipeline-stage-rank`: one stage per vLLM instance, torch world without PP; executors given by import path may declare `supports_pp` | `feat/external-pp-stages` (on top of the first) | separate stages |
| `KVConnectorOutput.lost_kv_req_ids` / `lost_all_kv`: the scheduler preempts the requests, drops their in-flight outputs and evicts their blocks from the prefix cache, so they are recomputed | `feat/recompute-lost-kv` | failover |

The branches are linked from the RFC; they were validated on vLLM main as
of 2026-10-26 with the V2 model runner (the default).

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
tensor parallelism per stage works (`-tp 2`; TP rank 0 is the stage's
endpoint). Restarting a killed replica with the same command makes it a
spare again.

| Variable | Default | Meaning |
|---|---|---|
| `CONNEX_PP_STAGES` | (stage 0, required) | JSON list, one entry per downstream stage, of the stage servers' control addresses; list position = replica index |
| `CONNEX_PP_REPLICA` | 0 | this instance's replica index (stage 0: preferred replica of each stage) |
| `CONNEX_PP_DATA_PORT` | ephemeral | port of the stage's data endpoint |
| `CONNEX_RS_LOCAL_IP` | vLLM's host IP | address to listen on (use the RDMA interface's) |
| `CONNEX_PP_RING_MB` | from the config | registered GPU buffer per direction for RDMA ((PP+1) steps of hidden states) |
| `CONNEX_PP_CUDA_IPC` | 1 | same-host stage pairs over CUDA IPC (0: RDMA loopback) |
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
| stage-0 TP > 1 | serving yes; failover hangs | with the resume proxy |

Stage pairs on one host exchange the hidden states and the sampled tokens
over CUDA IPC: the sender's GPU copies into a ring the receiver has mapped
and sets a flag in shared pinned memory with `cuStreamWriteValue32`; the
receiver's stream waits on it with `cuStreamWaitValue32`. Nothing on that
path runs in Python, so it is as asynchronous as NCCL, and on these PCIe A40s
it is faster than vLLM's NCCL path for large steps. `CONNEX_PP_CUDA_IPC=0`
uses RDMA loopback instead. When a peer dies its flags are released at once
and its memory is unmapped once the local streams have passed that point,
so a replica restarted on the same GPU finds its memory free.

Known limitations: failover with TP > 1 on stage 0 hangs (the TP ranks of
stage 0 block in their in-group broadcast of the sampled tokens; see
`docs/results.md` section 13). Between hosts, the receive paths make a CUDA
stream wait for a flag that a Python thread sets. A CUDA call that holds the GIL and synchronizes the
device (a first-time kernel module load) would deadlock against it, so
receives wait on the CPU during warmup (`ConnexPPWorker`), where kernels
load. A kernel that is first loaded later, while traffic flows, could still
hang the stage.

## The external-PP fork

| File | Purpose |
|---|---|
| `connexte_communicator.py` | ConnexTE: Mooncake TE data plane + Connex control plane (recommended). A re-export shim of `connex.adapters.vllm.connexte`, built on the engine-neutral core `connex.pp` (see `docs/design/engine-contract.md`); needs the Connex package importable |
| `connex_communicator.py` | Connex over UCX (RDMA/TCP) |
| `zeromq_communicator.py` | ZeroMQ base communicator the others build on (same as the fork's copy) |
| `pipeline_comm_metrics.py` | Prometheus metrics for pipeline communication (same as the fork's copy) |
| `patches/` | engine patches for the external-PP fork; `apply_all.py` applies them in order |

## Status

This is a reference integration, not an installable plugin. It targets the
**external-PP vLLM fork at commit `380186d57`** (stages run as separate
processes with `--pipeline-stage-mode external`) and imports vLLM internals
(`vllm.distributed.parallel_state`, `vllm.utils.network_utils`, the v1
worker/scheduler), which change between vLLM releases.

## Installing

The easiest way is the image in `docker/Dockerfile`, which does all of this
at build time. By hand:

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
