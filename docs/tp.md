# Tensor parallelism with Connex

Connex treats a tensor-parallel (TP) group as **one mobile endpoint**. The
all-reduce inside the group stays on NCCL/NVLink; Connex moves data between
groups (prefill to decode, stage to stage) and decides when a group as a
whole joins, fails or comes back. This page covers prefill-decode
disaggregation with TP > 1 (vLLM and SGLang, both as plugins on upstream
engines; validated with vLLM 0.32 and SGLang 0.5.22), pipeline stages with
TP > 1 (on upstream vLLM with `connex.adapters.vllm.pp`, see
[`integrations/vllm/README.md`](../integrations/vllm/README.md#upstream-vllm-no-fork);
the sections below on pipeline stages describe the legacy external-PP fork),
and what happens when one rank of a group dies. Measurements are in
[`results.md` section 11](results.md#11-tensor-parallelism).

## The model: one group, one identity

An instance (or a pipeline stage) with TP size N is N worker processes.
Every rank registers with the front-router as one **shard** of the same
logical endpoint, `LID(model, stage, shard=rank, replica)`, with
`group_size=N`:

* peers only see shard 0 of a **complete** group (`fetch_pools` for PD,
  `connex.epc_client.serving_routes` for pipelines); a group with a missing
  rank takes no new work, even if its rank 0 is still alive;
* any shard that misses its lease, leaves, or is reported dead fails the
  **whole group**: the front-router removes every shard and announces each
  as deregistered with a failure reason (`fail_group`);
* the group rejoins under the same identity once all N ranks have
  registered again, e.g. after a restart.

A rank's death is seen two ways, whichever is first: its lease lapses (the
process is gone or hung; 2 s with the settings used here), or a peer's
connex-rs data plane raises a peer-death event within milliseconds (the
kernel tears down the dead process's connections) and reports it
(`connex.pd.group.PeerDeathReporter` -> `report_failure`; SGLang names its
connex-rs endpoints by port, so its members also advertise `rs=host:port`
and the reporter resolves the port to the member). Engines do not
reliably take a group down fast when one rank dies: vLLM's engine with a
dead TP rank kept serving errors or hung for 4-10 s in our runs, the
external-PP stage process stayed up indefinitely. Group-level failure in the
control plane does not wait for that.

## PD disaggregation with TP > 1

### vLLM (`ConnexConnector`)

Each TP rank of the decode instance pulls **its own KV heads**. With equal
TP sizes, decode rank r reads whole blocks from prefill rank r. With
different sizes, `connex.pd.kv_layout.tp_plan` maps decode rank r to every
prefill rank whose head shard overlaps its own, and `head_read_ops` builds
one-sided reads that copy just those heads into the right place
(**KV head re-layout**):

| prefill TP -> decode TP | decode rank r reads |
|---|---|
| 2 -> 2 | prefill rank r, whole blocks |
| 2 -> 1 | heads 0-3 from prefill rank 0 and heads 4-7 from rank 1 (Llama-3-8B, 8 KV heads) |
| 1 -> 2 | heads 4r .. 4r+3 of prefill rank 0; both decode ranks read rank 0 |

The connector asks vLLM for the head-major `LBHNC` cache layout
(`get_required_kvcache_layout`), where one head of one block is a single
contiguous run (16 tokens x 128 dims x K and V x bf16 = 8 KB), so a head
range of a block is one read well above connex-rs' 4 KB zero-copy
threshold. Token-major layouts (`kv_layout: "none"`) still work but need one
read per token. When several decode ranks read the same prefill rank
(decode TP larger), the prefill rank frees its hold only after all of them
released it (`release` carries `of` and the reader's identity).

```bash
# prefill TP 2 on GPUs 0,1; decode TP 1 on GPU 2 (same command as TP 1 plus -tp)
CUDA_VISIBLE_DEVICES=0,1 vllm serve <model> -tp 2 --port 8100 --kv-transfer-config \
  '{"kv_connector":"ConnexConnector","kv_role":"kv_producer","kv_connector_extra_config":{"ctrl_port":17100}}'
CUDA_VISIBLE_DEVICES=2 vllm serve <model> --port 8200 --kv-transfer-config \
  '{"kv_connector":"ConnexConnector","kv_role":"kv_consumer"}'
```

TP rank r of an instance uses `ctrl_port + r` for its KV control channel and
`rs_port + r` for connex-rs, so leave room between instances on one host.

### SGLang (`connex` transfer backend)

The `connex` backend is SGLang's Mooncake backend over connex-rs, so equal
TP sizes need nothing new (each prefill rank writes to its decode rank).
Different TP sizes use SGLang's **staging buffer**
(`SGLANG_DISAGG_STAGING_BUFFER=1` on both sides): head slices are gathered
into contiguous GPU memory and written in bulk. Without it SGLang writes one
head slice per token (1 KB for Llama-3-8B), which connex-rs refuses (below
the zero-copy threshold a write from GPU memory would be a CPU copy). The
plugin lets `connex` through SGLang's check that otherwise allows staging
only for `mooncake` and `nixl`.

### Group failure in the PD router

Every TP rank of a vLLM instance registers its shard from its worker
process when `front_router` is set; SGLang ranks do the same with
`CONNEX_PD_FRONT_ROUTER`. The PD router lists only complete groups, and it
subscribes to the front-router's events (`--front-router-events`, default
port 15560): a deregistration with a failure reason marks the instance dead
at once, which cuts its in-flight streams and pending prefills. Decode
requests resume on another decode instance and re-pull the held prompt KV
(with KV head re-layout if that instance has another TP size); prefills
retry on another prefill instance. A graceful leave (no reason) still lets
the instance finish its requests.

```bash
connex-front-router --api-address tcp://*:15555 --event-address tcp://*:15560
python -m connex.serving.pd_router --front-router tcp://<router>:15555 --model <name>
# vLLM: "kv_connector_extra_config": {"front_router": "tcp://<router>:15555",
#        "http_url": "http://<host>:<port>", "replica": <i>}
```

## Pipeline stages with TP > 1 (legacy external-PP fork)

Engine patch 16 (`integrations/vllm/patches/patch_external_pp_tp.py`,
applied by `apply_all.py`) lets an external-PP stage run with
`--tensor-parallel-size N` (`TP=N` for `docker/launch_stage.sh`):

* **TP rank 0 is the stage's pipeline endpoint.** It alone creates the
  Connex communicator, receives each step from the upstream stage and sends
  each reply. A received step is broadcast inside the TP group
  (`broadcast_tensor_dict`: metadata over the CPU group, tensors over NCCL),
  a timeout is broadcast as well so all ranks retry or idle together. Other
  ranks drop their sends: hidden states and residual leave every layer
  replicated after the all-reduce, and stage 0's sampled tokens are
  identical on all ranks.
* **The group fails as one.** Rank 0's endpoint registers as shard 0 of a
  group of N; ranks 1..N-1 register liveness shards
  (`connex.pp.group.TPShardMember`). When any rank dies its lease lapses and
  the front-router deregisters the whole stage, so upstream peers fail its
  in-flight steps over to the same stage of another replica chain
  (re-prefilling requests whose KV was lost) without waiting for rank 0,
  which typically hangs in NCCL instead of exiting.
* Limits: `CONNEX_PP_INFLIGHT` must be 1 on a TP stage (the pipelined loops
  bypass the group wrapper; start-up refuses otherwise); re-prefill ids come
  from rank 0's output, which vLLM's multiprocess executor pickles with its
  extra attribute intact.

```bash
# one chain, two stages of TP 2, one container per stage
python benchmarks/pp/local_pp.py --image connex-vllm:dev --model /models/Llama-3-8B \
    --chains "0+1,2+3" --hold
```

## Reproducing

* PD: `benchmarks/pd/tp_experiments.py correct|perf|kill` (GPU specs with
  `+` are TP groups, e.g. `--prefill 1+2 --decode 3`; `kill --victim decode1
  --rank 1 --at 30 --restart-at 75` SIGKILLs one rank under load and
  restarts the instance). Every generated token is scored teacher-forced on
  a non-disaggregated vLLM instance (`--score-gpus`).
* PP: `benchmarks/pp/tp_pp_experiments.py correct|perf|kill` with
  `--chains "0+1,2+3;0+1,2+3"` (two chains whose stage groups share GPUs,
  `--gpu-mem 0.33`) and `--victim 0:1 --rank 1`.

## Limits

* PD with TP groups is validated on one host and across hosts (prefill and
  decode groups on different A40 nodes over 100 Gb InfiniBand: TP 2 -> 2,
  2 -> 1, 1 -> 2 correct, steady state on par with one host, group failure
  out in 0.3-0.5 s; [results section 12](results.md#12-across-hosts-and-mixed-gpus)).
  Not validated: a TP group that itself spans nodes for PD, and pipelines
  over RDMA between hosts with TP stages.
* KV heads must divide evenly by both TP sizes (no replicated KV heads, e.g.
  TP 16 with 8 KV heads); MLA and hybrid caches are not covered.
* NIXL could not be compared at TP 2: vLLM's NixlConnector (NIXL 1.5 in its
  own environment) aborts at start-up in UCX (`cuda_md.c: Assertion status
  == UCS_OK` after `system device 1 is invalid`) when two GPUs are visible.
  The TP 1 comparison is in the results.
* SGLang P2D2 with connex-rs failed with the prefill group on GPUs 0,1 (the
  socket away from the NIC; peer failed on the first writes, 2 of 3 runs)
  while Mooncake worked there; with the prefill group on GPUs 2,3 it is
  correct and on par with Mooncake. Not root-caused.
* Instances on one host do not use UCX `cuda_ipc` between each other
  (`CONNEX_RS_CUDA_IPC=1` re-enables it): a decode instance that had read a
  prefill instance's KV through cuda_ipc kept that GPU memory allocated
  after the prefill died, and the prefill could not be restarted until the
  decode exited.
* Pipeline: the external-PP fork image has no connex-rs, so stage failure is
  seen by lease (2.4 s) rather than by a transport event; stage 0 with TP > 1
  dying is covered only by the resume proxy, as at TP 1.
