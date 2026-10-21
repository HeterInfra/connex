# ConnexTE on a 4-node GPU cluster: where the time goes

Setup: plain Kubernetes in a dedicated namespace.
Llama-3-8B, **4 external-PP stages**, one stage per A40 node (four nodes),
`hostNetwork` + `/dev/infiniband`, Mooncake TE over RDMA, etcd metadata,
front-router `connex-front-router`. Workload: gamma-arrival traces (CV 1.0,
QPS 2/4/8; `benchmarks/k8s/gen_trace.py` generates equivalent ones) replayed open-loop,
fixed output length (`ignore_eos`), and a 60 s client timeout. Goodput is the
share of requests that complete within the timeout.

Reproduce (see `deploy/k8s/README.md` for the generator options):

```bash
G() { python deploy/k8s/manifests.py --namespace $NS --image $IMAGE --model-host-path $MODEL "$@"; }
G router --router-node NODE_A | kubectl apply -f -
G pipeline --backend connexte --prefix cte --num-stages 4 --nodes NODE_A,NODE_B,NODE_C,NODE_D \
    --host-network --ib --protocol rdma [--metadata etcd --etcd ETCD:2379] | kubectl apply -f -
benchmarks/k8s/wait_ready.sh cte
python benchmarks/k8s/gen_trace.py --qps 4 --out traces/gamma_qps4.0.csv
benchmarks/k8s/run_trace.sh cte traces/gamma_qps4.0.csv OUT_DIR http://STAGE0_HOST:5100
python benchmarks/k8s/hop_profile.py cte-model STAGE0_HOST,STAGE1_HOST,STAGE2_HOST,STAGE3_HOST
# per-phase timing in the logs: --set CONNEXTE_PROFILE=1
```

## Results (steady state, 4 stages, 1 pipeline replica)

Mean E2E / P99 E2E / goodput, one run per cell during development; the
repeated measurements are in `docs/results.md`.

| QPS | K=1, host-copy data path | K=1, GPUDirect data path | **K=4 + downstream CUDA graphs** |
|---|---|---|---|
| 2 | 3.03 s / 10.4 s / 100% | 2.37 s / 8.5 s / 100% | **1.62 s / 5.7 s / 100%** |
| 4 | 6.08 s / 20.2 s / 100% | 3.09 s / 10.5 s / 100% | **2.03 s / 7.2 s / 100%** |
| 8 | 42 s / - / 34% | 5.84 s / 23.3 s / 100% | **2.61 s / 10.0 s / 100%** |

All columns use the engine patches in `integrations/vllm/patches`. Without
correlation ids and idempotent retries (item 1 below), stage 0 crashed at
QPS 4 and 8, so those mechanisms are a precondition for every column.

Closed loop, 64 tokens: K=1 -> K=4 is 1068 -> 1380 tok/s at 64 clients and
1693 -> 2522 tok/s (+49%) at 128 clients. Greedy outputs are identical.

What makes churn survivable (2 replicas, a replica going down and up every
60 s; numbers in `docs/results.md`):
- **sticky peers** (no rewiring of established chains on join/leave);
- **replica-aligned chains**: replicas are identified by
  `PIPELINE_REPLICA_IDX`, never derived from the node address, so two pods of
  one stage may share a node;
- **UNHEALTHY treated as departure**, plus receive retries, so failover
  finishes before stage 0 gives up on a step.

## What was slow or broken, and why

1. **Stage 0 crashed under load (fixed timeouts).** A fixed 3 s recv timeout
   fires on healthy steps that are merely slow: cold first prefill, big mixed
   batches, or a downstream stage that is still starting. Stage 0 then re-sends the
   step. The replica processes it twice, the late original reply is taken as the
   retry's reply, steps fall out of step, and garbage token ids end in a CUDA
   device-side assert. After 3 timeouts the engine's retry loop kills the engine.
   *Fix:* a correlation id in the signal metadata. Stage 0 stamps each step and
   reuses the id on retry. Every stage drops replies for other ids and answers a
   duplicate id from its cached last output instead of recomputing, so retries
   are idempotent. The recv timeout adapts (max(base, 3 × slowest recent step)).
2. **Host copies dominate the data path under load.** With the CPU path,
   each hop costs ~40–45 ms at QPS 4: `.cpu()` 20 ms, safetensors 15–25 ms,
   copy into the Mooncake buffer 2–6 ms, deserialize 4–8 ms. The RDMA read itself
   takes only 1–2 ms. *Fix:* the GPUDirect path (`CONNEXTE_GDR=1`, default).
   Tensors are copied device-to-device into a GPU ring registered with Mooncake
   (`register_memory` accepts CUDA pointers; nvidia_peermem is loaded), read with
   RDMA straight into the receiver's registered GPU buffer, and cloned on the GPU.
   The header (layout and non-tensor fields) rides in the ZMQ signal. Measured
   GPU→GPU RDMA: 16 KB 9 µs, 1 MB 116 µs, 16 MB 2 ms. Idle comm per hop drops to ~1.1 ms.
3. **Engine: block table filled element by element on the GPU.** py-spy on a
   downstream stage under load: 88% of step time is in
   `_allocate_blocks_with_prefix_cache_for_external_pp`, at
   `blk_table_tensor[i, j] = bid`, a CUDA tensor written from a Python double
   loop (one tiny H2D copy and kernel per slot per request per step). *Fix:*
   `integrations/vllm/patches/fast_external_pp_block_table.py` builds the table on the
   host and copies it once per step (`VLLM_PATCH_BLOCK_TABLE=1`). It applies to
   every backend.

## Cluster pitfalls hit along the way

- **Shared A40s are time-sliced** (`nvidia.com/gpu: 1` may hand you a busy GPU
  with 4 GB free). `docker/launch_stage.sh` (GPU_SELECT=nic-local) keeps the k8s request for accounting but
  picks the physical GPU itself: it needs enough free memory and prefers GPUs on
  the IB NIC's NUMA side (`nvidia-smi topo -m`; two of the four GPUs on these nodes), since
  "SYS" paths are slow for GPUDirect.
- **Mooncake advertises the first NIC for its handshake RPC** under host
  network (a management or overlay NIC instead of IB). Set `MC_TCP_BIND_ADDRESS=$HOST_IP`
  and `MC_METADATA_CLUSTER_ID` so our keys are separate in a shared etcd.
- In P2P handshake mode the Mooncake target name is `ip:rpc_port`, not the
  local server name.
- **Start load only after all stages are serving** (`wait_ready.sh`). Stage 0's
  readiness probe says nothing about stages 1–3.
- The tar overlay of the fork must skip `vllm/vllm_flash_attn/*` (it collides
  with files already in the image).
- `kubectl exec ... | tar` streams can be cut by the API server. Use `kubectl cp`
  and check md5.
- Pods run as user `app`, so py-spy in the pod gets EPERM. Profile from a
  `--pid host --privileged` container on the node, as root (`docker exec -u root`).

- **Smoke-test every chain before measuring.** After a deployment, send a
  few requests to each replica's stage 0 and check that all complete; a chain
  that cannot complete requests (for example because a stage pod is still
  starting) makes churn numbers measure a broken deployment, not churn.
- **First-request stall of 8-9 s comes from logprobs, not from the pipeline.** vLLM
  torch.compiles `batched_count_greater_than` on the first request with
  logprobs. A cold deployment answers plain requests in 0.5 s. Any harness
  that asks for logprobs (handover.py, failover_correctness.py, the resume
  proxy) pays the stall once per engine. Fixed by patch 15
  (`patch_external_pp_warmup_logprobs.py`, `CONNEX_WARMUP_LOGPROBS=1` by
  default): `_dummy_sampler_run` also runs `compute_logprobs` +
  `gather_logprobs` for 1, 2 and max_num_seqs rows during engine init, before
  /health is green. Why those sizes: `dynamic=True` still specialises size-1
  dims, so batch 1 is its own graph and all sizes >= 2 share one. Use fresh
  tensors, not slices of one tensor, or dynamo guards on views force a later
  recompile (reproduced on CPU in tests/test_warmup_logprobs.py with a
  counting backend). Cost: ~8.7 s more startup per engine; logprobs TTFT on
  a cold engine 9.1 s -> 0.08 s, resume-proxy steady-state mean/P99
  3.18/13.25 s -> 2.68/10.10 s. Check it ran: grep `logprobs warm-up took` in the stage-0 log
  (printed twice, the second ~0.04 s). With PROXY on a single endpoint, a
  ~1 s connection-reset blip can still exhaust the proxy's 4 attempts
  (seen once, 15 x HTTP 502, cause not captured).
- **The churn driver must leave the endpoints file consistent at exit.** An
  early `benchmarks/k8s/churn.py` wrote both replicas back into the
  endpoints file when it exited, so when the last kill fired near the end of
  the run the load generator kept sending to a dead replica, and hard-churn
  goodput at QPS 8 read ~89.5 % instead of 99.2 %. The driver now writes the
  file only on real membership changes.

## Still open

- With K=1, per-token time at idle is ~55 ms (≈13 ms engine per stage): one
  step in flight leaves 3 of 4 stages idle. `CONNEX_PP_INFLIGHT=K` (see
  `pp_inflight.md`) addresses this.
- The 2-stage configuration has not been measured on the cluster.
