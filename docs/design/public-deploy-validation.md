# Validating the public deployment path on GPUs

What we learned taking v0.1.0 from "works on our cluster with a code
overlay" to "built from `docker/Dockerfile` and deployed with
`deploy/k8s/manifests.py`", and how to repeat the check.

## What was run

* **Image.** `docker/Dockerfile` with `BASE_IMAGE=vllm/vllm-openai:v0.11.2`
  (torch 2.9.0+cu129, CUDA 12, the same torch as the external-PP fork at
  `380186d57`), `FORK_INSTALL=overlay`, the fork served from a local
  `git daemon` as `VLLM_FORK_REPO`, PyPI for ucx-py-cu12 0.45 and
  mooncake-transfer-engine 0.3.13. No code mounted at runtime: the pods run
  the image's `/opt/connex/docker/launch_stage.sh`.
* **Deployment.** Manifests from `deploy/k8s/manifests.py` only, two
  configurations on A40 nodes with time-shared GPUs:
  1. portable defaults: pod network, Mooncake over TCP with P2P handshake,
     `--gpu-select auto`, 1 replica x 4 stages (Llama-3-8B from a host
     directory);
  2. RDMA opt-in: `--host-network --ib --protocol rdma --gpu-select
     nic-local`, still P2P handshake (no etcd), 2 replicas x 4 stages, the
     same GPU placement as the measurements in `docs/results.md`
     (`PREFERRED_GPU` per stage via `--set`).
  The Hugging Face path (`--model-hf-id`) was exercised with a small
  ungated Llama-architecture model (TinyLlama 1.1B, `--num-layers 22`) on 2
  stages through an HTTP proxy: stage 1 downloaded the checkpoint, loaded
  its layers and joined the pipeline; stage 0's download was still crawling
  through the proxy (about 100 KB/s, restarted after each broken connection)
  when we stopped, so the HF path has no end-to-end completion yet.
* **Checks** (all against a deployment of the pre-release code with the
  same model and placement):

| check | result | reference |
|---|---|---|
| smoke (`docker/smoke_test.sh` through a port-forward of the stage-0 Service) | OK, both configurations | |
| 16 greedy prompts, 64 tokens, batch 1 | 16/16 identical (TCP config, and both replicas of the RDMA config) | 16/16 between the reference's two replicas |
| K=1, QPS 8 trace, 1 replica | 100% goodput, 5.80 s mean / 23.8 s P99 E2E | 99.9%, 6.0 / 24.1 s |
| K=4, QPS 8 trace, 1 replica | 100%, 2.57 / 9.93 s | 100%, 2.59 / 9.96 s |
| failover, K=4, 32 requests x 300 tokens, kill stage 1/2/3 of replica 1 at 6 s | 32/32 complete, 0 non-argmax tokens in 9,600 for each stage; longest token gap 2.4-2.8 s | 2-3 s handover |
| hard churn, K=4, QPS 8, resume proxy (`PROXY=local`) | 100% of 2,340, 2.53 / 10.2 s; 16 proxy recoveries (10 mid-stream resumes, 6 retries before the first token) | 100%, 2.60 / 10.5 s |

The longest token gap was measured with four extra streaming requests on the
affected replica during the kill; it starts within 30 ms of the kill.

## Problems found, and why they hid

* **The build-time import check needs a driver.** The last `RUN` imports
  Mooncake and ucx-py to fail early on a CUDA mismatch, but `docker build`
  has no GPU driver, so `libcuda.so.1` is missing even with a matching base.
  On our cluster nobody built the image; the pods mounted the code. The check
  now uses the toolkit's `stubs/libcuda.so` (only during the check).
* **Stages that start together pick the same GPU.** With `--gpu-select
  auto` every stage took the GPU with the most free memory; stages of one
  node start together and vLLM allocates only after loading weights, so two
  stages landed on one GPU. Our cluster scripts had always pinned
  `PREFERRED_GPU`. A first fix spread stages by index into the list sorted by
  free memory, which broke again as soon as the first stage had allocated a
  little memory; candidates are now ordered by GPU index (free memory is only
  a filter), so every pod sees the same order.
* **Router and proxy names were global.** Every pipeline's router was
  `connex-front-router` and every proxy `connex-proxy`: applying a second
  pipeline into a namespace replaced the first one's router. Our test
  namespace already had an unrelated `connex-front-router`. Now
  `--router-name` and `PREFIX-proxy`; `churn_matrix.sh` also only restarts
  its own router instead of every pod labelled `app=connex-front-router`.
* **Host-network ports collided within a replica.** Ports depended only on
  the replica index. The layout in `docs/results.md` puts two stages of one
  replica on one node, which only worked because our scripts overrode the
  ports by hand. With `--host-network`, the pipeline and metrics ports are now
  offset per stage too.
* **Docs.** `G="python ..."; $G router` does not word-split in zsh (the
  default shell on macOS); the examples use a shell function now. The
  reproduction commands in `docs/results.md` asked for RDMA without
  `--host-network`. The shipped sample traces have about 3x longer outputs
  than the traces behind the tables (uniform 32-256 generated tokens), so
  "expect numbers close to the tables" was wrong; the setup section now says
  what they give.
* **Model downloads through a proxy.** The base image enables `hf_transfer`
  (and recent `huggingface_hub` uses Xet); both broke off repeatedly through
  our HTTP proxy (`IncompleteRead`, `CAS service error`). Plain downloads
  (`HF_HUB_ENABLE_HF_TRANSFER=0`, `HF_HUB_DISABLE_XET=1`) are more robust;
  for an 8B model a shared cache or a pre-staged directory is the practical
  choice anyway, because every stage downloads the full checkpoint.

## How to repeat it

```bash
# image (fork reachable as a git URL; a local `git daemon --export-all` works)
make image IMAGE=REG/connex-vllm:test VLLM_FORK_REPO=git://HOST/vllm-fork.git \
    BASE_IMAGE=vllm/vllm-openai:v0.11.2 && make push IMAGE=REG/connex-vllm:test
# manifests: use your own prefix and router name in a shared namespace
G() { python deploy/k8s/manifests.py --namespace NS --image REG/connex-vllm:test \
      --num-stages 4 --prefix p --router-name p-router --router-node NODE_A \
      --model-host-path /models/Llama-3-8B "$@"; }
G router | kubectl apply -f -
G pipeline --gpu-select auto | kubectl apply -f -                       # portable
# or RDMA: add --host-network --ib --protocol rdma --gpu-select nic-local
CONNEX_NAMESPACE=NS benchmarks/k8s/wait_ready.sh p
```

Then `docker/smoke_test.sh`, `benchmarks/k8s/run_trace.sh`,
`benchmarks/k8s/failover_correctness.py run --kill ... --kill-after 6` and
`score`, and `PROXY=local benchmarks/k8s/run_churn.sh p hard ...` with
`CONNEX_ROUTER_NAME=p-router`, as in `docs/results.md`.

Not covered by this run: tier 2 (plain docker on several hosts), etcd
metadata, and the UCX backend. The compose tier was checked separately,
below.

## Compose tier, two replicas on one box

Setup: one host with A40s (46 GB), the image built as above, base
Llama-3-8B from a local directory (`compose.local-model.yml`), two replicas
x two stages packed on two GPUs (`GPU0=2 GPU1=3 GPU2=2 GPU3=3
GPU_MEM_UTIL=0.4`), Mooncake TCP, `CONNEX_PP_INFLIGHT=1`, router lease 2 s.
Replica 0 was started first (`up -d router stage0 stage1`, smoke test), then
the rest (`--profile proxy up -d`).

| Check | Result |
|---|---|
| smoke on replica 0, replica 1 and through the proxy | all ` Paris, which is located in the north of the country...` |
| greedy 64 tokens, two runs on each replica | identical across runs and replicas |
| `failover_correctness.py run --url :8010 --n 8 --max-tokens 300`, `kill stage1-r1` after 7 s | 8/8 complete; `score` on replica 0: 0 non-argmax of 2400 tokens; stage0-r1 confirmed the death by a receive timeout plus probe (8 s before the timing fix below, 3.3 s after), failed over to replica 0's stage 1 and re-prefilled the 8 requests |
| `up -d stage1-r1` (rejoin), then the same run on `:8000` killing `stage1` | the rejoined stage re-registered in ~20 s (warm page cache) and took both chains: 8/8 complete, 0 non-argmax of 2400 |
| greedy check after both kills | still identical across runs and replicas |
| proxy: 300-token greedy stream through `:5199`, kill the stage 0 serving it after 60 chunks (twice, `up -d stage0-r1` in between) | `resumes` +1 each time, 300 tokens delivered, `failed: 0`; text differs from an uninterrupted run at a near-tie (char 712, after the splice), and teacher-forcing finds 0 non-argmax tokens in both, the documented batch-invariance effect |

Findings that changed the repository:

* **Compose defaulted to one replica.** The single-box tier now runs two
  replicas of the 2-stage pipeline by default (the failover profile is
  gone); the proxy covers both stage-0 APIs.
* **Packing two stages per GPU needs a staggered start.** Started together,
  vLLM's memory profiling counted the neighbour's loading and one stage
  failed with `Available KV cache memory: -1.83 GiB`. Starting replica 1
  after replica 0 is up fixes it; `docs/deploy.md` says so.
* **Router ports were fixed** at 15555/15560, which a port-forward on the
  test host already held; `ROUTER_PORT` / `ROUTER_EVENT_PORT` now set them.
* **`MODEL_DIR` must hold real files visible to the Docker daemon.** A
  Hugging Face cache snapshot (symlinks into `../../blobs`) or a path that
  exists only inside the client's container mounts as an empty directory and
  vLLM fails with `Invalid repository ID or local directory`.
* **Compose lacked the k8s data-path timing.** The first runs handed over
  in ~8 s: with the default `CONNEXTE_RECV_TIMEOUT_MS=8000` the death was
  only confirmed by that receive timeout. The compose stages now set the
  same values as `deploy/k8s/manifests.py` (`CONNEXTE_SEND/RECV_TIMEOUT_MS=3000`,
  `CONNEXTE_SEND_RETRIES=3`, `CONNEXTE_RECV_RETRIES=15`). Rerun with them,
  same packed layout, killing `stage1-r1` 7 s into 8 concurrent 300-token
  greedy streams on replica 1:

  | K (`CONNEX_PP_INFLIGHT`) | completed | non-argmax (`score`) | handover: longest token gap | no-kill run, 8 streams | false failovers |
  |---|---|---|---|---|---|
  | 1 | 8/8 | 0 / 2400 | 3.37 s | 67 s (1500 tokens each), longest gap 0.06 s | 0 |
  | 4 | 8/8 | 0 / 2400 | 3.31 s | 214 s (3000 tokens each), longest gap 0.13 s | 0 |

  Handover is measured as the longest gap between streamed tokens (kill to
  first token after the stall; tokens already in the pipe arrive right after
  the kill). Detection was still the 3 s data-path timeout confirmed by the
  probe, so ~3.3 s here against 2.4-2.8 s in the 4-stage k8s run, where the
  2 s router lease usually fires first. Completion and correctness come from
  `failover_correctness.py run --n 8 --max-tokens 300 --kill-after 7` and
  `score` on replica 0, one run per K; false failovers are "confirmed dead" /
  re-prefill lines in both stage-0 logs.
