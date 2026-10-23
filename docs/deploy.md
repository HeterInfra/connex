# Deploying Connex

Three ways to run a Connex pipeline with vLLM, from one GPU box to
Kubernetes, plus a CPU-only demo. Nothing here depends on a particular
cluster: you build the image yourself from public sources, the model comes
from Hugging Face (or a directory you mount), and neither data plane needs a
metadata service (connex-rs addresses peers by URL; Mooncake runs in P2P
handshake mode, so no etcd).

The data plane is chosen per stage (`CONNEX_PP_BACKEND`, default `auto`):
connex-rs when the host has an RDMA device, Mooncake otherwise. connex-rs
needs the IB devices in the container (`/dev/infiniband`, host network) and,
for GPUDirect, `nvidia_peermem` on the host; it reports a dead peer to every
stage within milliseconds and lets a single restarted stage rejoin.

| Tier | Hardware | Data plane | What you get |
|---|---|---|---|
| 0. CPU demo | any Linux/macOS machine, Python 3.10+ | ZeroMQ / UCX-TCP | directory discovery between two stages; the resume proxy surviving a backend kill (fake engines) |
| 1. single GPU box | 1 host with 4 NVIDIA GPUs (≥ 24 GB each), or 2 GPUs of ≥ 40 GB, NVIDIA container toolkit; Llama-3-8B | Mooncake TE over TCP (loopback), RDMA optional | two replicas of a 2-stage vLLM pipeline with in-flight handover between them; optional resume proxy |
| 2. multi-node docker | ≥ 2 hosts with NVIDIA GPUs; ConnectX/InfiniBand (RoCE) NICs for RDMA, or plain Ethernet with TCP | connex-rs over RDMA (GPUDirect when `nvidia_peermem` is loaded); Mooncake TE over TCP without RDMA | the same pipeline across hosts; sub-second handover after a stage dies; single-stage rejoin |
| 3. Kubernetes | a cluster with the NVIDIA device plugin; RDMA optional | TCP on the pod network by default; RDMA opt-in (`--host-network --ib --protocol rdma`) | generated plain manifests: router, stages, stage-0 Services, optional proxy |

## Build the image (tiers 1-3)

The vLLM integration needs the external-PP vLLM fork (see the README,
[Integration](../README.md#integration-multi-replica-pipeline-parallel-serving-with-vllm)
and [Limitations](../README.md#limitations)).

```bash
make image IMAGE=<registry>/connex-vllm:0.1.0 VLLM_FORK_REPO=<git URL of the fork> \
    [VLLM_FORK_REF=<commit>] [BASE_IMAGE=vllm/vllm-openai:<tag>] [FORK_INSTALL=source]
make push IMAGE=<registry>/connex-vllm:0.1.0       # tiers 2 and 3, if hosts pull from a registry
```

The image also builds [connex-rs](https://github.com/HeterInfra/connex-rs)
in a builder stage: UCX (`UCX_VERSION`, default 1.18.1, with verbs, rdmacm
and CUDA) into `/opt/ucx`, and the Rust wheel against it (`RUST_TOOLCHAIN`);
the pip UCX wheels lack InfiniBand transports. The source is the
`third_party/connex-rs` submodule of the build context (run
`git submodule update --init` first, or clone with `--recurse-submodules`);
without it the build clones `CONNEX_RS_REPO` at `CONNEX_RS_REF` (default
the connex-rs GitHub repository, `v0.1.0`) and fails with a message if that
is not reachable. `--build-arg BUILD_CONNEX_RS=0` skips connex-rs
(Mooncake only).

The default `FORK_INSTALL=overlay` copies the fork's Python tree over the
compiled vLLM in `BASE_IMAGE`, so the base image must match the fork's torch
and kernel ABI; `FORK_INSTALL=source` compiles the fork instead (slow). The
build fails at once with a clear message if `VLLM_FORK_REPO` is unset.

Gated models (Llama 3) need `HF_TOKEN`. The stages download the model at
start into the Hugging Face cache; mount that cache so it is reused. Every
stage downloads the whole checkpoint into its own cache unless they share
one (a host directory in tiers 1-2, `--hf-cache-pvc` in tier 3), so for
Llama-3-8B a pre-staged model directory is much faster to start. Behind an
HTTP proxy, pass `HTTPS_PROXY`/`NO_PROXY` to the stages (`--set` in tier 3;
keep cluster-internal addresses in `NO_PROXY`), and if downloads break off
(`IncompleteRead`, `CAS service error`, an `hf_transfer` error) also set
`HF_HUB_ENABLE_HF_TRANSFER=0` and `HF_HUB_DISABLE_XET=1`: the base image
enables the fast multi-connection downloader, which retries poorly through
proxies. A stage whose download fails exits; under Kubernetes it is
restarted and tries again.

## Tier 0: CPU-only demo

```bash
pip install -e ".[proxy]"   # from a checkout (the demo scripts live in the repository)
make cpu-demo             # = examples/cpu_demo.sh
```

Expected output ends with `identical to uninterrupted reference: True` and
`resumes: 1`, then `== demo OK`. The engine-side handover logic (correlation
ids, retries, re-prefill on a replica departure) is exercised on CPU by the
unit tests: `python -m pytest -q tests/test_pp_core.py tests/test_fast_handover.py`.

## Tier 1: one GPU box (docker compose)

The compose file runs the front-router and **two replicas of a 2-stage
pipeline** on the host network: replica 0 is `stage0` (API on :8000, layers
0-15) + `stage1` (layers 16-31), replica 1 is `stage0-r1` (API on :8010) +
`stage1-r1`. Both register with the same front-router, so a stage of one
replica fails over to the same stage of the other with requests in flight.
Mooncake runs over TCP with P2P handshake.

```bash
export CONNEX_IMAGE=<registry>/connex-vllm:0.1.0     # or build: add --build and VLLM_FORK_REPO
export HF_TOKEN=hf_...                               # MODEL defaults to meta-llama/Meta-Llama-3-8B
docker compose -f docker/docker-compose.yml --profile proxy up -d
docker/smoke_test.sh                                 # replica 0, OK: ' Paris...'
docker/smoke_test.sh http://127.0.0.1:8010           # replica 1
docker/smoke_test.sh http://127.0.0.1:5199           # through the resume proxy (both replicas)
```

GPUs: replica 0 uses `GPU0`/`GPU1` (default 0 and 1), replica 1
`GPU2`/`GPU3` (default 2 and 3). With two GPUs of 40 GB or more, pack both
replicas on them: `GPU2=0 GPU3=1 GPU_MEM_UTIL=0.4` (Llama-3-8B, about 8 GB
of weights per stage). When packing, start the replicas one after another:
vLLM's memory profiling counts the other process's allocations while both
load, and the second stage on a GPU then fails with negative KV cache memory.

```bash
docker compose -f docker/docker-compose.yml up -d router stage0 stage1
docker/smoke_test.sh
docker compose -f docker/docker-compose.yml --profile proxy up -d
docker/smoke_test.sh http://127.0.0.1:8010
```
 One replica only: `docker compose -f
docker/docker-compose.yml up -d router stage0 stage1`. If the router ports
are taken, set `ROUTER_PORT` / `ROUTER_EVENT_PORT` (default 15555 / 15560).

```bash
# local model directory instead of a download. It must hold real files (not
# a Hugging Face cache snapshot, whose files are symlinks into ../../blobs)
# and be visible to the Docker daemon.
MODEL_DIR=/path/to/Llama-3-8B docker compose -f docker/docker-compose.yml \
    -f docker/compose.local-model.yml --profile proxy up -d

# in-flight handover across replicas: start long requests on replica 1,
# then kill its stage 1; they continue on replica 0's stage 1 after a ~3 s
# stall (3 s data-path timeout + probe, as in the k8s manifests)
for i in 1 2 3 4; do curl -s http://127.0.0.1:8010/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"connex-demo","prompt":"Tell a long story.","max_tokens":400,"temperature":0}' & done
sleep 5; docker compose -f docker/docker-compose.yml kill stage1-r1; wait
# replica 1's stage 1 back (scale up): it re-registers and is used by the
# next failover; stage 0 stays sticky on its current peer until that dies.
# With Mooncake a re-added stage can break its peers (seen with RDMA, see
# Limitations in the README): restart the whole chain (stage0-r1 too).
# connex-rs (RDMA hosts) handles a single-stage rejoin.
docker compose -f docker/docker-compose.yml up -d stage1-r1

# stage-0 death behind the resume proxy: stream through :5199 and kill the
# stage 0 serving it (the one whose /metrics shows a running request)
curl -sN http://127.0.0.1:5199/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"connex-demo","prompt":"Tell a long story.","max_tokens":400,"temperature":0,"stream":true}' &
sleep 5; docker compose -f docker/docker-compose.yml kill stage0-r1   # or stage0
curl -s http://127.0.0.1:5199/proxy/stats                            # resumes: 1

# K steps in flight
CONNEX_PP_INFLIGHT=2 docker compose -f docker/docker-compose.yml up -d
# RDMA on one box: MOONCAKE_PROTOCOL=rdma and uncomment the /dev/infiniband lines
```

`make down` stops everything, including the proxy.

## Tier 2: several hosts (docker, RDMA or TCP)

Run one container per stage with `docker/launch_stage.sh` (the image's
default command). Every container uses the host network, so set `HOST_IP` to
the address peers should use (the RDMA NIC's IP for RDMA). Example: 4 stages
on hosts A, B, C, D, router on A.

```bash
IMG=<registry>/connex-vllm:0.1.0
COMMON="--network host --ipc host --shm-size 16g
  -e HF_TOKEN -v $HOME/.cache/huggingface:/root/.cache/huggingface
  -e MODEL_PATH=meta-llama/Meta-Llama-3-8B -e MODEL_NAME=llama3
  -e NUM_STAGES=4 -e ROUTER_ADDR=HOST_A_IP:15555
  -e CONNEX_HEARTBEAT_INTERVAL_S=0.5 -e CONNEX_PP_INFLIGHT=4"
RDMA="--device /dev/infiniband --cap-add IPC_LOCK -e MOONCAKE_PROTOCOL=rdma"   # omit for TCP
# with /dev/infiniband present the stages use connex-rs (CONNEX_PP_BACKEND=auto);
# MOONCAKE_PROTOCOL only matters for CONNEX_PP_BACKEND=mooncake

# host A: router (2 s lease) and stage 0
docker run -d --name connex-router --network host -e CONNEX_HEARTBEAT_TIMEOUT_S=2 \
  -e CONNEX_UNHEALTHY_GRACE_S=10 $IMG python3 -m connex.front_router \
  --api-address tcp://*:15555 --event-address tcp://*:15560
docker run -d --name s0 $COMMON $RDMA --gpus device=0 -e STAGE=0 -e HOST_IP=HOST_A_IP $IMG
# hosts B, C, D
docker run -d --name s1 $COMMON $RDMA --gpus device=0 -e STAGE=1 -e HOST_IP=HOST_B_IP $IMG
docker run -d --name s2 $COMMON $RDMA --gpus device=0 -e STAGE=2 -e HOST_IP=HOST_C_IP $IMG
docker run -d --name s3 $COMMON $RDMA --gpus device=0 -e STAGE=3 -e HOST_IP=HOST_D_IP $IMG
# a second replica of stage 2 on host B (ports shift with REPLICA)
docker run -d --name s2r1 $COMMON $RDMA --gpus device=1 -e STAGE=2 -e REPLICA=1 -e HOST_IP=HOST_B_IP $IMG

docker/smoke_test.sh http://HOST_A_IP:5100 llama3
# optional resume proxy (any host): python -m connex.serving.proxy --endpoints http://HOST_A_IP:5100,...
```

Ports per stage replica: Connex `25500 + 10*REPLICA`, Mooncake
`50000 + 100*REPLICA` (+ stage index), connex-rs `51000 + 100*REPLICA`
(+ stage index), stage-0 API `5100 + REPLICA`; open them between hosts. With several NICs, `MC_TCP_BIND_ADDRESS` defaults to
`HOST_IP`. For GPUDirect RDMA, load `nvidia_peermem` on the hosts; with
`GPU_SELECT=nic-local` and `--gpus all` the entrypoint prefers GPUs on the
NIC's PCIe side. All knobs: [`configuration.md`](configuration.md).

## Tier 3: Kubernetes

```bash
NS=connex IMAGE=<registry>/connex-vllm:0.1.0
kubectl create namespace $NS
kubectl -n $NS create secret generic hf-token --from-literal=HF_TOKEN=$HF_TOKEN
G() { python deploy/k8s/manifests.py --namespace $NS --image $IMAGE --num-stages 4 \
   --model-hf-id meta-llama/Meta-Llama-3-8B --hf-token-secret hf-token "$@"; }   # bash and zsh
G router | kubectl apply -f -
G pipeline --set CONNEX_PP_INFLIGHT=4 | kubectl apply -f -
G stage --stage 2 --replica 1 --set CONNEX_PP_INFLIGHT=4 | kubectl apply -f -   # failover replica
G proxy --proxy-endpoints http://cte-s0-r0:5100 | kubectl apply -f -           # optional, Deployment cte-proxy
CONNEX_NAMESPACE=$NS benchmarks/k8s/wait_ready.sh cte
kubectl -n $NS port-forward svc/cte-s0-r0 5100:5100 &
docker/smoke_test.sh http://127.0.0.1:5100 cte-model
```

Defaults are portable: pod network, Mooncake TCP with P2P handshake, one
`nvidia.com/gpu` per stage, no node pinning, an `emptyDir` HF cache
(`--hf-cache-pvc` to reuse a PVC you created; `--model-pvc` /
`--model-host-path` for a pre-staged model). For RDMA add
`--host-network --ib --protocol rdma` (privileged pods with
`/dev/infiniband`). If the device plugin time-shares GPUs (several pods per
physical GPU), add `--gpu-select auto` (or `nic-local` for GPUDirect RDMA)
so each stage picks a GPU with free memory itself. If the namespace already
runs Connex, use your own `--prefix` and `--router-name` (on every `G` call):
`kubectl apply` replaces objects of the same name. All options: [`deploy/k8s/README.md`](../deploy/k8s/README.md).

## Benchmarks against any deployment

The benchmark scripts take endpoints from flags or environment variables
and use your current `kubectl` context for churn; sample traces are in
`benchmarks/traces/` (`make traces` regenerates them).

```bash
python benchmarks/k8s/trace_load.py --trace benchmarks/traces/gamma_qps2.0.csv \
    --endpoints http://127.0.0.1:8000,http://127.0.0.1:8010 --model connex-demo --out run.json
python benchmarks/k8s/closed_loop.py --url http://127.0.0.1:8000 --model connex-demo --clients 32
python benchmarks/k8s/failover_correctness.py run --url http://127.0.0.1:8010 --model connex-demo \
    --n 16 --max-tokens 300 --kill "docker compose -f docker/docker-compose.yml kill stage1-r1" \
    --kill-after 7 --out kill.json      # tier 1
python benchmarks/k8s/failover_correctness.py score --url http://127.0.0.1:8000 --model connex-demo kill.json
python benchmarks/proxy_overhead.py --concurrency 16    # CPU only
```

`benchmarks/k8s/run_trace.sh`, `run_churn.sh` and `churn_matrix.sh` drive
the Kubernetes tier (see their headers and `docs/results.md`).
