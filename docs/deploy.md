# Deploying Connex

Three ways to run a Connex pipeline with vLLM, from one GPU box to
Kubernetes, plus a CPU-only demo. Nothing here depends on a particular
cluster: you build the image yourself from public sources, the model comes
from Hugging Face (or a directory you mount), and neither data plane needs a
metadata service (connex-rs addresses peers by URL).

Pipeline stages run on the latest vLLM main with the three changes proposed
in RFC [vllm#63818](https://github.com/vllm-project/vllm/issues/63818)
([vllm#63897](https://github.com/vllm-project/vllm/pull/63897),
[#63898](https://github.com/vllm-project/vllm/pull/63898),
[#63899](https://github.com/vllm-project/vllm/pull/63899)) and the plugin
`connex.adapters.vllm.pp` ([`integrations/vllm/README.md`](../integrations/vllm/README.md#upstream-vllm-no-fork)).
Stage 0 serves the API and schedules; it sends each step to one replica of
every downstream stage over connex-rs (CUDA IPC between GPUs of one host,
UCX/RDMA between hosts) and moves the route to another replica when one
dies. The legacy external-PP fork is still available, see
[Legacy: external-PP fork](#legacy-external-pp-fork).

| Tier | Hardware | Data plane | What you get |
|---|---|---|---|
| 0. CPU demo | any Linux/macOS machine, Python 3.10+ | ZeroMQ / UCX-TCP | directory discovery between two stages; the resume proxy surviving a backend kill (fake engines) |
| 1. single GPU box | 1 host with 3 NVIDIA GPUs (≥ 24 GB each), NVIDIA container toolkit; Llama-3-8B | connex-rs: CUDA IPC between the stages | a 2-stage vLLM pipeline whose stage 1 has two replicas; in-flight failover between them |
| 2. multi-node docker | ≥ 2 hosts with NVIDIA GPUs; ConnectX/InfiniBand (RoCE) NICs for RDMA, or plain Ethernet with TCP | connex-rs over RDMA (GPUDirect when `nvidia_peermem` is loaded) or TCP | the same pipeline across hosts; failover to a replica on another host; a restarted replica rejoins as a spare |
| 3. Kubernetes | a cluster with the NVIDIA device plugin; RDMA optional | connex-rs on the pod network by default; RDMA opt-in (`--host-network --ib`) | generated plain manifests: stages, stage-0 Service, headless Services for the downstream replicas |

## Build the image (tiers 1-3)

```bash
make image IMAGE=<registry>/connex-vllm:dev [VLLM_REF=<vLLM main commit>] \
    [BASE_IMAGE=<vLLM image>] [VLLM_INSTALL=precompiled|source|none]
make push IMAGE=<registry>/connex-vllm:dev       # tiers 2 and 3, if hosts pull from a registry
```

The default build (`PP_ENGINE=upstream`) starts from `vllm/vllm-openai:latest`,
clones vLLM main (`VLLM_REPO` at `VLLM_REF`, default `main`) and merges the
three PP branches from `https://github.com/yanyinglin/vllm` <!-- release-scan: allow -->
(`VLLM_PP_BRANCHES`: `feat/pluggable-pp-group`, `feat/external-pp-stages`,
`feat/recompute-lost-kv`). The changes are Python only, so
`VLLM_INSTALL=precompiled` installs the merged tree with the compiled kernels
of the nearest vLLM main nightly wheel (`VLLM_USE_PRECOMPILED=1`);
`VLLM_INSTALL=source` compiles vLLM (slow; needs a CUDA devel base image and
`MAX_JOBS`), and `VLLM_INSTALL=none` keeps the vLLM of `BASE_IMAGE` (an image
that already has the changes). If main has moved so far that a branch no
longer merges, the build stops and names it; pass an older `VLLM_REF`. The
build then checks that vLLM has `--pp-group-cls`, `--pipeline-stage-rank` and
`KVConnectorOutput.lost_all_kv`. Once the changes are merged upstream, a
plain vLLM image will do.

The image also builds [connex-rs](https://github.com/HeterInfra/connex-rs)
in a builder stage: UCX (`UCX_VERSION`, with verbs, rdmacm
and CUDA) into `/opt/ucx`, and the Rust wheel against it (`RUST_TOOLCHAIN`);
the pip UCX wheels lack InfiniBand transports. The source is the
`third_party/connex-rs` submodule of the build context (run
`git submodule update --init` first, or clone with `--recurse-submodules`);
without it the build clones `CONNEX_RS_REPO` at `CONNEX_RS_REF` (default
the connex-rs GitHub repository) and fails with a message if that
is not reachable. The upstream PP path needs connex-rs; `BUILD_CONNEX_RS=0`
only makes sense for the legacy fork (Mooncake only).

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

`docker/docker-compose.yml` runs a 2-stage pipeline on the host network:
`stage0` (API on :8000, GPU0) and two replicas of stage 1, `stage1` (GPU1,
the preferred route) and `stage1-r1` (GPU2, spare). vLLM splits the layers.

```bash
export CONNEX_IMAGE=<registry>/connex-vllm:dev       # or build: add --build
export HF_TOKEN=hf_...                               # MODEL defaults to meta-llama/Meta-Llama-3-8B
docker compose -f docker/docker-compose.yml up -d
docker/smoke_test.sh                                 # OK: ' Paris...'

# local model directory instead of a download. It must hold real files (not
# a Hugging Face cache snapshot, whose files are symlinks into ../../blobs)
# and be visible to the Docker daemon.
MODEL_DIR=/path/to/Llama-3-8B docker compose -f docker/docker-compose.yml \
    -f docker/compose.local-model.yml up -d

# in-flight failover: start long requests, then kill the stage-1 replica on
# the route; stage 0 moves the route to stage1-r1, the running requests are
# recomputed there and complete (a delay, no error)
for i in 1 2 3 4; do curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"connex-demo","prompt":"Tell a long story.","max_tokens":400,"temperature":0}' & done
sleep 5; docker compose -f docker/docker-compose.yml kill stage1; wait
# the killed replica back: same command, it rejoins as a spare
docker compose -f docker/docker-compose.yml up -d stage1
```

GPUs: `GPU0`, `GPU1`, `GPU2` (default 0, 1, 2). One stage-1 replica only:
`docker compose -f docker/docker-compose.yml up -d stage1 stage0` (stage 0
still lists both and uses the live one). Every stage must use the same
`NUM_GPU_BLOCKS` (default 8192; the block ids in the steps are stage 0's).
The compose file pins connex-rs to TCP, shared memory and CUDA IPC
(`UCX_TLS`); for RDMA uncomment the `/dev/infiniband` lines and unset
`UCX_TLS`. `make down` stops everything.

## Tier 2: several hosts (docker, RDMA or TCP)

Run one container per stage replica with `docker/launch_stage.sh` (the
image's default command). Every container uses the host network, so set
`HOST_IP` to the address peers should use (the RDMA NIC's IP for RDMA).
Example: 3 stages, stage 0 on host A, two replicas of stages 1 and 2 on
hosts B and C.

```bash
IMG=<registry>/connex-vllm:dev
COMMON="--network host --ipc host --shm-size 16g
  -e HF_TOKEN -v $HOME/.cache/huggingface:/root/.cache/huggingface
  -e MODEL_PATH=meta-llama/Meta-Llama-3-8B -e MODEL_NAME=llama3 -e NUM_STAGES=3"
RDMA="--device /dev/infiniband --cap-add IPC_LOCK"   # omit for TCP

# downstream replicas: control port 47000 + 1000*STAGE + 16*REPLICA (data port of TP rank r: +1+r)
docker run -d --name s1r0 $COMMON $RDMA --gpus device=0 -e STAGE=1 -e REPLICA=0 -e HOST_IP=HOST_B_IP $IMG
docker run -d --name s2r0 $COMMON $RDMA --gpus device=1 -e STAGE=2 -e REPLICA=0 -e HOST_IP=HOST_B_IP $IMG
docker run -d --name s1r1 $COMMON $RDMA --gpus device=0 -e STAGE=1 -e REPLICA=1 -e HOST_IP=HOST_C_IP $IMG
docker run -d --name s2r1 $COMMON $RDMA --gpus device=1 -e STAGE=2 -e REPLICA=1 -e HOST_IP=HOST_C_IP $IMG
# host A: stage 0, OpenAI API on :5100
docker run -d --name s0 $COMMON $RDMA --gpus device=0 -e STAGE=0 -e HOST_IP=HOST_A_IP \
  -e PP_STAGES='[["HOST_B_IP:48000","HOST_C_IP:48016"],["HOST_B_IP:49000","HOST_C_IP:49016"]]' $IMG

docker/smoke_test.sh http://HOST_A_IP:5100 llama3
```

Open the control and data ports and the API port between hosts. A killed
replica restarted with the same command rejoins as a spare. For GPUDirect
RDMA, load `nvidia_peermem` on the hosts; with `GPU_SELECT=nic-local` and
`--gpus all` the entrypoint prefers GPUs on the NIC's PCIe side. Tensor
parallelism per stage: `-e TP=2` with two GPUs per container. Variables:
the header of `docker/launch_stage.sh` and
[`integrations/vllm/README.md`](../integrations/vllm/README.md#upstream-vllm-no-fork).

## Tier 3: Kubernetes

```bash
NS=connex IMAGE=<registry>/connex-vllm:dev
kubectl create namespace $NS
kubectl -n $NS create secret generic hf-token --from-literal=HF_TOKEN=$HF_TOKEN
G() { python deploy/k8s/manifests.py --namespace $NS --image $IMAGE --num-stages 4 \
   --model-hf-id meta-llama/Meta-Llama-3-8B --hf-token-secret hf-token "$@"; }   # bash and zsh
G pipeline --stage-replicas 2 | kubectl apply -f -
CONNEX_NAMESPACE=$NS benchmarks/k8s/wait_ready.sh cte
kubectl -n $NS port-forward svc/cte-s0-r0 5100:5100 &
docker/smoke_test.sh http://127.0.0.1:5100 cte-model
```

Defaults are portable: pod network, one `nvidia.com/gpu` per stage, no node
pinning, an `emptyDir` HF cache (`--hf-cache-pvc` to reuse a PVC you created;
`--model-pvc` / `--model-host-path` for a pre-staged model). Each downstream
replica gets a headless Service that stage 0 resolves at start; for replicas
that rejoin after a restart under the same address, and for RDMA, use
`--host-network --nodes ...` (and `--ib`). If the device plugin time-shares
GPUs, add `--gpu-select auto` (or `nic-local` for GPUDirect RDMA). If the
namespace already runs Connex, use your own `--prefix`. All options:
[`deploy/k8s/README.md`](../deploy/k8s/README.md).

## Legacy: external-PP fork

The original integration ran each stage on the external-PP vLLM fork
(commit `380186d57`, `--pipeline-stage-mode external`) with 16 engine patches
(`integrations/vllm/patches/`), the Connex front-router for membership and
Mooncake TE, connex-rs, UCX or ZeroMQ as data plane. It is kept for the
features the upstream path does not have yet (see the comparison in
[`integrations/vllm/README.md`](../integrations/vllm/README.md#compared-with-the-fork-path)).
The fork is not part of this repository: `VLLM_FORK_REPO` is the git URL of
a copy you can access.

```bash
make image PP_ENGINE=fork IMAGE=<registry>/connex-vllm:fork VLLM_FORK_REPO=<git URL of the fork> \
    BASE_IMAGE=<vLLM image matching the fork's torch> [VLLM_FORK_REF=<commit>] [FORK_INSTALL=source]
```

`FORK_INSTALL=overlay` (default) copies the fork's Python tree over the
compiled vLLM in `BASE_IMAGE`, so the base image must match the fork's torch
and kernel ABI (the vLLM release the fork is based on); `FORK_INSTALL=source`
compiles the fork instead (slow). The build fails at once with a clear
message if `VLLM_FORK_REPO` is unset. The image then applies the engine
patches and copies the communicators, and its stages start with
`PP_ENGINE=fork`. The data plane is chosen per stage (`CONNEX_PP_BACKEND`,
default `auto`): connex-rs when the host has an RDMA device, Mooncake
otherwise (P2P handshake, no etcd).

The tiers below are the fork's versions of tiers 1-3: compose file
`docker/docker-compose.fork.yml` (`make up PP_ENGINE=fork`), `PP_ENGINE=fork`
and `ROUTER_ADDR` for `docker run`, and `--engine fork` for the manifests.

### Fork tier 1: one GPU box (docker compose)

The compose file runs the front-router and **two replicas of a 2-stage
pipeline** on the host network: replica 0 is `stage0` (API on :8000, layers
0-15) + `stage1` (layers 16-31), replica 1 is `stage0-r1` (API on :8010) +
`stage1-r1`. Both register with the same front-router, so a stage of one
replica fails over to the same stage of the other with requests in flight.
Mooncake runs over TCP with P2P handshake.

```bash
export CONNEX_IMAGE=<registry>/connex-vllm:fork     # or build: add --build, VLLM_FORK_REPO and BASE_IMAGE
export HF_TOKEN=hf_...                               # MODEL defaults to meta-llama/Meta-Llama-3-8B
docker compose -f docker/docker-compose.fork.yml --profile proxy up -d
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
docker compose -f docker/docker-compose.fork.yml up -d router stage0 stage1
docker/smoke_test.sh
docker compose -f docker/docker-compose.fork.yml --profile proxy up -d
docker/smoke_test.sh http://127.0.0.1:8010
```
 One replica only: `docker compose -f
docker/docker-compose.fork.yml up -d router stage0 stage1`. If the router ports
are taken, set `ROUTER_PORT` / `ROUTER_EVENT_PORT` (default 15555 / 15560).

```bash
# local model directory instead of a download. It must hold real files (not
# a Hugging Face cache snapshot, whose files are symlinks into ../../blobs)
# and be visible to the Docker daemon.
MODEL_DIR=/path/to/Llama-3-8B docker compose -f docker/docker-compose.fork.yml \
    -f docker/compose.local-model.fork.yml --profile proxy up -d

# in-flight handover across replicas: start long requests on replica 1,
# then kill its stage 1; they continue on replica 0's stage 1 after a ~3 s
# stall (3 s data-path timeout + probe, as in the k8s manifests)
for i in 1 2 3 4; do curl -s http://127.0.0.1:8010/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"connex-demo","prompt":"Tell a long story.","max_tokens":400,"temperature":0}' & done
sleep 5; docker compose -f docker/docker-compose.fork.yml kill stage1-r1; wait
# replica 1's stage 1 back (scale up): it re-registers and is used by the
# next failover; stage 0 stays sticky on its current peer until that dies.
# With Mooncake a re-added stage can break its peers (seen with RDMA, see
# Limitations in the README): restart the whole chain (stage0-r1 too).
# connex-rs (RDMA hosts) handles a single-stage rejoin.
docker compose -f docker/docker-compose.fork.yml up -d stage1-r1

# stage-0 death behind the resume proxy: stream through :5199 and kill the
# stage 0 serving it (the one whose /metrics shows a running request)
curl -sN http://127.0.0.1:5199/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"connex-demo","prompt":"Tell a long story.","max_tokens":400,"temperature":0,"stream":true}' &
sleep 5; docker compose -f docker/docker-compose.fork.yml kill stage0-r1   # or stage0
curl -s http://127.0.0.1:5199/proxy/stats                            # resumes: 1

# K steps in flight
CONNEX_PP_INFLIGHT=2 docker compose -f docker/docker-compose.fork.yml up -d
# RDMA on one box: MOONCAKE_PROTOCOL=rdma and uncomment the /dev/infiniband lines
```

`make down` stops everything, including the proxy.

### Fork tier 2: several hosts

Run one container per stage with `docker/launch_stage.sh` (the image's
default command). Every container uses the host network, so set `HOST_IP` to
the address peers should use (the RDMA NIC's IP for RDMA). Example: 4 stages
on hosts A, B, C, D, router on A.

```bash
IMG=<registry>/connex-vllm:fork
COMMON="--network host --ipc host --shm-size 16g
  -e HF_TOKEN -v $HOME/.cache/huggingface:/root/.cache/huggingface
  -e MODEL_PATH=meta-llama/Meta-Llama-3-8B -e MODEL_NAME=llama3
  -e PP_ENGINE=fork -e NUM_STAGES=4 -e ROUTER_ADDR=HOST_A_IP:15555
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

### Fork tier 3: Kubernetes

```bash
NS=connex IMAGE=<registry>/connex-vllm:fork
kubectl create namespace $NS
kubectl -n $NS create secret generic hf-token --from-literal=HF_TOKEN=$HF_TOKEN
G() { python deploy/k8s/manifests.py --namespace $NS --image $IMAGE --num-stages 4 \
   --model-hf-id meta-llama/Meta-Llama-3-8B --hf-token-secret hf-token "$@"; }   # bash and zsh
G router --engine fork | kubectl apply -f -
G pipeline --engine fork --set CONNEX_PP_INFLIGHT=4 | kubectl apply -f -
G stage --engine fork --stage 2 --replica 1 --set CONNEX_PP_INFLIGHT=4 | kubectl apply -f -   # failover replica
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
    --n 16 --max-tokens 300 --kill "docker compose -f docker/docker-compose.fork.yml kill stage1-r1" \
    --kill-after 7 --out kill.json      # fork tier 1
python benchmarks/k8s/failover_correctness.py score --url http://127.0.0.1:8000 --model connex-demo kill.json
python benchmarks/proxy_overhead.py --concurrency 16    # CPU only
```

`benchmarks/k8s/run_trace.sh`, `run_churn.sh` and `churn_matrix.sh` drive
the Kubernetes tier (see their headers and `docs/results.md`).
For the upstream path, `benchmarks/pp_upstream/failover_load.py` kills a
downstream replica under load and measures the handover (tier 1:
`--kill "docker compose -f docker/docker-compose.yml kill stage1"`).
