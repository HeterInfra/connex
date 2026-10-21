# Deploying an external-PP pipeline on Kubernetes

`manifests.py` prints a `v1/List` of plain Kubernetes objects (no Helm, no
operator). Each pipeline stage replica is one `Deployment` with one GPU that
runs `docker/launch_stage.sh` from the image built by `docker/Dockerfile`; the
Connex front-router is a small `Deployment` + `Service` named
`connex-front-router` (`--router-name` to change it, e.g. when several
pipelines share a namespace).

## Quick start

```bash
NS=connex IMAGE=registry.example.com/connex-vllm:dev    # your image (make image / make push)
kubectl create namespace $NS
kubectl -n $NS create secret generic hf-token --from-literal=HF_TOKEN=$HF_TOKEN
G() { python deploy/k8s/manifests.py --namespace $NS --image $IMAGE \
   --model-hf-id meta-llama/Meta-Llama-3-8B --hf-token-secret hf-token "$@"; }   # bash and zsh

G router | kubectl apply -f -
G pipeline --prefix demo --num-stages 2 | kubectl apply -f -
CONNEX_NAMESPACE=$NS benchmarks/k8s/wait_ready.sh demo

# a second replica of stage 1 (replica index 1):
G stage --prefix demo --num-stages 2 --stage 1 --replica 1 | kubectl apply -f -

# optional request retry/resume proxy (connex/serving/proxy.py) on :5199
G proxy --prefix demo --proxy-endpoints http://demo-s0-r0:5100 | kubectl apply -f -

# smoke test through a port-forward
kubectl -n $NS port-forward svc/demo-s0-r0 5100:5100 &
docker/smoke_test.sh http://127.0.0.1:5100 demo-model
```

Stage 0 serves the OpenAI API on port `5100 + replica`, behind a Service
named `PREFIX-s0-rREPLICA`, and has a readiness probe on `/v1/models`. Stages 1+ are ready once their log
shows `entering tight pipeline loop` (`wait_ready.sh` checks both).

## Options

Every option can be given as a flag, as an environment variable
`CONNEX_K8S_<OPTION>` (e.g. `CONNEX_K8S_IMAGE`), or in a JSON file passed
with `--config` (keys are option names with `_`).

| Option | Default | Meaning |
|---|---|---|
| `--namespace` | `connex` | namespace of every object |
| `--image` | `connex-vllm:latest` | stage image (docker/Dockerfile) |
| `--router-image` | same as `--image` | front-router image (needs only `connex` + pyzmq/msgpack) |
| `--image-pull-secret` | none | pull secret name |
| `--model-hf-id` | | Hugging Face model id, downloaded by each stage at start (one of the three model options is required) |
| `--hf-token-secret` | none | Secret with key `HF_TOKEN` for gated models (`kubectl create secret generic hf-token --from-literal=HF_TOKEN=...`) |
| `--hf-cache-pvc` | emptyDir | PVC you created for the HF cache, so restarts do not re-download |
| `--model-pvc` / `--model-host-path` | | a PVC you created / a node directory holding the model, mounted read-only at `/model` |
| `--model-name` | `<prefix>-model` | served model name |
| `--num-stages` / `--num-layers` | 2 / 32 | layers are split evenly; the last stage takes the remainder |
| `--nodes` | unpinned | comma-separated node per stage for `pipeline` |
| `--node` | unpinned | node for `stage` |
| `--router-node` | unpinned | pin the router (recommended next to the stages with host network) |
| `--host-network` / `--no-host-network` | off | host network (for RDMA without an RDMA-aware CNI); ports are offset per replica and per stage, so any placement works |
| `--ib` | off | mount `/dev/infiniband`, run privileged (RDMA) |
| `--backend` | `connexte` | `connexte` (Mooncake TE data plane) or `connex` (UCX) |
| `--protocol` | `tcp` | Mooncake protocol, `tcp` or `rdma` |
| `--metadata` | `p2p` | `p2p` = `MOONCAKE_METADATA_SERVER=P2PHANDSHAKE` (no etcd); `etcd` needs `--etcd HOST:PORT` |
| `--gpu-select` | `none` | `none`: the device plugin assigns the GPU; `auto` / `nic-local`: the entrypoint picks one itself (time-shared GPUs, see `docker/launch_stage.sh`) |
| `--gpu-resource` | `nvidia.com/gpu` | extended resource requested per stage |
| `--set KEY=VAL` | | extra stage environment, repeatable (e.g. `--set CONNEX_PP_INFLIGHT=4`) |
| `--router-name` | `connex-front-router` | router Deployment/Service name; stages point `ROUTER_ADDR` at it. Pass the same value to `router`, `pipeline` and `stage` |
| `--proxy-endpoints` | (required for `proxy`) | stage-0 URLs for the resume proxy Deployment `PREFIX-proxy` |
| `--proxy-port` | 5199 | resume proxy port |

With `--gpu-select auto|nic-local` the pod gets `NVIDIA_VISIBLE_DEVICES=all`
and the entrypoint picks the GPU with the most free memory. This is meant for
clusters where GPUs are time-shared and the device plugin may hand out a busy
GPU. With exclusive GPUs use `--gpu-select none`.

## Notes and pitfalls

- With host networking and several NICs, Mooncake advertises the first NIC
  for its handshake RPC. The entrypoint sets `MC_TCP_BIND_ADDRESS=$HOST_IP`
  (the node IP from the downward API); override it if peers must use another
  interface.
- With `--metadata etcd`, `MC_METADATA_CLUSTER_ID` is set per namespace and
  prefix so several pipelines can share one etcd.
- Object names are `PREFIX-sS-rR` for stages, `PREFIX-proxy` for the proxy
  and `--router-name` for the router. Before applying into a namespace that
  already runs Connex, pick a prefix and router name that are not taken:
  `kubectl apply` replaces an existing object of the same name.
- Start load only after every stage is serving. Stage 0's readiness says
  nothing about stages 1+.
- If host-network pods cannot reach the router's Service from some nodes
  (overlay issues), pin the router with `--router-node`.
- The router lease is 2 s (`CONNEX_HEARTBEAT_TIMEOUT_S`) with 0.5 s stage
  heartbeats. A busy engine can miss a lease; that only makes the peer
  suspect, and it is probed (TCP connect) before anything fails over.
- Killing a single pod and re-adding it under the same replica identity can
  break its Mooncake peers; restart the whole replica chain instead.
