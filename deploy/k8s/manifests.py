# SPDX-License-Identifier: Apache-2.0
"""Generate plain Kubernetes manifests for a vLLM pipeline on Connex.

Every stage replica is its own Deployment and runs docker/launch_stage.sh from
the Connex image (docker/Dockerfile).

--engine upstream (default): vLLM main with the proposed PP changes and the
plugin connex.adapters.vllm.pp. Stage 0 serves the API and drives one replica
of every downstream stage; `pipeline` makes --stage-replicas replicas of each
downstream stage, each behind a headless Service (PREFIX-sK-rR) that stage 0
resolves at start. No front-router is needed.

--engine fork (legacy external-PP fork): a small front-router Deployment +
Service provides the control plane.

    python deploy/k8s/manifests.py pipeline --namespace demo --image REG/connex-vllm:TAG \\
        --stage-replicas 2 --model-hf-id meta-llama/Meta-Llama-3-8B > pipe.yaml
    # legacy fork:
    python deploy/k8s/manifests.py router --engine fork --namespace demo --image REG/connex-vllm:TAG > router.yaml
    python deploy/k8s/manifests.py pipeline --engine fork --namespace demo --image REG/connex-vllm:TAG \\
        --nodes node-a,node-b --num-stages 2 --model-host-path /models/Llama-3-8B > pipe.yaml
    # fork: extra replica of stage 1 on node-a (replica index 1):
    python deploy/k8s/manifests.py stage --engine fork --stage 1 --replica 1 --node node-a ... > s1r1.yaml
    # optional request retry/resume proxy (connex/serving/proxy.py) on :5199
    python deploy/k8s/manifests.py proxy --endpoints http://NODE_A_IP:5100,http://NODE_B_IP:5101 > proxy.yaml
    kubectl apply -f router.yaml -f pipe.yaml

All options can also come from a config file (--config, a JSON
object whose keys are the long option names with '-' replaced by '_') or from
CONNEX_K8S_* environment variables (e.g. CONNEX_K8S_IMAGE, CONNEX_K8S_NAMESPACE).

Defaults are portable: pod network, (fork) Mooncake over TCP with P2P handshake (no
etcd), one `nvidia.com/gpu` per stage from the device plugin, no node pinning.
Opt in to the rest:
  --host-network      run stages on the host network (needed for RDMA/IB
                      without a CNI that exposes the HCA); ports are spread per
                      replica so replicas can share a node.
  --ib                mount /dev/infiniband and run privileged (RDMA); use
                      with --protocol rdma.
  --nodes / --node    pin stages to nodes.
  --gpu-select auto   let the entrypoint pick the freest GPU itself (for
                      clusters whose GPUs are time-shared).
  --metadata p2p      Mooncake P2PHANDSHAKE metadata (no etcd needed, default)
  --metadata etcd     Mooncake etcd metadata at --etcd ADDR:PORT
"""
import argparse
import json
import os

DEFAULTS = {
    "engine": "upstream",          # upstream | fork (legacy external-PP fork)
    "stage_replicas": 1,           # upstream `pipeline`: replicas per downstream stage
    "num_gpu_blocks": 8192,        # upstream: --num-gpu-blocks-override, same on every stage
    "namespace": "connex",
    "image": "connex-vllm:latest",
    "router_image": "",            # empty: same as --image
    "model_host_path": "",
    "model_pvc": "",
    "model_hf_id": "",             # Hugging Face id, downloaded by each stage
    "hf_token_secret": "",         # Secret with key HF_TOKEN (gated models)
    "hf_cache_pvc": "",            # PVC for the HF cache (default: emptyDir)
    "model_name": "",              # empty: PREFIX-model
    "num_layers": 32,
    "num_stages": 2,
    "nodes": "",
    "router_node": "",
    "host_network": False,
    "ib": False,
    "metadata": "p2p",
    "etcd": "",
    "protocol": "tcp",
    "gpu_select": "none",          # see docker/launch_stage.sh
    "gpu_resource": "nvidia.com/gpu",
    "shm_size": "16Gi",
    "image_pull_secret": "",
    "proxy_port": 5199,
    "proxy_endpoints": "",         # comma-separated stage-0 URLs for `proxy`
    # Object name of the router Deployment/Service. Give each pipeline that
    # shares a namespace its own (e.g. PREFIX-front-router) so applying one
    # pipeline's router never replaces another's.
    "router_name": "connex-front-router",
    "router_port": 15555,
    "events_port": 15560,
}

BACKEND_ENV = {
    # UCX transport (backend "connex")
    "connex": {
        "UCX_TLS": "rc,sm,self,cuda_copy", "UCX_RNDV_THRESH": "1048576",
        "UCX_WARN_UNUSED_ENV_VARS": "n", "UCX_MEMTYPE_CACHE": "y",
        "UCX_CUDA_COPY_MAX_REG_RATIO": "1.0", "CONNEX_GPU_DIRECT": "0",
        "CONNEX_UCX_DIRECT_RECV": "0",
    },
    # ConnexTE: Mooncake TE data plane + Connex control plane (backend "connexte")
    "connexte": {
        "MOONCAKE_DEVICE_NAME": "", "CONNEXTE_SEND_TIMEOUT_MS": "3000",
        "CONNEXTE_RECV_TIMEOUT_MS": "3000", "CONNEXTE_SEND_RETRIES": "3",
        # 15 x adaptive timeout outlasts lease-based failover (~3-4 s)
        "CONNEXTE_RECV_RETRIES": "15",
    },
}


def env_list(d):
    return [{"name": k, "value": str(v)} for k, v in d.items()]


def control_port(idx, replica):
    """Upstream: control port of downstream stage `idx`, replica `replica`
    (data port = control port + 1); distinct per stage and replica so any
    placement works on the host network."""
    return 47020 + 100 * idx + 2 * replica


def upstream_env(c, prefix, idx, replica):
    ns = c.namespace
    env = {
        "PP_ENGINE": "upstream", "STAGE": idx, "NUM_STAGES": c.num_stages, "REPLICA": replica,
        "MODEL_NAME": c.model_name or f"{prefix}-model", "MODEL_PATH": c.model_hf_id or "/model",
        "GPU_SELECT": c.gpu_select, "NUM_GPU_BLOCKS": c.num_gpu_blocks,
        "API_PORT": 5100 + replica,
    }
    if idx == 0:
        env["PP_STAGES"] = json.dumps([
            [f"{prefix}-s{k}-r{r}.{ns}.svc.cluster.local:{control_port(k, r)}"
             for r in range(c.stage_replicas)] for k in range(1, c.num_stages)])
    else:
        env["CONTROL_PORT"] = control_port(idx, replica)
    return env


def stage(c, backend, prefix, idx, node, replica=0, overrides=None):
    ns = c.namespace
    name = f"{prefix}-s{idx}-r{replica}"
    labels = {"app": "connex-stage", "pipeline": prefix, "stage": str(idx), "replica": str(replica)}
    if c.engine == "upstream":
        env = upstream_env(c, prefix, idx, replica)
    else:
        env = fork_env(c, backend, prefix, idx, replica)
    if c.gpu_select != "none":
        env["NVIDIA_VISIBLE_DEVICES"] = "all"  # launch_stage.sh picks the GPU
    env.update(overrides or {})
    return stage_objects(c, prefix, idx, replica, name, labels, env, node)


def fork_env(c, backend, prefix, idx, replica):
    ns = c.namespace
    env = {
        "BACKEND": backend, "STAGE": idx, "NUM_STAGES": c.num_stages, "REPLICA": replica,
        "NUM_LAYERS": c.num_layers,
        "MODEL_NAME": c.model_name or f"{prefix}-model", "MODEL_PATH": c.model_hf_id or "/model",
        "ROUTER_ADDR": f"{c.router_name}.{ns}.svc.cluster.local:{c.router_port}",
        "GPU_SELECT": c.gpu_select,
        # Router lease 2 s. A busy K>1 engine can miss it (starved heartbeat
        # thread); that is only a suspicion: the peer is probed (TCP connect)
        # before anything fails over (CONNEXTE_FAST_HANDOVER).
        "CONNEX_HEARTBEAT_INTERVAL_S": "0.5",
        # Ports stay distinct when replicas share a node on the host network
        # (Mooncake adds the stage index itself). With --host-network two
        # stages of one replica may also share a node, so the pipeline port
        # and the metrics port of stages > 0 are offset per stage as well;
        # stage 0's API port stays 5100 + replica.
        "BASE_PORT": 25500 + 10 * replica + (100 * idx if c.host_network else 0),
        "API_PORT": 5100 + replica + (10 * idx if c.host_network else 0),
        "MOONCAKE_BASE_PORT": 50000 + 100 * replica,
        **BACKEND_ENV[backend],
    }
    if backend == "connexte":
        env["MOONCAKE_PROTOCOL"] = c.protocol
        if c.metadata == "etcd":
            if not c.etcd:
                raise SystemExit("--metadata etcd needs --etcd HOST:PORT")
            env.update(MOONCAKE_METADATA_BACKEND="etcd", MOONCAKE_METADATA_SERVER=c.etcd,
                       MC_METADATA_CLUSTER_ID=f"connex-{ns}-{prefix}")
        else:
            env["MOONCAKE_METADATA_SERVER"] = "P2PHANDSHAKE"
    return env


def stage_objects(c, prefix, idx, replica, name, labels, env, node):
    ns = c.namespace
    container = {
        "name": "stage", "image": c.image, "imagePullPolicy": "IfNotPresent",
        "command": ["/opt/connex/docker/launch_stage.sh"],
        "env": env_list(env) + [{"name": "HOST_IP", "valueFrom": {"fieldRef": {
            "fieldPath": "status.hostIP" if c.host_network else "status.podIP"}}}] + (
            [{"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {
                "name": c.hf_token_secret, "key": "HF_TOKEN"}}}] if c.hf_token_secret else []),
        "resources": {"limits": {c.gpu_resource: 1}},
        "volumeMounts": [{"name": "shm", "mountPath": "/dev/shm"}],
    }
    volumes = [{"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": c.shm_size}}]
    if c.model_hf_id:
        container["volumeMounts"].append({"name": "hf-cache", "mountPath": "/root/.cache/huggingface"})
        volumes.append({"name": "hf-cache", "persistentVolumeClaim": {"claimName": c.hf_cache_pvc}}
                       if c.hf_cache_pvc else {"name": "hf-cache", "emptyDir": {}})
    else:
        container["volumeMounts"].append({"name": "model", "mountPath": "/model", "readOnly": True})
    if c.model_hf_id:
        pass  # downloaded at start into the hf-cache volume
    elif c.model_pvc:
        volumes.append({"name": "model", "persistentVolumeClaim": {"claimName": c.model_pvc, "readOnly": True}})
    elif c.model_host_path:
        volumes.append({"name": "model", "hostPath": {"path": c.model_host_path}})
    else:
        raise SystemExit("one of --model-hf-id / --model-pvc / --model-host-path is required")
    if c.ib:
        container["securityContext"] = {"privileged": True}
        container["volumeMounts"].append({"name": "ib", "mountPath": "/dev/infiniband"})
        volumes.append({"name": "ib", "hostPath": {"path": "/dev/infiniband"}})
    if idx == 0:
        container["readinessProbe"] = {
            "httpGet": {"path": "/v1/models", "port": env["API_PORT"]},
            "periodSeconds": 5, "failureThreshold": 3}
    pod = {"terminationGracePeriodSeconds": 15, "containers": [container], "volumes": volumes}
    if c.host_network:
        pod.update(hostNetwork=True, dnsPolicy="ClusterFirstWithHostNet")
    if node:
        pod["nodeSelector"] = {"kubernetes.io/hostname": node}
    if c.image_pull_secret:
        pod["imagePullSecrets"] = [{"name": c.image_pull_secret}]
    dep = {
        "apiVersion": "apps/v1", "kind": "Deployment",
        "metadata": {"name": name, "namespace": ns, "labels": labels},
        "spec": {"replicas": 1, "strategy": {"type": "Recreate"},
                 "selector": {"matchLabels": labels},
                 "template": {"metadata": {"labels": labels}, "spec": pod}},
    }
    if idx != 0:
        if c.engine != "upstream":
            return [dep]
        # Upstream: a headless Service gives stage 0 a stable name for this
        # replica's control endpoint (resolved to the pod address at start).
        cp = int(env["CONTROL_PORT"])
        return [dep, {
            "apiVersion": "v1", "kind": "Service",
            "metadata": {"name": name, "namespace": ns, "labels": labels},
            "spec": {"clusterIP": "None", "selector": labels, "publishNotReadyAddresses": True,
                     "ports": [{"name": "control", "port": cp, "targetPort": cp},
                               {"name": "data", "port": cp + 1, "targetPort": cp + 1}]},
        }]
    # Stage 0 serves the OpenAI API: a Service per replica gives clients and
    # the resume proxy a stable name (http://PREFIX-s0-rR:API_PORT).
    svc = {
        "apiVersion": "v1", "kind": "Service",
        "metadata": {"name": name, "namespace": ns, "labels": labels},
        "spec": {"selector": labels, "ports": [
            {"name": "api", "port": env["API_PORT"], "targetPort": env["API_PORT"]}]},
    }
    return [dep, svc]


def router(c):
    labels = {"app": "connex-front-router", "router": c.router_name}
    pod = {"containers": [{
        "name": "router", "image": c.router_image or c.image, "imagePullPolicy": "IfNotPresent",
        "command": ["python3", "-m", "connex.front_router",
                    "--api-address", f"tcp://*:{c.router_port}",
                    "--event-address", f"tcp://*:{c.events_port}"],
        "env": [{"name": "CONNEX_HEARTBEAT_TIMEOUT_S", "value": "2"},
                {"name": "CONNEX_UNHEALTHY_GRACE_S", "value": "10"}],
        "resources": {"requests": {"cpu": "1"}, "limits": {"cpu": "2"}},
    }]}
    # Pinning the router next to the stages avoids depending on cross-node
    # overlay connectivity from host-network pods.
    if c.router_node:
        pod["nodeSelector"] = {"kubernetes.io/hostname": c.router_node}
    if c.image_pull_secret:
        pod["imagePullSecrets"] = [{"name": c.image_pull_secret}]
    dep = {
        "apiVersion": "apps/v1", "kind": "Deployment",
        "metadata": {"name": c.router_name, "namespace": c.namespace, "labels": labels},
        "spec": {"replicas": 1, "selector": {"matchLabels": labels},
                 "template": {"metadata": {"labels": labels}, "spec": pod}},
    }
    svc = {
        "apiVersion": "v1", "kind": "Service",
        "metadata": {"name": c.router_name, "namespace": c.namespace, "labels": labels},
        "spec": {"selector": labels, "ports": [
            {"name": "api", "port": c.router_port, "targetPort": c.router_port},
            {"name": "events", "port": c.events_port, "targetPort": c.events_port}]},
    }
    return [dep, svc]


def proxy(c, prefix, node=""):
    """Optional data-path proxy: retries/resumes requests when a stage-0 replica
    dies. Membership comes from its own /v1/models prober, so it needs only the
    list of candidate stage-0 URLs."""
    if not c.proxy_endpoints:
        raise SystemExit("proxy needs --proxy-endpoints URL[,URL...]")
    name = f"{prefix}-proxy"
    labels = {"app": "connex-proxy", "proxy-for": prefix}
    pod = {"containers": [{
        "name": "proxy", "image": c.image, "imagePullPolicy": "IfNotPresent",
        "command": ["python3", "-m", "connex.serving.proxy", "--port", str(c.proxy_port),
                    "--endpoints", c.proxy_endpoints],
        "readinessProbe": {"httpGet": {"path": "/health", "port": c.proxy_port}, "periodSeconds": 2},
        "resources": {"requests": {"cpu": "1"}, "limits": {"cpu": "2"}},
    }]}
    if c.host_network:
        pod.update(hostNetwork=True, dnsPolicy="ClusterFirstWithHostNet")
    if node:
        pod["nodeSelector"] = {"kubernetes.io/hostname": node}
    if c.image_pull_secret:
        pod["imagePullSecrets"] = [{"name": c.image_pull_secret}]
    return [{
        "apiVersion": "apps/v1", "kind": "Deployment",
        "metadata": {"name": name, "namespace": c.namespace, "labels": labels},
        "spec": {"replicas": 1, "selector": {"matchLabels": labels},
                 "template": {"metadata": {"labels": labels}, "spec": pod}},
    }, {
        "apiVersion": "v1", "kind": "Service",
        "metadata": {"name": name, "namespace": c.namespace, "labels": labels},
        "spec": {"selector": labels, "ports": [
            {"name": "http", "port": c.proxy_port, "targetPort": c.proxy_port}]},
    }]


def _bool(v):
    return str(v).lower() in ("1", "true", "yes", "on")


def build_parser(defaults):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["router", "pipeline", "stage", "proxy"])
    ap.add_argument("--config", help="JSON file with option defaults")
    ap.add_argument("--backend", choices=["connex", "connexte"], default="connexte")
    ap.add_argument("--prefix", default="cte", help="pipeline name / label")
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--replica", type=int, default=0)
    ap.add_argument("--node", default="", help="node for `stage`/`proxy` (empty: unpinned)")
    ap.add_argument("--set", action="append", default=[], help="extra stage env KEY=VAL")
    for k, v in defaults.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, dest=k, action=argparse.BooleanOptionalAction, default=v)
        else:
            ap.add_argument(flag, dest=k, type=type(v), default=v)
    return ap


def load_defaults(argv=None):
    d = dict(DEFAULTS)
    for k, v in d.items():
        e = os.environ.get("CONNEX_K8S_" + k.upper())
        if e is not None:
            d[k] = _bool(e) if isinstance(v, bool) else type(v)(e)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)
    if known.config:
        with open(known.config) as f:
            for k, v in json.load(f).items():
                if k not in d:
                    raise SystemExit(f"unknown config key {k!r}")
                d[k] = v
    return d


def generate(argv=None):
    c = build_parser(load_defaults(argv)).parse_args(argv)
    if c.engine not in ("upstream", "fork"):
        raise SystemExit("--engine must be upstream or fork")
    if c.metadata not in ("p2p", "etcd"):
        raise SystemExit("--metadata must be p2p or etcd")
    ov = dict(kv.split("=", 1) for kv in c.set)
    if c.what == "router":
        items = router(c)
    elif c.what == "proxy":
        items = proxy(c, c.prefix, c.node)
    elif c.what == "pipeline":
        nodes = [n for n in c.nodes.split(",") if n] if c.nodes else []
        if nodes and len(nodes) != c.num_stages:
            raise SystemExit(f"--nodes lists {len(nodes)} nodes for {c.num_stages} stages")
        reps = c.stage_replicas if c.engine == "upstream" else 1
        items = [o for i in range(c.num_stages) for r in range(1 if i == 0 else reps)
                 for o in stage(c, c.backend, c.prefix, i, nodes[i] if nodes else "", r, ov)]
    else:
        items = stage(c, c.backend, c.prefix, c.stage, c.node, c.replica, ov)
    return {"apiVersion": "v1", "kind": "List", "items": items}


def main():
    print(json.dumps(generate(), indent=1))


if __name__ == "__main__":
    main()
