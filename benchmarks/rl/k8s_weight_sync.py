# SPDX-License-Identifier: Apache-2.0
"""Emit pods for the distributed weight-sync benchmark (weight_sync_bench.py
dist): one pod per rank, host network, one physical GPU picked with
CUDA_VISIBLE_DEVICES (privileged pods see every GPU of the node, so pick
idle ones), code and the connex-rs wheel from a shared hostPath.

  python k8s_weight_sync.py --namespace NS --image IMG --share /shared/dir \\
      --master MASTER_IP --ranks node:gpu,node:gpu,... --size 1.5B | kubectl apply -f -

Rank 0 (the first entry) is the trainer and must run on the node whose
address is MASTER_IP. Results are printed by the rank-0 pod (kubectl logs).
"""
import argparse
import json


def pod(a, rank, node, gpu, world):
    cmd = (f"pip install -q --no-deps {a.share}/connex_rs-*.whl && cd {a.share}/src && "
           f"CUDA_VISIBLE_DEVICES={gpu} python3 benchmarks/rl/weight_sync_bench.py {a.cmd} "
           f"--rank {rank} --world {world} --master {a.master} --port {a.port} "
           f"--size {a.size} --rounds {a.rounds} --fanout {a.fanout} --chunk-mb {a.chunk_mb} "
           f"--window {a.window} --pub-buffers {a.pub_buffers} --out /tmp/{a.tag}{a.size}.jsonl "
           f"{a.extra}")
    return {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"name": f"{a.prefix}-{rank}", "namespace": a.namespace,
                     "labels": {"app": "connex-bench", "connex-rl": a.prefix}},
        "spec": {
            "restartPolicy": "Never", "nodeName": node, "hostNetwork": True,
            "containers": [{
                "name": "bench", "image": a.image, "imagePullPolicy": "IfNotPresent",
                "command": ["bash", "-c", cmd],
                "env": [{"name": "NCCL_IB_HCA", "value": a.ib_hca},
                        {"name": "NCCL_SOCKET_IFNAME", "value": a.ifname},
                        {"name": "NCCL_DEBUG", "value": a.nccl_debug},
                        # the image sets an external IB net plugin that does not
                        # match the NCCL torch ships; use the built-in IB transport
                        {"name": "NCCL_NET_PLUGIN", "value": "none"},
                        {"name": "NCCL_IB_DISABLE", "value": "0" if a.nccl_ib else "1"},
                        {"name": "UCX_WARN_UNUSED_ENV_VARS", "value": "n"},
                        {"name": "CONNEX_ADVERTISE_IP", "valueFrom": {
                            "fieldRef": {"fieldPath": "status.hostIP"}}}]
                       + ([{"name": "CONNEX_ADVERTISE_IP", "value": a.advertise[rank]}]
                          if a.advertise else []),
                "resources": {"limits": {"nvidia.com/gpu": 1}},
                "securityContext": {"privileged": True},
                "volumeMounts": [{"name": "share", "mountPath": a.share},
                                 {"name": "ib", "mountPath": "/dev/infiniband"},
                                 {"name": "shm", "mountPath": "/dev/shm"}],
            }],
            "volumes": [{"name": "share", "hostPath": {"path": a.share}},
                        {"name": "ib", "hostPath": {"path": "/dev/infiniband"}},
                        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}}],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", choices=["dist", "churn"], default="dist")
    ap.add_argument("--extra", default="", help="extra weight_sync_bench.py arguments")
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--share", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--ranks", required=True, help="node:gpu,... (first = trainer)")
    ap.add_argument("--advertise", default="", help="per-rank data-plane IPs, comma-separated")
    ap.add_argument("--prefix", default="connex-rl-ws")
    ap.add_argument("--port", type=int, default=29611)
    ap.add_argument("--size", default="1.5B")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--fanout", type=int, default=2)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--pub-buffers", type=int, default=2)
    ap.add_argument("--ib-hca", default="mlx5_0")
    ap.add_argument("--nccl-debug", default="WARN")
    ap.add_argument("--nccl-ib", type=int, default=1,
                    help="0 sets NCCL_IB_DISABLE=1 (NCCL over IPoIB sockets)")
    ap.add_argument("--ifname", default="ib", help="NCCL bootstrap interface (prefix)")
    a = ap.parse_args()
    a.advertise = [x for x in a.advertise.split(",") if x]
    ranks = [r.split(":") for r in a.ranks.split(",")]
    docs = [pod(a, i, n, g, len(ranks)) for i, (n, g) in enumerate(ranks)]
    for d in docs:
        if a.advertise:   # drop the hostIP default when explicit addresses are given
            env = d["spec"]["containers"][0]["env"]
            env[:] = [e for e in env if not (e["name"] == "CONNEX_ADVERTISE_IP" and "valueFrom" in e)]
    print(json.dumps({"apiVersion": "v1", "kind": "List", "items": docs}))


if __name__ == "__main__":
    main()
