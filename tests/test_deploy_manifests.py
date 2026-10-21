# SPDX-License-Identifier: Apache-2.0
"""deploy/k8s/manifests.py: configurable, no environment-specific defaults."""
import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("manifests", os.path.join(ROOT, "deploy", "k8s", "manifests.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def _env(item):
    return {e["name"]: e.get("value") for e in item["spec"]["template"]["spec"]["containers"][0]["env"]}


def test_pipeline_p2p_defaults():
    d = M.generate(["pipeline", "--nodes", "a,b", "--model-host-path", "/m"])
    assert [(i["kind"], i["metadata"]["name"]) for i in d["items"]] == [
        ("Deployment", "cte-s0-r0"), ("Service", "cte-s0-r0"), ("Deployment", "cte-s1-r0")]
    env = _env(d["items"][2])
    assert env["MOONCAKE_METADATA_SERVER"] == "P2PHANDSHAKE" and env["STAGE"] == "1"
    assert env["ROUTER_ADDR"] == "connex-front-router.connex.svc.cluster.local:15555"
    pod = d["items"][0]["spec"]["template"]["spec"]
    assert "hostNetwork" not in pod and pod["nodeSelector"] == {"kubernetes.io/hostname": "a"}  # release-scan: allow
    assert "NVIDIA_VISIBLE_DEVICES" not in _env(d["items"][0])  # device plugin picks the GPU
    assert "ib" not in {v["name"] for v in pod["volumes"]}


def test_etcd_ib_pvc_and_config(tmp_path, monkeypatch):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"namespace": "ns1", "model_pvc": "models"}))
    monkeypatch.setenv("CONNEX_K8S_IMAGE", "img:1")
    d = M.generate(["stage", "--config", str(cfg), "--stage", "1", "--replica", "1", "--ib",
                    "--metadata", "etcd", "--etcd", "etcd:2379", "--no-host-network"])
    (item,) = d["items"]
    pod = item["spec"]["template"]["spec"]
    c = pod["containers"][0]
    assert c["image"] == "img:1" and item["metadata"]["namespace"] == "ns1"
    assert c["securityContext"]["privileged"] and "hostNetwork" not in pod
    env = _env(item)
    assert env["MOONCAKE_METADATA_BACKEND"] == "etcd" and env["API_PORT"] == "5101"
    assert {"name": "model", "persistentVolumeClaim": {"claimName": "models", "readOnly": True}} in pod["volumes"]


def test_errors():
    with pytest.raises(SystemExit):
        M.generate(["pipeline", "--nodes", "a,b"])            # no model source
    with pytest.raises(SystemExit):
        M.generate(["pipeline", "--nodes", "a", "--model-host-path", "/m"])  # 1 node, 2 stages
    with pytest.raises(SystemExit):
        M.generate(["stage", "--model-host-path", "/m", "--metadata", "etcd"])


def test_router():
    d = M.generate(["router", "--router-node", "a"])
    dep, svc = d["items"]
    assert dep["spec"]["template"]["spec"]["containers"][0]["command"][:3] == ["python3", "-m", "connex.front_router"]
    assert svc["metadata"]["name"] == "connex-front-router"


def test_portable_defaults_hf_model():
    d = M.generate(["pipeline", "--model-hf-id", "org/model", "--hf-token-secret", "hf"])
    dep = d["items"][0]
    pod = dep["spec"]["template"]["spec"]
    assert "nodeSelector" not in pod and "hostNetwork" not in pod
    c = pod["containers"][0]
    assert _env(dep)["MODEL_PATH"] == "org/model"
    assert {"name": "hf-cache", "emptyDir": {}} in pod["volumes"]
    assert any(e["name"] == "HF_TOKEN" and e["valueFrom"]["secretKeyRef"]["name"] == "hf" for e in c["env"])
    assert c["resources"]["limits"] == {"nvidia.com/gpu": 1}


def test_proxy():
    with pytest.raises(SystemExit):
        M.generate(["proxy"])
    dep, svc = M.generate(["proxy", "--proxy-endpoints", "http://cte-s0-r0:5100"])["items"]
    cmd = dep["spec"]["template"]["spec"]["containers"][0]["command"]
    assert cmd[:3] == ["python3", "-m", "connex.serving.proxy"] and "http://cte-s0-r0:5100" in cmd
    assert svc["kind"] == "Service" and svc["spec"]["ports"][0]["port"] == 5199


def test_two_pipelines_share_a_namespace_without_clobbering():
    """Router and proxy object names must not collide across pipelines: applying
    one pipeline's router used to replace any `connex-front-router` already in
    the namespace."""
    a = M.generate(["router", "--router-name", "a-front-router"])["items"]
    b = M.generate(["router"])["items"]
    assert {i["metadata"]["name"] for i in a} == {"a-front-router"}
    assert {i["metadata"]["name"] for i in b} == {"connex-front-router"}
    sel = a[0]["spec"]["selector"]["matchLabels"]
    assert sel["router"] == "a-front-router" and a[1]["spec"]["selector"] == sel
    st = M.generate(["stage", "--prefix", "a", "--router-name", "a-front-router",
                     "--model-host-path", "/m"])["items"][0]
    assert _env(st)["ROUTER_ADDR"] == "a-front-router.connex.svc.cluster.local:15555"
    pa = M.generate(["proxy", "--prefix", "a", "--proxy-endpoints", "http://a-s0-r0:5100"])["items"]
    pb = M.generate(["proxy", "--proxy-endpoints", "http://cte-s0-r0:5100"])["items"]
    assert {i["metadata"]["name"] for i in pa} == {"a-proxy"}
    assert {i["metadata"]["name"] for i in pb} == {"cte-proxy"}
    # the proxy must not carry the pipeline label (wait_ready.sh selects stage pods by it)
    assert "pipeline" not in pa[0]["spec"]["template"]["metadata"]["labels"]


def test_host_network_ports_unique_per_node():
    """Two stages of one replica, and replicas of a stage, may share a node on
    the host network: no port may be used twice."""
    seen = set()
    for r in (0, 1):
        for st in range(4):
            dep = M.generate(["stage", "--num-stages", "4", "--stage", str(st), "--replica", str(r),
                              "--host-network", "--model-host-path", "/m"])["items"][0]
            e = _env(dep)
            ports = {int(e["BASE_PORT"]), int(e["API_PORT"]), int(e["MOONCAKE_BASE_PORT"]) + st}
            assert len(ports) == 3 and not seen & ports, (dep["metadata"]["name"], ports)
            seen |= ports
    s0 = _env(M.generate(["stage", "--host-network", "--model-host-path", "/m"])["items"][0])
    assert s0["API_PORT"] == "5100" and s0["BASE_PORT"] == "25500"
