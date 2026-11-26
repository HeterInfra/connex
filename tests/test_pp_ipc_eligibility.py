# SPDX-License-Identifier: Apache-2.0
"""Same-host CUDA IPC only when both endpoints can see each other's GPU (#40)."""

import json

import pytest

link = pytest.importorskip("connex.adapters.vllm.pp.link")


@pytest.fixture
def gpudir(tmp_path, monkeypatch):
    monkeypatch.setattr(link, "_GPU_DIR", str(tmp_path))
    link._IPC_CACHE.clear()
    return tmp_path


def _pub(url, gpu, visible, p2p=None):
    with open(link._gpu_file(url), "w") as f:
        json.dump({"gpu": gpu, "visible": visible, "p2p": visible if p2p is None else p2p}, f)


A = "10.0.0.5:48001#cxpp|0|0|0"
B = "10.0.0.5:48017#cxpp|1|0|0"
C = "10.0.0.6:48017#cxpp|1|0|0"


def test_separate_containers_one_gpu_each_use_rdma(gpudir):
    _pub(A, "GPU-a", ["GPU-a"])
    _pub(B, "GPU-b", ["GPU-b"])
    assert not link.ipc_ok(A, B) and not link.ipc_ok(B, A)


def test_shared_visibility_uses_ipc(gpudir):
    _pub(A, "GPU-a", ["GPU-a", "GPU-b"])
    _pub(B, "GPU-b", ["GPU-a", "GPU-b"])
    assert link.ipc_ok(A, B) and link.ipc_ok(B, A)


def test_no_peer_access_uses_rdma(gpudir):
    _pub(A, "GPU-a", ["GPU-a", "GPU-b"], p2p=["GPU-a"])
    _pub(B, "GPU-b", ["GPU-a", "GPU-b"], p2p=["GPU-b"])
    assert not link.ipc_ok(A, B) and not link.ipc_ok(B, A)


def test_same_gpu_two_processes_uses_ipc(gpudir):
    _pub(A, "GPU-a", ["GPU-a"], p2p=["GPU-a"])
    _pub(B, "GPU-a", ["GPU-a"], p2p=["GPU-a"])
    assert link.ipc_ok(A, B)


def test_one_sided_visibility_is_no_and_symmetric(gpudir):
    _pub(A, "GPU-a", ["GPU-a", "GPU-b"])
    _pub(B, "GPU-b", ["GPU-b"])
    assert link.ipc_ok(A, B) is link.ipc_ok(B, A) is False


def test_other_host_or_missing_identity_is_no(gpudir):
    _pub(A, "GPU-a", ["GPU-a", "GPU-b"])
    _pub(C, "GPU-b", ["GPU-a", "GPU-b"])
    assert not link.ipc_ok(A, C)            # different IP
    assert not link.ipc_ok(A, B)            # B never published (older peer)


def test_missing_peer_is_not_cached(gpudir):
    _pub(A, "GPU-a", ["GPU-a", "GPU-b"])
    assert not link._ipc_ok_cached(A, B)
    _pub(B, "GPU-b", ["GPU-a", "GPU-b"])
    assert link._ipc_ok_cached(A, B)         # decided again once B is up


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs CUDA")
def test_publish_gpu_writes_own_uuid(gpudir):
    import torch

    link.publish_gpu(A, torch.device("cuda", 0))
    gpu, seen = link._gpu_info(A)
    assert gpu == str(torch.cuda.get_device_properties(0).uuid) and gpu in seen
