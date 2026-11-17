"""Issue #21: every connex-rs engine in the package gets the same UCX
defaults (connex.pd.dataplane.setup_ucx_env); RL used to skip ^cuda_ipc and
CONNEX_RS_NET_DEVICES."""
import sys
import types

import pytest

_ENV = ("UCX_TLS", "UCX_NET_DEVICES", "UCX_ZCOPY_THRESH", "UCX_TCP_CM_REUSEADDR",
        "CONNEX_RS_CUDA_IPC", "CONNEX_RS_NET_DEVICES")


class _Engine:
    def __init__(self, *a, **kw):
        self.local_url = "fake://x"


@pytest.fixture
def clean_env(monkeypatch):
    import os
    saved = dict(os.environ)   # setup_ucx_env writes os.environ directly
    for k in _ENV:
        os.environ.pop(k, None)
    monkeypatch.setitem(sys.modules, "connex_rs", types.SimpleNamespace(Engine=_Engine))
    yield monkeypatch
    os.environ.clear()
    os.environ.update(saved)


def _rl(ip="10.0.0.1"):
    from connex.rl.transport import ConnexRsTransport
    return ConnexRsTransport(advertise_ip=ip)


def _sglang(ip="127.0.0.1"):
    from connex.adapters.sglang.pd_backend import ConnexRsTransferEngine
    return ConnexRsTransferEngine(ip, "prefill", engine_cls=_Engine)


@pytest.mark.parametrize("make", [_rl, _sglang])
def test_defaults(clean_env, make):
    import os
    clean_env.setenv("CONNEX_RS_NET_DEVICES", "mlx5_0:1")
    make()
    assert os.environ["UCX_TLS"] == "^cuda_ipc"
    assert os.environ["UCX_NET_DEVICES"] == "mlx5_0:1"
    assert os.environ["UCX_ZCOPY_THRESH"] == "4096"
    assert os.environ["UCX_TCP_CM_REUSEADDR"] == "y"


@pytest.mark.parametrize("make", [_rl, _sglang])
def test_explicit_settings_win(clean_env, make):
    import os
    clean_env.setenv("UCX_TLS", "rc,cuda_copy")
    clean_env.setenv("UCX_NET_DEVICES", "mlx5_1:1")
    make()
    assert os.environ["UCX_TLS"] == "rc,cuda_copy"
    assert os.environ["UCX_NET_DEVICES"] == "mlx5_1:1"


def test_cuda_ipc_opt_in(clean_env):
    import os
    clean_env.setenv("CONNEX_RS_CUDA_IPC", "1")
    clean_env.setenv("CONNEX_RS_NET_DEVICES", "")
    from connex.pd.dataplane import setup_ucx_env
    setup_ucx_env("10.0.0.1")
    assert "UCX_TLS" not in os.environ and "UCX_NET_DEVICES" not in os.environ
