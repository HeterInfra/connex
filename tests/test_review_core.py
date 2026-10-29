# SPDX-License-Identifier: Apache-2.0
"""Regression tests for core-module review fixes."""
import pickle
import struct

import torch

from connex import ucx_transport as ut


def _gdir(dtype_idx, shape):
    numel = 1
    for d in shape:
        numel *= d
    return b"GDIR" + struct.pack(f"!BBI{len(shape)}I", dtype_idx, len(shape), numel, *shape)


def test_gdir_roundtrip():
    m = ut._parse_gdir_metadata(_gdir(2, (3, 4)))
    assert m == {"shape": (3, 4), "dtype": torch.bfloat16, "nbytes": 24}


def test_gdir_rejects_malformed_and_plain_payloads():
    assert ut._parse_gdir_metadata(b"GDIR" + b"\x00" * 3) is None
    assert ut._parse_gdir_metadata(_gdir(0, (2, 2)) + b"x") is None      # length mismatch
    assert ut._parse_gdir_metadata(_gdir(99, (2,))) is None             # unknown dtype
    bad = b"GDIR" + struct.pack("!BBII", 0, 1, 7, 2)                   # numel != prod(shape)
    assert ut._parse_gdir_metadata(bad) is None


def test_peer_bytes_are_never_unpickled():
    # A pickled dict with is_cuda used to be honoured (and unpickled) by the
    # recv loop; now it is ordinary payload data.
    payload = pickle.dumps({"is_cuda": True, "shape": [1], "dtype": "torch.float32", "nbytes": 4})
    assert ut._parse_gdir_metadata(payload) is None
    import inspect
    assert "pickle.loads" not in inspect.getsource(ut.UCXTransport._ucx_recv_loop)
    assert "pickle.loads" not in inspect.getsource(ut.UCXTransport._recv_ucx_cuda)


def test_directory_client_discards_stale_reply():
    import msgpack
    import zmq
    from connex.directory import DirectoryClient

    ctx = zmq.Context.instance()
    router = ctx.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    port = router.bind_to_random_port("tcp://127.0.0.1")
    c = DirectoryClient(f"127.0.0.1:{port}", "m", stage_idx=0)
    c.start()
    try:
        # First request times out; its reply arrives late.
        assert c._control_call({"type": "a"}, timeout_ms=50) is None
        ident, raw = router.recv_multipart()
        router.send_multipart([ident, msgpack.packb({"reply": "a"})])
        import time
        time.sleep(0.1)
        # Second request must get its own reply, not the stale one.
        import threading

        def serve():
            ident2, raw2 = router.recv_multipart()
            req = msgpack.unpackb(raw2, raw=False)
            router.send_multipart([ident2, msgpack.packb({"reply": req["type"]})])

        t = threading.Thread(target=serve)
        t.start()
        assert c._control_call({"type": "b"}, timeout_ms=2000) == {"reply": "b"}
        t.join()
    finally:
        c.stop()
        router.close()
