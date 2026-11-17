# SPDX-License-Identifier: Apache-2.0
"""Regression tests for core-module review fixes."""

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
