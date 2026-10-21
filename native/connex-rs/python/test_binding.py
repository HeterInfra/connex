"""Smoke test of the connex_rs Python module: two processes, SIGKILL one,
restart it under the same logical id, transfers resume.

usage: PYTHONPATH=<dir with connex_rs> UCX_TLS=tcp python3 python/test_binding.py
"""
import os
import signal
import subprocess
import sys
import time

import connex_rs

SERVER = ("py", 0, 0, 0)
CLIENT = ("py", 1, 0, 0)


def client(port):
    ep = connex_rs.Endpoint(*CLIENT)
    ep.connect(SERVER, "127.0.0.1", port)
    corr = 1
    while True:  # echo forever until killed
        data = ep.recv(SERVER, corr, 1 << 20, timeout_ms=10000)
        ep.send(SERVER, corr, data, timeout_ms=10000)
        corr += 1


def server():
    ep = connex_rs.Endpoint(*SERVER)
    port = ep.listen("127.0.0.1", 0)
    detect, rejoin = [], []
    corr = 1
    for incarnation in range(3):
        child = subprocess.Popen([sys.executable, __file__, "client", str(port)])
        assert ep.wait_peer(CLIENT, 20000), "client did not join"
        for _ in range(20):
            payload = os.urandom(4096 + corr)
            ep.send(CLIENT, corr, payload, timeout_ms=5000)
            assert ep.recv(CLIENT, corr, 1 << 20, timeout_ms=5000) == payload
            corr += 1
        t_kill = connex_rs.now_ns()
        child.send_signal(signal.SIGKILL)
        while True:
            ev = ep.wait_event(5000)
            assert ev is not None, "kill not detected"
            if ev["type"] == "joined":
                rejoin.append(ev["rejoin"])
            if ev["type"] == "failed" and ev["peer"] == CLIENT:
                detect.append((ev["ts_ns"] - t_kill) / 1e6)
                break
        child.wait()
        # the next incarnation re-uses the same corr space
        corr = 1
    st = ep.stats()
    print("detect ms:", [f"{d:.1f}" for d in detect], "rejoin flags:", rejoin, "stats:", st)
    assert len(detect) == 3 and st["live_peers"] == 0 and st["eps_created"] == st["eps_closed"]
    print("OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        client(int(sys.argv[2]))
    else:
        server()
