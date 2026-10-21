# SPDX-License-Identifier: Apache-2.0
"""UCX_TCP per-message send time and bandwidth by size. Compare env settings, e.g. UCX_RNDV_THRESH=auto."""
import multiprocessing as mp, socket, time
from connex.ucx_transport import UCXTransport, TransportMode
def fp():
    s=socket.socket(); s.bind(("127.0.0.1",0)); p=s.getsockname()[1]; s.close(); return p
def rx(port, ev, n, done):
    t=UCXTransport("127.0.0.1",port,mode=TransportMode.UCX_TCP,device="cpu",buffer_size=1<<16,num_buffers=1); t.start_sync(); ev.set()
    while True:
        for _ in range(n.value): t.recv_from_queue(timeout=30)
        done.set()
if __name__=="__main__":
    ctx=mp.get_context("spawn"); port=fp(); ev=ctx.Event(); done=ctx.Event(); n=ctx.Value('i',0)
    p=ctx.Process(target=rx,args=(port,ev,n,done),daemon=True)
    tx=UCXTransport("127.0.0.1",fp(),mode=TransportMode.UCX_TCP,device="cpu",buffer_size=1<<16,num_buffers=1); tx.start_sync()
    for size in [64<<10, 1<<20, 8<<20, 32<<20]:
        cnt=max(20, min(2000, (512<<20)//size))
        n.value=cnt; done.clear()
        if not p.is_alive(): p.start(); ev.wait(60)
        ep=tx.connect_sync("127.0.0.1",port); buf=bytearray(size)
        for _ in range(5): tx.send_sync(ep,buf)
        n.value=cnt; done.clear()
        # drain warmup
        time.sleep(0.2)
        a=time.time()
        for _ in range(cnt): tx.send_sync(ep,buf)
        lat=(time.time()-a)/cnt
        print(f"size={size>>10}KiB per-msg={lat*1e3:.2f}ms bw={size/lat/1e9:.2f}GB/s", flush=True)
    p.kill()
