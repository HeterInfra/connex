# Across hosts and mixed GPUs: what changed when the peers left the host

PD disaggregation, tensor parallelism and expert parallelism were first
validated on one 4-GPU host, where every "remote" read went through the
NIC in loopback and every process could be watched locally. This note
records what had to change when prefill, decode and EP ranks moved to
different nodes, and when a node with a different GPU and no GPUDirect RDMA
joined. Measurements: [`results.md` section 12](../results.md#12-across-hosts-and-mixed-gpus).

## What stayed the same

The data path. A one-sided connex-rs read of the KV of a 1024-token
Llama-3-8B prompt (128 MB) takes about 13 ms over 100 Gb InfiniBand, next
to a 300 ms prefill, so latency and throughput across nodes match one host
and stay on par with NIXL (vLLM) and Mooncake (SGLang). Failure handling
too: a killed decode or TP rank on another node is reported by the
surviving peers' connex-rs engines within milliseconds (the dead process's
kernel resets its connections across the fabric as it does locally), and
the PD router hands its requests over with 0.4-0.6 s stalls, as on one host.

## What changed

**1. Watching processes does not cross hosts.** `EPGroupSupervisor`
detected a dead EP rank by polling its pid. With rank 1 on another node
there is no pid to poll, and vLLM's API server on the rank-0 node keeps
answering `/health` after the remote rank died: in a 200 s run the group was
never removed. The fix uses the same signal PD and TP already use: the
connex-rs transports of the EP ranks. `RankDeathWatcher` links one engine
to every rank's EP transport (identity `("connex-ep", 0, rank, group)`,
URL from `connex_ep_init`) and fails the group on the peer-death event;
timeouts stay suspicion only, as in `GroupFailureTracker`. Detection went
from never to 0.29 s. After a rebuild the new ranks have new addresses
under the same identities, so the watcher re-links (`EPGroup.on_rejoin`).

**2. A wire-up path that never ran on one host.** The first cross-node
expert move (rejoin of rank 1, 12.9 GB) killed rank 1 with a segfault in
`ucp_am_zcopy_single_reply` after `ibv_reg_mr ... failed: Bad address`.
Inside vLLM workers UCX has no registration cache (OpenCV is imported and
UCX cannot install its memory hooks), its zero-copy threshold collapses,
and across hosts (rc transport instead of shared memory) UCX sent the
1-byte connex-rs heartbeat zero-copy. That buffer is a static read-only
constant; registering it for RDMA fails and UCX dereferences the failed
registration. The PD data plane had pinned `UCX_ZCOPY_THRESH=4096` for a
related reason; the EP/RL transport now does the same. (The heartbeat
buffer in connex-rs should also be writable; that is a connex-rs change.)

**3. A node without GPUDirect RDMA.** The RTX A6000 node has the same NIC
but no `nvidia_peermem`. UCX still accepts a registration of GPU memory
there (through its CUDA memory domain), so `register_memory` returns 0, and
then every RDMA read of it fails (`cannot find remote protocol ... get into
cuda memory from cuda`). Two things follow:

* the decision cannot rest on the registration result:
  `gpudirect_available()` checks for a loaded peer-memory driver
  (`/sys/module/nvidia_peermem`, `nv_peer_mem`);
* the only working path was UCX over TCP, where UCX stages GPU memory
  through the host on every read: 0.4-0.5 GB/s, saturating at about QPS 3,
  and an owner crashed in UCX's TCP RMA emulation at QPS 4.

Host staging (`connex/pd/host_staging.py`) keeps the transfer one-sided RDMA
and moves the GPU copy out of UCX:

```
prefill (no GPUDirect)                              decode
 KV cache (GPU) --D2H on the engine stream-->  pinned mirror (registered)
                         acquire waits for the copy event      |
                                         one-sided RDMA read <-+
                     decode mirror (registered) --H2D, side stream--> KV cache
```

Addresses map by a constant offset per registered span, so the producer
only rewrites the base of each advertised `TensorLayout` and the consumer
rewrites the destination of each read; block lists, holds, re-pulls and TP
head re-layout are unchanged. The device -> mirror copy is queued in
`hold()` on the engine's current stream, i.e. after the prefill that wrote
those blocks and before any later step can reuse them; the acquire reply is
sent only after its event completes. On the consumer the mirror -> device
copy runs on a side stream and is synchronised before the pull reports
done, which is when vLLM may read the blocks. Cost: pinned host memory the
size of the registered KV cache (about 11 GB per instance here) and one
PCIe copy per side; under load (QPS 4) TTFT p50 is 531 ms against 462 ms
with GPUDirect on both sides, and the same staging with an A40 on that node
gives the same numbers, so the GPU type is not what matters. Either side
can stage independently; a GPUDirect peer reads a staged producer directly.

## Smaller things the harness had to learn

* vLLM's NixlConnector defaults its side channel to localhost; across hosts
  set `VLLM_NIXL_SIDE_CHANNEL_HOST`, or every handshake is reset.
* Python's `urllib` ignores CIDR entries in `NO_PROXY`: a proxy configured
  for the cluster silently swallowed the readiness probes of remote
  instances.
* The router re-reads pool files every second; writing them in place let it
  read an empty file once (requests failed with "no prefill instances").
  Pool files are now replaced atomically.
* UCX's CUDA transport (`libuct_cuda`) of HPC-X needs `libcudart.so.12`; in
  a CUDA 13 environment it is silently absent, which only matters once a
  path needs it (TCP or staging through UCX).
* GPUs shared with other tenants come and go: check free memory right
  before each run; several runs here failed at start-up because another
  job had taken the GPU minutes earlier.

## Reproducing

```bash
# hosts.json: how to run a shell on each node and its RDMA address
# {"A": {"ip": "10.0.0.1", "exec": ["ssh", "node-a"]}, "B": {...}, "C": {...}}
T=benchmarks/pd/tp_experiments.py
python $T correct --hosts hosts.json --model M --prefill A:2 --decode B:2 --score-gpus C:2 --out r/c
python $T perf    --hosts hosts.json --model M --prefill A:2 --decode B:2 --qps 1,2,4,6 --out r/p
python $T kill    --hosts hosts.json --model M --prefill A:2 --decode B:2,C:3 \
    --victim decode1 --rank -1 --at 30 --n 240 --max-tokens 256 --score-gpus C:2 --out r/k
python $T scale   --hosts hosts.json --model M --prefill A:2 --decode B:2 \
    --add decode=C:3 --victim decode1 --at 5 --drain-at 50 --stop-at 80 --score-gpus C:2 --out r/s
python $T kill    --hosts hosts.json --model M --front-router --prefill A:2 --decode B:2+3,C:2 \
    --victim decode1 --rank 1 --at 30 --restart-at 75 --score-gpus C:3 --out r/tp
# a node without GPUDirect RDMA as D: nothing to set (CONNEX_PD_HOST_STAGING=auto)
python $T perf    --hosts hosts.json --model M --prefill D:0 --decode B:2 --qps 1,2,4,6 --out r/h
```

Every instance runs in its own session on its host; the harness kills by
process group through the same shell prefix and copies the logs back.
EP across nodes: start rank 0 and rank 1 with `--nnodes 2` as in
[`ep.md`](../ep.md#across-hosts), then `benchmarks/ep/ep_bench.py run
--url http://<rank-0 host>:<port> --event drain|migrate`.
