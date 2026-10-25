# Expert parallelism with recovery and elasticity (`connex.ep`)

In expert-parallel (EP) MoE serving, every EP rank hosts a set of experts and
all ranks exchange tokens every layer (dispatch / combine all-to-all). The
set of ranks, and which rank serves which expert, should be able to change
while the deployment serves: a rank drains for maintenance, a rank dies,
hot experts get more copies, a new rank joins. `connex.ep` treats each EP
rank as a mobile endpoint and gives the EP group a control plane for this;
the per-layer all-to-all stays on the engine's fast path. Numbers:
[`results.md`](results.md#10-expert-parallelism-moe).

## Model

| Piece | What it does | Where |
|---|---|---|
| `PlacementTable` | versioned placement: per MoE layer, which logical expert each physical slot holds (`slot = rank * slots_per_rank + i`), replicas, rank states `up` / `draining` / `down`, per-slot routable mask; repair planning | `connex/ep/placement.py` |
| `ExpertStore` | one rank's expert weights (per layer, tensors indexed by local slot) registered with a one-sided transport; pulls an expert's rows from a peer straight into a local slot (CUDA IPC peer-to-peer for a peer on the same host, connex-rs otherwise), copies locally, or loads from the checkpoint when no live copy exists | `connex/ep/mover.py` |
| `EPController` | runs every change through two epochs, re-plans moves whose source fails | `connex/ep/controller.py` |
| `EPGroupSupervisor` | replica-level recovery when the engine cannot survive a rank loss: health probes, drop the group from the routing set, rebuild, rejoin | `connex/ep/groups.py` |
| vLLM glue | worker extension (`collective_rpc` methods), engine handle for the controller, EPLB support for OLMoE | `connex/ep/vllm_ext.py` |

**Epochs and the cutover point.** Every change becomes visible as a new
epoch of the table. An engine installs a view only at a step boundary and
only if its epoch is newer than the one it runs, like `connex.pp` chain
epochs: all ranks switch placement between the same two steps, and a stale
or repeated view is ignored. A reconfiguration is

```
epoch k+1   stop routing to down ranks, to draining ranks (for experts that
            have a replica elsewhere) and to the destination slots of the
            repair moves                                   -> engines apply
copy        destination ranks pull expert weights with one-sided reads
            (connex-rs; GPUDirect where available) from a live replica, or
            load them from the checkpoint
epoch k+2   destination slots routable with their new expert -> apply
```

In-flight requests keep running throughout: no step ever routes a token to
a slot that is being overwritten or that sits on a dead rank, and a step
never mixes two placements. A draining rank keeps serving the experts that
exist only on it until their copies are in place, so a drain never strands
a token.

**Repair.** `plan_repair(min_replicas)` gives every (layer, expert) at least
`min_replicas` routable replicas on `up` ranks. A destination is an empty
slot, else a slot whose expert keeps another replica (most replicated
first), preferring a rank that does not hold the expert yet and the least
loaded rank. The source is a live replica (same rank first, then the
draining rank, for a drain), else the checkpoint. If a source dies during
the copy, the move is re-planned from another source; repair is best effort
when the up ranks have too few slots (`uncovered()` reports what is left).

## vLLM (0.32, upstream, no fork)

vLLM's EP keeps a physical-to-logical expert map for its EPLB (expert
parallel load balancer) and routes each token to one replica of its logical
expert through `logical_to_physical_map` and `logical_replica_count`, read
by the router kernel every step. Connex drives exactly these tensors:

```bash
VLLM_SERVER_DEV_MODE=1 PYTHONPATH=/path/to/connex vllm serve allenai/OLMoE-1B-7B-0924 \
    -tp 2 --enable-expert-parallel --enable-eplb \
    --eplb-config '{"num_redundant_experts": 64, "step_interval": 100000000, "use_async": false}' \
    --worker-extension-cls connex.ep.vllm_ext.ConnexEPWorker
```

The worker extension also makes vLLM's OLMoE EPLB-capable. Other vLLM
processes are left alone; to get the OLMoE EPLB support without the
extension, set `CONNEX_EP=1` (the `connex_ep` vLLM plugin then applies it).

```python
from connex.ep import EPController
from connex.ep.vllm_ext import VLLMEPEngine

eng = VLLMEPEngine("http://127.0.0.1:8000")
ctl = EPController(eng.table(), eng)     # adopt the layout vLLM loaded
ctl.refresh_peers(); ctl.push()          # register expert weights with connex-rs, epoch 0
ctl.drain_rank(1)                        # rank 1 gets no tokens from the next step on
ctl.migrate(expert=5, dst_slot=64)       # copy expert 5 into slot 64, all layers
```

* `step_interval` must exceed the run so vLLM's own rebalancing never fires:
  Connex owns the map. `--enable-eplb` needs a model with vLLM's
  `MixtureOfExperts` interface; importing `connex.ep.vllm_ext` adds it to
  OLMoE (whose upstream MoE layer is built without EPLB). Qwen3-MoE,
  DeepSeek-V2/V3, Mixtral and the other models that have it need nothing.
* The view is installed by `collective_rpc` (the HTTP route needs
  `VLLM_SERVER_DEV_MODE=1`); vLLM runs it on every worker between two
  engine steps, which is the step-boundary cutover. Maps are updated in
  place, so captured CUDA graphs stay valid.
* Weights are copied into a slot that the previous epoch made unroutable, so
  the copy can never be read half-written.
* `VLLMEPEngine.table()` adopts the epoch the engine runs, so a restarted or
  new controller continues the sequence; a view an engine rejects as stale
  raises `PlacementError` (two controllers must not drive one group).
* Between ranks of one host, experts move over CUDA IPC (`CONNEX_EP_IPC=0`
  turns it off); connex-rs over UCX went through the RDMA NIC in loopback
  at about a third of the peer-to-peer bandwidth.
* Weights are copied in a background thread of each worker (one-sided reads
  land directly in the slots; local copies on a side stream) while the
  engine keeps stepping; `VLLMEPEngine` polls until every rank is done
  (`async_moves=False` copies inside the RPC instead, which stalls serving).
* Redundant experts cost decode throughput when every replica is routed:
  vLLM spreads tokens over all replicas, so each step reads the weights of
  every copy. `table.active_replicas = 1` routes one replica per expert and
  keeps the others as warm standby, which become routable in the next epoch
  when a rank drains or fails (no copy needed).

### The boundary in vLLM: rank death

The ranks of a vLLM EP group share collectives every step (the MoE
all-to-all or all-gather/reduce-scatter, and the tensor-parallel
all-reduce). vLLM's elastic EP (`--enable-elastic-ep`) rescales the data
parallel size on request; it does not handle a rank that dies. When one EP
rank's process dies, the engine stops the whole group. Within one group,
`connex.ep` therefore covers planned events (drain, migrate, rejoin,
rebalance, a degraded rank taken out of routing while it is still alive).
For a hard failure, recovery is one level up:

1. every EP group is an endpoint of the resume proxy (`connex.serving.proxy`);
   requests in flight on the dead group resume on a healthy group from the
   tokens already produced (greedy output identical);
2. `EPGroupSupervisor` drops the group from the routing set, restarts it and
   puts it back once it serves.

`benchmarks/ep/ep_failover.py` measures this end to end.

## SGLang (0.5.22)

SGLang has EP (`--ep-size`), EPLB with redundant experts
(`--enable-eplb`, `--ep-num-redundant-experts`) and an expert location
updater that moves expert weights with NCCL P2P during a rebalance. Its
elastic EP (`--elastic-ep-backend mooncake`) masks failed ranks out of the
all-to-all and rebalances onto the survivors, which is the in-group rank
failover vLLM lacks; it needs the Mooncake EP kernels, and the Mooncake
build in the tested environment does not load (`Mooncake PG was not built
against torch==2.14.1`), so it was not run. Driving SGLang's expert location
from the Connex placement table needs a scheduler hook (SGLang exposes no
request to install an external map); not implemented yet. Replica-level
failover with the resume proxy and `EPGroupSupervisor` is engine neutral and
applies to SGLang EP groups unchanged.

## Not covered yet

* Placement-driven expert maps in SGLang (see above); SGLang elastic EP.
* Cross-host EP groups (all numbers come from one 4-GPU host), DeepEP /
  pplx all-to-all kernels, data parallel attention + EP.
* Checkpoint loading of an expert into a live vLLM rank (`ExpertStore`
  supports a loader; the vLLM glue does not wire one, since a vLLM rank that
  lost its copies is a dead rank and the group restarts anyway).
