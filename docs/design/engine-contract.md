# Connex and the inference engine: architecture and contract

Connex PP lets an inference engine run each pipeline stage as an independent
process (its own pod, its own GPU, its own lifetime) and keep serving while
stage replicas join, leave or die. Requests that are mid-decode when their
chain changes are handed over without client-visible errors: Connex detects
which requests lost their KV somewhere downstream, and the scheduling stage
recomputes them on the new chain.

This document defines the split of responsibilities between Connex and the
engine, and the wire protocol, precisely enough that a second engine can
implement its side and interoperate with existing stages.

## 1. Why a contract, and where the line is

Inside one engine, pipeline parallelism assumes a fixed group: ranks are
created together, `send`/`recv` never fail, and every stage holds the KV of
every request in flight. Elastic PP breaks each of these assumptions:

* membership changes at run time, so the neighbour a stage talks to is a
  routing decision, not a rank;
* messages can be lost, duplicated, delayed or arrive after the step they
  belong to was abandoned, so every step needs an identity;
* after a failover the new neighbour does not hold the KV of requests it
  never saw, so correctness needs an explicit ownership check.

Only the engine knows its KV cache and its scheduler; only the transport
knows about peers, retries and failures. The contract is therefore narrow:
Connex owns *steps* (moving a tensor dict to the right peer exactly once per
step id) and *membership*; the engine owns *requests* (what KV it holds,
what to recompute). They meet in four non-tensor fields carried with every
step: `request_ids`, `cx_ctx`, `cx_missing`, `finished_req_ids`.

## 2. Architecture

```
            +---------------- engine process, stage s -----------------+
            |  scheduler / model runner (engine)                       |
            |      |  step boundary: tensor dict + metadata            |
            |      v                                                   |
            |  engine adapter  (connex/adapters/<engine>)              |
            |      |  EngineHooks: kv_held / release / reprefill/abort |
            |      v                                                   |
            |  connex.pp  (engine neutral)                             |
            |   link.py      Stage0Link / StageWorker  (step drivers)  |
            |   endpoint.py  ConnexTEEndpoint = PipelineEndpoint       |
            |   steps.py     ids, CorrFilter, ReplyCache               |
            |   handover.py  cx_ctx / cx_missing rules, KVLedger       |
            |   routing.py   select_peer, StaticMembership             |
            |   ring.py      send-ring leases, ack-based reuse         |
            |   wire.py      frames, headers, versioning               |
            |   transports.py Mooncake TE | loopback | (ZeroMQ only)   |
            |   zmq_base.py  ROUTER/DEALER sockets, ZeroMQ fallback    |
            +----------------------------------------------------------+
                  |  ZeroMQ signals            |  RDMA / TCP reads
                  v                            v
            neighbour stages             Front Router (membership,
                                         leases, events; connex.epc_client)
```

Package layout and public API (all importable without any engine; the
endpoint loads torch, pyzmq and safetensors on first use, Mooncake is
optional):

| module | public names | role |
|---|---|---|
| `connex.pp.endpoint` | `ConnexTEEndpoint` (alias `PipelineEndpoint`) | one stage's endpoint: data plane, step API, failover |
| `connex.pp.link` | `Stage0Link`, `StageWorker`, `StepFailed` | the engine loop of stage 0 / stages 1.. |
| `connex.pp.hooks` | `EngineHooks`, `StepTransport`, `MetricsSink`, `NetUtils` | protocols the engine / adapter fill in |
| `connex.pp.handover` | `CX_CTX`, `CX_MISSING`, `FINISHED`, `REQUEST_IDS`, `find_missing`, `merge_missing`, `blank_rows`, `KVLedger`, `FinishedOutbox`, `RewindBudget` | handover rules |
| `connex.pp.routing` | `select_peer`, `StaticMembership`, `Route`, `Lid` | membership / peer choice |
| `connex.pp.steps` | `StepIds`, `CorrFilter`, `ReplyCache` | step identity, dedupe, stale filter |
| `connex.pp.wire` | `encode_signal`, `decode_signal`, `encode_g2`, ... | wire protocol |
| `connex.pp.testing` | `LoopbackTransferEngine`, `fake_chain` | test doubles for adapters |
| `connex.adapters.vllm` | `connexte.ConnexTECommunicator` | the vLLM fork's communicator |
| `connex.adapters.sglang` | `ConnexPPGroup`, `make_pp_group` | SGLang `pp_group` (prototype) |

### Topology

External PP has one scheduling stage (stage 0) per chain. Stage 0 forms
batches, sends each step forward, and receives the step's result (sampled
tokens) from the last stage. Stages 1..n-1 are passive: they receive a step,
run their layers, and send the result on. Every stage may have several
replicas; each stage picks one replica of its downstream neighbour (sticky,
see 4.3), so a deployment is a set of chains that share replicas only after
a failover. The last stage returns each reply directly to the stage-0
replica that originated it (its ROUTER endpoint travels with the step).

## 3. What Connex provides

**Data plane.** `send_step` / `send_reply` take a dict of tensors and
picklable values. Tensors go through registered memory of a transfer engine
(Mooncake TE: GPUDirect RDMA from a GPU send ring, or pinned host staging,
or host buffers), the receiver pulls them with one one-sided read, and a
ZeroMQ signal carries the header. Without a transfer engine the endpoint
falls back to a ZeroMQ-only path. Received tensors are on the endpoint's
device. A send ring region is reused only after the receiver's ack.

**Step protocol.** Each step has a 64-bit correlation id chosen by stage 0
(`new_step_id`). Up to K steps may be outstanding (`CONNEX_PP_INFLIGHT`);
replies are matched by id in any order (`recv_step(id)`), replies of other
outstanding steps are buffered, replies of abandoned or unknown steps are
dropped. A retry re-sends the same id (`send_step(td, id)` again). Every
later stage keeps its last K replies and answers a duplicate from that cache
without recomputing, and drops ids K or more behind the newest id of the
same stage-0 incarnation (stale). Retries are therefore idempotent end to
end.

**Membership and routing.** Each stage registers with the Front Router
(lease + heartbeats) and subscribes to membership events. Peer choice is
sticky; it changes only when the current peer departs (deregistration
event, or a missed lease confirmed by a data-path timeout), its address
changes, or a send to it fails. Replicas seen departing are skipped even if
the router still lists them. Socket identities are unique per DEALER, so a
re-created connection is never silently dropped by the peer's ROUTER.

**Failure detection.** Receive timeouts adapt to step time (3x the slowest
recent step, at least `CONNEXTE_RECV_TIMEOUT_MS`). A missed lease
(`ENDPOINT_UNHEALTHY`) or a data-path timeout is only a suspicion; Connex
confirms it with a TCP connect to the peer's ZeroMQ port (fast handover,
`CONNEXTE_FAST_HANDOVER=1` by default). The kernel accepts for a live but
busy engine, and refuses at once for a dead one, so: refused means fail
over now, accepted means keep the peer and re-send. Stage 0 also probes
unhealthy replicas of stages >= 2. With the probe disabled (or no peer
address) the old rule applies: timeout plus missed lease = failover; only
a second timeout of the same step reconnects. `downstream_healthy()` tells
the engine whether to keep retrying or give up.

**Fail all in flight at once.** A confirmed death of stage 0's own peer
bumps a chain epoch; a death farther down bumps a remote epoch. Every step
sent before the bump is lost: a waiting `recv_step` / `recv_tensor_dict`
(polled in 50 ms slices) returns `None` at once for it (for a remote death,
only steps already overdue), and `step_stale(corr)` reports it.
`Stage0Link` re-sends all stale in-flight steps together on the first
`None`; an engine that drives K > 1 itself should do the same (optional;
without it each slot still returns quickly, one at a time). At K = 1 stage
0 drops a reply whose id is <= the last one answered, so a late duplicate
of a re-sent step is never taken for the next one, even when the receiving
thread starts before the sending thread has stamped the new id.

**Handover signalling.** Connex carries `cx_ctx`, `cx_missing` and
`finished_req_ids` and applies the rules in section 5 (in `StageWorker` /
`Stage0Link`, or the engine applies them itself with the helpers in
`connex.pp.handover`).

**Metrics** are optional: a `MetricsSink` receives named observations
(`bytes_sent`, `recv_latency`, ...); the vLLM adapter maps them to the
fork's Prometheus collectors.

## 4. What the engine provides

### 4.1 Step boundary (required)

Stage 0: per step, a dict with the hidden states (and whatever the next
stage needs to run its layers: positions, per-row lengths, block info), plus
`request_ids` (one per row) and `cx_ctx` (row i: tokens of `request_ids[i]`
computed before this step). It hands the dict to `Stage0Link.submit` (or
`send_step`), later consumes `Stage0Link.wait(id)` (or `recv_step`), which
returns the last stage's dict (sampled tokens).

Stages 1..: receive the dict (`recv_forward`), run the stage's layers on
it, and send the result (`send_reply(td, ctx)` with the `ctx` returned by
the receive). The engine must forward its own metadata (e.g. scheduled
token counts) in the result dict; Connex forwards `request_ids`, `cx_ctx`
and `finished_req_ids` itself. Reserved keys: `__stage0_endpoint__`,
`__send_timestamp__`, and everything starting with `__cx`.

### 4.2 Per-request KV ownership (required on stages 1..)

`EngineHooks.kv_held(request_id) -> int`: the number of leading tokens of
the request whose KV this stage computed itself and still holds. A replica
that never saw the request returns 0; one that served it earlier and was
bypassed returns its older, shorter length; one that reclaimed it (TTL,
LRU, finished notice) returns 0. For prefill rows an engine with a prefix
cache may count the blocks it computed itself (vLLM does). `KVLedger` is a
ready-made implementation for engines that track nothing themselves.

### 4.3 Recompute of missing requests (required on stage 0)

`EngineHooks.reprefill(request_ids)`: discard the token sampled for these
requests in this step and recompute them from scratch (prompt + generated
tokens) on the current chain, e.g. by setting `num_computed_tokens = 0` and
letting the scheduler issue a (chunked) prefill. Rewinding to 0, not to a
stage-0 prefix-cache hit, matters: a partial context would be missing again
on the new downstream replica. `EngineHooks.abort(request_ids)`: finish with
an error after `reprefill_max` recomputes of the same request.

### 4.4 Release (required on stages 1..)

`EngineHooks.release(request_ids)`: free the per-request KV. Called for
requests reported missing (their next appearance must be a fresh prefill)
and for `finished_req_ids` that are not part of the current step.

### 4.5 K in flight (optional)

With K > 1 stage 0 keeps K steps outstanding and must make them disjoint in
requests (a request whose previous token has not returned is not
scheduled); stages 1.. may prefetch the next step while computing the
current one (one receiving thread, sends may come from another). K = 1
needs no engine change beyond 4.1.

## 5. Handover rules

1. Stage 0 sends `cx_ctx` with every step.
2. A stage s >= 1 computes `missing = [r_i | cx_ctx[i] > 0 and kv_held(r_i) < cx_ctx[i]]`.
3. It releases `missing`, blanks their token ids (nothing it computes for
   them may enter a content-addressed cache), still runs the whole batch
   (the shape is fixed), and sends `cx_missing = upstream cx_missing U missing`.
4. Stage 0 receives `cx_missing` with the reply, drops the tokens sampled
   for those rows, and re-prefills them (rule 4.3); after `reprefill_max`
   rewinds of the same request it aborts it.
5. After a step a stage holds `cx_ctx[i] + scheduled_i` tokens of every
   non-missing row, never less than before (a re-executed older step must
   not shrink it).
6. `finished_req_ids` (each id on 2 consecutive steps) let stages 1.. free
   KV early; ids scheduled in the current step are released at a later step.

## 6. Sequences

### 6.1 Steady state (K = 2, three stages)

```mermaid
sequenceDiagram
    participant S0 as stage 0
    participant S1 as stage 1
    participant S2 as stage 2 (last)
    S0->>S1: step c1 {h, request_ids, cx_ctx}
    S0->>S1: step c2 {h, request_ids, cx_ctx}
    S1->>S2: c1 {h', request_ids, cx_ctx}
    S2-->>S1: CXACK(token) (ring region of c1 free)
    S1->>S2: c2 {h', ...}
    S2->>S0: reply c1 {tokens}  (to c1's origin endpoint)
    S0->>S0: recv_step(c1) -> tokens; schedule next
    S2->>S0: reply c2 {tokens}
```

Each arrow is one ZeroMQ signal; the tensor bytes are pulled by the
receiver with one read from the sender's registered memory, then acked.

### 6.2 Lost reply, retry and duplicate

```mermaid
sequenceDiagram
    participant S0 as stage 0
    participant S1 as stage 1
    participant S2 as stage 2 (last)
    S0->>S1: step c7
    S1->>S2: c7
    S2-xS0: reply c7 (lost)
    Note over S0: recv_step(c7) times out; downstream healthy
    S0->>S1: step c7 (same id)
    Note over S1: c7 in reply cache: re-send cached reply, no recompute
    S1->>S2: c7 (cached bytes, ring region still pinned)
    Note over S2: c7 in reply cache
    S2->>S0: reply c7 (cached)
    Note over S0: a late duplicate reply of c7 is dropped (not outstanding)
```

### 6.3 Failover with re-prefill

```mermaid
sequenceDiagram
    participant S0 as stage 0
    participant A as stage 1 replica A
    participant B as stage 1 replica B
    participant S2 as stage 2
    participant FR as Front Router
    S0->>A: steps for r1, r2 (A and S2 hold their KV)
    Note over A: A dies
    FR-->>S0: ENDPOINT_UNHEALTHY(A) -> suspect
    S0-xA: TCP probe of A's port: refused
    Note over S0: confirmed dead: drop A, choose B, chain epoch + 1;<br/>in-flight c9 (sent to A) is stale: re-sent now
    S0->>B: step c9 {request_ids [r1, r2], cx_ctx [57, 31]}
    Note over B: kv_held(r1) = kv_held(r2) = 0 < ctx: missing; release, blank rows
    B->>S2: c9 {..., cx_missing [r1, r2]}
    S2->>S0: reply c9 {tokens, cx_missing [r1, r2]}
    Note over S0: drop tokens of r1, r2; reprefill(r1, r2)
    S0->>B: step c10 {prefill r1, r2: prompt + generated, cx_ctx [0, 0]}
    B->>S2: c10
    Note over S2: holds r1, r2 already; rewrites their KV on recompute
    S2->>S0: reply c10 {tokens}
```

### 6.4 Graceful leave

```mermaid
sequenceDiagram
    participant S0 as stage 0
    participant A as stage 1 replica A (leaving)
    participant B as stage 1 replica B
    participant FR as Front Router
    A->>FR: deregister (SIGTERM path)
    FR-->>S0: ENDPOINT_DEREGISTERED(A)
    Note over S0: drop sticky peer A, close its DEALER, choose B
    Note over A: may still answer steps it already received
    A->>S0: replies of in-flight steps (accepted if still outstanding)
    S0->>B: next steps; requests whose KV was on A come back in cx_missing
    S0->>B: re-prefill of those requests
```

A step whose reply was in flight on A is either answered by A before it
exits (and accepted, because it is still outstanding) or re-sent to B under
the same id; both orders are safe because ids identify steps.

## 7. Wire protocol

All integers are big-endian. One ZeroMQ message per signal; the sender is a
DEALER, the receiver's ROUTER adds the identity frame; receivers look only at
the last frame.

### 7.1 Data signal

```
offset  size  field
0       8     magic   = "CONNEXTE"
8       8     src_ptr   address of the payload in the sender's registered memory
16      8     total     payload bytes
24      4     url_len
28      L     url       transfer-engine segment of the sender (utf-8)
28+L    8     corr      correlation id (0 = none)
36+L    ...   tail      header revision, below
```

Receivers accept `corr` missing (legacy senders, no matching/dedupe).

**Correlation id.** `corr = incarnation << 32 | seq`: `incarnation` is 32
random bits chosen by each stage-0 process at start, `seq` counts its steps
from 1. Stages 1.. echo the id of the step they answer. Stale rule: drop an
id whose seq is K or more below the highest seq seen from the same
incarnation (K = steps in flight).

### 7.2 Tail revisions

* **host (no tag)**: the payload at `src_ptr` is
  `!Q tensor_len | !d send_ts | safetensors(tensors) | pickle(non-tensors)`.
* **G1**: `"G1" | pickle({"layout": [...], "nt": {...}})`; payload = raw
  tensor bytes per layout. Legacy, still accepted.
* **G2** (default):

```
"G2" | !Q token | !I layout_len | !I nt_len | !H ack_ep_len | !H origin_len
     | layout | ack_ep | origin | nt
layout = !H n, then n x ( !H keylen | !B dtype | !B ndim | key | ndim x !q dim | !Q off | !Q nbytes )
```

`dtype` codes (append-only): 0 float32, 1 float16, 2 bfloat16, 3 float64,
4 int64, 5 int32, 6 int16, 7 int8, 8 uint8, 9 bool, 10 float8_e4m3fn,
11 float8_e5m2. Tensor offsets are 256-byte aligned. `nt` is the pickled
non-tensor dict, empty when there is none (the steady state does no
pickling). `origin` is the stage-0 ROUTER endpoint the last stage replies
to. `token`/`ack_ep`: if non-empty, the receiver acks after its read.

### 7.3 Ack

`"CXACK" | !Q token`, sent to `ack_ep` (the sender's ROUTER). Unacked
regions are reclaimed after `CONNEXTE_GDR_ACK_TIMEOUT_S`.

### 7.4 Non-tensor keys owned by Connex

| key | set by | meaning |
|---|---|---|
| `__stage0_endpoint__` | stage 0 (G2: `origin`) | reply address |
| `request_ids` | stage 0 | row -> request id |
| `cx_ctx` | stage 0 | row -> tokens computed before this step |
| `cx_missing` | stages 1.. | requests some stage lacks KV for (union) |
| `finished_req_ids` | stage 0 | requests finished since recent steps |

### 7.5 Versioning

The header revision is the 2-byte tag after `corr`. Current receivers
accept G1 and G2 and treat any other tail as the host format, so a new
revision must only be sent to receivers that know it; a sender can be pinned
to an older revision (`CONNEXTE_GDR_WIRE=1`) during mixed-version rollouts.
An implementation of a new engine should reject unknown tags explicitly.
New fields go into a new tag; new dtype codes are appended; new non-tensor
keys are ignored by receivers that do not know them. The ZeroMQ-only
fallback (no transfer engine) uses a separate multi-frame format with
pickled `TensorMetadata` (`connex.pp.zmq_base`) and is meant for single-engine
development, not interop.

## 8. Status

* vLLM: production path. `connex.adapters.vllm.connexte.ConnexTECommunicator`
  is the moved communicator with vLLM's logger, network helpers, tensor
  metadata class and Prometheus collectors bound; the engine side is the
  vLLM fork plus `integrations/vllm/patches` (which apply the handover rules
  inside the model runner with the same keys). Equivalence with the
  pre-refactor file is checked byte for byte in
  `tests/test_vllm_adapter_equivalence.py`.
* SGLang: communication prototype (`connex.adapters.sglang`), see
  `sglang-adapter.md`; the engine side does not exist yet.
* UCX: `integrations/vllm/connex_communicator.py` (UCX data plane, own wire
  format) has not been moved onto `connex.pp` yet.
