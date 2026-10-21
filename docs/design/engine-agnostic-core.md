# Engine-agnostic core (connex.pp) and the vLLM adapter

## Why the split looks like this
The production communicator mixed three things in one file inside the vLLM
namespace: vLLM glue (logger, network helpers, TensorMetadata, Prometheus
collectors), the ConnexTE protocol, and the base ZeroMQ socket layer from the
fork. Only the first is engine specific, so the core takes the other two and
exposes the glue as class attributes (`log`, `net`, `codec`, `metrics`) that
an adapter sets; methods were moved verbatim (logger calls rewritten to
`self.log`, metrics calls to a named `MetricsSink`), which is what makes a
byte-level equivalence test possible. The fork still imports
`vllm.distributed.device_communicators.connexte_communicator`, so
`integrations/vllm/connexte_communicator.py` stays as a re-export shim and
the deployment (copy the file, repo on PYTHONPATH) is unchanged.

Handover semantics (cx_ctx / cx_missing) stay inside the vLLM patches
(inlined into the runner); the core has an engine-neutral copy of the rules
(`connex.pp.handover`), checked against `ConnexPPKVGuard` on a failover
script (`tests/test_pp_core.py::test_ledger_matches_vllm_guard_rules`).

## How to check equivalence
`tests/test_vllm_adapter_equivalence.py` loads the frozen pre-refactor file
(`tests/reference/connexte_communicator_reference.py`, refreshed when the
communicator gained fast handover) next to the adapter
and compares wire bytes, returned dicts and protocol state across host and
GPUDirect sends, receive rules, the K>1 step API, peer selection and
membership events. Mutation check: forcing `_STALE_DROP=False` and changing
the layout alignment in the core makes 4 of its 12 tests fail.

## Useful for any new engine adapter
`connex.pp.testing.LoopbackTransferEngine` + `StaticMembership` run the real
endpoint over real ZeroMQ sockets on localhost without Mooncake, a router or
a GPU (`device="cpu"` takes the host-buffer path), including a replica
departure with re-prefill. Set `CONNEX_HOST_IP=127.0.0.1` and patch
`connex.pp.endpoint._CONNEXTE_RECV_TIMEOUT_MS` low for fast timeouts.

## Gotcha found while writing the fakes
Connex forwards only its own keys (`request_ids`, `cx_ctx`,
`finished_req_ids`) to the next stage; engine metadata a later stage needs
(e.g. scheduled tokens per row for its KV accounting) must be put in the
engine's output dict by the engine. Without it a downstream stage
under-counts what it holds and reports every request missing on the next
decode step (all requests re-prefilled forever).

## Porting a communicator change into the core (fast handover)
The endpoint is still a near-verbatim copy of the old communicator, so a
communicator diff applies to `connex/pp/endpoint.py` with
`patch -F3` after rewriting `logger.` to `self.log.`; only hunks whose
context is class-level attributes need hand placement (fuzz put two of them
inside the wrong method). Where each piece of fast handover lives: probe,
chain/remote epochs, `step_stale`, the 50 ms epoch poll and both K=1 race
fixes are in the endpoint; "re-send every stale in-flight step at once" is
a driver rule, so it is in `Stage0Link` (`_resend_stale`, used only if the
transport has the optional `step_stale`) and in the vLLM patch for the fork.
The fake `FakeLink` mirrors the epoch rule (`fail_over`, `step_stale`) so
the driver can be tested without sockets.

Testing the race fixes on the real core endpoint (no vLLM stubs): a late
duplicate reply is produced by having the last stage send its reply twice,
then starting stage 0's recv before its next send; the `_drain_acks` steal
is simulated by wrapping the ROUTER so the first `recv_multipart` after a
poll moves the frames into `_stash` and raises `zmq.Again`. Both tests fail
with the fixes reverted. Note: without `_resend_stale`, the multi-slot test
still passes on wall time (each recv returns at once on a stale epoch), so
it asserts that all K steps were re-sent before the first reply came back.

## Runtime check on the cluster
* Greedy output equality between builds must be measured with sequential
  requests: concurrent runs differ even ref-vs-ref (batching; K=4 N=16 only
  10-14/16 identical). Sequentially, one prompt (#3, token 21) is a near tie
  that flips between deploys of the *same* build (new vs new), while the
  pre-refactor build vs the new one on a fresh pair of deploys was 16/16
  identical.
* Mixed builds interoperate: the pre-refactor communicator file can be
  swapped in per stage, everything else identical, since the vLLM patches
  did not change. Stage 2 old at K=4, stage 0 old and stages 1+3 old at K=1:
  all requests complete, no re-sends, no errors.
* On shared GPUs, capacity changes under you (other tenants take GPUs);
  a pod that cannot get 27 GB crashloops and `wait_ready` times out, which
  makes `fo_matrix.sh` skip the whole K silently except for "ref deploy
  FAILED". Check free memory per GPU before a matrix and move the pod.
* Steady state, failover and churn numbers match the pre-refactor build. One K=1 QPS 4 run lost 1/450 requests to
  `ServerDisconnectedError` 1 ms after sending with 0 tokens (HTTP
  keep-alive reuse race in the aiohttp client; no stage log event); two
  reruns were 100%. Do not read single client-side disconnects as pipeline
  failures without the stage logs; capture logs before the next deploy
  restarts the pods.
