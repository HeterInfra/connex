# Design overview

## Why endpoint mobility

Collective communication libraries assume that the set of peers is fixed
when the job starts: a rank is an address, and a lost rank ends the job.
Dynamic LLM serving breaks that assumption in several ways at once.
Instances scale up and down with load, fail and are replaced, are moved to
other nodes or other hardware, and a single request is increasingly served
by several role-specialised instances that talk to each other over the
network (pipeline stages, prefill and decode instances, attention and FFN
workers). These instances run as many replicas of the same distributed model (several
pipeline chains, pools of prefill and decode instances), and in all of these
the communication partner of a running request can change while data for
that request is in flight: a request should hand over to another replica
when its peer fails or is scaled down, and a new replica should take traffic
as soon as it joins.

Connex treats this as a communication problem rather than an orchestration
problem. Its missing abstraction is the **mobile endpoint**: a stable
logical identity whose physical address, liveness and role-specific state
may change, plus the contract that lets senders keep traffic correct across
such a change.

## Primitives

* **Logical endpoint id.** `LID = (model, stage, shard, replica[, pd_role])`
  (`connex.protocol.LogicalEndpointID`). Addressing, routing and failure
  handling refer to LIDs, never to sockets.
* **Membership with leases.** Endpoints register with the front-router or
  directory (`connex.front_router`, `connex.directory`), heartbeat, and are
  evicted when their lease expires. Peers receive JOIN, LEAVE, failure and
  re-registration (address change) events, or poll.
* **Versioned routes and cutover.** Routes carry epochs
  (`connex.mobility_control.VersionedRouteTable`); a newer epoch for the same
  LID supersedes the old one, a draining route takes no new traffic, and the
  cutover state machine (`connex.cutover_fsm`) orders the switch.
* **Step-level delivery contract.** Every message step carries a correlation
  id; sends and receives are idempotent and retried, a reply cache answers
  duplicates, and stale replies are filtered (`connex.pp`). This is what lets
  a sender re-issue in-flight work to a new endpoint without double
  processing.
* **Sticky peer choice and fast handover.** Traffic stays on one peer until
  that peer is confirmed dead (lease miss or receive timeout, confirmed by a
  TCP probe); then every affected step fails over at once.
* **State ownership hooks.** The engine reports what state a peer holds for
  each request (in the pipeline case, how much KV); work whose state died
  with the old endpoint is recomputed rather than silently continued.

## Scenarios

The same primitives serve several deployment shapes. Their status in this
release:

| Scenario | Uses | Status |
|---|---|---|
| Pipeline parallelism, many replica chains | all primitives; `connex.pp`, vLLM adapter | implemented and validated on GPUs (`docs/results.md`) |
| Scale up / down | membership, sticky choice | implemented and validated (pipeline replicas; `ElasticPeerSet` benchmark in `elastic-scaling-performance.md`) |
| Failover with in-flight handover across replicas | step contract, fast handover, state hooks, resume proxy | implemented and validated for pipeline stages |
| Migration (same LID, new address) | re-registration events, versioned routes | implemented, partly unit-tested, not validated at scale |
| Heterogeneous transports / hardware | ConnexTE (Mooncake RDMA/TCP), UCX, ZeroMQ, capability negotiation | transports validated; mixed GPU types not validated |
| Prefill-decode disaggregation (prefill and decode pools) | PD roles in LIDs, membership, KV holds and pulls (`connex.pd`), vLLM connector, SGLang backend, PD router | implemented and validated on upstream vLLM 0.32.0 and SGLang 0.5.22 on one host (`pd-disaggregation.md`) |
| Attention-FFN disaggregation | generic primitives | roadmap |

Pipeline parallelism came first because it stresses every primitive: a
request touches every stage on every decode step, so any endpoint change is
visible to all in-flight requests within milliseconds. A single chain is
only the building block: the validated deployments run several replica
chains whose stages fail over to each other.

## Where to read next

* `engine-contract.md`: what an engine adapter must implement.
* `engine-agnostic-core.md`: how `connex.pp` was split out of the vLLM
  communicator and how equivalence is checked.
* `vllm-churn-resilience.md`, `reprefill-on-failover.md`, `pp_inflight.md`:
  the failover and K-in-flight mechanisms.
* `request-resume-proxy.md`: hiding front-end replica deaths from clients.
* `elastic-scaling-performance.md`, `connexte-cluster-performance.md`:
  measurements and the reasons behind the defaults.
* `public-deploy-validation.md`: the image and k8s path checked end to end
  on GPUs, and what that check found.
