# RL weight sync and agent session migration

Status: design plus first implementation (`connex.rl`, `connex.agents`).
Upstream snapshots read for this note: vLLM `main` at `89faadf7` and
SGLang `main` at `74c08f3` (both 2026-10-23).

## 1. What changes in the system

Two workloads stress the assumption that the set of serving instances and
the state they hold are fixed for the lifetime of a request.

RL post-training runs a trainer that produces a new weight version every
step and a pool of rollout engines that must serve the new version before
the next batch of rollouts. Rollout pools are elastic: spot or preemptible
GPUs come and go, engines crash, and the pool is resized between phases.
The failure assumptions of today's update paths are that every engine is
present for the whole broadcast (one NCCL group per update round) and that
the update is a barrier across all engines.

Agent serving keeps multi-turn sessions whose prefix state (KV cache)
lives on one instance. Load balancing, scale-down and failures move the
session, and today that means the next turn re-prefills the whole
conversation on another instance, while a generation in flight on the old
instance is lost.

The missing abstraction in both cases is the same one Connex already uses
for pipeline stages: a logical endpoint id whose placement can change, with
an epoch that orders changes and a cutover that resumes in-flight work.

## 2. What the engines expose today

### vLLM

* **Pluggable weight transfer** (`vllm/distributed/weight_transfer/`):
  `WeightTransferEngine` (worker side: `init_transfer_engine`,
  `start_weight_update`, `update_weights` / `receive_weights`,
  `finish_weight_update`, `shutdown`) and `TrainerWeightTransferEngine`
  (`trainer_init`, `send_weights`), selected by
  `--weight-transfer-config '{"backend": ...}'` through
  `WeightTransferEngineFactory` / `WeightTransferTrainerFactory`
  (`register_engine(name, module_or_cls)`). Built in: `nccl`, `ipc`,
  `sparse_nccl`, `sharded_rdt`, `nccl_m2n`, and `modelexpress`, which is a
  re-export of an external package. The base class docstring lists RDMA as
  a TODO. HTTP routes (with `VLLM_SERVER_DEV_MODE=1`): `/pause?mode=abort|wait|keep`,
  `/resume`, `/init_weight_transfer_engine`, `/start_weight_update`,
  `/update_weights`, `/finish_weight_update`, `/update_weight_version`.
  Examples: `examples/rl/rlhf_http_nccl.py`, `rlhf_http_ipc.py`,
  `rlhf_async_new_apis.py`, `rlhf_m2n.py`.
* **KV connectors**: `KVConnectorBase_V1` (scheduler side
  `get_num_new_matched_tokens`, `update_state_after_alloc`,
  `build_connector_meta`, `request_finished`; worker side `start_load_kv`,
  `wait_for_layer_load`, `save_kv_layer`, `wait_for_save`), loadable out of
  tree with `kv_connector_module_path`. Per-request `kv_transfer_params`
  flow through the OpenAI server (the P/D path). There is no per-session
  export or import API; prefix-hash keyed connectors (LMCache,
  OffloadingConnector, MooncakeStore) give "save on finish, load on prefix
  hit" when instances share a store.
* Related open items: RFC #48049 (first-class session id for routing),
  RFC #57103 (programmable KV cache for agentic serving), RFC #57373 (shared
  KV transfer planning primitives), PRs #52152 / #61083 (new disk-based
  weight-transfer backends). No open PR adds an RDMA weight-transfer backend
  or session/request migration.

### SGLang

* **Weight updates**: `update_weights_from_distributed` (NCCL group from
  `init_weights_update_group`), `update_weights_from_tensor` (colocated,
  CUDA IPC or flattened buckets), `update_weights_from_disk`,
  `update_weights_from_ipc` (checkpoint-engine), plus the remote-instance
  loader and the weight-cache daemon. `pause_generation(mode=abort|retract|in_place)`
  / `continue_generation`, `release/resume_memory_occupation`, and
  `weight_version` reported per request (with per-token version spans when a
  request is retracted and continued across an update). There is no backend
  registry; `update_weights_from_tensor` accepts `load_format` naming a
  function listed in `--custom-weight-loader`, which is called as
  `loader(model, named_tensors)` in every TP scheduler process. That is the
  out-of-tree hook Connex uses.
* **KV state**: radix cache, HiCache (GPU / host / L3 storage with
  pluggable `HiCacheStorage` backends keyed by chained page hashes),
  sessions (`open_session`, `session_params`). No export/import or
  migration API. PD transfer interfaces: `BaseKVManager/Sender/Receiver`.
* Related open items: PR #35712 (NIXL weight-transfer backend), PR #42103
  (ModelExpress backend), PR #36279 (weight-cache resharding via Mooncake),
  PR #39990 (NCCL M2N receiver), tracking issue #34510 (PD protocol
  unification). No open item on session or KV migration.

### RL frameworks

verl runs colocated (release memory, train, resume, `update_weights_from_tensor`);
slime/Miles and AReaL run disaggregated (NCCL group, `update_weights_from_distributed`,
wrapped in pause(retract)/continue, `weight_version` for off-policy
accounting). All of them treat the rollout pool as fixed for an update
round: an engine that dies inside the NCCL broadcast hangs or aborts the
round.

## 3. Design: elastic versioned weight sync (`connex.rl`)

**Contract.** Each rollout engine is a member with a logical id and a
weight version. A request is stamped with the pool version at admission
and only runs on an instance at exactly that version; an instance changes
version only when it has nothing in flight. So no request is ever served
by two versions, which is stronger than per-token version spans (it needs
no off-policy correction) at the cost of draining or restarting in-flight
rollouts at an update.

**Data path.** The trainer packs the weights into one registered arena
(double-buffered: a slow reader of `v` is never overwritten by `v+1`; the
publisher waits for readers of the slot it reuses) and publishes `v` with a
manifest (name, dtype, shape, offset). Engines pull with one-sided reads
(connex-rs `get_async`, a window of chunks in flight). The registry hands
each puller a source with spare fan-out, preferring engines that already
hold `v`; every engine that finishes becomes a source, so the transfer is a
tree that grows as it completes and the trainer's NIC serves only the first
`fanout` readers. Pulls are idempotent byte-range reads: a source that dies
or leaves is reported, dropped from the holder set, and the batch is
re-read from another source. The trainer always holds `v`, so the
survivors finish as long as the trainer is alive. There is no group to
form and no barrier: a member joining mid-training pulls the latest
version before it serves, a member leaving simply stops pulling.

**Apply.** In vLLM the pull and `load_weights` run inside
`update_weights` (between scheduler steps, inside vLLM's layerwise-reload
lifecycle); in SGLang inside the custom weight loader. After apply the
engine commits `v` to the registry (and becomes a source).

**Rollout pool.** `RolloutPool` admits requests at the pool version, routes
to instances at that version, drains instances (or aborts past a timeout
and restarts those rollouts at the new version) before updating them, and
updates `k` at a time (`rolling=k`) so capacity never drops to zero. An
instance that fails mid-request is dropped; its request re-runs on another
instance at the same version (with the generated prefix when the instance
supports resume), or restarts at the pool version if none is left.

**Where Connex adds value vs the engine default.** NCCL broadcast is a
collective over a fixed group: fastest when the group is fixed and
healthy, but a join needs a new group and a failure mid-broadcast hangs
the round. Connex trades a little peak bandwidth for: no group setup on
membership change, tree fan-out without a collective, per-chunk failover,
and catch-up of late joiners from peers. Section 6 has the measurements.

## 4. Design: agent session migration (`connex.agents`)

**Session LIDs.** `SessionProxy` (a subclass of the resume proxy) keeps a
table `session_id -> (placement, epoch, in-flight requests, last context)`.
The session id comes from `session_id` in the body or the
`X-Connex-Session` header and is stripped before the request reaches the
engine. Requests are routed to the placement (sticky).

**migrate(session, target).** (1) Prepare the target: a `KVMover` makes the
target hold the session's prefix state. Movers are tried in order; the
built-in `ReprefillMover` re-prefills the last known context on the target
(one generated token) so the target's automatic prefix cache (vLLM APC,
SGLang radix cache) holds it. A KV-transfer mover plugs in here (section 5).
(2) Bump the epoch and repoint the session. (3) Cut every in-flight
generation over with the proxy's resume: close the backend stream (the
engine aborts the request), re-issue prompt + tokens emitted so far on the
target. The client sees one stream with the same id. Greedy outputs are
identical to an unmigrated run; with sampling the continuation is a valid
sample, not the same RNG stream.

**Failure.** If the placement dies, the next attempt goes to another
instance, the epoch increases and the move is recorded as a failover; that
instance re-prefills on demand.

**Drain.** `drain(url)` migrates every session off an instance (scale-down)
before it is removed.

Proxy hooks added for this, all small: `Proxy.pick(st, tried)`,
`Proxy.move(st)` (planned cutover that is not counted as a failure and does
not blacklist), and `begin`/`end` request hooks.

## 5. Upstream changes (proposed, not submitted)

Keep engine changes minimal and engine-native; everything Connex-specific
(registry, membership, router, connex-rs) stays in this repository.

* **vLLM, weight transfer**: register a `connex` backend that re-exports
  `connex.rl.vllm_backend` (same pattern as `modelexpress_engine.py`), plus
  docs and a factory test. Out of tree it already works today through the
  `vllm.general_plugins` entry point this package installs; the upstream
  change only makes it discoverable without the plugin.
* **SGLang**: no change needed for weight sync (custom weight loader).
* **KV export/import for migration**: vLLM's connector API is already
  pluggable out of tree; SGLang's HiCache storage backends are too. The PD
  disaggregation work (branch `feat/pd-disagg`) builds a KV export/import
  layer over `KVConnectorBase_V1`; a session migration mover should reuse it
  rather than add a second connector. Overlap is noted in the report.

## 6. Results

See `docs/results.md` (RL weight sync, agent migration). Numbers in this
note are not duplicated there.

## 7. Limits and open points

* The registry is a single process (HTTP/JSON). It holds soft state only
  (members heartbeat, holders are re-announced on commit), so restarting it
  loses nothing a new publish does not restore, but it is not replicated.
* Each TP rank of an engine pulls the full model and lets `load_weights`
  shard it, as vLLM's NCCL path does. Pulling only the rank's shard needs
  a resharding plan (what `nccl_m2n` / `sharded_rdt` do).
* `RolloutPool` restarts rollouts that were aborted for an update; it does
  not keep them at the old version on a straggler instance.
* The SGLang path passes a control record through
  `update_weights_from_tensor`; it relies on SGLang's safe unpickler
  accepting a plain CPU tensor.

## 8. Insights from bring-up (why, and how to reproduce)

* **The registry client must ignore `HTTP(S)_PROXY`.** On hosts with a proxy
  in the environment, `urllib` sends cluster-internal registry calls to the
  proxy because it does not understand CIDR entries in `NO_PROXY`
  (a subnet entry is ignored). The symptom was a non-JSON error body on
  `publish`. `RegistryClient` builds an opener with an empty `ProxyHandler`.
* **Pin UCX to the RDMA device that carries the advertised address.**
  Without `UCX_NET_DEVICES`, connex-rs in a pod also tried the calico and
  tailscale interfaces and cross-node reads failed with "peer failed".
  `ConnexRsTransport` sets `UCX_NET_DEVICES` from
  `connex.pp.transports.ucx_net_devices_for_ip` (e.g. `mlx5_0:1,ibs102`)
  when it is unset, as `connex.pp` already does. The advertised address
  defaults to the first `ib*` IPv4 address (`CONNEX_ADVERTISE_IP` overrides).
* **Stale keep-alive connections look like dead engines.** uvicorn closes
  idle connections after 5 s; aiohttp keeps them for 15 s. After a training
  step longer than 5 s, the next control call (`/start_weight_update`) went
  out on a closed connection, raised `ServerDisconnectedError`, and the pool
  removed a healthy engine. Fix: client keep-alive of 2 s and one retry on
  `ServerDisconnectedError` (a dead server also refuses the retry's new
  connection, so real failures are still reported).
* **Cold-migration numbers need unique prefixes per session.** The first
  benchmark reused one filler text for every session, so both engines had it
  cached after the first sessions and "cold" migrations looked as fast as
  sticky ones. Each session's context now starts with a unique header.
* **Do not run the migration inside the client's stream reader.** Awaiting
  `migrate` from the token callback stopped the client from reading, so the
  measured stall equalled the re-prefill time. The benchmark now migrates in
  a separate task; the stall the client sees is only the cut-over.
* **NCCL over IB verbs fails between pods of the benchmark image**
  (`vllm-pipeline-base-images:v1.10`, NCCL 2.27.5 and 2.28.8, with or without
  the image's external net plugin): `socketFinalizeAccept ... wrong type 3 != 4`
  at the first collective. NCCL with `NCCL_IB_DISABLE=1` (IPoIB sockets)
  works, so the cross-node NCCL baseline in `results.md` is over IPoIB and
  is labelled as such; connex-rs uses IB verbs there.
* **Intra-node, NCCL wins.** On one A40 host without NVLink, a chunked NCCL
  broadcast of 0.99 GB to two GPUs takes 0.05 s (PCIe peer path), while
  connex-rs reads go through the NIC loopback (0.2 s). Use NCCL or CUDA IPC
  inside a node; Connex's case is across nodes and under churn.
* **Environment traps on the test host, not Connex bugs.** vLLM 0.32.0 with
  flashinfer 0.7.0 JIT-compiles its sampling kernels on the first sampled
  (temperature > 0) request and failed with the host's older `nvcc`
  (`Unknown option '--compress-mode=size'`), killing the engine right after
  a weight update; greedy requests do not hit it. SGLang 0.5.22 in a
  shared environment refused to start on a flashinfer / flashinfer-cubin
  version mismatch; the runs used `FLASHINFER_DISABLE_VERSION_CHECK=1` with
  the Triton attention backend and the PyTorch sampler.
* **SGLang 0.5.22 requires an update session.** `update_weights_from_tensor`
  outside `begin_weight_update` / `end_weight_update` returns 400 there;
  `SGLangInstance` opens and closes the session (and skips it on servers
  that answer 404, such as 0.5.9).
