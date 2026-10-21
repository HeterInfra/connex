# Connex under dynamic scale-out / scale-in

Measured 2026-10-16 on one A40 host ( busy box with load avg 10–16), UCX
1.20 / ucx-py 0.45, loopback TCP, 1 MiB messages at 500 msg/s from one
upstream sender to a pool of downstream replicas.

## How to reproduce

```bash
benchmarks/run_elastic_matrix.sh /tmp/elastic-out 5   # ~5 min, prints a table
python benchmarks/elastic_bench.py --membership elastic --duration 12   # one run
```

Timeline per run: r0 and r1 up → T/4 r2 JOINs → T/2 r0 deregisters
(graceful) → 3T/4 r1 is SIGKILLed. Receivers log every message with its send
timestamp, so loss and one-way latency are exact. All processes are on one
host, so a single clock serves every timestamp.

## Results (median of 5 runs)

| config | msg/s | join ready | sends after LEAVE | kill stall | lost |
|---|---|---|---|---|---|
| baseline (poll 100 ms, TLS=all, rndv) | 306 | 172 ms | 15 | 5008 ms | 0 |
| transport_opt (poll + TCP defaults) | 316 | 109 ms | 3 | 5009 ms | 3 |
| full_opt (ElasticPeerSet) | 496 | 54 ms | 0 | 34 ms | 3 |

Target rate is 500 msg/s. The baseline delivers only ~300 msg/s, because the
sender is blocked for seconds during scale events.

## What was wrong, and why

1. **Connect took ~0.7 s.** `UCX_TCP` mode still passed `TLS=all` and
   `SOCKADDR_TLS_PRIORITY=rdmacm,tcp`, so on a host with IB devices every
   `create_endpoint` probed the RDMA transports first. With
   `TLS=tcp,sm,self,cuda_copy` a connect takes ~40 ms. Lazy connect-on-first-send put
   this cost straight on the data path of every scale-out.
2. **Membership events were never delivered.** `DirectoryClient.subscribe_events`
   stored handlers, but nothing ever opened a SUB socket, and the directory had
   no way to list a stage's replicas. Senders could only poll, so they kept
   sending to a leaving replica for up to one poll period.
3. **A killed replica stayed in the directory forever.** There was no lease and no
   failure report. Each poll re-added the dead member, and the next lazy
   reconnect to its closed port blocked ~5 s (this is the 5 s "kill stall" in the
   polling configs).
4. **UCX segfault on peer death.** With a rendezvous send in flight to a peer
   that dies, UCX crashes the *sender* in
   `ucp_ep_reqs_purge → ucp_proto_rndv_rts_abort → ucp_request_memh_invalidate`
   (NULL+0xb8). The rate was 3/20 runs, and 5/20 even without our explicit
   `close_endpoint`, so the crash is inside UCX. With `UCX_RNDV_THRESH=inf` it was 0/20, and after the fix
   0/45 under parallel load. Our explicit close does not cause it.
5. **ucx-py rewrites the environment.** `import ucp` sets `UCX_RNDV_THRESH=8192`
   (and others) in `os.environ`. A "did the user set it?" check after the import
   always says yes. Capture env before importing ucp (`_USER_RNDV_THRESH`).
   Separately, setting `UCX_SOCKADDR_TLS_PRIORITY` in the environment makes
   ucx-py's `ucx_config_to_dict` raise, and Connex then *silently* falls back to
   `TLS=tcp,cuda_copy`.
6. **`close_endpoint` blocked up to 5 s** on a dead peer. It now schedules the
   close on the UCX loop and returns immediately.

## What changed

- `UCX_TCP` mode: TCP-class TLS and eager-only (`RNDV_THRESH=inf` unless the
  user sets `UCX_RNDV_THRESH`).
- Directory: `list_stage`, a PUB event stream, leases (`start_heartbeat`),
  `report_failure`, and automatic re-register of a falsely evicted member.
- `connex.elastic.ElasticPeerSet` connects new members in the background
  and routes to them only once connected. LEAVE removes a member from routing
  at once, and it is closed after a drain grace. A send failure triggers
  `report_failure`, so every sender drops the member.

## Costs and open issues

- **Eager-only costs bandwidth**: ~20–30 % at 1–8 MiB over TCP, and it is
  slightly faster at 64 KiB (`benchmarks/ucx_bandwidth.py`, noisy box).
- **Eager loses in-flight messages on abrupt kill** (median 3 at 500 msg/s).
  An eager send completes once the data is buffered locally, whereas a rendezvous send completes
  only after the receiver has pulled the data. Upper layers must retry on peer
  failure. Graceful leave loses nothing.
- **p50 one-way latency 0.8 → 1.6 ms on loopback**: `TLS=all` uses IB loopback
  on one host, which is faster for 1 MiB (2.3 vs 1.5 GB/s). In a cross-node
  TCP deployment this advantage does not exist. If you have RDMA, use `UCX_RDMA` mode.
- **`UCX_RDMA` mode still uses rendezvous**, so the segfault in (4) can still
  hit there. It was not reproduced, because no cross-node IB setup was available. It needs an
  upstream UCX report or an app-level guard.
- `connect_sync` still has a 10 s timeout. `ElasticPeerSet` keeps it off the data path,
  but a lazy-connect user (like the vLLM communicator today) still pays it
  when a dead member is listed.
