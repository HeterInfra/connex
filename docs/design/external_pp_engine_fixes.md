# External-PP engine fixes (patches 11-13)

Source: `integrations/vllm/patches/patch_external_pp_{chunked_seqlens,finished_ids,host_logits_idx}.py`, pure logic in `external_pp_engine_fixes.py`, tests in `tests/test_engine_fixes.py`. Fork: external-PP vLLM fork 380186d57.

## Why the downstream KV pool filled with dead requests
Stages 1+ never run `_update_states`, so `scheduler_output.finished_req_ids` exists only on stage 0. The downstream allocator could only drop a request after `CONNEX_PP_REQ_TTL_S` (120 s) of silence or by LRU on exhaustion, so live_reqs grows at roughly arrival_rate x 120 s and dead requests evict useful prefix-cache blocks. Stage 0 now attaches `finished_req_ids` (a pickled non-tensor field, only when non-empty) to the next 2 sends; downstream releases them before the step allocates, defers ids that the step itself still references, and re-releases requests that a re-sent older step re-creates (tombstones). Duplicates are no-ops, so retries and repeats are safe.

How to check on GPUs: stage 1+ logs `External PP stage N: finished notices=.. released=.. deferred=.. revived=.. live_reqs=.. free=..` every 256 notices; live_reqs should stay around the number of concurrently running requests, and the allocator's own line (`prefix cache hits=... reclaimed_reqs=... live_reqs=...`) should show reclaimed_reqs staying flat (the TTL fallback is no longer doing the work).

## Why seq_len - query_len must equal num_computed_tokens
The allocator patch overwrote `num_computed_tokens_list` in place with this stage's block-granular prefix-cache view. That list object came from the received tensor_dict, and intermediate stages forward the same object, so stage 2+ got stage 1's view (wrong seq_lens with >2 stages). The runner now copies the list, and `num_computed_tokens_cpu` is derived as seq_lens - query_lens, vLLM's own invariant.

## Why logits_indices was a device sync
`logits_indices.cpu().tolist()` on stage 0 waits for the whole forward before the send is enqueued. Without spec decode it equals `query_start_loc.np[1:num_reqs+1] - 1`, already on the host. `CONNEX_PP_HOST_LOGITS_IDX=0` restores the old path.
