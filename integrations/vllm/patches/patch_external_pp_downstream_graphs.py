# SPDX-License-Identifier: Apache-2.0
"""Patch 10 (perf, stages 1+): replay CUDA graphs for decode steps.

    python patch_external_pp_downstream_graphs.py /workspace/vllm/vllm/v1/worker/gpu_model_runner.py

Why: stages 1+ capture the usual cudagraphs at startup (capture_model runs
_dummy_run with self.intermediate_tensors / self.positions / the runner's
seq_lens, query_start_loc and input_batch.block_table buffers), but
execute_external_pipeline_step calls set_forward_context without a
cudagraph mode or batch descriptor, so every step runs the inductor code
eagerly (~15.9 ms vs ~13.5 ms for the same 16 Llama-3-8B layers on stage 0).

What: with CONNEX_PP_DOWNSTREAM_GRAPHS=1, a pure decode step (step_type
"decode", one token per request) whose padded size has a captured graph:
  * hidden_states / residual are copied into self.intermediate_tensors and
    positions into self.positions.gpu (pad rows of positions zeroed), i.e.
    the addresses the graphs were captured with;
  * the dispatcher picks the mode for BatchDescriptor(pad, uniform_decode);
  * FULL (FA2 default FULL_AND_PIECEWISE): the per-step metadata the
    existing minimal builder produced (block table rows, slot mapping,
    seq_lens) is copied into the persistent buffers the decode graph reads;
    padded requests get seq_len 0, an empty query range and slot -1, which is
    exactly what the stock runner does for padding;
  * PIECEWISE: attention runs eagerly between pieces with the minimal
    metadata, only the layer inputs need the persistent buffers;
  * the output (a graph-pool tensor reused by the next replay) is sliced to
    the input length and cloned.
Prefill / mixed steps, LoRA, mrope, >1 KV cache group and sizes above the
largest capture size stay eager. All allocator bookkeeping still goes
through the unchanged minimal builder.

CONNEX_PP_PROFILE=1 additionally logs median/p90 GPU forward time of stage
1+ decode steps (CUDA events, read back lazily, every 256 steps).
"""
import os
import sys

MARK = "# connex: downstream-stage CUDA graphs (CONNEX_PP_DOWNSTREAM_GRAPHS)"

FLAG_OLD = "_external_pp_send_executor: ThreadPoolExecutor | None = None\n"
FLAG_NEW = (MARK + "\n"
            "_CONNEX_PP_DOWNSTREAM_GRAPHS = os.environ.get(\"CONNEX_PP_DOWNSTREAM_GRAPHS\", \"0\") == \"1\"\n"
            "_CONNEX_PP_FWD_PROFILE = os.environ.get(\"CONNEX_PP_PROFILE\", \"0\") == \"1\"\n"
            + FLAG_OLD)

METHOD_OLD = '''    def execute_external_pipeline_step(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
    ) -> dict[str, torch.Tensor | Any]:
'''
METHOD_NEW = '''    def _connex_pp_graph_forward(self, step_type, num_reqs, num_actual_tokens,
                                 hidden_states, residual, positions, attn_metadata):
        ''' + MARK + '''
        # Returns the model output, or None to run the step eagerly.
        if (step_type != "decode" or not num_reqs or num_actual_tokens != num_reqs
                or positions is None or self.intermediate_tensors is None
                or self.lora_config is not None or self.uses_mrope
                or not isinstance(attn_metadata, dict) or not attn_metadata):
            return None
        n_in = int(hidden_states.shape[0])
        cc = self.compilation_config
        if (cc.cudagraph_mode is None or cc.cudagraph_mode == CUDAGraphMode.NONE
                or n_in < num_reqs or n_in > cc.max_cudagraph_capture_size
                or positions.shape[0] < n_in or residual.shape[0] != n_in):
            return None
        pad = self.vllm_config.pad_for_cudagraph(n_in)
        mode, desc = self.cudagraph_dispatcher.dispatch(
            BatchDescriptor(num_tokens=pad, uniform_decode=True, has_lora=False))
        if mode == CUDAGraphMode.NONE:
            return None
        if mode == CUDAGraphMode.FULL:
            if len(self.kv_cache_config.kv_cache_groups) != 1:
                return None
            md = next(iter(attn_metadata.values()))
            bt = getattr(md, "block_table", None)
            sm = getattr(md, "slot_mapping", None)
            sl = getattr(md, "seq_lens", None)
            blk = self.input_batch.block_table[0]
            dst_bt = blk.block_table.gpu
            if (bt is None or sm is None or sl is None or bt.shape[1] > dst_bt.shape[1]
                    or pad > dst_bt.shape[0] or pad > self.seq_lens.gpu.shape[0]):
                return None
            dst_bt[:num_reqs, :bt.shape[1]].copy_(bt[:num_reqs], non_blocking=True)
            self.seq_lens.gpu[:num_reqs].copy_(sl[:num_reqs], non_blocking=True)
            self.seq_lens.gpu[num_reqs:pad].zero_()
            ar = getattr(self, "_connex_pp_arange", None)
            if ar is None:
                ar = torch.arange(self.query_start_loc.gpu.shape[0], dtype=torch.int32,
                                  device=self.device)
                self._connex_pp_arange = ar
            qsl = self.query_start_loc.gpu
            qsl[:num_reqs + 1].copy_(ar[:num_reqs + 1])
            qsl[num_reqs + 1:pad + 1].fill_(num_reqs)
            sm_dst = blk.slot_mapping.gpu
            sm_dst[:num_reqs].copy_(sm[:num_reqs], non_blocking=True)
            sm_dst[num_reqs:pad].fill_(-1)
        it = self.intermediate_tensors
        it["hidden_states"][:n_in].copy_(hidden_states, non_blocking=True)
        it["residual"][:n_in].copy_(residual, non_blocking=True)
        pos = self.positions.gpu
        pos[:n_in].copy_(positions[:n_in], non_blocking=True)
        if pad > n_in:
            pos[n_in:pad].zero_()
        inter = IntermediateTensors({k: v[:pad] for k, v in it.tensors.items()})
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=pad,
            cudagraph_runtime_mode=mode,
            batch_descriptor=desc,
        ):
            out = self._model_forward(
                input_ids=None,
                positions=pos[:pad],
                intermediate_tensors=inter,
                inputs_embeds=None,
                **self._init_model_kwargs(pad),
            )
        # Graph outputs live in the graph pool and are overwritten by the next replay.
        if isinstance(out, IntermediateTensors):
            return IntermediateTensors({k: v[:n_in].clone() for k, v in out.tensors.items()})
        return out[:n_in].clone()

    def _connex_pp_fwd_profile(self, ev0, ev1, graphed):
        # CONNEX_PP_PROFILE: stage 1+ decode GPU forward time, read back lazily.
        q = getattr(self, "_connex_pp_fwd_events", None)
        if q is None:
            q = self._connex_pp_fwd_events = []
            self._connex_pp_fwd_ms = {True: [], False: []}
        q.append((ev0, ev1, graphed))
        while q and q[0][1].query():
            a, b, g = q.pop(0)
            self._connex_pp_fwd_ms[g].append(a.elapsed_time(b))
        for g, ms in self._connex_pp_fwd_ms.items():
            if len(ms) >= 256:
                ms.sort()
                logger.info("connex: stage-%d decode forward graphs=%s n=%d median=%.2f ms p90=%.2f ms",
                            get_pp_group().rank_in_group, g, len(ms), ms[len(ms) // 2],
                            ms[int(len(ms) * 0.9)])
                ms.clear()

''' + METHOD_OLD

FWD_OLD = '''        with set_forward_context(
            attn_metadata=attn_metadata,
            vllm_config=self.vllm_config,
            num_tokens=num_tokens,
        ):
            model_output = self._model_forward(
                input_ids=None,  # Not used for intermediate stages
                positions=positions,  # Generated from hidden_states shape
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=None,
                **model_kwargs,
            )
'''
FWD_NEW = '''        ''' + MARK + '''
        _cx_prof = _CONNEX_PP_FWD_PROFILE and step_type == "decode"
        if _cx_prof:
            _cx_ev0 = torch.cuda.Event(enable_timing=True)
            _cx_ev1 = torch.cuda.Event(enable_timing=True)
            _cx_ev0.record()
        model_output = None
        if _CONNEX_PP_DOWNSTREAM_GRAPHS:
            model_output = self._connex_pp_graph_forward(
                step_type, num_reqs, num_actual_tokens, hidden_states, residual,
                positions, attn_metadata)
        _cx_graphed = model_output is not None
        if model_output is None:
          with set_forward_context(
            attn_metadata=attn_metadata,
            vllm_config=self.vllm_config,
            num_tokens=num_tokens,
          ):
            model_output = self._model_forward(
                input_ids=None,  # Not used for intermediate stages
                positions=positions,  # Generated from hidden_states shape
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=None,
                **model_kwargs,
            )
        if _cx_prof:
            _cx_ev1.record()
            self._connex_pp_fwd_profile(_cx_ev0, _cx_ev1, _cx_graphed)
'''

EDITS = [(FLAG_OLD, FLAG_NEW), (METHOD_OLD, METHOD_NEW), (FWD_OLD, FWD_NEW)]


def patch(src: str):
    if MARK in src:
        return src, "already patched"
    for old, _ in EDITS:
        if src.count(old) != 1:
            return None, f"anchor not found/unique: {old.strip()[:60]!r}"
    for old, new in EDITS:
        src = src.replace(old, new)
    return src, "patched"


def main(path: str) -> int:
    src = open(path).read()
    new, status = patch(src)
    print(f"{os.path.basename(__file__)}: {status}")
    if new is None:
        return 1
    if new is not src:
        open(path, "w").write(new)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
