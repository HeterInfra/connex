# SPDX-License-Identifier: Apache-2.0
"""Patch 15 (cold start): pre-compile the logprobs path before /health is green.

vllm/v1/sample/ops/logprobs.py wraps batched_count_greater_than (token rank in
Sampler.gather_logprobs) in @torch.compile(dynamic=True). It is compiled lazily
by the first request that asks for logprobs (sample logprobs or echo/prompt
logprobs), which costs 8.6-9.1 s of TTFT on the cluster. The resume proxy
(logprobs=0 for token ids), failover_correctness and handover all hit it, on
every fresh deploy, rejoin after hard churn and scale-up.

_dummy_sampler_run (called from Worker.compile_or_warm_up_model on the stage
that samples, before the engine reports ready) never sets max_num_logprobs, so
the compile is not covered. This patch makes it run Sampler.compute_logprobs +
gather_logprobs on the dummy logits for row counts {1, 2, num_reqs}: dynamic=True
still specialises dims of size 0/1, so batch 1 (single decode, 1-token prompt)
is its own graph and every size >= 2 shares the symbolic one. num_logprobs 0
and 1 are both exercised (topk k only changes the eager part, but cheap).
Gated by CONNEX_WARMUP_LOGPROBS (default 1; 0 disables).

    python patch_external_pp_warmup_logprobs.py /workspace/vllm/vllm/v1/worker/gpu_model_runner.py
"""
import os
import sys

MARK = "# connex: logprobs compile warm-up"

ANCHOR = """        if self.speculative_config:
            draft_token_ids = [[0] for _ in range(num_reqs)]
"""

BLOCK = f"""        {MARK}
        if os.environ.get("CONNEX_WARMUP_LOGPROBS", "1") != "0":
            _t0 = time.perf_counter()
            for _n in sorted({{n for n in (1, 2, num_reqs) if 1 <= n <= num_reqs}}):
                # fresh (non-view) tensors, as in the real sampling path:
                # dynamo guards differ for views and would recompile later
                _lp = self.sampler.compute_logprobs(logits[:_n])
                _ids = _lp.argmax(dim=-1)
                for _k in (0, 1):
                    self.sampler.gather_logprobs(_lp, _k, _ids)
            if _lp.is_cuda:
                torch.cuda.synchronize()
            del _lp, _ids
            logger.info("connex: logprobs warm-up took %.2f s",
                        time.perf_counter() - _t0)
"""


def patch(src: str):
    if MARK in src:
        return src, "already patched"
    if "def _dummy_sampler_run(" not in src:
        return None, "anchor not found: _dummy_sampler_run"
    head, tail = src.split("def _dummy_sampler_run(", 1)
    if tail.count(ANCHOR) < 1:
        return None, "anchor not found: speculative block in _dummy_sampler_run"
    tail = tail.replace(ANCHOR, BLOCK + ANCHOR, 1)
    return head + "def _dummy_sampler_run(" + tail, "patched"


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
