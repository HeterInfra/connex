# PP x EP correctness gate

Checks that a pipeline whose stages each run expert parallelism (vLLM `--enable-expert-parallel`
on a TP group) produces the same greedy tokens as a non-PP server with the same TP width and EP.
Needs one host with 4 GPUs, a vLLM build with the PP branches (see `integrations/vllm/README.md`)
and Connex.

```bash
# optional: a random-init tiny Qwen3-MoE with a real tokenizer, for a smoke without weights
python make_tiny_moe.py /path/to/Qwen3-30B-A3B /tmp/tiny-moe
# reference twice (null), Connex one-instance-per-stage, Connex one-instance, stock NCCL PP, re-score
EXTRA=--enforce-eager ./run_gate.sh /path/to/Qwen3-30B-A3B a3b 20 "ref1 ref2 pp2 pp1 ppn score"
```

`gate.py cmp A.json B.json` compares two runs token by token. `tf_score.py URL RUN.json` feeds
prompt + generated tokens of a run to a reference server and counts generated tokens more than
0.5 nats below the reference's argmax.

Run the exact gate with `--enforce-eager`. With CUDA graphs on, stock vLLM PP is not deterministic
from one fresh server to the next on the vLLM main of 2026-10-26, with or without EP, so an
exact comparison then fails for stock vLLM as well (`docs/results.md`, section 13).
