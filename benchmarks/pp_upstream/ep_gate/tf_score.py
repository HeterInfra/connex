# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Teacher-forced re-score: feed prompt+generated ids of a candidate run (gate.py JSON) to a reference server and
count generated tokens more than 0.5 nats below the reference argmax.  usage: tf_score.py URL CAND.json"""
import json, sys
import os
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import post  # noqa: E402

url, cand = sys.argv[1], json.load(open(sys.argv[2]))
off, total, worst = 0, 0, 0.0
for r in cand:
    p, g = r["prompt_ids"], r["ids"]
    if not p or not g:
        sys.exit("candidate lacks token ids (return_token_ids unsupported?)")
    out = post(url + "/v1/completions", {"model": "m", "prompt": p + g, "max_tokens": 1, "temperature": 0,
                                         "prompt_logprobs": 1})
    pl = out["choices"][0].get("prompt_logprobs") or out.get("prompt_logprobs")
    for i, tok in enumerate(g):
        d = pl[len(p) + i]
        lp_tok = d[str(tok)]["logprob"]
        lp_top = max(v["logprob"] for v in d.values())
        gap = lp_top - lp_tok
        worst = max(worst, gap)
        off += gap > 0.5
        total += 1
print(f"{off}/{total} tokens off argmax (>0.5 nats); worst gap {worst:.3f}")
sys.exit(1 if off else 0)
