# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Serial greedy run against an OpenAI-compatible vLLM server; writes token ids per prompt.
usage: gate.py run URL MODEL OUT.json [N] | gate.py cmp A.json B.json"""
import json, sys, time, urllib.request

PROMPTS = [
    "The capital of France is", "Write a short poem about the sea.", "Explain how a transistor works.",
    "List the first ten prime numbers:", "def fibonacci(n):", "Translate to German: The weather is nice today.",
    "Once upon a time in a small village,", "What is the derivative of x^3 + 2x?", "Summarize the theory of relativity.",
    "The three primary colors are", "Give three tips for writing clean code.", "In 1969, the first humans",
    "A haiku about autumn leaves:", "SELECT name FROM users WHERE", "Why is the sky blue?",
    "Recipe for pancakes: ingredients", "The mitochondria is", "Describe a busy train station at rush hour in detail, "
    "including the sounds, the people, the announcements and the smells, and continue for a long while.",
    "Write a long story about a robot who learns to paint, with dialogue and several chapters.",
    "Explain step by step how TCP congestion control works, covering slow start, congestion avoidance, fast "
    "retransmit and fast recovery, with examples.",
]


def post(url, body, timeout=600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run(url, model, out, n=len(PROMPTS), max_tokens=128):
    res = []
    for p in PROMPTS[:n]:
        t = time.time()
        r = post(url + "/v1/completions", {"model": model, "prompt": p, "max_tokens": max_tokens,
                                          "temperature": 0, "return_token_ids": True})
        ch = r["choices"][0]
        res.append({"prompt": p, "text": ch["text"], "ids": ch.get("token_ids"),
                    "prompt_ids": ch.get("prompt_token_ids") or r.get("prompt_token_ids"), "s": time.time() - t})
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out, len(res))


def cmp(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    exact, firsts = 0, []
    for x, y in zip(A, B):
        xa, yb = x["ids"] or x["text"], y["ids"] or y["text"]
        if xa == yb:
            exact += 1
        else:
            k = next((i for i, (u, v) in enumerate(zip(xa, yb)) if u != v), min(len(xa), len(yb)))
            firsts.append(k)
    print(f"{exact}/{len(A)} exact; first diffs {firsts}")
    return exact == len(A)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], *(int(x) for x in sys.argv[5:6]))
    else:
        sys.exit(0 if cmp(sys.argv[2], sys.argv[3]) else 1)
