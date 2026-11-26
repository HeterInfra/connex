# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Random-init tiny Qwen3-MoE (fixed seed) with the real Qwen3-30B-A3B tokenizer, for a no-weights smoke."""
import json, os, shutil, sys
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

src, dst = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(src, "config.json")))
torch.manual_seed(0)
c = Qwen3MoeConfig(vocab_size=cfg["vocab_size"], hidden_size=256, intermediate_size=512, moe_intermediate_size=128,
                   num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                   num_experts=16, num_experts_per_tok=4, decoder_sparse_step=1, mlp_only_layers=[],
                   max_position_embeddings=4096, rope_theta=cfg.get("rope_theta", 1e6), tie_word_embeddings=False,
                   bos_token_id=cfg.get("bos_token_id"), eos_token_id=cfg.get("eos_token_id"), torch_dtype="bfloat16")
m = Qwen3MoeForCausalLM(c).to(torch.bfloat16)
with torch.no_grad():
    for p in m.parameters():
        p.normal_(0, 0.05)
m.save_pretrained(dst, safe_serialization=True)
for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"):
    if os.path.exists(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), dst)
print("tiny model at", dst)
