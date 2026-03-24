#!/usr/bin/env python3
"""CoPA parameter sweep — find optimal lambda/alpha/temp.

Run with: /opt/anaconda3/bin/python3.13 scripts/copa_sweep.py
"""

from __future__ import annotations

import re
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

# ── Test texts (different AI styles) ─────────────────────────────────────────

TEXTS = {
    "academic": (
        "Artificial intelligence has fundamentally transformed the landscape "
        "of modern education. Furthermore, the integration of machine learning "
        "algorithms into educational platforms has enabled personalized learning "
        "experiences that cater to individual student needs. Moreover, these "
        "technological advancements have facilitated the development of adaptive "
        "assessment tools that provide real-time feedback to both educators and "
        "learners. Consequently, the educational sector has witnessed a paradigm "
        "shift in pedagogical approaches."
    ),
    "blog": (
        "Climate change is one of the most pressing challenges facing humanity "
        "today. The rising global temperatures have led to unprecedented weather "
        "events, including devastating hurricanes, prolonged droughts, and massive "
        "wildfires. Scientists have consistently warned that without immediate and "
        "decisive action, the consequences will be catastrophic and irreversible. "
        "It is imperative that governments, corporations, and individuals work "
        "together to reduce carbon emissions and transition to renewable energy."
    ),
    "technical": (
        "The transformer architecture has revolutionized natural language processing "
        "by introducing the self-attention mechanism. Unlike recurrent neural networks, "
        "transformers can process entire sequences in parallel, significantly reducing "
        "training time. The key innovation lies in the multi-head attention mechanism, "
        "which allows the model to attend to different positions simultaneously. "
        "This architecture has become the foundation for large language models such "
        "as GPT and BERT, achieving state-of-the-art results across numerous benchmarks."
    ),
}

# ── Parameter grid ───────────────────────────────────────────────────────────

LAMBDAS = [0.3, 0.5, 0.7, 1.0, 1.5]
ALPHAS = [1e-5, 1e-3, 0.01]
TEMPS = [0.9, 1.0, 1.1]

MODEL_ID = "mlx-community/Qwen3.5-4B-4bit"


def build_prompt(tokenizer, system: str, user: str) -> mx.array:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    return mx.array(tokenizer.encode(text))


SYS_H = (
    "You rewrite text casually, as if explaining to a friend. "
    "Use simple words, vary sentence length, add personal touches."
)
SYS_M = (
    "You are a formal academic writer. Rewrite text using sophisticated "
    "vocabulary, complex sentence structures, and transition words like "
    "furthermore, moreover, consequently, additionally."
)


def copa_generate(model, tokenizer, text, lam, alpha, temp, max_tokens=200):
    tokens_h = build_prompt(tokenizer, SYS_H, f"Put this in your own words:\n{text}")
    tokens_m = build_prompt(tokenizer, SYS_M, f"Rewrite this formally and academically:\n{text}")

    cache_h = make_prompt_cache(model)
    cache_m = make_prompt_cache(model)

    logits_h = model(tokens_h[None, :], cache=cache_h)[:, -1, :]
    logits_m = model(tokens_m[None, :], cache=cache_m)[:, -1, :]

    eos_tokens = set()
    if hasattr(tokenizer, 'eos_token_id'):
        if isinstance(tokenizer.eos_token_id, list):
            eos_tokens = set(tokenizer.eos_token_id)
        elif tokenizer.eos_token_id is not None:
            eos_tokens = {tokenizer.eos_token_id}
    im_end = tokenizer.encode("<|im_end|>")
    if im_end:
        eos_tokens.add(im_end[-1])

    generated = []
    for _ in range(max_tokens):
        lh, lm = logits_h[0], logits_m[0]

        # Plausibility constraint
        probs_h = mx.softmax(lh)
        threshold = alpha * mx.max(probs_h)
        valid = probs_h >= threshold

        # Contrastive logits
        contrastive = mx.where(valid, (1 + lam) * lh - lam * lm, mx.array(float('-inf')))
        contrastive = contrastive / temp
        probs = mx.softmax(contrastive)

        token = mx.random.categorical(mx.log(probs + 1e-20)).item()

        if token in eos_tokens:
            break
        ts = tokenizer.decode([token])
        if "<|im_end|>" in ts or "<|endoftext|>" in ts:
            break

        generated.append(token)

        token_in = mx.array([[token]])
        logits_h = model(token_in, cache=cache_h)[:, -1, :]
        logits_m = model(token_in, cache=cache_m)[:, -1, :]

    result = tokenizer.decode(generated)
    return re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()


def analyze(text, model, tokenizer):
    tokens = mx.array(tokenizer.encode(text))
    if tokens.shape[0] < 5:
        return {"ppl": 0, "ent": 0, "burst": 0, "gltr": 0, "words": 0}

    cache = make_prompt_cache(model)
    logits = model(tokens[None, :], cache=cache)[0].astype(mx.float32)

    # Perplexity
    lps = []
    for i in range(1, tokens.shape[0]):
        p = mx.softmax(logits[i-1])
        lps.append(np.log(max(p[tokens[i]].item(), 1e-20)))
    ppl = np.exp(-np.mean(lps))

    # Entropy
    ents = []
    for i in range(logits.shape[0]):
        p = np.array(mx.softmax(logits[i]).astype(mx.float32))
        p = np.clip(p, 1e-20, 1.0)
        ents.append(-np.sum(p * np.log(p)))
    ent = np.mean(ents)

    # Burstiness
    sents = [s.strip() for s in text.replace("!",".").replace("?",".").split(".") if s.strip()]
    burst = np.std([len(s.split()) for s in sents]) / max(np.mean([len(s.split()) for s in sents]), 1) if len(sents) > 1 else 0

    # GLTR
    t10 = 0
    for i in range(1, tokens.shape[0]):
        top10 = set(mx.argpartition(logits[i-1], kth=-10)[-10:].tolist())
        if tokens[i].item() in top10:
            t10 += 1
    gltr = t10 / max(tokens.shape[0]-1, 1) * 100

    return {"ppl": ppl, "ent": ent, "burst": burst, "gltr": gltr, "words": len(text.split())}


def main():
    print("Loading model...")
    model, tokenizer = load(MODEL_ID)
    print("Loaded.\n")

    # First: analyze original texts
    print("=" * 80)
    print("ORIGINAL TEXT BASELINES")
    print("=" * 80)
    for name, text in TEXTS.items():
        m = analyze(text, model, tokenizer)
        print(f"  {name:12s}  PPL={m['ppl']:5.1f}  ENT={m['ent']:.2f}  BURST={m['burst']:.2f}  GLTR={m['gltr']:.0f}%")

    # Sweep parameters on the "academic" text
    print("\n" + "=" * 80)
    print("PARAMETER SWEEP (academic text)")
    print(f"{'λ':>5s} {'α':>8s} {'T':>5s} | {'PPL':>6s} {'ENT':>5s} {'BURST':>5s} {'GLTR':>5s} | First 80 chars")
    print("-" * 80)

    best_score = -999
    best_params = None

    for lam in LAMBDAS:
        for alpha in ALPHAS:
            for temp in TEMPS:
                try:
                    result = copa_generate(model, tokenizer, TEXTS["academic"],
                                          lam=lam, alpha=alpha, temp=temp)
                    m = analyze(result, model, tokenizer)

                    # Score: how close to human range?
                    # PPL target: 20-50, ENT target: 2.5-3.5, BURST target: 0.35-0.65, GLTR target: <75%
                    ppl_score = -abs(m['ppl'] - 35)     # closer to 35 is better
                    ent_score = -abs(m['ent'] - 3.0)    # closer to 3.0 is better
                    burst_score = -abs(m['burst'] - 0.5) # closer to 0.5 is better
                    gltr_score = -max(0, m['gltr'] - 75) # under 75 is best
                    score = ppl_score + ent_score * 10 + burst_score * 10 + gltr_score

                    marker = ""
                    if score > best_score:
                        best_score = score
                        best_params = (lam, alpha, temp)
                        marker = " ★"

                    preview = result[:80].replace('\n', ' ')
                    print(f"{lam:5.1f} {alpha:8.0e} {temp:5.1f} | {m['ppl']:6.1f} {m['ent']:5.2f} {m['burst']:5.2f} {m['gltr']:4.0f}% | {preview}{marker}")

                except Exception as e:
                    print(f"{lam:5.1f} {alpha:8.0e} {temp:5.1f} | ERROR: {e}")

    print("-" * 80)
    print(f"BEST: λ={best_params[0]}, α={best_params[1]:.0e}, T={best_params[2]} (score={best_score:.1f})")

    # Run best params on all 3 texts
    print("\n" + "=" * 80)
    print(f"BEST PARAMS ON ALL TEXTS (λ={best_params[0]}, α={best_params[1]:.0e}, T={best_params[2]})")
    print("=" * 80)

    lam, alpha, temp = best_params
    for name, text in TEXTS.items():
        result = copa_generate(model, tokenizer, text, lam=lam, alpha=alpha, temp=temp)
        m = analyze(result, model, tokenizer)

        print(f"\n--- {name} ---")
        print(f"  PPL={m['ppl']:5.1f}  ENT={m['ent']:.2f}  BURST={m['burst']:.2f}  GLTR={m['gltr']:.0f}%  ({m['words']} words)")
        print(f"  Output: {result[:300]}")

    # Human reference targets
    print("\n" + "=" * 80)
    print("REFERENCE: Human PPL=20-50, ENT=2.5-3.5, BURST=0.35-0.65, GLTR<75%")
    print("=" * 80)


if __name__ == "__main__":
    main()
