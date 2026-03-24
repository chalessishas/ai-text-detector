#!/usr/bin/env python3
"""CoPA v2 — contrastive decoding with critical fixes.

Fixes over v1:
  1. English-only vocab mask (no Chinese token leakage)
  2. Randomized prompt templates (no "Honestly" monoculture)
  3. Semantic similarity scoring (cosine sim vs original)
  4. Multi-candidate best-of-N selection
  5. Composite scoring: metrics in-range + semantic preservation

Run with: /opt/anaconda3/bin/python3.13 scripts/copa_proof.py
"""

from __future__ import annotations

import random
import re
import time
from collections import Counter

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from sentence_transformers import SentenceTransformer

MODEL_ID = "mlx-community/Qwen3.5-4B-4bit"

# ── Randomized prompt pool ───────────────────────────────────────────────────
# Why multiple: single template causes pattern lock-in (43/45 started "Honestly")

HUMAN_PROMPTS = [
    {
        "system": "Rewrite the text below in plain, conversational English. Vary your sentence lengths naturally.",
        "user": "Say this differently:\n{text}",
    },
    {
        "system": "You explain complex ideas simply. Use everyday words and a relaxed tone.",
        "user": "How would you explain this to someone?\n{text}",
    },
    {
        "system": "Paraphrase the following. Write like a college student taking notes — brief, direct, sometimes incomplete.",
        "user": "Rewrite:\n{text}",
    },
    {
        "system": "Rewrite this passage. Mix short punchy sentences with longer ones. Use contractions. Be direct.",
        "user": "Put this in different words:\n{text}",
    },
    {
        "system": "You are summarizing a reading for a study group. Be clear and natural, not formal.",
        "user": "Rephrase this for the group:\n{text}",
    },
]

MACHINE_PROMPTS = [
    {
        "system": "You are a formal academic writer. Use sophisticated vocabulary, complex syntax, and transitions like furthermore, moreover, consequently.",
        "user": "Rewrite formally:\n{text}",
    },
    {
        "system": "Write in polished, professional prose. Use passive voice and abstract nouns. Be thorough and precise.",
        "user": "Make this more academic:\n{text}",
    },
]


def build_english_mask(tokenizer, model) -> mx.array:
    """True for English/punctuation/digit tokens, False for CJK/Cyrillic/Arabic etc."""
    # Use model's actual logit output size, not tokenizer.vocab_size (can differ)
    dummy = mx.array([[1]])
    vocab_size = model(dummy).shape[-1]
    del dummy
    mask = np.ones(vocab_size, dtype=bool)

    CJK_RANGES = [
        (0x3000, 0x303F), (0x3040, 0x30FF), (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF), (0xAC00, 0xD7AF), (0xF900, 0xFAFF),
        (0xFF00, 0xFFEF), (0x0400, 0x04FF), (0x0600, 0x06FF),
        (0x0E00, 0x0E7F),
    ]

    for token_id in range(vocab_size):
        try:
            decoded = tokenizer.decode([token_id])
        except Exception:
            mask[token_id] = False
            continue
        for ch in decoded:
            cp = ord(ch)
            if any(lo <= cp <= hi for lo, hi in CJK_RANGES):
                mask[token_id] = False
                break

    blocked = np.sum(~mask)
    print(f"  English mask: blocked {blocked}/{vocab_size} tokens ({blocked/vocab_size*100:.1f}%)")
    return mx.array(mask)


def copa_generate(
    model, tokenizer, text: str,
    lam: float, alpha: float, temp: float,
    eng_mask: mx.array,
    max_tokens: int = 250,
) -> str:
    """CoPA contrastive decoding with English mask and randomized prompts."""
    h_tmpl = random.choice(HUMAN_PROMPTS)
    m_tmpl = random.choice(MACHINE_PROMPTS)

    h_msgs = [
        {"role": "system", "content": h_tmpl["system"]},
        {"role": "user", "content": h_tmpl["user"].format(text=text)},
    ]
    m_msgs = [
        {"role": "system", "content": m_tmpl["system"]},
        {"role": "user", "content": m_tmpl["user"].format(text=text)},
    ]

    tokens_h = mx.array(tokenizer.apply_chat_template(
        h_msgs, add_generation_prompt=True, tokenize=True, enable_thinking=False
    ))
    tokens_m = mx.array(tokenizer.apply_chat_template(
        m_msgs, add_generation_prompt=True, tokenize=True, enable_thinking=False
    ))

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

        probs_h = mx.softmax(lh)
        threshold = alpha * mx.max(probs_h)
        valid = (probs_h >= threshold) & eng_mask

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
    """Compute PPL, entropy, burstiness, GLTR for a text."""
    tokens = mx.array(tokenizer.encode(text))
    if tokens.shape[0] < 5:
        return {"ppl": 0, "ent": 0, "burst": 0, "gltr": 0, "words": 0}

    cache = make_prompt_cache(model)
    logits = model(tokens[None, :], cache=cache)[0].astype(mx.float32)

    lps = []
    for i in range(1, tokens.shape[0]):
        p = mx.softmax(logits[i-1])
        lps.append(np.log(max(p[tokens[i]].item(), 1e-20)))
    ppl = np.exp(-np.mean(lps))

    ents = []
    for i in range(logits.shape[0]):
        p = np.array(mx.softmax(logits[i]).astype(mx.float32))
        p = np.clip(p, 1e-20, 1.0)
        ents.append(-np.sum(p * np.log(p)))
    ent = np.mean(ents)

    sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sents) > 1:
        lens = [len(s.split()) for s in sents]
        burst = np.std(lens) / max(np.mean(lens), 1)
    else:
        burst = 0.0

    t10 = 0
    for i in range(1, tokens.shape[0]):
        top10 = set(mx.argpartition(logits[i-1], kth=-10)[-10:].tolist())
        if tokens[i].item() in top10:
            t10 += 1
    gltr = t10 / max(tokens.shape[0]-1, 1) * 100

    return {"ppl": ppl, "ent": ent, "burst": burst, "gltr": gltr, "words": len(text.split())}


def semantic_sim(a: str, b: str, st_model) -> float:
    embs = st_model.encode([a, b])
    return float(np.dot(embs[0], embs[1]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1])))


def score_candidate(m: dict, sim: float) -> float:
    """Composite score: reward in-range metrics + high semantic similarity."""
    def range_score(val, lo, hi, center):
        if lo <= val <= hi:
            return 1.0
        return -min(abs(val - center), center) / center

    ppl_s = range_score(m["ppl"], 20, 50, 35)
    ent_s = range_score(m["ent"], 2.5, 3.5, 3.0)
    burst_s = range_score(m["burst"], 0.35, 0.65, 0.5)
    gltr_s = 1.0 if m["gltr"] < 75 else -(m["gltr"] - 75) / 25

    return sim * 3.0 + ppl_s * 2.0 + ent_s * 1.5 + gltr_s * 1.0 + burst_s * 0.5


# ── Test texts ───────────────────────────────────────────────────────────────

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

# ── Best params from v1 sweep ────────────────────────────────────────────────

PARAM_SETS = [
    {"lam": 0.5, "alpha": 1e-3, "temp": 1.0},
    {"lam": 1.5, "alpha": 1e-2, "temp": 1.1},
    {"lam": 0.5, "alpha": 1e-5, "temp": 1.1},
    {"lam": 1.0, "alpha": 1e-5, "temp": 1.0},
]
N_PER_PARAM = 5  # candidates per param set


def main():
    print("=" * 80)
    print("CoPA v2 — Contrastive Decoding with Fixes")
    print("  [1] English-only vocab mask  [2] Randomized prompts")
    print("  [3] Semantic similarity      [4] Best-of-N selection")
    print("=" * 80)

    print("\nLoading LLM...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"  LLM loaded in {time.time() - t0:.1f}s")

    print("Loading sentence-transformers...")
    t0 = time.time()
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  ST loaded in {time.time() - t0:.1f}s")

    print("Building English vocabulary mask...")
    eng_mask = build_english_mask(tokenizer, model)

    for text_name, text in TEXTS.items():
        print(f"\n{'='*80}")
        print(f"TEXT: {text_name} ({len(text.split())} words)")
        print(f"{'='*80}")

        m_orig = analyze(text, model, tokenizer)
        print(f"  Original:  PPL={m_orig['ppl']:5.1f}  ENT={m_orig['ent']:.2f}  "
              f"BURST={m_orig['burst']:.2f}  GLTR={m_orig['gltr']:.0f}%")

        total = len(PARAM_SETS) * N_PER_PARAM
        print(f"\n  Generating {total} candidates ({len(PARAM_SETS)} param sets x {N_PER_PARAM})...")

        candidates = []
        for pi, params in enumerate(PARAM_SETS):
            for ci in range(N_PER_PARAM):
                idx = pi * N_PER_PARAM + ci + 1
                t0 = time.time()
                result = copa_generate(
                    model, tokenizer, text,
                    lam=params["lam"], alpha=params["alpha"], temp=params["temp"],
                    eng_mask=eng_mask,
                )
                elapsed = time.time() - t0

                if len(result.split()) < 10:
                    print(f"    [{idx:2d}] SKIP (too short: {len(result.split())} words)")
                    continue

                m = analyze(result, model, tokenizer)
                sim = semantic_sim(text, result, st_model)
                sc = score_candidate(m, sim)

                candidates.append({
                    "text": result, "metrics": m, "sim": sim,
                    "score": sc, "params": params, "time": elapsed,
                })

                # In-range indicators
                flags = (
                    ("P" if 20 <= m["ppl"] <= 50 else ".")
                    + ("E" if 2.5 <= m["ent"] <= 3.5 else ".")
                    + ("B" if 0.35 <= m["burst"] <= 0.65 else ".")
                    + ("G" if m["gltr"] < 75 else ".")
                )
                print(f"    [{idx:2d}] [{flags}] PPL={m['ppl']:5.1f} ENT={m['ent']:.2f} "
                      f"BURST={m['burst']:.2f} GLTR={m['gltr']:.0f}% "
                      f"SIM={sim:.3f} SC={sc:+.1f} ({elapsed:.1f}s)")

        if not candidates:
            print("  No valid candidates!")
            continue

        candidates.sort(key=lambda c: c["score"], reverse=True)

        print(f"\n  {'─'*70}")
        print(f"  TOP 3 / {len(candidates)}:")
        for i, c in enumerate(candidates[:3]):
            m = c["metrics"]
            p = c["params"]
            flags = (
                ("PPL" if 20 <= m["ppl"] <= 50 else "   ") + " "
                + ("ENT" if 2.5 <= m["ent"] <= 3.5 else "   ") + " "
                + ("BURST" if 0.35 <= m["burst"] <= 0.65 else "     ") + " "
                + ("GLTR" if m["gltr"] < 75 else "    ")
            )
            print(f"\n  #{i+1} score={c['score']:+.1f}  sim={c['sim']:.3f}  "
                  f"lam={p['lam']} a={p['alpha']:.0e} T={p['temp']}")
            print(f"     PPL={m['ppl']:5.1f}  ENT={m['ent']:.2f}  BURST={m['burst']:.2f}  "
                  f"GLTR={m['gltr']:.0f}%  in-range: [{flags}]")
            # Show first 300 chars of output
            preview = c["text"][:300].replace("\n", " ")
            print(f"     \"{preview}\"")

        # First-word diversity check
        first_words = [c["text"].split()[0].lower() for c in candidates if c["text"]]
        fw = Counter(first_words).most_common(3)
        pct = fw[0][1] / len(candidates) * 100 if candidates else 0
        print(f"\n  First-word diversity: {fw}")
        if pct > 50:
            print(f"  WARNING: '{fw[0][0]}' in {pct:.0f}% of outputs")

    print(f"\n{'='*80}")
    print("HUMAN TARGETS: PPL=20-50  ENT=2.5-3.5  BURST=0.35-0.65  GLTR<75%")
    print("=" * 80)


if __name__ == "__main__":
    main()
