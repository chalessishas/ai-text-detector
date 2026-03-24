"""Shared utilities for CoPA humanizer scripts.

Centralizes: text feature analysis, EOS token resolution,
CoPA decode loop, test fixtures, and constants.
"""

from __future__ import annotations

import re
from typing import Optional

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache

# ── Constants ────────────────────────────────────────────────────────────────

LOG_EPS = 1e-20

HUMAN_PPL = (13, 50)
HUMAN_ENT = (2.3, 3.5)
HUMAN_BURST = (0.32, 0.65)
HUMAN_GLTR = (0, 75)

# ── Test fixtures ────────────────────────────────────────────────────────────

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


# ── Feature analysis (vectorized) ────────────────────────────────────────────

def compute_features(text, model, tokenizer, max_tokens=512):
    """Compute PPL, entropy, entropy_std, burstiness, GLTR for a text.

    Returns None if text is too short (< 10 tokens).
    Uses vectorized ops instead of per-token Python loops.
    """
    tokens = mx.array(tokenizer.encode(text))
    if tokens.shape[0] < 10:
        return None
    if tokens.shape[0] > max_tokens:
        tokens = tokens[:max_tokens]

    cache = make_prompt_cache(model)
    logits = model(tokens[None, :], cache=cache)[0].astype(mx.float32)
    n = tokens.shape[0]

    # Single softmax, reused for PPL + entropy + GLTR
    all_probs = mx.softmax(logits, axis=-1)
    mx.eval(all_probs)
    probs_np = np.array(all_probs.astype(mx.float32))

    # PPL: vectorized gather of token probabilities
    token_ids = np.array(tokens)[1:]
    token_probs = probs_np[np.arange(n - 1), token_ids]
    ppl = float(np.exp(-np.mean(np.log(np.clip(token_probs, LOG_EPS, 1.0)))))

    # Entropy: vectorized -sum(p * log(p))
    clipped = np.clip(probs_np, LOG_EPS, 1.0)
    entropies = -np.sum(clipped * np.log(clipped), axis=-1)
    ent_mean = float(np.mean(entropies))
    ent_std = float(np.std(entropies))

    # Burstiness: CV of sentence lengths
    sents = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip().split()) > 2]
    if len(sents) > 1:
        lens = [len(s.split()) for s in sents]
        burst = float(np.std(lens) / max(np.mean(lens), 1))
    else:
        burst = 0.0

    # GLTR: vectorized top-10 check
    top10 = np.argpartition(probs_np[:-1], kth=-10, axis=-1)[:, -10:]
    in_top10 = np.any(top10 == token_ids[:, None], axis=1)
    gltr = float(np.mean(in_top10) * 100)

    return {
        "ppl": ppl, "ent_mean": ent_mean, "ent_std": ent_std,
        "burst": burst, "gltr": gltr, "words": len(text.split()),
    }


# ── EOS token resolution ────────────────────────────────────────────────────

def get_eos_tokens(tokenizer):
    """Get all EOS/stop token IDs for a tokenizer."""
    eos = set()
    if hasattr(tokenizer, 'eos_token_id'):
        if isinstance(tokenizer.eos_token_id, list):
            eos = set(tokenizer.eos_token_id)
        elif tokenizer.eos_token_id is not None:
            eos = {tokenizer.eos_token_id}
    im_end = tokenizer.encode("<|im_end|>")
    if im_end:
        eos.add(im_end[-1])
    return eos


# ── CoPA decode loop ─────────────────────────────────────────────────────────

def copa_decode(
    model, tokenizer,
    tokens_h, tokens_m,
    lam, alpha, temp,
    max_tokens=250,
    vocab_mask=None,
):
    """Core CoPA contrastive decoding loop.

    tokens_h/tokens_m: pre-tokenized human/machine prompts (mx.array).
    vocab_mask: optional boolean mask (True=allowed token).
    """
    cache_h = make_prompt_cache(model)
    cache_m = make_prompt_cache(model)
    logits_h = model(tokens_h[None, :], cache=cache_h)[:, -1, :]
    logits_m = model(tokens_m[None, :], cache=cache_m)[:, -1, :]

    eos = get_eos_tokens(tokenizer)
    generated = []

    for _ in range(max_tokens):
        lh, lm = logits_h[0], logits_m[0]

        probs_h = mx.softmax(lh)
        threshold = alpha * mx.max(probs_h)
        valid = probs_h >= threshold
        if vocab_mask is not None:
            valid = valid & vocab_mask

        contrastive = mx.where(valid, (1 + lam) * lh - lam * lm, mx.array(float('-inf')))
        contrastive = contrastive / temp
        probs = mx.softmax(contrastive)

        token = mx.random.categorical(mx.log(probs + LOG_EPS)).item()

        if token in eos:
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


# ── English vocab mask ───────────────────────────────────────────────────────

CJK_RANGES = [
    (0x3000, 0x303F), (0x3040, 0x30FF), (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF), (0xAC00, 0xD7AF), (0xF900, 0xFAFF),
    (0xFF00, 0xFFEF), (0x0400, 0x04FF), (0x0600, 0x06FF),
    (0x0E00, 0x0E7F),
]


def build_english_mask(tokenizer, model):
    """True for English/punctuation tokens, False for CJK/Cyrillic/Arabic."""
    dummy = mx.array([[1]])
    vocab_size = model(dummy).shape[-1]
    mask = np.ones(vocab_size, dtype=bool)

    decode_failures = 0
    for token_id in range(min(vocab_size, tokenizer.vocab_size)):
        try:
            decoded = tokenizer.decode([token_id])
        except Exception:
            mask[token_id] = False
            decode_failures += 1
            continue
        for ch in decoded:
            cp = ord(ch)
            if any(lo <= cp <= hi for lo, hi in CJK_RANGES):
                mask[token_id] = False
                break

    blocked = np.sum(~mask)
    print(f"  English mask: blocked {blocked}/{vocab_size} ({blocked/vocab_size*100:.1f}%)")
    if decode_failures > 0:
        print(f"  ({decode_failures} tokens failed to decode)")
    return mx.array(mask)
