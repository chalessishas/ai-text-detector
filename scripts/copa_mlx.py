#!/usr/bin/env python3
"""CoPA (Contrastive Paraphrase Attack) via MLX on Apple Silicon.

Uses Qwen3.5-4B (4-bit quantized) for contrastive decoding.
Requires Python 3.13+ with mlx-lm installed.

Run with: /opt/anaconda3/bin/python3.13 scripts/copa_mlx.py

Based on EMNLP 2025 paper:
"Your Language Model Can Secretly Write Like Humans"
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
from mlx_lm import load

# ── CoPA parameters ──────────────────────────────────────────────────────────

LAMBDA = 0.5         # contrast intensity
ALPHA = 1e-5         # plausibility threshold
MAX_TOKENS = 200
TEMPERATURE = 1.0
MODEL_ID = "mlx-community/Qwen3.5-4B-4bit"

# ── Prompts ──────────────────────────────────────────────────────────────────

def build_messages_human(text: str) -> list[dict]:
    return [
        {"role": "system", "content":
            "You rewrite text casually, as if explaining to a friend. "
            "Use simple words, vary sentence length, add personal touches. "
            "Do NOT use <think> tags. Respond directly."},
        {"role": "user", "content": f"Put this in your own words:\n{text}"},
    ]


def build_messages_machine(text: str) -> list[dict]:
    return [
        {"role": "system", "content":
            "You are a formal academic writer. Rewrite text using sophisticated "
            "vocabulary, complex sentence structures, and transition words like "
            "furthermore, moreover, consequently, additionally. "
            "Do NOT use <think> tags. Respond directly."},
        {"role": "user", "content":
            f"Rewrite this formally and academically:\n{text}"},
    ]


def copa_generate_mlx(
    model: nn.Module,
    tokenizer,
    text: str,
    max_tokens: int = MAX_TOKENS,
    lam: float = LAMBDA,
    alpha: float = ALPHA,
    temp: float = TEMPERATURE,
) -> str:
    """CoPA contrastive decoding using MLX.

    Since MLX doesn't natively support two separate KV caches for
    the same model, we use a different approach:
    1. Get full logits from the human prompt
    2. Get full logits from the machine prompt
    3. Apply contrastive formula token by token
    """
    # Build prompts using chat template
    # enable_thinking=False prevents qwen3.5 from entering thinking mode
    prompt_h = tokenizer.apply_chat_template(
        build_messages_human(text), add_generation_prompt=True,
        tokenize=False, enable_thinking=False
    )
    prompt_m = tokenizer.apply_chat_template(
        build_messages_machine(text), add_generation_prompt=True,
        tokenize=False, enable_thinking=False
    )

    tokens_h = mx.array(tokenizer.encode(prompt_h))
    tokens_m = mx.array(tokenizer.encode(prompt_m))

    print(f"  Prompt H: {tokens_h.shape[0]} tokens")
    print(f"  Prompt M: {tokens_m.shape[0]} tokens")

    # We need two separate caches for the two prompts
    # MLX's generate_step uses prompt_cache internally
    # We'll use the model directly for manual generation

    from mlx_lm.models.cache import make_prompt_cache

    cache_h = make_prompt_cache(model)
    cache_m = make_prompt_cache(model)

    # Prefill both caches
    logits_h = model(tokens_h[None, :], cache=cache_h)
    logits_h = logits_h[:, -1, :]  # last token logits

    logits_m = model(tokens_m[None, :], cache=cache_m)
    logits_m = logits_m[:, -1, :]  # last token logits

    generated_tokens = []
    eos_tokens = set()
    if hasattr(tokenizer, 'eos_token_id'):
        if isinstance(tokenizer.eos_token_id, list):
            eos_tokens = set(tokenizer.eos_token_id)
        elif tokenizer.eos_token_id is not None:
            eos_tokens = {tokenizer.eos_token_id}
    # Also check for im_end token
    im_end_id = tokenizer.encode("<|im_end|>")
    if im_end_id:
        eos_tokens.add(im_end_id[-1])

    for step in range(max_tokens):
        # ── CoPA contrastive logits ──────────────────────────────────
        lh = logits_h[0]  # (vocab_size,)
        lm = logits_m[0]

        contrastive = (1 + lam) * lh - lam * lm

        # ── Plausibility constraint ──────────────────────────────────
        probs_h = mx.softmax(lh)
        threshold = alpha * mx.max(probs_h)
        valid = probs_h >= threshold
        # Set invalid tokens to -inf
        contrastive = mx.where(valid, contrastive, mx.array(float('-inf')))

        # ── Temperature + sampling ───────────────────────────────────
        contrastive = contrastive / temp
        probs = mx.softmax(contrastive)

        # Sample from distribution
        token = mx.random.categorical(mx.log(probs + 1e-20))
        token_int = token.item()

        if token_int in eos_tokens:
            break

        # Check for think tags (qwen3.5 sometimes generates these)
        token_str = tokenizer.decode([token_int])
        if "<|im_end|>" in token_str or "<|endoftext|>" in token_str:
            break

        generated_tokens.append(token_int)

        # Advance both caches with the same token
        token_input = mx.array([[token_int]])
        logits_h = model(token_input, cache=cache_h)
        logits_h = logits_h[:, -1, :]
        logits_m = model(token_input, cache=cache_m)
        logits_m = logits_m[:, -1, :]

    result = tokenizer.decode(generated_tokens)
    # Strip any <think>...</think> blocks that may have leaked
    import re
    result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
    return result


def normal_generate_mlx(model, tokenizer, text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Standard generation for comparison (manual token-by-token)."""
    prompt = tokenizer.apply_chat_template(
        build_messages_human(text), add_generation_prompt=True,
        tokenize=False, enable_thinking=False
    )
    tokens = mx.array(tokenizer.encode(prompt))

    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)

    logits = model(tokens[None, :], cache=cache)
    logits = logits[:, -1, :]

    generated = []
    eos_tokens = set()
    if hasattr(tokenizer, 'eos_token_id'):
        if isinstance(tokenizer.eos_token_id, list):
            eos_tokens = set(tokenizer.eos_token_id)
        elif tokenizer.eos_token_id is not None:
            eos_tokens = {tokenizer.eos_token_id}
    im_end_id = tokenizer.encode("<|im_end|>")
    if im_end_id:
        eos_tokens.add(im_end_id[-1])

    for _ in range(max_tokens):
        probs = mx.softmax(logits[0] / TEMPERATURE)
        token = mx.random.categorical(mx.log(probs + 1e-20)).item()

        if token in eos_tokens:
            break
        token_str = tokenizer.decode([token])
        if "<|im_end|>" in token_str or "<|endoftext|>" in token_str:
            break

        generated.append(token)
        logits = model(mx.array([[token]]), cache=cache)
        logits = logits[:, -1, :]

    import re
    result = tokenizer.decode(generated)
    result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
    return result


def analyze_perplexity(text: str, model, tokenizer) -> dict:
    """Compute perplexity metrics for a text."""
    tokens = mx.array(tokenizer.encode(text))
    if tokens.shape[0] < 5:
        return {"perplexity": 0, "entropy": 0, "burstiness": 0, "gltr_top10": 0}

    # Forward pass to get logits for all positions
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    logits = model(tokens[None, :], cache=cache)
    logits = logits[0]  # (seq_len, vocab_size)

    # Perplexity — cast to float32 to avoid bfloat16 numpy issues
    logits = logits.astype(mx.float32)

    logprobs = []
    for i in range(1, tokens.shape[0]):
        probs = mx.softmax(logits[i - 1])
        token_prob = probs[tokens[i]].item()
        logprobs.append(np.log(max(token_prob, 1e-20)))

    avg_logprob = np.mean(logprobs)
    perplexity = np.exp(-avg_logprob)

    # Entropy
    entropies = []
    for i in range(logits.shape[0]):
        probs = mx.softmax(logits[i])
        # Convert to float32 numpy array (MLX may use bfloat16)
        probs_np = np.array(probs.astype(mx.float32))
        probs_np = np.clip(probs_np, 1e-20, 1.0)
        ent = -np.sum(probs_np * np.log(probs_np))
        entropies.append(ent)
    avg_entropy = np.mean(entropies)

    # Burstiness
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    if len(sentences) > 1:
        lengths = [len(s.split()) for s in sentences]
        burstiness = np.std(lengths) / max(np.mean(lengths), 1)
    else:
        burstiness = 0.0

    # GLTR top-10
    top10_count = 0
    for i in range(1, tokens.shape[0]):
        top10_ids = set(mx.argpartition(logits[i - 1], kth=-10)[-10:].tolist())
        if tokens[i].item() in top10_ids:
            top10_count += 1
    gltr_top10 = top10_count / max(tokens.shape[0] - 1, 1) * 100

    return {
        "perplexity": perplexity,
        "entropy": avg_entropy,
        "burstiness": burstiness,
        "gltr_top10": gltr_top10,
    }


def print_metrics(label: str, metrics: dict):
    print(f"  {label}:")
    print(f"    Perplexity: {metrics['perplexity']:.1f}  (human: 20-50, AI: 3-8)")
    print(f"    Entropy:    {metrics['entropy']:.2f}  (human: 2.5-3.5, AI: 1.0-2.0)")
    print(f"    Burstiness: {metrics['burstiness']:.2f}  (human: 0.35-0.65, AI: 0.10-0.20)")
    print(f"    GLTR top10: {metrics['gltr_top10']:.0f}%  (human: <75%, AI: >90%)")


def main():
    test_text = (
        "Artificial intelligence has fundamentally transformed the landscape "
        "of modern education. Furthermore, the integration of machine learning "
        "algorithms into educational platforms has enabled personalized learning "
        "experiences that cater to individual student needs. Moreover, these "
        "technological advancements have facilitated the development of adaptive "
        "assessment tools that provide real-time feedback to both educators and "
        "learners. Consequently, the educational sector has witnessed a paradigm "
        "shift in pedagogical approaches."
    )

    print("=" * 60)
    print("CoPA Contrastive Decoding — Qwen3.5-4B via MLX")
    print("=" * 60)
    print(f"\nModel: {MODEL_ID}")
    print(f"Parameters: lambda={LAMBDA}, alpha={ALPHA}, temp={TEMPERATURE}")
    print(f"\nOriginal AI text ({len(test_text.split())} words):")
    print(f"  {test_text[:200]}...")

    print("\n-- Loading model --")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Standard generation ──────────────────────────────────────────
    print("\n-- Standard paraphrase (no CoPA) --")
    t0 = time.time()
    baseline = normal_generate_mlx(model, tokenizer, test_text)
    print(f"  Generated in {time.time() - t0:.1f}s")
    print(f"  Result: {baseline[:500]}")

    # ── CoPA generation ──────────────────────────────────────────────
    print("\n-- CoPA contrastive paraphrase --")
    t0 = time.time()
    copa_result = copa_generate_mlx(model, tokenizer, test_text)
    print(f"  Generated in {time.time() - t0:.1f}s")
    print(f"  Result: {copa_result[:500]}")

    # ── Perplexity analysis ──────────────────────────────────────────
    print("\n-- Perplexity analysis --")
    m_orig = analyze_perplexity(test_text, model, tokenizer)
    m_base = analyze_perplexity(baseline, model, tokenizer)
    m_copa = analyze_perplexity(copa_result, model, tokenizer)

    print_metrics("Original (AI)", m_orig)
    print_metrics("Standard paraphrase", m_base)
    print_metrics("CoPA paraphrase", m_copa)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
