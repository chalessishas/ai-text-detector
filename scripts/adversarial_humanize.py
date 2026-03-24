#!/usr/bin/env python3
"""Adversarial Paraphrasing with DeBERTa guidance on Apple Silicon.

Adapts the Adversarial Paraphrasing method (NeurIPS 2025, chengez/Adversarial-Paraphrasing)
to use our DeBERTa 98.5% classifier as guidance detector, with MLX for fast paraphrasing
on M4 Mac.

Algorithm: At each token position, generate top-k candidates from the paraphraser,
score each candidate (appended to partial output) with DeBERTa, and select the token
that minimizes AI detection score.

Usage:
    /opt/anaconda3/bin/python3.13 scripts/adversarial_humanize.py
    /opt/anaconda3/bin/python3.13 scripts/adversarial_humanize.py --text "Your AI text here"
    /opt/anaconda3/bin/python3.13 scripts/adversarial_humanize.py --top_k 20  # faster, less optimal
"""

import argparse
import os
import re
import time

import mlx.core as mx
import mlx.nn as mxnn
import numpy as np
import torch
from mlx_lm import load as mlx_load
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTOR_DIR = os.path.join(SCRIPT_DIR, "..", "models", "detector")

SYSTEM_PROMPT = (
    "You are a rephraser. Given any input text, rephrase it without changing "
    "its meaning. Use a different style from the input — vary sentence structure, "
    "word choice, and rhythm. Output ONLY the rephrased text, nothing else."
)

TEST_TEXTS = {
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
        "decisive action, the consequences will be catastrophic and irreversible."
    ),
    "technical": (
        "The transformer architecture has revolutionized natural language processing "
        "by introducing the self-attention mechanism. Unlike recurrent neural networks, "
        "transformers can process entire sequences in parallel, significantly reducing "
        "training time. The key innovation lies in the multi-head attention mechanism, "
        "which allows the model to attend to different positions simultaneously."
    ),
}


class DeBERTaGuidance:
    """Our DeBERTa 98.5% classifier as guidance detector.

    Interface: get_scores(texts) -> numpy array of AI scores (higher = more AI).
    """

    def __init__(self, model_dir=DETECTOR_DIR, device=None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.float()  # fp32 required — fp16 explodes on DeBERTa-v3
        self.model.requires_grad_(False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_scores(self, texts):
        """Return AI probability scores for each text. Higher = more AI-like."""
        inputs = self.tokenizer(
            texts, truncation=True, max_length=512,
            padding=True, return_tensors="pt",
        ).to(self.device)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # ai_score = P(ai) + P(ai_polished) — classes 1 and 2
        ai_scores = probs[:, 1] + probs[:, 2]
        return ai_scores

    def score_single(self, text):
        """Score a single text, return float."""
        return float(self.get_scores([text])[0])


def _build_prompt(tokenizer, messages):
    """Build prompt string, disabling thinking mode if supported."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def _build_whitespace_tokens(tokenizer, sample_size=500):
    """Build set of whitespace-only and thinking token ids to filter during selection."""
    ws = set()
    for tok_id in range(min(sample_size, tokenizer.vocab_size)):
        if not tokenizer.decode([tok_id]).strip():
            ws.add(tok_id)
    for special in ["<think>", "</think>", "<|think|>"]:
        ws.update(tokenizer.encode(special))
    return ws


def adversarial_paraphrase(
    text,
    mlx_model,
    mlx_tokenizer,
    detector,
    top_k=50,
    top_p=0.99,
    temperature=0.6,
    max_tokens=512,
    deterministic=True,
    verbose=True,
):
    """Adversarial paraphrasing: detector-guided token selection.

    At each step:
    1. Get next-token logits from MLX paraphraser
    2. Filter to top-k candidates (after top-p)
    3. For each candidate, decode partial output + candidate -> score with DeBERTa
    4. Select token with lowest AI score
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt = _build_prompt(mlx_tokenizer, messages)
    prompt_tokens = mlx_tokenizer.encode(prompt)
    prompt_array = mx.array([prompt_tokens])
    whitespace_tokens = _build_whitespace_tokens(mlx_tokenizer)

    generated_tokens = []
    cache = mlx_model.make_cache()  # proper KV cache initialization
    input_ids = prompt_array
    t0 = time.time()

    eos_id = mlx_tokenizer.eos_token_id
    # Qwen models also use <|im_end|> as stop token
    stop_tokens = {eos_id}
    im_end = mlx_tokenizer.encode("<|im_end|>")
    if im_end:
        stop_tokens.add(im_end[-1])

    for step in range(max_tokens):
        logits = mlx_model(input_ids, cache=cache)  # cache updated in-place

        next_logits = logits[:, -1, :]
        scaled = next_logits[0] / temperature
        probs_np = np.array(mxnn.softmax(scaled).astype(mx.float32))

        # Top-p + top-k filtering
        sorted_indices = np.argsort(-probs_np)
        sorted_probs = probs_np[sorted_indices]
        cumsum = np.cumsum(sorted_probs)
        cutoff = int(np.searchsorted(cumsum, top_p)) + 1
        cutoff = min(cutoff, top_k)

        candidate_indices = sorted_indices[:cutoff].tolist()

        if not candidate_indices:
            candidate_indices = [sorted_indices[0]]

        # Remove stop tokens from candidates (unless we have enough output)
        if len(generated_tokens) < 20:
            candidate_indices = [c for c in candidate_indices if c not in stop_tokens]
            if not candidate_indices:
                candidate_indices = [sorted_indices[0]]

        # Filter out whitespace-only and thinking tokens from candidates
        content_candidates = [c for c in candidate_indices if c not in whitespace_tokens]
        if content_candidates:
            candidate_indices = content_candidates

        # Single candidate — skip scoring
        if len(candidate_indices) == 1:
            chosen = candidate_indices[0]
        else:
            # Score each candidate with DeBERTa (score on stripped text)
            partial_texts = []
            for cand in candidate_indices:
                trial_tokens = generated_tokens + [cand]
                trial_text = mlx_tokenizer.decode(trial_tokens).strip()
                if not trial_text:
                    trial_text = "."  # avoid empty string
                partial_texts.append(trial_text)

            ai_scores = detector.get_scores(partial_texts)

            if deterministic:
                chosen_idx = int(np.argmin(ai_scores))
            else:
                weights = 1.0 - ai_scores
                weights = np.maximum(weights, 1e-9)
                weights /= weights.sum()
                chosen_idx = int(np.random.choice(len(candidate_indices), p=weights))

            chosen = candidate_indices[chosen_idx]

            if verbose and step % 10 == 0:
                best_score = ai_scores[chosen_idx] * 100
                worst_score = float(np.max(ai_scores)) * 100
                tok_str = mlx_tokenizer.decode([chosen]).replace("\n", "\\n")
                elapsed = time.time() - t0
                print(
                    f"  step {step:3d} ({elapsed:.0f}s): AI={best_score:5.1f}% "
                    f"(worst: {worst_score:.1f}%) "
                    f"tok='{tok_str}'"
                )

        generated_tokens.append(chosen)

        # Check stop condition
        if chosen in stop_tokens:
            break

        # Next input: just the chosen token
        input_ids = mx.array([[chosen]])

    elapsed = time.time() - t0
    result_text = mlx_tokenizer.decode(generated_tokens)

    # Clean up artifacts
    for tag in ["</s>", "<|im_end|>", "<|endoftext|>", "<|end|>"]:
        result_text = result_text.replace(tag, "")
    result_text = result_text.strip()

    if verbose:
        tps = len(generated_tokens) / elapsed if elapsed > 0 else 0
        print(f"  Generated {len(generated_tokens)} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)")

    return result_text


def sentence_level_humanize(
    text,
    mlx_model,
    mlx_tokenizer,
    detector,
    n_candidates=5,
    temperature=1.0,
    max_tokens=256,
    verbose=True,
):
    """Sentence-level adversarial humanization.

    Strategy: split text into sentences, generate N paraphrase candidates per sentence,
    score each with DeBERTa in the context of accumulated output, pick the best.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return text

    result_sentences = []
    t0 = time.time()

    for i, sent in enumerate(sentences):
        if verbose:
            print(f"  Sentence {i+1}/{len(sentences)}: '{sent[:60]}...'")

        candidates = []
        for j in range(n_candidates):
            # Vary the prompt slightly for diversity
            prompts = [
                f"Rewrite this sentence naturally, changing the style but keeping the meaning:\n{sent}",
                f"Rephrase in your own words, use different sentence structure:\n{sent}",
                f"Say the same thing differently, like a college student would write it:\n{sent}",
                f"Express this idea in a completely different way:\n{sent}",
                f"Rewrite casually but keep it smart:\n{sent}",
            ]
            prompt_text = prompts[j % len(prompts)]

            messages = [
                {"role": "system", "content": "Rewrite the given sentence. Output ONLY the rewritten sentence, nothing else. Do not explain."},
                {"role": "user", "content": prompt_text},
            ]
            prompt = _build_prompt(mlx_tokenizer, messages)

            tokens = mlx_tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            cache = mlx_model.make_cache()

            generated = []
            for _ in range(max_tokens):
                logits = mlx_model(input_ids, cache=cache)
                next_logits = logits[:, -1, :]
                next_logits_f = next_logits[0].astype(mx.float32)

                # Temperature sampling
                scaled = next_logits_f / (temperature + 0.1 * j)  # increase temp per candidate
                probs = mxnn.softmax(scaled)
                token_id = int(mx.random.categorical(mx.log(probs)))

                if token_id == mlx_tokenizer.eos_token_id:
                    break
                # Stop on <|im_end|>
                decoded_tok = mlx_tokenizer.decode([token_id])
                if "<|im_end|>" in decoded_tok or "<|end|>" in decoded_tok:
                    break

                generated.append(token_id)
                input_ids = mx.array([[token_id]])

            candidate_text = mlx_tokenizer.decode(generated).strip()
            # Clean artifacts
            for tag in ["</s>", "<|im_end|>", "<|endoftext|>"]:
                candidate_text = candidate_text.replace(tag, "")
            candidate_text = candidate_text.strip()

            if candidate_text and len(candidate_text) > 5:
                candidates.append(candidate_text)

        if not candidates:
            result_sentences.append(sent)
            continue

        # Score all candidates in context of what we've already built
        context = " ".join(result_sentences) if result_sentences else ""
        full_texts = []
        for cand in candidates:
            full = f"{context} {cand}".strip() if context else cand
            full_texts.append(full)

        ai_scores = detector.get_scores(full_texts)
        best_idx = int(np.argmin(ai_scores))
        best_score = ai_scores[best_idx] * 100
        worst_score = float(np.max(ai_scores)) * 100

        if verbose:
            print(f"    Best: AI={best_score:.1f}% (worst: {worst_score:.1f}%) "
                  f"from {len(candidates)} candidates")
            print(f"    '{candidates[best_idx][:80]}...'")

        result_sentences.append(candidates[best_idx])

    elapsed = time.time() - t0
    result = " ".join(result_sentences)
    if verbose:
        print(f"  Total: {elapsed:.1f}s for {len(sentences)} sentences")
    return result


def main():
    parser = argparse.ArgumentParser(description="Adversarial Paraphrasing with DeBERTa guidance")
    parser.add_argument("--text", type=str, default=None, help="Text to paraphrase")
    parser.add_argument("--test", action="store_true", help="Run on all test texts")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3.5-4B-4bit",
                        help="MLX paraphraser model")
    parser.add_argument("--top_k", type=int, default=30,
                        help="Top-k candidates per token (less=faster, more=better evasion)")
    parser.add_argument("--top_p", type=float, default=0.99, help="Top-p nucleus filtering")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no_adversarial", action="store_true",
                        help="Disable adversarial guidance (baseline comparison)")
    parser.add_argument("--mode", type=str, choices=["token", "sentence"], default="sentence",
                        help="token=per-token guidance (original paper), sentence=per-sentence rejection sampling")
    parser.add_argument("--n_candidates", type=int, default=5,
                        help="Number of candidates per sentence (sentence mode only)")
    args = parser.parse_args()

    print("=" * 70)
    print("Adversarial Paraphrasing x DeBERTa 98.5% Guidance")
    print("=" * 70)

    # Load DeBERTa
    print("\n[1/2] Loading DeBERTa detector...")
    t0 = time.time()
    detector = DeBERTaGuidance()
    print(f"  Loaded in {time.time()-t0:.1f}s (device: {detector.device})")

    # Sanity check
    human_score = detector.score_single("I went to the store yesterday and bought some milk.")
    ai_score = detector.score_single(
        "Furthermore, the integration of advanced methodologies has facilitated "
        "the development of comprehensive frameworks for systematic analysis."
    )
    print(f"  Sanity: human text -> AI={human_score*100:.1f}%, AI text -> AI={ai_score*100:.1f}%")

    # Load paraphraser
    print(f"\n[2/2] Loading paraphraser: {args.model}...")
    t0 = time.time()
    mlx_model, mlx_tokenizer = mlx_load(args.model)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Determine texts
    if args.text:
        texts = {"custom": args.text}
    elif args.test:
        texts = TEST_TEXTS
    else:
        texts = {"academic": TEST_TEXTS["academic"]}

    print(f"\n{'=' * 70}")
    print(f"Config: top_k={args.top_k}, top_p={args.top_p}, temp={args.temperature}")
    print(f"Adversarial: {'OFF (baseline)' if args.no_adversarial else 'ON'}")
    print(f"{'=' * 70}")

    results = []
    for name, text in texts.items():
        print(f"\n{'=' * 70}")
        print(f"[{name.upper()}] Original ({len(text.split())} words):")
        print(f"  {text[:150]}...")

        orig_score = detector.score_single(text) * 100
        print(f"  Original AI score: {orig_score:.1f}%")

        mode_label = args.mode if not args.no_adversarial else "baseline"
        print(f"\n  Mode: {mode_label}")

        if args.mode == "sentence" and not args.no_adversarial:
            paraphrased = sentence_level_humanize(
                text, mlx_model, mlx_tokenizer, detector,
                n_candidates=args.n_candidates,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        else:
            effective_top_k = 1 if args.no_adversarial else args.top_k
            paraphrased = adversarial_paraphrase(
                text, mlx_model, mlx_tokenizer, detector,
                top_k=effective_top_k,
                top_p=args.top_p, temperature=args.temperature,
                max_tokens=args.max_tokens,
                deterministic=args.deterministic if not args.no_adversarial else False,
            )

        para_score = detector.score_single(paraphrased) * 100
        delta = orig_score - para_score
        status = "PASS" if para_score < 50 else "FAIL"

        print(f"\n  Paraphrased ({len(paraphrased.split())} words):")
        print(f"  {paraphrased}")
        print(f"\n  {'_' * 50}")
        print(f"  Original:     AI = {orig_score:5.1f}%")
        print(f"  Paraphrased:  AI = {para_score:5.1f}%  [{status}]")
        print(f"  Reduction:    {delta:+.1f}pp")
        print(f"  {'_' * 50}")

        results.append({
            "name": name, "orig_score": orig_score,
            "para_score": para_score, "status": status,
        })

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for r in results:
        print(f"  {r['name']:15s}  {r['orig_score']:5.1f}% -> {r['para_score']:5.1f}%  [{r['status']}]")
    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n  {passed}/{len(results)} passed DeBERTa detection")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
