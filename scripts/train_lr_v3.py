#!/usr/bin/env python3
"""Train LR v3 with 16 perplexity + 8 linguistic = 24 features.

Expands LR v2 with spaCy-based linguistic features that capture
human vs AI writing style differences (pronouns, contractions, tense).
Also trains on diverse human data (HC3, IMDB) for better calibration.

Run: /opt/anaconda3/bin/python3.13 scripts/train_lr_v3.py
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
import re
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import spacy

# ── Config ────────────────────────────────────────────────────────────────
MLX_MODEL_ID = "mlx-community/Qwen3.5-4B-4bit"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DATASET_PATH = os.path.join(PROJECT_DIR, "dataset_v4.jsonl")
EXTRA_HUMAN_SOURCES = [
    os.path.join(PROJECT_DIR, "data_human_hc3.jsonl"),
    os.path.join(PROJECT_DIR, "data_human_imdb.jsonl"),
]
NEW_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "perplexity_lr_v3.pkl")

N_SAMPLE = int(os.environ.get("LR_N_SAMPLE", "3000"))
N_EXTRA_HUMAN = int(os.environ.get("LR_N_EXTRA", "2000"))
MAX_TOKENS = 128
MIN_TOKENS = 10
RANDOM_SEED = 42

print("Loading spaCy...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])


# ── Perplexity features (same as v2) ───────────────────────────────────

def compute_token_data(text, model, tokenizer):
    import mlx.core as mx

    tokens = tokenizer.encode(text)
    if len(tokens) < MIN_TOKENS or len(tokens) > 2048:
        return None
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]

    x = mx.array([tokens])
    logits = model(x)[0]
    pred_logits = logits[:-1]
    target_ids = mx.array(tokens[1:])
    n = target_ids.shape[0]

    log_probs = pred_logits - mx.logsumexp(pred_logits, axis=-1, keepdims=True)
    target_logprobs = log_probs[mx.arange(n), target_ids]
    top1_ids = mx.argmax(pred_logits, axis=-1)
    is_top1 = (top1_ids == target_ids)
    top10_indices = mx.argpartition(pred_logits, kth=-10, axis=-1)[:, -10:]
    is_top10 = mx.any(top10_indices == target_ids[:, None], axis=-1)

    probs = mx.exp(log_probs)
    safe_log_probs = mx.where(probs > 1e-10, log_probs, mx.zeros_like(log_probs))
    safe_probs = mx.where(probs > 1e-10, probs, mx.zeros_like(probs))
    entropies = -mx.sum(safe_probs * safe_log_probs, axis=-1)

    mx.eval(target_logprobs, is_top1, is_top10, entropies)

    return {
        "logprobs": np.array(target_logprobs.astype(mx.float32)),
        "is_top1": np.array(is_top1),
        "is_top10": np.array(is_top10),
        "entropies": np.array(entropies.astype(mx.float32)),
        "n": n,
    }


def extract_basic_features(td):
    lp = td["logprobs"]
    ppl = math.exp(-float(np.mean(lp)))
    return {
        "log_ppl": math.log(max(ppl, 1e-5)),
        "top10": float(np.mean(td["is_top10"])) * 100,
        "mean_ent": float(np.mean(td["entropies"])),
        "top1": float(np.mean(td["is_top1"])) * 100,
        "ent_std": float(np.std(td["entropies"])),
    }


def extract_diveye_features(td):
    surprisal = -td["logprobs"]
    if len(surprisal) < 5:
        return {k: 0.0 for k in [
            "s_mean", "s_std", "s_var", "s_skew", "s_kurt",
            "d1_mean", "d1_std", "d2_var", "d2_entropy", "d2_autocorr"
        ]}

    diff1 = np.diff(surprisal)
    diff2 = np.diff(diff1) if len(diff1) > 1 else np.array([0.0])

    if len(diff2) > 5:
        hist, _ = np.histogram(diff2, bins=20, density=True)
        hist = hist[hist > 0]
        d2_entropy = float(-np.sum(hist * np.log(hist + 1e-10)))
    else:
        d2_entropy = 0.0

    if len(diff2) > 2:
        d2_autocorr = float(np.corrcoef(diff2[:-1], diff2[1:])[0, 1])
        if np.isnan(d2_autocorr):
            d2_autocorr = 0.0
    else:
        d2_autocorr = 0.0

    return {
        "s_mean": float(np.mean(surprisal)),
        "s_std": float(np.std(surprisal)),
        "s_var": float(np.var(surprisal)),
        "s_skew": float(skew(surprisal)),
        "s_kurt": float(kurtosis(surprisal)),
        "d1_mean": float(np.mean(diff1)) if len(diff1) > 0 else 0.0,
        "d1_std": float(np.std(diff1)) if len(diff1) > 0 else 0.0,
        "d2_var": float(np.var(diff2)) if len(diff2) > 0 else 0.0,
        "d2_entropy": d2_entropy,
        "d2_autocorr": d2_autocorr,
    }


def extract_specdetect_energy(td):
    lp = td["logprobs"]
    if len(lp) < 10:
        return 0.0
    surprisal = -lp
    surprisal = surprisal - np.mean(surprisal)
    fft = np.fft.rfft(surprisal)
    return float(np.sum(np.abs(fft) ** 2) / len(surprisal))


# ── NEW: Linguistic features (spaCy) ──────────────────────────────────

CONTRACTION_PATTERN = re.compile(
    r"\b(I'm|I've|I'll|I'd|don't|doesn't|didn't|won't|can't|couldn't|"
    r"wouldn't|shouldn't|haven't|hasn't|isn't|aren't|wasn't|weren't|"
    r"it's|that's|there's|they're|we're|you're|he's|she's|let's|"
    r"who's|what's|here's)\b", re.IGNORECASE
)

FIRST_PERSON = re.compile(r"\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b")


def extract_linguistic_features(text):
    """8 linguistic features that distinguish human from AI writing."""
    doc = nlp(text[:5000])  # limit for speed
    words = [t for t in doc if not t.is_punct and not t.is_space]
    n_words = max(len(words), 1)
    sentences = list(doc.sents)
    n_sents = max(len(sentences), 1)

    # 1. Contraction rate (AI rarely uses contractions)
    contractions = len(CONTRACTION_PATTERN.findall(text))
    contraction_rate = contractions / n_words

    # 2. First-person pronoun density (AI tends to avoid I/me/my)
    first_person = len(FIRST_PERSON.findall(text))
    fp_density = first_person / n_words

    # 3. Sentence length variance (AI is more uniform)
    sent_lens = [len([t for t in s if not t.is_punct]) for s in sentences]
    sent_len_cv = float(np.std(sent_lens) / max(np.mean(sent_lens), 1)) if sent_lens else 0

    # 4. Question rate (humans ask more rhetorical questions)
    questions = sum(1 for s in sentences if str(s).strip().endswith("?"))
    question_rate = questions / n_sents

    # 5. Past tense ratio (AI often defaults to present tense)
    past_tense = sum(1 for t in words if t.tag_ in ("VBD", "VBN"))
    past_ratio = past_tense / max(sum(1 for t in words if t.pos_ == "VERB"), 1)

    # 6. Adverb density (humans use more hedging adverbs)
    adverbs = sum(1 for t in words if t.pos_ == "ADV")
    adverb_density = adverbs / n_words

    # 7. Conjunction start rate (humans start sentences with "But", "And", "So")
    conj_starts = sum(1 for s in sentences
                      if len(list(s)) > 0 and list(s)[0].pos_ in ("CCONJ", "SCONJ"))
    conj_start_rate = conj_starts / n_sents

    # 8. Punctuation diversity (humans use more varied punctuation)
    punct_types = set(t.text for t in doc if t.is_punct)
    punct_diversity = len(punct_types) / max(n_sents, 1)

    return {
        "contraction_rate": contraction_rate,
        "fp_density": fp_density,
        "sent_len_cv": sent_len_cv,
        "question_rate": question_rate,
        "past_ratio": past_ratio,
        "adverb_density": adverb_density,
        "conj_start_rate": conj_start_rate,
        "punct_diversity": punct_diversity,
    }


# ── Feature names ──────────────────────────────────────────────────────

BASIC_NAMES = ["log_ppl", "top10", "mean_ent", "top1", "ent_std"]
DIVEYE_NAMES = ["s_mean", "s_std", "s_var", "s_skew", "s_kurt",
                "d1_mean", "d1_std", "d2_var", "d2_entropy", "d2_autocorr"]
LING_NAMES = ["contraction_rate", "fp_density", "sent_len_cv", "question_rate",
              "past_ratio", "adverb_density", "conj_start_rate", "punct_diversity"]
ALL_NAMES = BASIC_NAMES + DIVEYE_NAMES + ["spec_energy"] + LING_NAMES


def extract_all_features(token_data, text):
    basic = extract_basic_features(token_data)
    diveye = extract_diveye_features(token_data)
    spec = extract_specdetect_energy(token_data)
    ling = extract_linguistic_features(text)

    return ([basic[k] for k in BASIC_NAMES] +
            [diveye[k] for k in DIVEYE_NAMES] +
            [spec] +
            [ling[k] for k in LING_NAMES])


# ── Dataset loading ───────────────────────────────────────────────────

def load_balanced_samples(dataset_path, extra_sources, n_main, n_extra):
    human, ai = [], []

    with open(dataset_path) as f:
        for line in f:
            d = json.loads(line)
            lab = d["label"]
            if lab in (0, 3):
                human.append(d)
            elif lab in (1, 2):
                ai.append(d)

    print(f"  Main dataset: {len(human)} human, {len(ai)} AI")

    # Add extra diverse human texts
    extra_human = []
    for src in extra_sources:
        if os.path.exists(src):
            with open(src) as f:
                for line in f:
                    d = json.loads(line)
                    if len(d.get("text", "").split()) >= 80:
                        extra_human.append(d)
            print(f"  Extra: +{len(extra_human)} from {os.path.basename(src)}")

    random.seed(RANDOM_SEED)
    random.shuffle(human)
    random.shuffle(ai)
    random.shuffle(extra_human)

    n_per = n_main // 2
    n_extra_use = min(n_extra, len(extra_human))

    samples = human[:n_per] + extra_human[:n_extra_use] + ai[:n_per + n_extra_use]
    random.shuffle(samples)

    n_human_total = min(n_per, len(human)) + n_extra_use
    n_ai_total = min(n_per + n_extra_use, len(ai))
    print(f"  Sampled: {n_human_total} human ({n_per} main + {n_extra_use} extra), {n_ai_total} AI")
    return samples


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("LR v3 Training: 16 Perplexity + 8 Linguistic = 24 Features")
    print("=" * 80)

    # Load MLX model
    print("\nLoading MLX model...")
    t0 = time.time()
    import mlx.core as mx
    from mlx_lm import load
    model, tokenizer = load(MLX_MODEL_ID)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Load samples
    print(f"\nLoading samples (main: {N_SAMPLE}, extra human: {N_EXTRA_HUMAN})...")
    samples = load_balanced_samples(DATASET_PATH, EXTRA_HUMAN_SOURCES, N_SAMPLE, N_EXTRA_HUMAN)

    # Compute features
    print(f"\nComputing features for {len(samples)} samples...")
    X_all, y_all = [], []
    skipped = 0
    t_start = time.time()

    for i, sample in enumerate(samples):
        text = sample["text"]
        token_data = compute_token_data(text, model, tokenizer)
        if token_data is None or token_data["n"] < MIN_TOKENS:
            skipped += 1
            continue

        feats = extract_all_features(token_data, text)
        if any(not np.isfinite(f) for f in feats):
            skipped += 1
            continue

        X_all.append(feats)
        is_ai = 1 if sample["label"] in (1, 2) else 0
        y_all.append(is_ai)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1 - skipped) / elapsed
            eta = (len(samples) - i - 1) / max(rate, 0.1)
            print(f"  [{i+1}/{len(samples)}] {elapsed:.0f}s, {rate:.1f}/s, ETA {eta:.0f}s, skip {skipped}")

    X = np.array(X_all)
    y = np.array(y_all)
    print(f"\n  Valid: {len(X)} ({np.sum(y==0)} human, {np.sum(y==1)} AI), skipped {skipped}")
    print(f"  Features: {X.shape[1]} ({len(BASIC_NAMES)} basic + {len(DIVEYE_NAMES)} diveye + 1 spec + {len(LING_NAMES)} linguistic)")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )

    # Model A: v2 features only (16)
    print("\n" + "=" * 80)
    print("MODEL A: 16 v2 features + scaler (baseline)")
    print("=" * 80)
    v2_idx = list(range(16))
    pipe_a = Pipeline([("scaler", StandardScaler()),
                       ("lr", LogisticRegression(max_iter=5000, C=5.0, random_state=RANDOM_SEED))])
    pipe_a.fit(X_train[:, v2_idx], y_train)
    acc_a = accuracy_score(y_test, pipe_a.predict(X_test[:, v2_idx]))
    print(f"  Accuracy: {acc_a:.4f}")
    print(classification_report(y_test, pipe_a.predict(X_test[:, v2_idx]),
                                target_names=["human", "AI"]))

    # Model B: All 24 features
    print("=" * 80)
    print("MODEL B: 24 features (16 perplexity + 8 linguistic) + scaler")
    print("=" * 80)
    pipe_b = Pipeline([("scaler", StandardScaler()),
                       ("lr", LogisticRegression(max_iter=5000, C=5.0, random_state=RANDOM_SEED))])
    pipe_b.fit(X_train, y_train)
    acc_b = accuracy_score(y_test, pipe_b.predict(X_test))
    print(f"  Accuracy: {acc_b:.4f}")
    print(classification_report(y_test, pipe_b.predict(X_test),
                                target_names=["human", "AI"]))

    # Model C: Linguistic features only (8)
    print("=" * 80)
    print("MODEL C: 8 linguistic features only + scaler")
    print("=" * 80)
    ling_idx = list(range(16, 24))
    pipe_c = Pipeline([("scaler", StandardScaler()),
                       ("lr", LogisticRegression(max_iter=5000, C=5.0, random_state=RANDOM_SEED))])
    pipe_c.fit(X_train[:, ling_idx], y_train)
    acc_c = accuracy_score(y_test, pipe_c.predict(X_test[:, ling_idx]))
    print(f"  Accuracy: {acc_c:.4f}")
    print(classification_report(y_test, pipe_c.predict(X_test[:, ling_idx]),
                                target_names=["human", "AI"]))

    # Feature importance
    print("=" * 80)
    print("FEATURE IMPORTANCE (Model B)")
    print("=" * 80)
    lr_b = pipe_b.named_steps["lr"]
    sorted_feats = sorted(zip(ALL_NAMES, lr_b.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    for name, coef in sorted_feats:
        direction = "AI" if coef > 0 else "human"
        tag = "★" if name in LING_NAMES else " "
        print(f"  {tag} {name:<20s} {coef:>+10.4f}  higher → {direction}")

    # Summary
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"  v2 features (16):        {acc_a:.4f} ({acc_a*100:.1f}%)")
    print(f"  v3 features (24):        {acc_b:.4f} ({acc_b*100:.1f}%)  {(acc_b-acc_a)*100:+.1f}%")
    print(f"  Linguistic only (8):     {acc_c:.4f} ({acc_c*100:.1f}%)")

    # Save best model
    best = pipe_b if acc_b >= acc_a else pipe_a
    best_name = "v3 (24 features)" if acc_b >= acc_a else "v2 (16 features)"
    best_acc = max(acc_a, acc_b)
    best_features = ALL_NAMES if acc_b >= acc_a else ALL_NAMES[:16]

    print(f"\nSaving {best_name} to {NEW_MODEL_PATH}...")
    lr_data = {
        "model": best,
        "feature_names": best_features,
        "accuracy": float(best_acc),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "note": f"Pipeline(StandardScaler + LR), {len(best_features)} features",
    }
    with open(NEW_MODEL_PATH, "wb") as f:
        pickle.dump(lr_data, f)
    print(f"  Saved. Accuracy: {best_acc:.4f}")

    print("\nDONE")


if __name__ == "__main__":
    main()
