#!/usr/bin/env python3
"""Train LR v2 with 5 basic + 10 DivEye + 1 SpecDetect = 16 features.

Loads dataset_v3.jsonl, samples ~5000 texts, computes all features via MLX
qwen3.5:4b, trains LogisticRegression, compares with old 5-feature model.

Run: /opt/anaconda3/bin/python3.13 scripts/train_lr_v2.py
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
import sys
import time

# Force unbuffered stdout for background execution
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Config ────────────────────────────────────────────────────────────────
MLX_MODEL_ID = "mlx-community/Qwen3.5-4B-4bit"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DATASET_PATH = os.path.join(PROJECT_DIR, "dataset_v3.jsonl")
OLD_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "perplexity_lr.pkl")
NEW_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "perplexity_lr_v2.pkl")

N_SAMPLE = int(os.environ.get("LR_N_SAMPLE", "2000"))
MAX_TOKENS = 128      # truncate for speed (128 tokens ~ 1 sample/s on M-series)
MIN_TOKENS = 10
RANDOM_SEED = 42


# ── Feature computation ───────────────────────────────────────────────────

def compute_token_data(text, model, tokenizer):
    """Get token-level logprobs, top1/top10 flags, entropies via MLX.

    Returns dict with numpy arrays instead of list-of-dicts for speed.
    Avoids expensive full-rank computation -- only checks top1/top10.
    """
    import mlx.core as mx

    tokens = tokenizer.encode(text)
    if len(tokens) < MIN_TOKENS or len(tokens) > 2048:
        return None
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]

    x = mx.array([tokens])
    logits = model(x)[0]  # (seq_len, vocab_size)

    # Positions 0..n-2 predict tokens 1..n-1
    pred_logits = logits[:-1]  # (n-1, vocab)
    target_ids = mx.array(tokens[1:])  # (n-1,)
    n = target_ids.shape[0]

    # Log-softmax (numerically stable)
    log_probs = pred_logits - mx.logsumexp(pred_logits, axis=-1, keepdims=True)

    # Extract logprob for each actual next token
    target_logprobs = log_probs[mx.arange(n), target_ids]  # (n,)

    # Top-1: is actual token the argmax?
    top1_ids = mx.argmax(pred_logits, axis=-1)  # (n,)
    is_top1 = (top1_ids == target_ids)  # (n,) bool

    # Top-10: check if actual token is among top 10
    # Use argpartition to get top-10 indices efficiently
    top10_indices = mx.argpartition(pred_logits, kth=-10, axis=-1)[:, -10:]  # (n, 10)
    is_top10 = mx.any(top10_indices == target_ids[:, None], axis=-1)  # (n,) bool

    # Entropy: -sum(p * log(p))
    probs = mx.exp(log_probs)
    safe_log_probs = mx.where(probs > 1e-10, log_probs, mx.zeros_like(log_probs))
    safe_probs = mx.where(probs > 1e-10, probs, mx.zeros_like(probs))
    entropies = -mx.sum(safe_probs * safe_log_probs, axis=-1)  # (n,)

    # Force evaluation and convert
    mx.eval(target_logprobs, is_top1, is_top10, entropies)

    return {
        "logprobs": np.array(target_logprobs.astype(mx.float32)),
        "is_top1": np.array(is_top1),
        "is_top10": np.array(is_top10),
        "entropies": np.array(entropies.astype(mx.float32)),
        "n": n,
    }


def extract_basic_features(td):
    """5 basic features from vectorized token data dict."""
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
    """10 DivEye surprisal diversity features."""
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
    """SpecDetect DFT total energy."""
    lp = td["logprobs"]
    if len(lp) < 10:
        return 0.0
    surprisal = -lp
    surprisal = surprisal - np.mean(surprisal)
    fft = np.fft.rfft(surprisal)
    return float(np.sum(np.abs(fft) ** 2) / len(surprisal))


BASIC_NAMES = ["log_ppl", "top10", "mean_ent", "top1", "ent_std"]
DIVEYE_NAMES = ["s_mean", "s_std", "s_var", "s_skew", "s_kurt",
                "d1_mean", "d1_std", "d2_var", "d2_entropy", "d2_autocorr"]
ALL_NAMES = BASIC_NAMES + DIVEYE_NAMES + ["spec_energy"]


def extract_all_features(token_data):
    """Return flat list of all 16 features."""
    basic = extract_basic_features(token_data)
    diveye = extract_diveye_features(token_data)
    spec = extract_specdetect_energy(token_data)

    return [basic[k] for k in BASIC_NAMES] + \
           [diveye[k] for k in DIVEYE_NAMES] + \
           [spec]


# ── Dataset loading ───────────────────────────────────────────────────────

def load_balanced_samples(path, n_total):
    """Load balanced human/AI samples from dataset_v3.jsonl."""
    human = []  # label 0, 3
    ai = []     # label 1, 2

    with open(path) as f:
        for line in f:
            d = json.loads(line)
            lab = d["label"]
            if lab in (0, 3):
                human.append(d)
            elif lab in (1, 2):
                ai.append(d)

    random.seed(RANDOM_SEED)
    random.shuffle(human)
    random.shuffle(ai)

    n_per = n_total // 2
    samples = human[:n_per] + ai[:n_per]
    random.shuffle(samples)

    print(f"  Loaded {len(human)} human, {len(ai)} AI total")
    print(f"  Sampled {min(n_per, len(human))} human + {min(n_per, len(ai))} AI = {len(samples)}")
    return samples


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("LR v2 Training: 5 Basic + 10 DivEye + 1 SpecDetect = 16 Features")
    print("=" * 80)

    # Check for cached features (skip expensive MLX extraction)
    cache_path = os.path.join(PROJECT_DIR, "models", "lr_v2_features.npz")
    if os.path.exists(cache_path) and "--recompute" not in sys.argv:
        print(f"\nLoading cached features from {cache_path}...")
        data = np.load(cache_path)
        X = data["X"]
        y = data["y"]
        print(f"  Loaded {len(X)} samples, {X.shape[1]} features")
        print(f"  (Use --recompute to force re-extraction)")
    else:
        # Load MLX model
        print("\nLoading MLX model...")
        t0 = time.time()
        import mlx.core as mx
        from mlx_lm import load
        model, tokenizer = load(MLX_MODEL_ID)
        print(f"  Loaded in {time.time() - t0:.1f}s")

        # Load samples
        print(f"\nLoading balanced samples from dataset_v3.jsonl (target: {N_SAMPLE})...")
        samples = load_balanced_samples(DATASET_PATH, N_SAMPLE)

        # Compute features
        print(f"\nComputing features for {len(samples)} samples...")
        X_all = []
        y_all = []
        skipped = 0
        t_start = time.time()

        for i, sample in enumerate(samples):
            token_data = compute_token_data(sample["text"], model, tokenizer)
            if token_data is None or token_data["n"] < MIN_TOKENS:
                skipped += 1
                continue

            feats = extract_all_features(token_data)

            # Sanity: skip if any NaN/inf
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
                print(f"  [{i+1}/{len(samples)}] {elapsed:.0f}s elapsed, "
                      f"{rate:.1f} samples/s, ETA {eta:.0f}s, skipped {skipped}")

        X = np.array(X_all)
        y = np.array(y_all)
        print(f"\n  Valid samples: {len(X)} (skipped {skipped})")
        print(f"  Class distribution: {np.sum(y==0)} human, {np.sum(y==1)} AI")
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Total compute time: {time.time() - t_start:.0f}s")

        # Cache features to disk for fast retraining
        np.savez(cache_path, X=X, y=y)
        print(f"  Cached features to {cache_path}")

    # ── Train/test split ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )

    print(f"  Class distribution: {np.sum(y==0)} human, {np.sum(y==1)} AI")

    # ── Model A: Basic 5 features, no scaling (baseline) ─────────────
    print("\n" + "=" * 80)
    print("MODEL A: Basic 5 Features, no scaling (baseline)")
    print("=" * 80)

    basic_idx = list(range(5))
    clf_a = LogisticRegression(max_iter=5000, C=5.0, random_state=RANDOM_SEED)
    clf_a.fit(X_train[:, basic_idx], y_train)
    y_pred_a = clf_a.predict(X_test[:, basic_idx])
    acc_a = accuracy_score(y_test, y_pred_a)
    print(f"  Accuracy: {acc_a:.4f}")
    print(classification_report(y_test, y_pred_a, target_names=["human", "AI"]))

    # ── Model B: All 16 features, no scaling ──────────────────────────
    print("=" * 80)
    print("MODEL B: All 16 Features, no scaling")
    print("=" * 80)

    clf_b = LogisticRegression(max_iter=5000, C=5.0, random_state=RANDOM_SEED)
    clf_b.fit(X_train, y_train)
    y_pred_b = clf_b.predict(X_test)
    acc_b = accuracy_score(y_test, y_pred_b)
    print(f"  Accuracy: {acc_b:.4f}")
    print(classification_report(y_test, y_pred_b, target_names=["human", "AI"]))

    # ── Model C: All 16 features + StandardScaler (best) ─────────────
    print("=" * 80)
    print("MODEL C: All 16 Features + StandardScaler")
    print("=" * 80)

    pipe_c = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000, C=5.0, random_state=RANDOM_SEED)),
    ])
    pipe_c.fit(X_train, y_train)
    y_pred_c = pipe_c.predict(X_test)
    acc_c = accuracy_score(y_test, y_pred_c)
    print(f"  Accuracy: {acc_c:.4f}")
    print(classification_report(y_test, y_pred_c, target_names=["human", "AI"]))

    # ── Model D: Basic 5 + StandardScaler ─────────────────────────────
    print("=" * 80)
    print("MODEL D: Basic 5 Features + StandardScaler")
    print("=" * 80)

    pipe_d = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000, C=5.0, random_state=RANDOM_SEED)),
    ])
    pipe_d.fit(X_train[:, basic_idx], y_train)
    y_pred_d = pipe_d.predict(X_test[:, basic_idx])
    acc_d = accuracy_score(y_test, y_pred_d)
    print(f"  Accuracy: {acc_d:.4f}")
    print(classification_report(y_test, y_pred_d, target_names=["human", "AI"]))

    # ── Feature importance (Model C, scaled coefficients) ─────────────
    print("=" * 80)
    print("FEATURE IMPORTANCE (Model C, scaled coefs)")
    print("=" * 80)

    lr_c = pipe_c.named_steps["lr"]
    print(f"\n  {'Feature':<20s} {'Coeff':>10s}  {'|Coeff|':>8s}  Direction")
    print(f"  {'-'*20} {'-'*10}  {'-'*8}  {'-'*10}")
    sorted_feats = sorted(
        zip(ALL_NAMES, lr_c.coef_[0]),
        key=lambda x: abs(x[1]), reverse=True
    )
    for name, coef in sorted_feats:
        direction = "AI" if coef > 0 else "human"
        print(f"  {name:<20s} {coef:>+10.4f}  {abs(coef):>8.4f}  higher -> {direction}")
    print(f"  {'intercept':<20s} {lr_c.intercept_[0]:>+10.4f}")

    # ── Comparison summary ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    results = [
        ("A", "5 basic, no scale", acc_a),
        ("B", "16 full, no scale", acc_b),
        ("C", "16 full + scaler", acc_c),
        ("D", "5 basic + scaler", acc_d),
    ]
    for tag, desc, acc in sorted(results, key=lambda x: x[2], reverse=True):
        delta = acc - acc_a
        print(f"  Model {tag} ({desc:25s}):  {acc:.4f}  ({acc*100:.1f}%)  {delta:+.1%}")

    best = max(results, key=lambda x: x[2])
    print(f"\n  Best: Model {best[0]} at {best[2]*100:.1f}%")

    # ── Save best model (pipeline with scaler) ────────────────────────
    print(f"\nSaving best model to {NEW_MODEL_PATH}...")
    os.makedirs(os.path.dirname(NEW_MODEL_PATH), exist_ok=True)

    # Use Model C (16 features + scaler) as the v2 model
    lr_data = {
        "model": pipe_c,  # Pipeline includes scaler + LR
        "feature_names": ALL_NAMES,
        "accuracy": float(acc_c),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "baseline_accuracy": float(acc_a),
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "note": "Pipeline(StandardScaler + LogisticRegression)",
    }
    with open(NEW_MODEL_PATH, "wb") as f:
        pickle.dump(lr_data, f)
    print(f"  Saved. Feature count: {len(ALL_NAMES)}, accuracy: {acc_c:.4f}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
