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

# ── Config ────────────────────────────────────────────────────────────────
MLX_MODEL_ID = "mlx-community/Qwen3.5-4B-4bit"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DATASET_PATH = os.path.join(PROJECT_DIR, "dataset_v3.jsonl")
OLD_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "perplexity_lr.pkl")
NEW_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "perplexity_lr_v2.pkl")

N_SAMPLE = 5000       # total samples (balanced across classes)
MAX_TOKENS = 512      # truncate for speed
MIN_TOKENS = 10
RANDOM_SEED = 42


# ── Feature computation ───────────────────────────────────────────────────

def compute_token_data(text, model, tokenizer):
    """Get token-level logprobs, ranks, entropies via MLX."""
    import mlx.core as mx

    tokens = tokenizer.encode(text)
    if len(tokens) < MIN_TOKENS or len(tokens) > 2048:
        return None
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]

    x = mx.array([tokens])
    logits = model(x)
    probs_mx = mx.softmax(logits[0], axis=-1)
    mx.eval(probs_mx)
    probs_all = np.array(probs_mx.astype(mx.float32))

    results = []
    for i in range(1, len(tokens)):
        actual_id = tokens[i]
        actual_prob = float(probs_all[i - 1, actual_id])
        logprob = math.log(max(actual_prob, 1e-20))
        rank = int(np.sum(probs_all[i - 1] > actual_prob)) + 1

        p = probs_all[i - 1]
        valid = p > 1e-10
        entropy = float(-np.sum(p[valid] * np.log(p[valid])))

        results.append({"logprob": logprob, "rank": rank, "entropy": entropy})

    return results


def extract_basic_features(token_data):
    """5 basic features: log_ppl, top10, mean_ent, top1, ent_std."""
    logprobs = [t["logprob"] for t in token_data]
    ranks = [t["rank"] for t in token_data]
    entropies = [t["entropy"] for t in token_data]

    ppl = math.exp(-sum(logprobs) / len(logprobs))
    return {
        "log_ppl": math.log(max(ppl, 1e-5)),
        "top10": sum(1 for r in ranks if r <= 10) / len(ranks) * 100,
        "mean_ent": sum(entropies) / len(entropies),
        "top1": sum(1 for r in ranks if r == 1) / len(ranks) * 100,
        "ent_std": float(np.std(entropies)),
    }


def extract_diveye_features(token_data):
    """10 DivEye surprisal diversity features."""
    logprobs = [t["logprob"] for t in token_data]
    surprisal = np.array([-lp for lp in logprobs])

    if len(surprisal) < 5:
        return {k: 0.0 for k in [
            "s_mean", "s_std", "s_var", "s_skew", "s_kurt",
            "d1_mean", "d1_std", "d2_var", "d2_entropy", "d2_autocorr"
        ]}

    diff1 = np.diff(surprisal)
    diff2 = np.diff(diff1) if len(diff1) > 1 else np.array([0.0])

    # Entropy of second-order distribution
    if len(diff2) > 5:
        hist, _ = np.histogram(diff2, bins=20, density=True)
        hist = hist[hist > 0]
        d2_entropy = float(-np.sum(hist * np.log(hist + 1e-10)))
    else:
        d2_entropy = 0.0

    # Autocorrelation of second-order
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


def extract_specdetect_energy(token_data):
    """SpecDetect DFT total energy."""
    logprobs = [t["logprob"] for t in token_data]
    if len(logprobs) < 10:
        return 0.0
    surprisal = np.array([-lp for lp in logprobs])
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
        if token_data is None or len(token_data) < MIN_TOKENS:
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

    # ── Train/test split ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )

    # ── Model A: Basic 5 features only (baseline) ────────────────────
    print("\n" + "=" * 80)
    print("MODEL A: Basic 5 Features (baseline)")
    print("=" * 80)

    basic_idx = list(range(5))
    clf_basic = LogisticRegression(max_iter=2000, C=5.0, random_state=RANDOM_SEED)
    clf_basic.fit(X_train[:, basic_idx], y_train)
    y_pred_basic = clf_basic.predict(X_test[:, basic_idx])
    acc_basic = accuracy_score(y_test, y_pred_basic)

    print(f"  Accuracy: {acc_basic:.4f}")
    print(classification_report(y_test, y_pred_basic, target_names=["human", "AI"]))

    # ── Model B: All 16 features ──────────────────────────────────────
    print("=" * 80)
    print("MODEL B: All 16 Features (Basic + DivEye + SpecDetect)")
    print("=" * 80)

    clf_full = LogisticRegression(max_iter=2000, C=5.0, random_state=RANDOM_SEED)
    clf_full.fit(X_train, y_train)
    y_pred_full = clf_full.predict(X_test)
    acc_full = accuracy_score(y_test, y_pred_full)

    print(f"  Accuracy: {acc_full:.4f}")
    print(classification_report(y_test, y_pred_full, target_names=["human", "AI"]))

    # ── Feature importance ────────────────────────────────────────────
    print("=" * 80)
    print("FEATURE IMPORTANCE (Model B coefficients)")
    print("=" * 80)

    print(f"\n  {'Feature':<20s} {'Coeff':>10s}  {'|Coeff|':>8s}  Direction")
    print(f"  {'-'*20} {'-'*10}  {'-'*8}  {'-'*10}")
    sorted_feats = sorted(
        zip(ALL_NAMES, clf_full.coef_[0]),
        key=lambda x: abs(x[1]), reverse=True
    )
    for name, coef in sorted_feats:
        direction = "AI" if coef > 0 else "human"
        print(f"  {name:<20s} {coef:>+10.4f}  {abs(coef):>8.4f}  higher -> {direction}")
    print(f"  {'intercept':<20s} {clf_full.intercept_[0]:>+10.4f}")

    # ── Comparison summary ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    delta = acc_full - acc_basic
    print(f"\n  Model A (5 basic):     {acc_basic:.4f}  ({acc_basic*100:.1f}%)")
    print(f"  Model B (16 full):     {acc_full:.4f}  ({acc_full*100:.1f}%)")
    print(f"  Delta:                 {delta:+.4f}  ({delta*100:+.1f}pp)")

    if delta > 0:
        print(f"\n  Model B is better by {delta*100:.1f} percentage points")
    elif delta == 0:
        print(f"\n  Models are tied")
    else:
        print(f"\n  Model A is better (DivEye features may hurt -- possible overfitting?)")

    # ── Save new model ────────────────────────────────────────────────
    print(f"\nSaving Model B to {NEW_MODEL_PATH}...")
    os.makedirs(os.path.dirname(NEW_MODEL_PATH), exist_ok=True)

    lr_data = {
        "model": clf_full,
        "feature_names": ALL_NAMES,
        "accuracy": float(acc_full),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "baseline_accuracy": float(acc_basic),
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open(NEW_MODEL_PATH, "wb") as f:
        pickle.dump(lr_data, f)
    print(f"  Saved. Feature count: {len(ALL_NAMES)}, accuracy: {acc_full:.4f}")

    # Also save basic model as reference
    basic_path = NEW_MODEL_PATH.replace("_v2", "_v2_basic5")
    lr_basic_data = {
        "model": clf_basic,
        "feature_names": BASIC_NAMES,
        "accuracy": float(acc_basic),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open(basic_path, "wb") as f:
        pickle.dump(lr_basic_data, f)
    print(f"  Also saved baseline to {basic_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
