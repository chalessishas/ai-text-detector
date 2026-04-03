#!/usr/bin/env python3
"""Calibrate a perplexity-based AI detector using dataset_v4.jsonl.

Samples N texts per class from dataset_v4.jsonl, computes PPL/ENT/BURST/GLTR
via Qwen3.5-4B (MLX), then finds optimal thresholds via logistic regression.

Outputs:
  - Feature distributions per class (human vs AI)
  - Calibrated thresholds
  - Accuracy on held-out set
  - Evaluation of CoPA humanizer outputs

Run: /opt/anaconda3/bin/python3.13 scripts/calibrate_detector.py
"""

from __future__ import annotations

import json
import os
import random
import re
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

MODEL_ID = "mlx-community/Qwen3.5-4B-4bit"
DATASET = os.path.join(os.path.dirname(__file__), "..", "dataset_v4.jsonl")
N_PER_CLASS = 100  # samples per class for calibration (keep small for speed)


def compute_features(text: str, model, tokenizer) -> dict | None:
    """Compute PPL, entropy, burstiness, GLTR for a text."""
    tokens = mx.array(tokenizer.encode(text))
    if tokens.shape[0] < 10 or tokens.shape[0] > 2048:
        return None

    # Truncate to 512 tokens for speed
    if tokens.shape[0] > 512:
        tokens = tokens[:512]

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
    ent_mean = np.mean(ents)
    ent_std = np.std(ents)

    # Burstiness
    sents = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip().split()) > 2]
    if len(sents) > 1:
        lens = [len(s.split()) for s in sents]
        burst = np.std(lens) / max(np.mean(lens), 1)
    else:
        burst = 0.0

    # GLTR top-10 percentage
    t10 = 0
    for i in range(1, tokens.shape[0]):
        top10 = set(mx.argpartition(logits[i-1], kth=-10)[-10:].tolist())
        if tokens[i].item() in top10:
            t10 += 1
    gltr = t10 / max(tokens.shape[0]-1, 1) * 100

    return {
        "ppl": float(ppl),
        "ent_mean": float(ent_mean),
        "ent_std": float(ent_std),
        "burst": float(burst),
        "gltr": float(gltr),
    }


def load_samples(path: str, n_per_class: int) -> list[dict]:
    """Load balanced samples from dataset_v4.jsonl. Shuffles before capping."""
    by_label = {"human": [], "ai": [], "ai_polished": [], "human_polished": []}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            label = d["label_name"]
            if label in by_label:
                by_label[label].append(d)

    # Shuffle BEFORE capping to avoid deterministic first-N bias
    all_samples = []
    for label, samples in by_label.items():
        random.shuffle(samples)
        all_samples.extend(samples[:n_per_class])

    random.shuffle(all_samples)
    return all_samples


def main():
    print("=" * 80)
    print("Perplexity-Based Detector Calibration")
    print("=" * 80)

    # ── Load model ───────────────────────────────────────────────────
    print("\nLoading Qwen3.5-4B...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Load samples ─────────────────────────────────────────────────
    print(f"\nLoading {N_PER_CLASS} samples per class from dataset_v4.jsonl...")
    samples = load_samples(DATASET, N_PER_CLASS)
    print(f"  Loaded {len(samples)} samples")

    # ── Compute features ─────────────────────────────────────────────
    print("\nComputing features (this takes a while)...")
    features_list = []
    labels_list = []
    binary_labels = []  # 0=human, 1=AI

    for i, sample in enumerate(samples):
        feats = compute_features(sample["text"], model, tokenizer)
        if feats is None:
            continue

        features_list.append(feats)
        labels_list.append(sample["label_name"])
        # Binary: human/human_polished = 0, ai/ai_polished = 1
        is_ai = 1 if sample["label_name"] in ("ai", "ai_polished") else 0
        binary_labels.append(is_ai)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(samples)}] {elapsed:.0f}s elapsed")

    print(f"  Done. {len(features_list)} valid samples.")

    # ── Feature distributions ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FEATURE DISTRIBUTIONS")
    print("=" * 80)

    human_feats = [f for f, l in zip(features_list, binary_labels) if l == 0]
    ai_feats = [f for f, l in zip(features_list, binary_labels) if l == 1]

    for key in ["ppl", "ent_mean", "ent_std", "burst", "gltr"]:
        h_vals = [f[key] for f in human_feats]
        a_vals = [f[key] for f in ai_feats]
        print(f"\n  {key}:")
        print(f"    Human:  mean={np.mean(h_vals):7.2f}  std={np.std(h_vals):7.2f}  "
              f"median={np.median(h_vals):7.2f}  [p25={np.percentile(h_vals,25):6.2f}, p75={np.percentile(h_vals,75):6.2f}]")
        print(f"    AI:     mean={np.mean(a_vals):7.2f}  std={np.std(a_vals):7.2f}  "
              f"median={np.median(a_vals):7.2f}  [p25={np.percentile(a_vals,25):6.2f}, p75={np.percentile(a_vals,75):6.2f}]")

    # ── Train logistic regression ────────────────────────────────────
    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION CALIBRATION")
    print("=" * 80)

    X = np.array([[f["ppl"], f["ent_mean"], f["ent_std"], f["burst"], f["gltr"]]
                  for f in features_list])
    y = np.array(binary_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print(f"\n  Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["human", "AI"]))

    # Feature importance (logistic regression coefficients)
    feature_names = ["ppl", "ent_mean", "ent_std", "burst", "gltr"]
    print("  Feature coefficients (positive = AI-like):")
    for name, coef in sorted(zip(feature_names, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        direction = "AI" if coef > 0 else "human"
        print(f"    {name:12s}: {coef:+.4f}  (higher → {direction})")
    print(f"    {'intercept':12s}: {clf.intercept_[0]:+.4f}")

    # ── Simple threshold-based detector ──────────────────────────────
    print("\n" + "=" * 80)
    print("THRESHOLD-BASED DETECTOR (for comparison)")
    print("=" * 80)

    # Find optimal single-feature thresholds
    for key_idx, key in enumerate(feature_names):
        vals = X[:, key_idx]
        best_acc = 0
        best_thresh = 0
        best_dir = ">"
        for thresh in np.percentile(vals, range(5, 96, 5)):
            # Try both directions
            pred_gt = (vals > thresh).astype(int)
            pred_lt = (vals < thresh).astype(int)
            acc_gt = accuracy_score(y, pred_gt)
            acc_lt = accuracy_score(y, pred_lt)
            if acc_gt > best_acc:
                best_acc = acc_gt
                best_thresh = thresh
                best_dir = ">"
            if acc_lt > best_acc:
                best_acc = acc_lt
                best_thresh = thresh
                best_dir = "<"
        print(f"  {key:12s}: AI if {best_dir} {best_thresh:.2f}  (accuracy: {best_acc:.3f})")

    # ── Save calibrated model ────────────────────────────────────────
    calibration = {
        "feature_names": feature_names,
        "coefficients": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "feature_distributions": {
            "human": {k: {"mean": float(np.mean([f[k] for f in human_feats])),
                          "std": float(np.std([f[k] for f in human_feats])),
                          "p25": float(np.percentile([f[k] for f in human_feats], 25)),
                          "p75": float(np.percentile([f[k] for f in human_feats], 75))}
                     for k in feature_names},
            "AI": {k: {"mean": float(np.mean([f[k] for f in ai_feats])),
                        "std": float(np.std([f[k] for f in ai_feats])),
                        "p25": float(np.percentile([f[k] for f in ai_feats], 25)),
                        "p75": float(np.percentile([f[k] for f in ai_feats], 75))}
                   for k in feature_names},
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "models", "calibration.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"\n  Calibration saved to {out_path}")

    # ── Evaluate CoPA outputs ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COPA HUMANIZER EVALUATION")
    print("=" * 80)

    copa_texts = {
        "copa_academic_1": (
            "AI changed edtech big time. ML algos = personalized learning "
            "(tailored to individual needs). Also: adaptive assessments give "
            "real-time feedback to teachers & students. Result? Total paradigm "
            "shift in pedagogy. Big one."
        ),
        "copa_academic_2": (
            "AI changed everything. Machine learning takes it even further. "
            "Platforms now crank out personalized lessons tailored just for your "
            "kid. No more one-size-fits-all drills. New adaptive tools? Real-time "
            "feedback for teachers and students."
        ),
        "copa_blog_1": (
            "Key points: Climate change is a top threat. Rising temps lead to bad "
            "weather including hurricanes, drought, wildfires. Scientists say we "
            "need action now or it's catastrophic and irreversible. Must act: "
            "governments, corporations, and individuals. Goals: cut carbon, "
            "switch to renewables."
        ),
        "copa_technical_1": (
            "Transformers flipped natural language processing. They kill the slow "
            "sequential slowness of old RNNs. Entire sequences process in parallel. "
            "Training skyrockets in speed. That's the multi-head attention magic. "
            "You can track multiple spots at once."
        ),
        # Original AI texts for comparison
        "original_ai": (
            "Artificial intelligence has fundamentally transformed the landscape "
            "of modern education. Furthermore, the integration of machine learning "
            "algorithms into educational platforms has enabled personalized learning "
            "experiences that cater to individual student needs."
        ),
    }

    for name, text in copa_texts.items():
        feats = compute_features(text, model, tokenizer)
        if feats is None:
            print(f"  {name}: too short/long to analyze")
            continue

        x = np.array([[feats["ppl"], feats["ent_mean"], feats["ent_std"],
                        feats["burst"], feats["gltr"]]])
        prob_ai = clf.predict_proba(x)[0][1]
        label = "AI" if prob_ai > 0.5 else "HUMAN"

        print(f"\n  {name}:")
        print(f"    PPL={feats['ppl']:5.1f}  ENT={feats['ent_mean']:.2f}  "
              f"ENT_STD={feats['ent_std']:.2f}  BURST={feats['burst']:.2f}  GLTR={feats['gltr']:.0f}%")
        print(f"    Verdict: {label} (AI probability: {prob_ai:.1%})")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
