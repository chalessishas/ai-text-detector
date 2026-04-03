#!/usr/bin/env python3
"""Train XGBoost meta-learner for 4-signal fusion.

Collects DeBERTa, PPL, LR, and Stat scores from the running detection server,
then trains an XGBoost classifier to optimally combine them.

Usage:
    python3 scripts/train_xgboost_fusion.py

Requires:
    - Detection server running on port 5001
    - dataset_v4.jsonl (or dataset_v3.jsonl) in project root
    - xgboost, scikit-learn installed
"""
import json
import os
import pickle
import random
import sys
import urllib.request

import numpy as np

SERVER = "http://127.0.0.1:5001/analyze"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")


def analyze(text, retries=3):
    """Get detection signals from the running server with retry."""
    import time
    data = json.dumps({"text": text}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(SERVER, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def extract_features(result):
    """Extract the 7 features used by XGBoost fusion."""
    clf = result.get("classification", {})
    ppl_stats = result.get("perplexity_stats") or {}
    fused = result.get("fused", {})

    deb_ai = clf.get("ai_score", 50)
    ppl_val = ppl_stats.get("perplexity")
    top10 = ppl_stats.get("top10_pct")
    mean_ent = ppl_stats.get("mean_entropy")
    lr_ai = ppl_stats.get("lr_ai_probability", 50)

    if ppl_val is None:
        return None  # PPL not computed, skip

    # Reconstruct ppl_score from the fusion logic
    if ppl_val < 6 and top10 > 88:
        ppl_score = 95
    elif ppl_val < 8 and top10 > 85:
        ppl_score = 80
    elif ppl_val > 25 and top10 < 75:
        ppl_score = 10
    elif ppl_val > 18 and top10 < 80:
        ppl_score = 20
    elif ppl_val > 12:
        ppl_score = 35
    elif ppl_val < 10 and top10 > 82:
        ppl_score = 65
    else:
        ppl_score = 50

    # Extract stat_score from signal_source string
    fused = result.get("fused", {})
    signal_src = fused.get("signal_source", "")
    stat_score = 50
    if "stat=" in signal_src:
        try:
            stat_score = int(signal_src.split("stat=")[1].split(")")[0].split(",")[0])
        except (ValueError, IndexError):
            pass

    return [deb_ai, ppl_score, lr_ai, stat_score, ppl_val, top10, mean_ent]


def load_dataset(max_samples=500):
    """Load balanced samples from dataset files.

    Prioritizes OOD (out-of-domain) data to avoid DeBERTa overfitting.
    Falls back to main dataset if OOD not available.
    """
    samples = {"human": [], "ai": []}

    # Prefer OOD data (texts NOT in DeBERTa training set)
    ood_path = os.path.join(PROJECT_DIR, "data_ood_xgboost.jsonl")
    if os.path.exists(ood_path):
        print(f"Loading OOD data from data_ood_xgboost.jsonl...")
        with open(ood_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                label = obj.get("label")
                if len(text) < 100:
                    continue
                if label in (0, "human"):
                    samples["human"].append(text)
                elif label in (1, "ai"):
                    samples["ai"].append(text)
        if len(samples["human"]) >= 20 and len(samples["ai"]) >= 20:
            n = min(len(samples["human"]), len(samples["ai"]), max_samples // 2)
            random.seed(42)
            return random.sample(samples["human"], n), random.sample(samples["ai"], n)
        print(f"  OOD data insufficient ({len(samples['human'])} human, {len(samples['ai'])} AI), falling back...")
        samples = {"human": [], "ai": []}

    # Fallback: main dataset
    for name in ["dataset_v4.jsonl", "dataset_v3.jsonl", "dataset.jsonl"]:
        path = os.path.join(PROJECT_DIR, name)
        if os.path.exists(path):
            print(f"Loading from {name}...")
            with open(path) as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = obj.get("text", "")
                    label = obj.get("label")
                    label_name = obj.get("label_name", "")
                    if len(text) < 100:
                        continue
                    # Support both numeric (0/1/2/3) and string labels
                    is_human = label in (0, "human") or label_name in ("human", "human_polished")
                    is_ai = label in (1, 2, "ai", "ai_polished") or label_name in ("ai", "ai_polished")
                    if is_human:
                        samples["human"].append(text)
                    elif is_ai:
                        samples["ai"].append(text)
            break

    # Balance and sample
    n = min(len(samples["human"]), len(samples["ai"]), max_samples // 2)
    if n < 20:
        print(f"Not enough samples: {len(samples['human'])} human, {len(samples['ai'])} ai")
        sys.exit(1)

    random.seed(42)
    human_sample = random.sample(samples["human"], n)
    ai_sample = random.sample(samples["ai"], n)

    print(f"Selected {n} human + {n} AI = {2*n} samples")
    return human_sample, ai_sample


def main():
    print("=" * 60)
    print("XGBoost Fusion Meta-Learner Training")
    print("=" * 60)

    human_texts, ai_texts = load_dataset(max_samples=400)

    # Collect features from server
    X, y = [], []
    total = len(human_texts) + len(ai_texts)

    print(f"\nCollecting signals from server ({total} texts)...")
    for i, text in enumerate(human_texts):
        result = analyze(text[:5000])  # truncate to avoid timeout
        if result is None:
            continue
        features = extract_features(result)
        if features is None:
            continue
        X.append(features)
        y.append(0)  # human
        if (i + 1) % 20 == 0:
            print(f"  Human: {i+1}/{len(human_texts)}")

    for i, text in enumerate(ai_texts):
        result = analyze(text[:5000])
        if result is None:
            continue
        features = extract_features(result)
        if features is None:
            continue
        X.append(features)
        y.append(1)  # AI
        if (i + 1) % 20 == 0:
            print(f"  AI: {i+1}/{len(ai_texts)}")

    X = np.array(X)
    y = np.array(y)
    print(f"\nCollected {len(X)} valid samples ({sum(y==0)} human, {sum(y==1)} AI)")

    # Train classifier (try XGBoost, fall back to sklearn)
    from sklearn.model_selection import cross_val_score
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )
        print("Using XGBoost classifier")
    except (ImportError, Exception):
        print("XGBoost unavailable, using sklearn GradientBoosting...")
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )

    # Cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"\n5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"  Per-fold: {[f'{s:.3f}' for s in scores]}")

    # Train on full data
    model.fit(X, y)

    # Feature importance
    feature_names = ["deb_ai", "ppl_score", "lr_ai", "stat_score", "ppl_val", "top10", "mean_ent"]
    importances = model.feature_importances_
    print(f"\nFeature importances:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"  {name:15s}: {imp:.3f}")

    # Save model
    output_path = os.path.join(PROJECT_DIR, "models", "xgboost_fusion.pkl")
    with open(output_path, "wb") as f:
        pickle.dump({
            "model": model,
            "accuracy": round(scores.mean(), 3),
            "cv_std": round(scores.std(), 3),
            "feature_names": feature_names,
            "n_samples": len(X),
        }, f)
    print(f"\nModel saved to {output_path}")
    print(f"Restart detection server to load the new fusion model.")


if __name__ == "__main__":
    main()
