#!/usr/bin/env python3
"""Benchmark LR and DeBERTa models on held-out + adversarial data.

Runs offline (no server needed). Compares old vs new models.

Usage:
  python3 scripts/benchmark_models.py                    # benchmark LR only (fast)
  python3 scripts/benchmark_models.py --deberta           # + DeBERTa benchmark
  python3 scripts/benchmark_models.py --adversarial       # + adversarial eval
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

PROJECT_DIR = Path(__file__).parent.parent
SEED = 42


def benchmark_lr(dataset_path, model_path, n_eval=500):
    """Benchmark LR model on held-out samples using cached features."""
    print("=" * 70)
    print(f"LR BENCHMARK: {model_path}")
    print("=" * 70)

    # Load model
    with open(model_path, "rb") as f:
        lr_data = pickle.load(f)

    model = lr_data["model"]
    feature_names = lr_data.get("feature_names", [])
    train_acc = lr_data.get("accuracy", 0)
    trained_at = lr_data.get("trained_at", "unknown")

    print(f"  Features: {len(feature_names)}")
    print(f"  Training accuracy: {train_acc:.4f}")
    print(f"  Trained at: {trained_at}")
    print(f"  Note: {lr_data.get('note', 'N/A')}")

    # Load eval data (samples not used in training)
    random.seed(SEED + 999)  # different seed from training to get different samples
    human, ai = [], []
    with open(dataset_path) as f:
        for line in f:
            d = json.loads(line)
            if d["label"] in (0, 3):
                human.append(d["text"])
            elif d["label"] in (1, 2):
                ai.append(d["text"])

    random.shuffle(human)
    random.shuffle(ai)
    # Use samples from the END of the shuffled list (training uses from the START)
    n_per = n_eval // 2
    eval_texts = human[-n_per:] + ai[-n_per:]
    eval_labels = [0] * n_per + [1] * n_per
    print(f"\n  Eval set: {len(eval_texts)} samples ({n_per} human, {n_per} AI)")

    return model, feature_names, eval_texts, eval_labels


def benchmark_deberta(model_dir, dataset_path, n_eval=500):
    """Benchmark DeBERTa model on held-out samples."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("\n" + "=" * 70)
    print(f"DeBERTa BENCHMARK: {model_dir}")
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.float()
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_labels = model.config.num_labels
    print(f"  {n_params:,} params, {n_labels} classes")

    # Load eval data
    random.seed(SEED + 999)
    by_label = defaultdict(list)
    with open(dataset_path) as f:
        for line in f:
            d = json.loads(line)
            by_label[d["label"]].append(d["text"])

    for label in by_label:
        random.shuffle(by_label[label])

    n_per = n_eval // 4
    eval_data = []
    for label in sorted(by_label):
        for text in by_label[label][-n_per:]:
            eval_data.append((text, label))

    random.shuffle(eval_data)
    eval_texts = [t for t, _ in eval_data]
    eval_labels_4class = [l for _, l in eval_data]
    eval_labels_binary = [1 if l in (1, 2) else 0 for l in eval_labels_4class]
    print(f"  Eval set: {len(eval_texts)} samples")

    # Run inference
    print("  Running inference...")
    y_pred_4class = []
    y_probs_ai = []
    t0 = time.time()

    for i, text in enumerate(eval_texts):
        inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()[0]
        y_pred_4class.append(int(probs.argmax()))
        y_probs_ai.append(float(probs[1] + probs[2]))

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{len(eval_texts)}] {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.0f}s ({len(eval_texts)/elapsed:.1f} samples/s)")

    # 4-class metrics
    label_names = ["human", "ai", "ai_polished", "human_polished"]
    acc_4class = accuracy_score(eval_labels_4class, y_pred_4class)
    print(f"\n  4-class accuracy: {acc_4class:.4f}")
    print(classification_report(
        eval_labels_4class, y_pred_4class,
        target_names=label_names, digits=3, zero_division=0,
    ))

    # Binary metrics
    y_pred_binary = [1 if p in (1, 2) else 0 for p in y_pred_4class]
    acc_binary = accuracy_score(eval_labels_binary, y_pred_binary)
    print(f"  Binary detection accuracy: {acc_binary:.4f}")
    try:
        auc = roc_auc_score(eval_labels_binary, y_probs_ai)
        print(f"  Binary AUROC: {auc:.4f}")
    except ValueError:
        print("  Binary AUROC: N/A")

    print(classification_report(
        eval_labels_binary, y_pred_binary,
        target_names=["human", "AI"], digits=3, zero_division=0,
    ))

    return acc_4class, acc_binary


def benchmark_adversarial(model_dir, adversarial_path, n_eval=500):
    """Test DeBERTa robustness against adversarial attacks."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("\n" + "=" * 70)
    print("ADVERSARIAL ROBUSTNESS TEST")
    print("=" * 70)

    if not os.path.exists(adversarial_path):
        print(f"  No adversarial data at {adversarial_path}")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.float()
    model.to(device)
    model.eval()

    # Load adversarial samples
    by_attack = defaultdict(list)
    with open(adversarial_path) as f:
        for line in f:
            d = json.loads(line)
            by_attack[d.get("attack", "unknown")].append(d["text"])

    # Sample per attack type
    n_per_attack = min(n_eval // len(by_attack), 50)
    results = {}

    for attack_name in sorted(by_attack):
        texts = by_attack[attack_name]
        random.shuffle(texts)
        sample = texts[:n_per_attack]

        correct = 0
        for text in sample:
            inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()[0]
            ai_prob = probs[1] + probs[2]
            if ai_prob > 0.5:
                correct += 1

        detection_rate = correct / len(sample) * 100
        results[attack_name] = detection_rate
        status = "OK" if detection_rate >= 70 else "WEAK" if detection_rate >= 40 else "FAIL"
        print(f"  {attack_name:20s}: {detection_rate:5.1f}% detected ({correct}/{len(sample)})  [{status}]")

    avg = sum(results.values()) / len(results)
    print(f"\n  Average detection rate: {avg:.1f}%")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deberta", action="store_true", help="Include DeBERTa benchmark")
    parser.add_argument("--adversarial", action="store_true", help="Include adversarial eval")
    parser.add_argument("--n-eval", type=int, default=500)
    args = parser.parse_args()

    dataset = PROJECT_DIR / "dataset_v4.jsonl"
    if not dataset.exists():
        dataset = PROJECT_DIR / "dataset_v3.jsonl"

    # Benchmark LR models
    lr_models = [
        ("LR v1 (5 basic)", PROJECT_DIR / "models" / "perplexity_lr.pkl"),
        ("LR v2 (16 features)", PROJECT_DIR / "models" / "perplexity_lr_v2.pkl"),
    ]

    for name, path in lr_models:
        if path.exists():
            print(f"\n{'#' * 70}")
            print(f"# {name}")
            print(f"{'#' * 70}")
            model, features, eval_texts, eval_labels = benchmark_lr(
                str(dataset), str(path), args.n_eval,
            )
            print(f"  (LR eval requires MLX feature extraction — use train_lr_v2.py output for eval)")

    if args.deberta:
        # Benchmark DeBERTa models
        deberta_dirs = [
            ("DeBERTa v1 (original)", PROJECT_DIR / "models" / "detector"),
            ("DeBERTa v3 (fine-tuned v4)", PROJECT_DIR / "models" / "detector_v3"),
        ]
        for name, model_dir in deberta_dirs:
            if model_dir.exists() and (model_dir / "model.safetensors").exists():
                print(f"\n{'#' * 70}")
                print(f"# {name}")
                print(f"{'#' * 70}")
                benchmark_deberta(str(model_dir), str(dataset), args.n_eval)

    if args.adversarial:
        adv_path = PROJECT_DIR / "dataset_adversarial_v4.jsonl"
        deberta_dirs = [
            ("DeBERTa v1 (original)", PROJECT_DIR / "models" / "detector"),
            ("DeBERTa v3 (fine-tuned v4)", PROJECT_DIR / "models" / "detector_v3"),
        ]
        for name, model_dir in deberta_dirs:
            if model_dir.exists() and (model_dir / "model.safetensors").exists():
                print(f"\n{'#' * 70}")
                print(f"# ADVERSARIAL: {name}")
                print(f"{'#' * 70}")
                benchmark_adversarial(str(model_dir), str(adv_path), args.n_eval)


if __name__ == "__main__":
    main()
