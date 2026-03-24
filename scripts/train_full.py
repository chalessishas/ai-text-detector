#!/usr/bin/env python3
"""Full retraining of DeBERTa on complete dataset (v2).

Unlike finetune_local.py (incremental), this trains from the base
DeBERTa-v3-base checkpoint on ALL data, avoiding catastrophic forgetting.

Supports:
  - Local Apple M4 MPS (slow but works, batch=1)
  - Colab GPU (fast, batch=16-64)

For binary mode: maps 4-class labels to binary (human=0,3 → 0; ai=1,2 → 1)

Run:
    python3 scripts/train_full.py --dataset dataset_v2.jsonl --binary
    python3 scripts/train_full.py --dataset dataset_v2_mini.jsonl --binary --epochs 1  # quick test
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
BASE_MODEL = "microsoft/deberta-v3-base"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "models", "detector_v3")


def load_dataset(path, binary=False):
    """Load JSONL dataset, optionally converting to binary labels."""
    data = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            text = entry.get("text", "").strip()
            label = entry.get("label", -1)
            if len(text) < 50 or label not in (0, 1, 2, 3):
                continue
            if binary:
                # human(0) + human_polished(3) → 0, ai(1) + ai_polished(2) → 1
                label = 0 if label in (0, 3) else 1
            data.append({"text": text, "label": label})
    return data


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_v2.jsonl")
    parser.add_argument("--binary", action="store_true", help="Binary mode (human vs AI)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=0, help="0=auto-detect")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--output", default=OUTPUT_DIR)
    args = parser.parse_args()

    dataset_path = os.path.join(PROJECT_DIR, args.dataset)
    if not os.path.exists(dataset_path):
        print(f"ERROR: {dataset_path} not found")
        sys.exit(1)

    num_labels = 2 if args.binary else 4
    label_names = ["human", "ai"] if args.binary else ["human", "ai", "ai_polished", "human_polished"]
    mode = "binary" if args.binary else "4-class"

    print(f"=== Full DeBERTa Retraining ({mode}) ===")
    print(f"Dataset: {dataset_path}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")

    # Load data
    data = load_dataset(dataset_path, binary=args.binary)
    print(f"Loaded {len(data)} entries")
    label_dist = Counter(d["label"] for d in data)
    for label, count in sorted(label_dist.items()):
        print(f"  {label_names[label]}: {count}")

    # Split
    train_data, eval_data = train_test_split(data, test_size=0.1, random_state=42, stratify=[d["label"] for d in data])
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_len)

    train_ds = Dataset.from_list(train_data).map(tokenize, batched=True, remove_columns=["text"])
    eval_ds = Dataset.from_list(eval_data).map(tokenize, batched=True, remove_columns=["text"])

    # Model - from base checkpoint (NOT from fine-tuned)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=num_labels)
    model.float()  # DeBERTa fp16 explodes on MPS

    # Auto-detect batch size
    if args.batch == 0:
        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory
            if mem > 70e9:  # A100-80GB
                args.batch = 64
            elif mem > 30e9:  # A100-40GB
                args.batch = 32
            else:  # T4 16GB
                args.batch = 16
        elif torch.backends.mps.is_available():
            args.batch = 2  # M4 MPS
        else:
            args.batch = 4  # CPU

    print(f"Batch size: {args.batch}")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Training
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=100,
        fp16=False,  # DeBERTa fp16 explodes
        bf16=torch.cuda.is_available(),  # bf16 only on CUDA
        warmup_ratio=0.1,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")

    # Evaluate
    results = trainer.evaluate()
    print(f"Final eval accuracy: {results['eval_accuracy']:.4f}")

    # Save
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Model saved to {args.output}")

    # Detailed report
    preds = trainer.predict(eval_ds)
    pred_labels = np.argmax(preds.predictions, axis=-1)
    true_labels = preds.label_ids
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=label_names))


if __name__ == "__main__":
    main()
