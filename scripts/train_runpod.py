#!/usr/bin/env python3
"""Full DeBERTa retrain on RunPod 4090 GPU.

Trains from scratch on balanced dataset_v4.jsonl with 4-class labels.
Optimized for 4090: fp16, batch_size=32, gradient accumulation=2.

Expected: ~20-30 min on RTX 4090, cost ~$0.17.

Setup on RunPod pod:
  pip install transformers datasets accelerate scikit-learn
  # Upload dataset_v4.jsonl to /workspace/
  python3 train_runpod.py
"""

import json
import os
import random
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
    EarlyStoppingCallback,
)

SEED = 42
LABEL_NAMES = ["human", "ai", "ai_polished", "human_polished"]

# Paths (adjust for RunPod)
WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
DATASET = os.path.join(WORKSPACE, "dataset_v4.jsonl")
OUTPUT_DIR = os.path.join(WORKSPACE, "detector_v4")
BASE_MODEL = "microsoft/deberta-v3-base"

# Training config optimized for 4090 24GB
EPOCHS = 3
BATCH_SIZE = 8
GRAD_ACCUM = 8
LR = 2e-5
WARMUP_RATIO = 0.1
MAX_LEN = 512
N_SAMPLES = 0  # 0 = use all data


def load_dataset(path, n_samples=0):
    by_label = {0: [], 1: [], 2: [], 3: []}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            label = d["label"]
            text = d.get("text", "")
            if label in by_label and len(text.split()) >= 100:
                by_label[label].append({"text": text, "label": label})

    print("Raw counts:")
    for label in sorted(by_label):
        print(f"  {LABEL_NAMES[label]}: {len(by_label[label])}")

    # Balance
    min_count = min(len(v) for v in by_label.values())
    if n_samples > 0:
        per_class = min(n_samples // 4, min_count)
    else:
        per_class = min_count

    random.seed(SEED)
    samples = []
    for label in sorted(by_label):
        pool = by_label[label]
        random.shuffle(pool)
        samples.extend(pool[:per_class])
        print(f"  Selected {LABEL_NAMES[label]}: {min(per_class, len(pool))}")

    random.shuffle(samples)
    return samples


def compute_metrics(pred_output):
    logits, labels = pred_output
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def main():
    print("=" * 70)
    print("DeBERTa Full Retrain on RunPod 4090")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("WARNING: No GPU detected, running on CPU (will be very slow)")

    # Load data
    print(f"\nLoading dataset from {DATASET}...")
    data = load_dataset(DATASET, N_SAMPLES)
    print(f"Total: {len(data)} samples")

    train_data, val_data = train_test_split(
        data, test_size=0.15, random_state=SEED,
        stratify=[d["label"] for d in data],
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Tokenize
    print(f"\nLoading tokenizer: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)

    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"], num_proc=4)
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"], num_proc=4)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load model
    print(f"\nLoading model: {BASE_MODEL} (4 classes)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=4,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters")

    # Training
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    effective_batch = BATCH_SIZE * GRAD_ACCUM
    n_steps = len(train_data) * EPOCHS // BATCH_SIZE
    print(f"\nTraining config:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Effective batch: {BATCH_SIZE} × {GRAD_ACCUM} = {effective_batch}")
    print(f"  LR: {LR}")
    print(f"  Max length: {MAX_LEN}")
    print(f"  Total steps: ~{n_steps}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        seed=SEED,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        fp16=(device == "cuda"),
        bf16=False,
        report_to="none",
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\nStarting training...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Evaluate
    metrics = trainer.evaluate()
    print(f"\nFinal accuracy: {metrics.get('eval_accuracy', 0):.4f}")

    # Detailed report
    preds = trainer.predict(val_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = preds.label_ids
    print("\nClassification report (4-class):")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=3))

    # Binary accuracy
    binary_pred = [1 if p in (1, 2) else 0 for p in y_pred]
    binary_true = [1 if t in (1, 2) else 0 for t in y_true]
    binary_acc = accuracy_score(binary_true, binary_pred)
    print(f"Binary detection accuracy: {binary_acc:.4f}")

    # Save best model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")

    # Sanity test
    print("\n=== Sanity Tests ===")
    model = model.to(device)
    model.eval()
    tests = [
        ("AI standard", "The rapid advancement of artificial intelligence has fundamentally transformed how we approach complex problem-solving in modern society. Machine learning algorithms now process vast amounts of data with unprecedented efficiency."),
        ("Human casual", "lol my roommate tried to cook pasta last night and somehow set off the fire alarm. we had to stand outside for like 20 min"),
        ("AI polished", "Climate change represents one of the most pressing challenges of our time. Rising global temperatures contribute to more frequent extreme weather events."),
        ("Human essay", "I remember sitting in my grandfather's workshop as a kid, watching him fix radios. He never threw anything away. That stuck with me."),
    ]
    for desc, text in tests:
        inputs = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()[0]
        pred = LABEL_NAMES[int(probs.argmax())]
        ai_pct = (probs[1] + probs[2]) * 100
        print(f"  [{desc:15s}] → {pred:15s} (AI: {ai_pct:.0f}%)")

    print(f"\n{'='*70}")
    print(f"DONE. Model at {OUTPUT_DIR}")
    print(f"Download: tar -czf detector_v4.tar.gz -C {WORKSPACE} detector_v4/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
