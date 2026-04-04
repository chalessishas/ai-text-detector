#!/usr/bin/env python3
"""DeBERTa v5 adversarial retrain — local M4 16GB version.

Aggressive memory optimization:
- batch_size=1, grad_accum=64
- gradient_checkpointing=True
- max_len=256 (vs 512 on GPU)
- fp32 (DeBERTa requires it)

Usage:
  python3 scripts/train_local_v5.py              # full training
  python3 scripts/train_local_v5.py --test 100   # test with 100 samples
"""

import argparse
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_CLEAN = os.path.join(BASE_DIR, "dataset_v4.jsonl")
DATASET_ADV = os.path.join(BASE_DIR, "dataset_adversarial_v4.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "detector_v5")
BASE_MODEL = "microsoft/deberta-v3-base"

EPOCHS = 4
BATCH_SIZE = 1
GRAD_ACCUM = 64
LR = 2e-5
WARMUP_RATIO = 0.1
MAX_LEN = 256


def load_clean(path, max_per_class=0):
    by_label = {0: [], 1: [], 2: [], 3: []}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            label = d["label"]
            text = d.get("text", "")
            if label in by_label and len(text.split()) >= 50:
                by_label[label].append({"text": text, "label": label})

    print("Clean dataset:")
    for label in sorted(by_label):
        print(f"  {LABEL_NAMES[label]}: {len(by_label[label])}")
    return by_label


def load_adversarial(path):
    samples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            text = d.get("text", "")
            if len(text.split()) >= 50:
                samples.append({
                    "text": text,
                    "label": 1,
                    "attack": d.get("attack", "unknown"),
                })
    print(f"Adversarial dataset: {len(samples)} total")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=0, help="Test with N samples (0=full)")
    args = parser.parse_args()

    print("=" * 70)
    print("DeBERTa v5 Local Training (M4 16GB)")
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    clean = load_clean(DATASET_CLEAN)
    adv = load_adversarial(DATASET_ADV)

    min_count = min(len(v) for v in clean.values())

    if args.test > 0:
        per_class = args.test // 5  # 4 clean classes + 1 adv portion
        print(f"\n*** TEST MODE: {args.test} samples ({per_class} per class) ***")
    else:
        per_class = min_count

    random.seed(SEED)
    all_samples = []
    for label in sorted(clean):
        pool = clean[label]
        random.shuffle(pool)
        selected = pool[:per_class]
        all_samples.extend(selected)

    # Stratified adversarial sampling
    by_attack = {}
    for s in adv:
        by_attack.setdefault(s["attack"], []).append(s)
    adv_per_attack = max(1, per_class // len(by_attack))
    adv_sampled = []
    for atk in sorted(by_attack):
        pool = by_attack[atk]
        random.shuffle(pool)
        adv_sampled.extend(pool[:adv_per_attack])
    adv_sampled = adv_sampled[:per_class]

    for s in adv_sampled:
        all_samples.append({"text": s["text"], "label": s["label"]})

    print(f"\nTotal samples: {len(all_samples)}")
    label_counts = Counter(s["label"] for s in all_samples)
    for label in sorted(label_counts):
        print(f"  {LABEL_NAMES[label]}: {label_counts[label]}")

    # Class weights
    total = len(all_samples)
    n_classes = 4
    class_weights = {}
    for label in range(n_classes):
        count = label_counts.get(label, 1)
        class_weights[label] = total / (n_classes * count)

    # Split
    random.shuffle(all_samples)
    train_data, val_data = train_test_split(
        all_samples, test_size=0.1, random_state=SEED,
        stratify=[d["label"] for d in all_samples],
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Tokenize
    print(f"\nLoading tokenizer: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)

    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load model
    print(f"\nLoading model: {BASE_MODEL} (4 classes)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=4,
    )
    model.float()  # DeBERTa MUST be fp32
    model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters (fp32 + gradient checkpointing)")

    # Custom Trainer with class weights
    weight_tensor = torch.tensor(
        [class_weights[i] for i in range(n_classes)], dtype=torch.float32
    ).to(device)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    effective_batch = BATCH_SIZE * GRAD_ACCUM
    steps_per_epoch = len(train_data) // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    est_hours = total_steps * 0.8 / 3600  # ~0.8s per step estimate on M4

    print(f"\nTraining config:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Effective batch: {BATCH_SIZE} x {GRAD_ACCUM} = {effective_batch}")
    print(f"  Steps/epoch: {steps_per_epoch}, Total: {total_steps}")
    print(f"  Estimated time: {est_hours:.1f} hours")
    print(f"  LR: {LR}, Max length: {MAX_LEN}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        seed=SEED,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=20,
        fp16=False,
        bf16=False,
        report_to="none",
        dataloader_num_workers=0,  # MPS doesn't benefit from multiprocess
        use_mps_device=(device == "mps"),
    )

    def compute_metrics(pred_output):
        logits, labels = pred_output
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    trainer = WeightedTrainer(
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

    preds_out = trainer.predict(val_ds)
    y_pred = np.argmax(preds_out.predictions, axis=-1)
    y_true = preds_out.label_ids
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=3))

    binary_pred = [1 if p in (1, 2) else 0 for p in y_pred]
    binary_true = [1 if t in (1, 2) else 0 for t in y_true]
    binary_acc = accuracy_score(binary_true, binary_pred)
    print(f"Binary accuracy: {binary_acc:.4f}")

    # Save
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")
    print(f"Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
