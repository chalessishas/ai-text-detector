#!/usr/bin/env python3
"""Fine-tune DeBERTa on balanced dataset_v4 with proper 4-class labels.

Fixes from finetune_local.py:
- Uses 4-class labels (matching model architecture) instead of binary
- Much larger training set (5000 samples from balanced dataset_v4)
- Proper stratified sampling across all 4 classes
- Saves to detector_v3/ (keeps existing detector/ as backup)

Run: python3 scripts/finetune_v4.py [--samples 5000] [--epochs 2]
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
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "detector")
DATASET = os.path.join(PROJECT_DIR, "dataset_v4.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "models", "detector_v3")

LABEL_NAMES = ["human", "ai", "ai_polished", "human_polished"]
SEED = 42


def load_balanced_samples(path, n_total):
    """Load stratified samples maintaining 4-class balance."""
    by_label = {0: [], 1: [], 2: [], 3: []}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            label = d["label"]
            text = d.get("text", "")
            if label in by_label and len(text.split()) >= 150:
                by_label[label].append({"text": text, "label": label})

    per_class = n_total // 4
    random.seed(SEED)

    samples = []
    for label in sorted(by_label):
        pool = by_label[label]
        random.shuffle(pool)
        selected = pool[:per_class]
        samples.extend(selected)
        print(f"  {LABEL_NAMES[label]}: {len(selected)}/{len(pool)} available")

    random.shuffle(samples)
    return samples


def compute_metrics(pred_output):
    logits, labels = pred_output
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    print("=" * 60)
    print("DeBERTa Fine-Tuning v4 (4-class, balanced, Apple M4 MPS)")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.float()  # fp16 explodes on DeBERTa-v3
    n_params = sum(p.numel() for p in model.parameters())
    n_labels = model.config.num_labels
    print(f"  {n_params:,} params, {n_labels} classes")

    if n_labels != 4:
        print(f"  WARNING: expected 4 classes, got {n_labels}")

    # Load data
    print(f"\nLoading {args.samples} balanced samples from dataset_v4...")
    data = load_balanced_samples(DATASET, args.samples)
    print(f"  Total: {len(data)} samples")

    label_dist = Counter(d["label"] for d in data)
    print(f"  Distribution: {dict(sorted(label_dist.items()))}")

    # Split
    train_data, val_data = train_test_split(
        data, test_size=0.2, random_state=SEED,
        stratify=[d["label"] for d in data],
    )
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # Tokenize
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Pre-training eval
    print("\nEvaluating pre-training accuracy...")
    pre_trainer = Trainer(
        model=model, eval_dataset=val_ds,
        data_collator=collator, compute_metrics=compute_metrics,
    )
    pre_metrics = pre_trainer.evaluate()
    pre_acc = pre_metrics.get("eval_accuracy", 0)
    print(f"  Pre-training accuracy: {pre_acc:.3f}")

    # Detailed pre-training classification report
    preds = pre_trainer.predict(val_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = preds.label_ids
    print("\n  Pre-training classification report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=3))

    # Training
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    effective_batch = args.batch_size * args.grad_accum
    n_steps = len(train_data) * args.epochs // args.batch_size
    n_opt_steps = n_steps // args.grad_accum

    print(f"\nTraining config:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch_size} × {args.grad_accum} grad_accum = {effective_batch} effective")
    print(f"  LR: {args.lr}")
    print(f"  Steps: ~{n_steps} forward, ~{n_opt_steps} optimizer")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        seed=SEED,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=False,  # avoid gamma/beta naming bug
        logging_steps=20,
        fp16=False,
        bf16=False,
        use_mps_device=(device == "mps"),
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print(f"\nStarting training...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Post-training eval
    post_metrics = trainer.evaluate()
    post_acc = post_metrics.get("eval_accuracy", 0)
    print(f"  Post-training accuracy: {post_acc:.3f}")
    print(f"  Improvement: {post_acc - pre_acc:+.3f}")

    # Detailed post-training report
    preds = trainer.predict(val_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = preds.label_ids
    print("\n  Post-training classification report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=3))

    # Save
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")

    # Binary accuracy (how detection actually works)
    binary_pred = np.array([1 if p in (1, 2) else 0 for p in y_pred])
    binary_true = np.array([1 if t in (1, 2) else 0 for t in y_true])
    binary_acc = accuracy_score(binary_true, binary_pred)
    print(f"\n  Binary detection accuracy: {binary_acc:.3f}")

    # Sanity test
    print("\n=== Sanity Tests ===")
    model.to(device)
    test_cases = [
        ("AI essay", "The proliferation of artificial intelligence has fundamentally transformed the landscape of modern technology. Machine learning algorithms now permeate virtually every sector, from healthcare diagnostics to financial modeling. This paradigm shift necessitates a comprehensive reevaluation of existing regulatory frameworks."),
        ("AI news", "The Federal Reserve held interest rates steady at its latest meeting, signaling that officials remain cautious about cutting borrowing costs amid persistent inflation concerns and a resilient labor market."),
        ("Human casual", "lol my roommate tried to cook pasta last night and somehow set off the fire alarm. we had to stand outside for like 20 min. at least it was funny"),
        ("Human academic", "In my experience working with first-generation college students, I've noticed they often struggle not with the coursework itself but with navigating institutional expectations that their peers absorbed growing up."),
        ("AI polished", "Climate change represents one of the most pressing challenges of our time. Rising global temperatures contribute to more frequent extreme weather events, threatening food security and displacing vulnerable populations worldwide."),
    ]

    for desc, text in test_cases:
        inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()[0]
        pred = LABEL_NAMES[int(probs.argmax())]
        ai_pct = (probs[1] + probs[2]) * 100
        human_pct = (probs[0] + probs[3]) * 100
        print(f"  [{desc:15s}] → {pred:15s} (AI: {ai_pct:.0f}%, Human: {human_pct:.0f}%)")


if __name__ == "__main__":
    main()
