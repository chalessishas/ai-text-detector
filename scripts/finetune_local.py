#!/usr/bin/env python3
"""Local incremental fine-tuning of DeBERTa on Apple M4 (MPS).

Fine-tunes the existing detector model on new genre data (RAID + DeepSeek)
to improve cross-domain generalization without full retraining.

Strategy: Low LR (5e-6), 2 epochs, small batch, only new data.
This preserves existing knowledge while adding new genre awareness.

Run: python3 scripts/finetune_local.py
"""

import json
import os
import random
import time
from collections import Counter

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "detector")
AUGMENTED = os.path.join(os.path.dirname(__file__), "..", "dataset_augmented.jsonl")
RAID_EXTRACT = os.path.join(os.path.dirname(__file__), "..", "dataset_raid_extract.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "detector_v2")

LABEL_NAMES = ["human", "ai", "ai_polished", "human_polished"]


ORIGINAL = os.path.join(os.path.dirname(__file__), "..", "dataset.jsonl")
MIX_OLD_SAMPLES = 1500  # How many old samples to mix in (0 = new-only mode)


def load_new_data():
    """Load augmented + RAID data + optional old data mix.

    Mixed training prevents catastrophic forgetting by including original
    domain samples alongside new domain data.
    """
    data = []

    if os.path.exists(AUGMENTED):
        with open(AUGMENTED) as f:
            for line in f:
                entry = json.loads(line)
                text = entry.get("text", "")
                if len(text) > 50:
                    data.append({"text": text, "label": 1})

    if os.path.exists(RAID_EXTRACT):
        with open(RAID_EXTRACT) as f:
            for line in f:
                entry = json.loads(line)
                text = entry.get("text", "")
                label = entry.get("label", -1)
                if len(text) > 50 and label in (0, 1):
                    data.append({"text": text, "label": label})

    # Mix in old domain samples to prevent catastrophic forgetting
    if MIX_OLD_SAMPLES > 0 and os.path.exists(ORIGINAL):
        old_lines = []
        with open(ORIGINAL) as f:
            old_lines = f.readlines()
        random.shuffle(old_lines)
        per_class = MIX_OLD_SAMPLES // 4
        class_counts = Counter()
        for line in old_lines:
            entry = json.loads(line)
            label = entry.get("label", -1)
            text = entry.get("text", "")
            if len(text) > 50 and label in (0, 1, 2, 3) and class_counts[label] < per_class:
                # Map 4-class to binary for consistency
                binary = 0 if label in (0, 3) else 1
                data.append({"text": text, "label": binary})
                class_counts[label] += 1
            if sum(class_counts.values()) >= MIX_OLD_SAMPLES:
                break
        print(f"  Mixed in {sum(class_counts.values())} old samples ({dict(class_counts)})")

    random.shuffle(data)
    return data


def compute_metrics(pred_output):
    logits, labels = pred_output
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def main():
    print("=" * 60)
    print("Local Incremental Fine-Tuning (Apple M4 MPS)")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nLoading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.float()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  {param_count:,} params")

    print("\nLoading new data...")
    data = load_new_data()
    print(f"  {len(data)} samples")

    if len(data) < 20:
        print("Not enough data for fine-tuning.")
        return

    label_counts = Counter(d["label"] for d in data)
    print(f"  Label distribution: {dict(label_counts)}")

    train_data, val_data = train_test_split(
        data, test_size=0.2, random_state=42,
        stratify=[d["label"] for d in data],
    )
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("\nPre-training accuracy on new data...")
    pre_trainer = Trainer(
        model=model,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    pre_metrics = pre_trainer.evaluate()
    pre_acc = pre_metrics.get("eval_accuracy", 0)
    print(f"  Pre-training accuracy: {pre_acc:.3f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        seed=42,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=5e-6,
        warmup_steps=50,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        fp16=False,
        bf16=False,
        use_mps_device=(device == "mps"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print(f"\nTraining: 2 epochs, batch=4, grad_accum=4, lr=5e-6")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.0f}s")

    post_metrics = trainer.evaluate()
    post_acc = post_metrics.get("eval_accuracy", 0)
    print(f"  Post-training accuracy: {post_acc:.3f}")
    print(f"  Improvement: {post_acc - pre_acc:+.3f}")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")

    print("\n=== Sanity Test ===")
    model.to(device)
    test_texts = [
        ("AI tech", "Kubernetes orchestrates containerized applications across clusters. The control plane consists of the API server, etcd, scheduler."),
        ("AI news", "The Federal Reserve held interest rates steady, signaling officials remain cautious about cutting borrowing costs."),
        ("Human", "my cat knocked the coffee over this morning. just sat there staring at the mess."),
    ]
    for desc, text in test_texts:
        inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()[0]
        pred = LABEL_NAMES[int(probs.argmax())]
        ai_pct = (probs[1] + probs[2]) * 100
        print(f"  [{desc}] -> {pred} (AI: {ai_pct:.0f}%)")


if __name__ == "__main__":
    main()
