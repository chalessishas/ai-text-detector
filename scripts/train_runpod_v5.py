#!/usr/bin/env python3
"""DeBERTa v5 adversarial retrain on RunPod GPU.

Trains on dataset_v4.jsonl (69K clean) + dataset_adversarial_v4.jsonl (17K sampled).
Key improvement over v4: adversarial data teaches model to see through 14 attack types
(typos, casual injection, homoglyphs, human_sandwich, etc.)

Expected: ~40 min on RTX 4090, cost ~$0.30.

Setup on RunPod pod:
  pip install transformers datasets accelerate scikit-learn
  # Upload dataset_v4.jsonl + dataset_adversarial_v4.jsonl to /workspace/
  python3 train_runpod_v5.py
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

WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
DATASET_CLEAN = os.path.join(WORKSPACE, "dataset_v4.jsonl")
DATASET_ADV = os.path.join(WORKSPACE, "dataset_adversarial_v4.jsonl")
OUTPUT_DIR = os.path.join(WORKSPACE, "detector_v5")
BASE_MODEL = "microsoft/deberta-v3-base"

EPOCHS = 4
BATCH_SIZE = 8
GRAD_ACCUM = 8
LR = 2e-5
WARMUP_RATIO = 0.1
MAX_LEN = 512
ADV_SAMPLE_PER_CLASS = 17294  # match clean per-class count


def load_clean(path):
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


def load_adversarial(path, n_sample):
    samples = []
    attacks = Counter()
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            text = d.get("text", "")
            if len(text.split()) >= 50:
                samples.append({
                    "text": text,
                    "label": 1,  # all adversarial are AI
                    "attack": d.get("attack", "unknown"),
                })
                attacks[d.get("attack", "unknown")] += 1

    print(f"\nAdversarial dataset: {len(samples)} total")
    for atk, cnt in attacks.most_common():
        print(f"  {atk}: {cnt}")

    # Stratified sample by attack type to ensure diversity
    random.seed(SEED)
    by_attack = {}
    for s in samples:
        by_attack.setdefault(s["attack"], []).append(s)

    per_attack = max(1, n_sample // len(by_attack))
    sampled = []
    for atk in sorted(by_attack):
        pool = by_attack[atk]
        random.shuffle(pool)
        sampled.extend(pool[:per_attack])

    # Fill remainder if needed
    random.shuffle(samples)
    remaining = n_sample - len(sampled)
    if remaining > 0:
        used = set(id(s) for s in sampled)
        for s in samples:
            if id(s) not in used:
                sampled.append(s)
                if len(sampled) >= n_sample:
                    break

    sampled = sampled[:n_sample]
    print(f"  Sampled: {len(sampled)} (balanced with clean per-class)")
    return sampled


def main():
    print("=" * 70)
    print("DeBERTa v5 Adversarial Retrain")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("WARNING: No GPU detected — will be extremely slow")

    # Load clean data
    clean = load_clean(DATASET_CLEAN)
    min_count = min(len(v) for v in clean.values())
    print(f"  Balanced per-class: {min_count}")

    # Load adversarial data (sample to match per-class count)
    adv = load_adversarial(DATASET_ADV, min_count)

    # Build balanced dataset:
    # class 0 (human): 17K clean
    # class 1 (ai): 17K clean + 17K adversarial = 34K → keep all (model sees more AI variety)
    # class 2 (ai_polished): 17K clean
    # class 3 (human_polished): 17K clean
    # Use sample weights to handle class 1 having 2x samples
    random.seed(SEED)
    all_samples = []
    for label in sorted(clean):
        pool = clean[label]
        random.shuffle(pool)
        selected = pool[:min_count]
        all_samples.extend(selected)
        print(f"  Added {len(selected)} clean {LABEL_NAMES[label]}")

    # Add adversarial (all label=1)
    for s in adv:
        all_samples.append({"text": s["text"], "label": s["label"]})
    print(f"  Added {len(adv)} adversarial AI")
    print(f"  Total: {len(all_samples)}")

    # Class distribution
    label_counts = Counter(s["label"] for s in all_samples)
    print(f"\n  Class distribution:")
    for label in sorted(label_counts):
        print(f"    {LABEL_NAMES[label]}: {label_counts[label]}")

    # Compute class weights for imbalanced class 1
    total = len(all_samples)
    n_classes = 4
    class_weights = {}
    for label in range(n_classes):
        count = label_counts.get(label, 1)
        class_weights[label] = total / (n_classes * count)
    print(f"\n  Class weights: { {LABEL_NAMES[k]: f'{v:.2f}' for k, v in class_weights.items()} }")

    # Split
    random.shuffle(all_samples)
    train_data, val_data = train_test_split(
        all_samples, test_size=0.1, random_state=SEED,
        stratify=[d["label"] for d in all_samples],
    )
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

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
    # DeBERTa must run fp32 on some ops — gradient checkpointing saves memory
    model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters (gradient checkpointing ON)")

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

    # Training
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    effective_batch = BATCH_SIZE * GRAD_ACCUM
    print(f"\nTraining config:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Effective batch: {BATCH_SIZE} x {GRAD_ACCUM} = {effective_batch}")
    print(f"  LR: {LR}, Warmup: {WARMUP_RATIO}")
    print(f"  Max length: {MAX_LEN}")

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

    # Save
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")

    # Sanity test
    print("\n=== Sanity Tests ===")
    model = model.to(device).float()
    model.eval()
    tests = [
        ("AI standard", "The rapid advancement of artificial intelligence has fundamentally transformed how we approach complex problem-solving in modern society. Machine learning algorithms now process vast amounts of data with unprecedented efficiency."),
        ("Human casual", "lol my roommate tried to cook pasta last night and somehow set off the fire alarm. we had to stand outside for like 20 min"),
        ("AI polished", "Climate change represents one of the most pressing challenges of our time. Rising global temperatures contribute to more frequent extreme weather events."),
        ("Human essay", "I remember sitting in my grandfather's workshop as a kid, watching him fix radios. He never threw anything away. That stuck with me."),
        ("AI + typos", "Teh rapid advanecment of artficial intellgence has fundamnentally transformd how we aproach complx problem-solving."),
        ("AI + casual", "so basically, AI is like, really changing everything right? like machine learning can process tons of data super fast lol"),
        ("AI + sandwich", "I was thinking about this the other day. The integration of artificial intelligence into healthcare systems promises significant improvements. Anyway that's just my take on it."),
    ]
    for desc, text in tests:
        inputs = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()[0]
        pred = LABEL_NAMES[int(probs.argmax())]
        ai_pct = (probs[1] + probs[2]) * 100
        print(f"  [{desc:15s}] -> {pred:15s} (AI: {ai_pct:.0f}%)")

    # Package
    print(f"\n{'='*70}")
    print(f"DONE. Model at {OUTPUT_DIR}")
    print(f"Download: tar -czf detector_v5.tar.gz -C {WORKSPACE} detector_v5/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
