#!/usr/bin/env python3
"""Incremental fine-tune DeBERTa on cross-domain data. Runs locally on CPU (M4 Mac).

Loads existing DeBERTa model from models/detector/, adds domain-diverse training data,
trains for 2 epochs with LoRA-like low learning rate to preserve existing essay detection
while adding new domain coverage. Saves to models/detector_domain/.

Usage:
    python3 scripts/finetune_domain.py

Expected time: ~2 hours on M4 Mac (CPU, FP32). ~30 min on Colab A100.
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_DIR / "models" / "detector"
OUTPUT_DIR = PROJECT_DIR / "models" / "detector_domain"
DOMAIN_AI = SCRIPT_DIR / "data" / "domain_ai_samples.jsonl"
DOMAIN_HUMAN = SCRIPT_DIR / "data" / "domain_human_samples.jsonl"
EXISTING_DATASET = PROJECT_DIR / "dataset_v3.jsonl"

LABEL_MAP = {"human": 0, "ai": 1, "ai_polished": 2, "human_polished": 3}
MAX_LEN = 512
BATCH_SIZE = 4  # small batch for CPU
EPOCHS = 2
LR = 5e-6  # very low LR to preserve existing knowledge
DOMAIN_WEIGHT = 2.0  # upweight domain samples vs existing data


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, max_length=MAX_LEN,
            padding="max_length", return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def load_domain_data():
    """Load new domain-diverse AI + human samples."""
    texts, labels = [], []

    if DOMAIN_AI.exists():
        for line in DOMAIN_AI.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(LABEL_MAP["ai"])
        print(f"  Domain AI samples: {sum(1 for l in labels if l == 1)}")

    if DOMAIN_HUMAN.exists():
        for line in DOMAIN_HUMAN.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(LABEL_MAP["human"])
        print(f"  Domain human samples: {sum(1 for l in labels if l == 0)}")

    return texts, labels


def load_existing_sample(n=200):
    """Load a small balanced sample from existing dataset to prevent catastrophic forgetting."""
    if not EXISTING_DATASET.exists():
        print(f"  No existing dataset at {EXISTING_DATASET}, skipping replay")
        return [], []

    all_data = []
    with open(EXISTING_DATASET) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            label_str = obj.get("label", obj.get("classification", ""))
            if label_str in LABEL_MAP:
                all_data.append((obj["text"], LABEL_MAP[label_str]))

    random.shuffle(all_data)
    # Balance: n/2 human + n/2 ai
    humans = [(t, l) for t, l in all_data if l in (0, 3)][:n // 2]
    ais = [(t, l) for t, l in all_data if l in (1, 2)][:n // 2]
    selected = humans + ais
    random.shuffle(selected)

    texts = [t for t, _ in selected]
    labels = [l for _, l in selected]
    print(f"  Replay samples from existing dataset: {len(texts)}")
    return texts, labels


def main():
    print("=== DeBERTa Domain Fine-tuning (Local CPU) ===\n")

    if not MODEL_DIR.exists():
        print(f"ERROR: Model not found at {MODEL_DIR}")
        sys.exit(1)

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.float()  # FP32 required for DeBERTa-v3
    model.train()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())} parameters")

    # Load data
    print("\nLoading training data...")
    domain_texts, domain_labels = load_domain_data()
    replay_texts, replay_labels = load_existing_sample(200)

    if not domain_texts:
        print("ERROR: No domain data found. Run generate_domain_data.py first.")
        sys.exit(1)

    # Combine: domain samples (upweighted) + replay samples
    # Upweight by repeating domain samples
    repeat = int(DOMAIN_WEIGHT)
    all_texts = domain_texts * repeat + replay_texts
    all_labels = domain_labels * repeat + replay_labels

    # Shuffle
    combined = list(zip(all_texts, all_labels))
    random.shuffle(combined)
    all_texts, all_labels = zip(*combined)
    all_texts, all_labels = list(all_texts), list(all_labels)

    print(f"\nTotal training samples: {len(all_texts)}")
    from collections import Counter
    label_dist = Counter(all_labels)
    for label_id, count in sorted(label_dist.items()):
        name = [k for k, v in LABEL_MAP.items() if v == label_id][0]
        print(f"  {name} ({label_id}): {count}")

    # Create dataset
    dataset = TextDataset(all_texts, all_labels, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Train
    print(f"\nTraining: {EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LR}")
    print(f"Estimated time: ~{len(loader) * EPOCHS * 3 // 60} minutes on M4 CPU\n")

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += len(batch["labels"])

            if (step + 1) % 10 == 0:
                avg_loss = total_loss / (step + 1)
                acc = correct / total * 100
                print(f"  Epoch {epoch+1}/{EPOCHS} Step {step+1}/{len(loader)} — loss: {avg_loss:.4f} acc: {acc:.1f}%")

        avg_loss = total_loss / len(loader)
        acc = correct / total * 100
        print(f"  Epoch {epoch+1} done — loss: {avg_loss:.4f} acc: {acc:.1f}%\n")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Model saved to {OUTPUT_DIR}")

    # Quick sanity check
    print("\nSanity check:")
    model.eval()
    test_texts = [
        ("AI essay", "The proliferation of artificial intelligence has transformed modern society in unprecedented ways."),
        ("Human casual", "lol my cat just knocked over my coffee again. third time this week smh"),
        ("AI creative", "The forest whispered ancient secrets as moonlight pooled like liquid silver upon the earth."),
        ("AI legal", "Pursuant to Section 7(a) of the Securities Act, the undersigned registrant hereby certifies."),
    ]
    for label, text in test_texts:
        inputs = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits / 2.0, dim=-1).cpu().numpy()[0]  # T=2.0 to match production
        ai_score = (probs[1] + probs[2]) * 100
        pred = "AI" if ai_score > 50 else "Human"
        print(f"  {label}: {pred} ({ai_score:.1f}%)")

    print("\nDone. Next: upload to Railway with GitHub Release.")


if __name__ == "__main__":
    main()
