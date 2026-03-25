#!/usr/bin/env python3
"""Full DeBERTa retrain on 83K+ balanced dataset with cross-domain data.

Downloads dataset from GitHub Release, trains from DeBERTa-v3-base (not our fine-tuned),
saves to /workspace/models/detector_v3/.

Usage (RunPod 4090, ~45 min):
    python3 train_full_v3.py

Usage (Colab A100, ~20 min):
    !python3 train_full_v3.py
"""

import json, os, sys, random
from collections import Counter
from pathlib import Path

# Config
MODEL_NAME = "microsoft/deberta-v3-base"
OUTPUT_DIR = Path("/workspace/models/detector_v3")
DATA_URL = "https://github.com/chalessishas/ai-text-detector/releases/download/data-v1/dataset_full_retrain.jsonl"
DATA_PATH = Path("/workspace/dataset_full_retrain.jsonl")
LABEL_MAP = {"human": 0, "ai": 1, "ai_polished": 2, "human_polished": 3}
MAX_LEN = 512
BATCH_SIZE = 32  # A100: 32, 4090: 16
EPOCHS = 4
LR = 2e-5
WARMUP_RATIO = 0.1

# Download data if needed
if not DATA_PATH.exists():
    print("Downloading dataset...")
    import urllib.request
    urllib.request.urlretrieve(DATA_URL, str(DATA_PATH))
    print(f"Downloaded: {DATA_PATH.stat().st_size / 1024 / 1024:.0f} MB")

# Load data
texts, labels = [], []
with open(DATA_PATH) as f:
    for line in f:
        if not line.strip(): continue
        obj = json.loads(line)
        label = obj.get("label", "human")
        if label in LABEL_MAP:
            texts.append(obj["text"])
            labels.append(LABEL_MAP[label])

print(f"Dataset: {len(texts)} samples")
for lid, count in sorted(Counter(labels).items()):
    name = [k for k, v in LABEL_MAP.items() if v == lid][0]
    print(f"  {name} ({lid}): {count}")

# Split train/eval (95/5)
combined = list(zip(texts, labels))
random.shuffle(combined)
split = int(len(combined) * 0.95)
train_data = combined[:split]
eval_data = combined[split:]
print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

# Setup
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print(f"\nLoading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
model.float()  # FP32 required for DeBERTa-v3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device: {device}")

# Auto batch size based on GPU memory
if torch.cuda.is_available():
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if mem_gb < 20: BATCH_SIZE = 8
    elif mem_gb < 30: BATCH_SIZE = 16
    else: BATCH_SIZE = 32
    print(f"GPU: {torch.cuda.get_device_name(0)}, {mem_gb:.0f}GB → batch_size={BATCH_SIZE}")

class TextDS(Dataset):
    def __init__(self, data, tokenizer):
        texts, labels = zip(*data)
        self.enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        self.labels = torch.tensor(list(labels), dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return {k: self.enc[k][i] for k in self.enc} | {"labels": self.labels[i]}

train_ds = TextDS(train_data, tokenizer)
eval_ds = TextDS(eval_data, tokenizer)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

# Linear warmup + decay scheduler
def get_lr(step):
    if step < warmup_steps:
        return step / warmup_steps
    return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

print(f"\nTraining: {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}, warmup={warmup_steps}")
print(f"Steps/epoch: {len(train_loader)}, total: {total_steps}\n")

model.train()
global_step = 0
best_eval_acc = 0

for epoch in range(EPOCHS):
    tl = c = n = 0
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(**batch)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        global_step += 1

        tl += out.loss.item()
        c += (out.logits.argmax(-1) == batch["labels"]).sum().item()
        n += len(batch["labels"])

        if (step + 1) % 50 == 0:
            lr_now = scheduler.get_last_lr()[0] * LR
            print(f"  E{epoch+1} S{step+1}/{len(train_loader)} loss={tl/(step+1):.4f} acc={c/n*100:.1f}% lr={lr_now:.2e}")

    train_acc = c / n * 100
    print(f"  Epoch {epoch+1} train: loss={tl/len(train_loader):.4f} acc={train_acc:.1f}%")

    # Eval
    model.eval()
    ec = en = 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            ec += (out.logits.argmax(-1) == batch["labels"]).sum().item()
            en += len(batch["labels"])
    eval_acc = ec / en * 100
    print(f"  Epoch {epoch+1} eval:  acc={eval_acc:.1f}%")

    if eval_acc > best_eval_acc:
        best_eval_acc = eval_acc
        # Save best model
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))
        print(f"  Saved best model (eval={eval_acc:.1f}%)")

    model.train()

print(f"\nBest eval accuracy: {best_eval_acc:.1f}%")
print(f"Model saved to: {OUTPUT_DIR}")

# Package
os.system(f"cd {OUTPUT_DIR} && tar czf /workspace/detector_v3.tar.gz .")
print("Packaged: /workspace/detector_v3.tar.gz")
print("DONE.")
