#!/usr/bin/env python3
"""Full DeBERTa retrain using HuggingFace Trainer (more reliable than manual loop).

Usage (RunPod):
    pip install transformers accelerate sentencepiece datasets
    python3 train_full_v3.py
"""

import json, os, sys, random
from collections import Counter
from pathlib import Path

MODEL_NAME = "microsoft/deberta-v3-base"
OUTPUT_DIR = Path("/workspace/models/detector_v3")
DATA_URL = "https://github.com/chalessishas/ai-text-detector/releases/download/data-v1/dataset_full_retrain.jsonl"
DATA_PATH = Path("/workspace/dataset_full_retrain.jsonl")
LABEL_MAP = {"human": 0, "ai": 1, "ai_polished": 2, "human_polished": 3}
MAX_LEN = 512

# Download data if needed
if not DATA_PATH.exists():
    print("Downloading dataset...")
    import urllib.request
    urllib.request.urlretrieve(DATA_URL, str(DATA_PATH))

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

# Split
combined = list(zip(texts, labels))
random.seed(42)
random.shuffle(combined)
split = int(len(combined) * 0.95)
train_data = combined[:split]
eval_data = combined[split:]
print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from torch.utils.data import Dataset

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
model.float()  # FP32 required for DeBERTa-v3

# Auto batch size
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    bs = 8 if mem_gb < 20 else (16 if mem_gb < 30 else 32)
    print(f"GPU: {torch.cuda.get_device_name(0)}, {mem_gb:.0f}GB → batch={bs}")
else:
    bs = 4
    print(f"CPU mode, batch={bs}")

class TextDS(Dataset):
    def __init__(self, data):
        self.texts = [t for t, _ in data]
        self.labels = [l for _, l in data]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        enc = tokenizer(self.texts[i], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()} | {"labels": torch.tensor(self.labels[i])}

import numpy as np
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=2,
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=False,  # DeBERTa-v3 gamma/beta bug
    save_safetensors=False,  # Avoid safetensors conversion bug
    logging_steps=50,
    fp16=False,  # FP32 required
    dataloader_num_workers=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=TextDS(train_data),
    eval_dataset=TextDS(eval_data),
    compute_metrics=compute_metrics,
)

print(f"\nStarting training...")
trainer.train()

# Save manually (avoid safetensors bug)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), str(OUTPUT_DIR / "pytorch_model.bin"))
model.config.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

# Also save as safetensors for our deployment
model.save_pretrained(str(OUTPUT_DIR))

# Eval
results = trainer.evaluate()
print(f"\nFinal eval: {results}")

# Package
os.system(f"cd {OUTPUT_DIR} && tar czf /workspace/detector_v3.tar.gz .")
print("Packaged: /workspace/detector_v3.tar.gz")
print("DONE.")
