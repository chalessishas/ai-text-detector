#!/usr/bin/env python3
"""DeBERTa v6 binary classifier with domain-adversarial training (DANN) on RunPod GPU.

Key changes from v5:
  - BINARY classification: 0=human, 1=AI (merged: old 0+3->human, old 1+2->AI)
  - Model: microsoft/deberta-v3-large (304M params)
  - Single dataset file: dataset_v6.jsonl (1M samples), falls back to dataset_v4.jsonl
  - Domain-Adversarial Neural Network (DANN): gradient reversal layer on domain classifier
    so the model learns domain-invariant features
  - Evaluation at both Temperature=1.0 and Temperature=2.0 (production config)

Expected: ~2-3 hours on A100 80GB with 1M samples.

Setup on RunPod pod (A100 80GB recommended):
  pip install transformers datasets accelerate scikit-learn
  # Upload dataset_v6.jsonl to /workspace/
  python3 train_runpod_v6.py
"""

import json
import os
import random
import tarfile
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# ── Config ────────────────────────────────────────────────────────────

SEED = 42
LABEL_NAMES_V6 = ["human", "AI"]
LABEL_NAMES_V4 = ["human", "ai", "ai_polished", "human_polished"]

WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
DATASET_V6 = os.path.join(WORKSPACE, "dataset_v6.jsonl")
DATASET_V4 = os.path.join(WORKSPACE, "dataset_v4.jsonl")
OUTPUT_DIR = os.path.join(WORKSPACE, "detector_v6")
BASE_MODEL = "microsoft/deberta-v3-large"

EPOCHS = 3
BATCH_SIZE = 16
GRAD_ACCUM = 4
LR = 1e-5
WARMUP_RATIO = 0.1
MAX_LEN = 512
MIN_WORDS = 50

# DANN config
DANN_HIDDEN = 768
DANN_LAMBDA_MAX = 1.0


# ── Gradient Reversal Layer ───────────────────────────────────────────

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_val = 0.0

    def set_lambda(self, val: float):
        self.lambda_val = val

    def forward(self, x):
        return GradientReversalFn.apply(x, self.lambda_val)


# ── DeBERTa + DANN Model ─────────────────────────────────────────────

class DeBERTaDANN(nn.Module):
    """DeBERTa-v3-large with a task classifier head and an optional
    domain-adversarial classifier head (DANN).

    The domain classifier receives features through a gradient reversal
    layer so the encoder learns domain-invariant representations.
    """

    def __init__(self, model_name: str, num_labels: int, num_domains: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Task classifier (binary: human vs AI)
        self.dropout = nn.Dropout(0.1)
        self.task_classifier = nn.Linear(hidden_size, num_labels)

        # Domain-adversarial head (only used when num_domains > 1)
        self.use_dann = num_domains > 1
        if self.use_dann:
            self.grl = GradientReversalLayer()
            self.domain_classifier = nn.Sequential(
                nn.Linear(hidden_size, DANN_HIDDEN),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(DANN_HIDDEN, num_domains),
            )

        self.num_labels = num_labels
        self.num_domains = num_domains

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        domain_labels=None,
        **kwargs,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        task_logits = self.task_classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(task_logits, labels)

        if self.use_dann and domain_labels is not None:
            reversed_features = self.grl(pooled)
            domain_logits = self.domain_classifier(reversed_features)
            # Only compute domain loss for samples that have valid domain labels (>= 0)
            valid_mask = domain_labels >= 0
            if valid_mask.any():
                domain_loss = nn.CrossEntropyLoss()(
                    domain_logits[valid_mask], domain_labels[valid_mask]
                )
                if loss is not None:
                    loss = loss + domain_loss
                else:
                    loss = domain_loss

        return {"loss": loss, "logits": task_logits}


# ── Data Loading ──────────────────────────────────────────────────────

def convert_label_to_binary(label: int) -> int:
    """Convert 4-class label to binary: 0+3 -> 0 (human), 1+2 -> 1 (AI)."""
    if label in (0, 3):
        return 0  # human
    if label in (1, 2):
        return 1  # AI
    raise ValueError(f"Unknown label: {label}")


def load_dataset(path: str) -> Tuple[List[dict], List[str], bool]:
    """Load JSONL dataset, convert labels to binary.

    Returns (samples, unique_domains, has_domains).
    Each sample: {"text": str, "label": int, "domain": int or -1}
    """
    raw_domain_counts = Counter()
    samples = []
    skipped = 0

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                if skipped <= 5:
                    print(f"  WARN: skipped malformed JSON at line {line_num}")
                continue

            text = d.get("text", "").strip()
            if len(text.split()) < MIN_WORDS:
                skipped += 1
                continue

            raw_label = d.get("label")
            if raw_label is None:
                skipped += 1
                continue

            try:
                binary_label = convert_label_to_binary(int(raw_label))
            except ValueError:
                skipped += 1
                continue

            domain = d.get("source") or d.get("domain") or None
            if domain is not None:
                domain = str(domain)
            raw_domain_counts[domain] += 1

            samples.append({
                "text": text,
                "label": binary_label,
                "raw_domain": domain,
            })

    if skipped > 0:
        print(f"  Skipped {skipped} rows (short text, bad JSON, or missing label)")

    # Build domain index
    unique_domains = sorted([d for d in raw_domain_counts if d is not None])
    has_domains = len(unique_domains) > 1

    if has_domains:
        domain_to_idx = {d: i for i, d in enumerate(unique_domains)}
        for s in samples:
            s["domain_label"] = domain_to_idx.get(s["raw_domain"], -1)
        print(f"\n  Domain-adversarial training ENABLED ({len(unique_domains)} domains):")
        for d in unique_domains:
            print(f"    {d}: {raw_domain_counts[d]}")
        no_domain = raw_domain_counts.get(None, 0)
        if no_domain > 0:
            print(f"    (no domain): {no_domain} -> domain_label=-1 (excluded from DANN loss)")
    else:
        for s in samples:
            s["domain_label"] = -1
        print("\n  No domain field found -> DANN disabled (backward compatible)")

    # Remove temporary key
    for s in samples:
        del s["raw_domain"]

    return samples, unique_domains, has_domains


def balance_classes(samples: List[dict]) -> List[dict]:
    """Downsample majority class to match minority class count."""
    by_label = {}
    for s in samples:
        by_label.setdefault(s["label"], []).append(s)

    counts = {label: len(items) for label, items in by_label.items()}
    min_count = min(counts.values())
    print(f"\n  Class counts before balancing:")
    for label in sorted(counts):
        print(f"    {LABEL_NAMES_V6[label]}: {counts[label]}")
    print(f"  Balancing to {min_count} per class")

    random.seed(SEED)
    balanced = []
    for label in sorted(by_label):
        pool = by_label[label]
        random.shuffle(pool)
        balanced.extend(pool[:min_count])

    return balanced


# ── Custom Trainer ────────────────────────────────────────────────────

class DANNTrainer(Trainer):
    """Trainer that handles domain_labels column and updates GRL lambda."""

    def __init__(self, *args, model_ref=None, total_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_ref = model_ref
        self.total_steps = total_steps

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract domain_labels before passing to model
        domain_labels = inputs.pop("domain_labels", None)

        # Update GRL lambda based on training progress
        if hasattr(model, "module"):
            actual_model = model.module
        else:
            actual_model = model

        if actual_model.use_dann and self.total_steps > 0:
            progress = min(1.0, self.state.global_step / self.total_steps)
            # Ramp lambda from 0 to DANN_LAMBDA_MAX over training
            current_lambda = DANN_LAMBDA_MAX * progress
            actual_model.grl.set_lambda(current_lambda)

        outputs = model(domain_labels=domain_labels, **inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


class DANNDataCollator(DataCollatorWithPadding):
    """Collator that also pads domain_labels."""

    def __call__(self, features):
        domain_labels = [f.pop("domain_labels") for f in features]
        batch = super().__call__(features)
        batch["domain_labels"] = torch.tensor(domain_labels, dtype=torch.long)
        return batch


# ── Evaluation Helpers ────────────────────────────────────────────────

def evaluate_with_temperature(
    model: nn.Module,
    val_ds: Dataset,
    tokenizer,
    device: str,
    temperature: float,
    max_len: int = MAX_LEN,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference with softmax temperature, return (preds, probs, labels)."""
    model.eval()
    all_logits = []
    all_labels = []

    # Process in batches
    for start in range(0, len(val_ds), batch_size):
        end = min(start + batch_size, len(val_ds))
        batch_texts = val_ds[start:end]["text"]
        batch_labels = val_ds[start:end]["label"]

        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs["logits"]
            all_logits.append(logits.cpu())
            all_labels.extend(batch_labels)

    all_logits = torch.cat(all_logits, dim=0)
    # Apply temperature scaling
    scaled_logits = all_logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1).numpy()
    preds = np.argmax(probs, axis=-1)
    labels = np.array(all_labels)

    return preds, probs, labels


# ── Sanity Tests ──────────────────────────────────────────────────────

SANITY_TESTS = [
    ("Human casual",
     "lol my roommate tried to cook pasta last night and somehow set off the fire alarm. "
     "we had to stand outside for like 20 min in our pajamas"),
    ("Human academic",
     "The findings suggest a correlation between sleep deprivation and decreased cognitive "
     "performance, though the sample size limits generalizability. Further longitudinal "
     "studies are warranted to establish causation."),
    ("Human legal",
     "Pursuant to Section 14(b) of the Agreement, the indemnifying party shall hold harmless "
     "and defend the indemnified party against any third-party claims arising from gross "
     "negligence or willful misconduct."),
    ("Human poetry",
     "The creek behind our house ran dry last August. I still hear it sometimes, "
     "a phantom rushing between the stones, like my grandmother humming after she forgot the words."),
    ("AI standard",
     "The rapid advancement of artificial intelligence has fundamentally transformed how we "
     "approach complex problem-solving in modern society. Machine learning algorithms now "
     "process vast amounts of data with unprecedented efficiency."),
    ("AI casual",
     "so basically, AI is like, really changing everything right? like machine learning can "
     "process tons of data super fast and it's making a huge impact on industries worldwide lol"),
    ("AI formal",
     "In conclusion, the implementation of comprehensive environmental policies requires a "
     "multifaceted approach that balances economic growth with ecological preservation. "
     "Stakeholders must collaborate to develop sustainable frameworks that address both "
     "immediate and long-term challenges."),
]


def run_sanity_tests(
    model: nn.Module, tokenizer, device: str, temperatures: List[float]
):
    """Run sanity tests at multiple temperatures."""
    model.eval()
    for temp in temperatures:
        print(f"\n  Temperature = {temp}:")
        for desc, text in SANITY_TESTS:
            inputs = tokenizer(
                text, truncation=True, max_length=MAX_LEN, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                logits = model(**inputs)["logits"]
                scaled = logits / temp
                probs = torch.softmax(scaled, dim=-1).cpu().numpy()[0]
            pred_label = LABEL_NAMES_V6[int(probs.argmax())]
            ai_pct = probs[1] * 100
            print(f"    [{desc:15s}] -> {pred_label:6s} (AI: {ai_pct:5.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("DeBERTa v6 Binary + DANN Training")
    print("=" * 70)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("WARNING: No GPU detected -- training will be extremely slow")

    # ── Load dataset ──────────────────────────────────────────────────

    dataset_path = None
    for candidate in [DATASET_V6, DATASET_V4]:
        if os.path.exists(candidate):
            dataset_path = candidate
            break

    if dataset_path is None:
        raise FileNotFoundError(
            f"No dataset found. Expected {DATASET_V6} or {DATASET_V4}"
        )

    print(f"\nLoading dataset: {dataset_path}")
    samples, unique_domains, has_domains = load_dataset(dataset_path)
    print(f"  Loaded {len(samples)} samples")

    # Balance classes
    samples = balance_classes(samples)
    total_samples = len(samples)
    print(f"  Balanced total: {total_samples}")

    # Verify distribution
    label_counts = Counter(s["label"] for s in samples)
    print(f"\n  Final class distribution:")
    for label in sorted(label_counts):
        print(f"    {LABEL_NAMES_V6[label]}: {label_counts[label]}")

    # ── Split ─────────────────────────────────────────────────────────

    random.shuffle(samples)
    train_data, val_data = train_test_split(
        samples,
        test_size=0.1,
        random_state=SEED,
        stratify=[s["label"] for s in samples],
    )
    print(f"\n  Train: {len(train_data)}, Val: {len(val_data)}")

    # ── Tokenize ──────────────────────────────────────────────────────

    print(f"\nLoading tokenizer: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)

    # Keep text column for val (needed for temperature evaluation)
    val_texts = [s["text"] for s in val_data]
    val_labels_list = [s["label"] for s in val_data]

    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    # Store raw val data for temperature evaluation
    val_ds_raw = Dataset.from_dict({"text": val_texts, "label": val_labels_list})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    # For training, remove text column; keep domain_labels renamed
    train_ds = train_ds.rename_column("domain_label", "domain_labels")
    val_ds = val_ds.rename_column("domain_label", "domain_labels")
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"], num_proc=4)
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"], num_proc=4)

    collator = DANNDataCollator(tokenizer=tokenizer)

    # ── Model ─────────────────────────────────────────────────────────

    num_domains = len(unique_domains) if has_domains else 0
    print(f"\nLoading model: {BASE_MODEL} (binary, {num_domains} domains)...")
    model = DeBERTaDANN(
        model_name=BASE_MODEL,
        num_labels=2,
        num_domains=num_domains,
    )
    model.encoder.gradient_checkpointing_enable()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {n_params:,} total parameters ({n_trainable:,} trainable)")
    print(f"  Gradient checkpointing: ON")
    print(f"  DANN: {'ON' if model.use_dann else 'OFF'}")

    # ── Training args ─────────────────────────────────────────────────

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    effective_batch = BATCH_SIZE * GRAD_ACCUM
    steps_per_epoch = len(train_ds) // effective_batch
    total_steps = steps_per_epoch * EPOCHS

    print(f"\nTraining config:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Effective batch: {BATCH_SIZE} x {GRAD_ACCUM} = {effective_batch}")
    print(f"  LR: {LR}, Warmup: {WARMUP_RATIO}")
    print(f"  Max length: {MAX_LEN}")
    print(f"  Steps/epoch: ~{steps_per_epoch}, Total: ~{total_steps}")
    if model.use_dann:
        print(f"  DANN lambda ramp: 0 -> {DANN_LAMBDA_MAX} over {total_steps} steps")

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
        remove_unused_columns=False,
    )

    def compute_metrics(pred_output):
        logits, labels = pred_output
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    trainer = DANNTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        model_ref=model,
        total_steps=total_steps,
    )

    # ── Train ─────────────────────────────────────────────────────────

    print(f"\nStarting training...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # ── Evaluate at T=1.0 ────────────────────────────────────────────

    metrics = trainer.evaluate()
    print(f"\nFinal eval accuracy (T=1.0): {metrics.get('eval_accuracy', 0):.4f}")

    preds_output = trainer.predict(val_ds)
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    y_true = preds_output.label_ids

    print("\nClassification report (T=1.0):")
    print(classification_report(
        y_true, y_pred, target_names=LABEL_NAMES_V6, digits=3
    ))
    print(f"Binary accuracy (T=1.0): {accuracy_score(y_true, y_pred):.4f}")

    # ── Evaluate at T=2.0 (production config) ────────────────────────

    print("\n--- Temperature=2.0 evaluation (production config) ---")
    model_eval = model.to(device).float()
    preds_t2, probs_t2, labels_t2 = evaluate_with_temperature(
        model_eval, val_ds_raw, tokenizer, device, temperature=2.0
    )
    print(f"\nClassification report (T=2.0):")
    print(classification_report(
        labels_t2, preds_t2, target_names=LABEL_NAMES_V6, digits=3
    ))
    print(f"Binary accuracy (T=2.0): {accuracy_score(labels_t2, preds_t2):.4f}")

    # ── Save model ────────────────────────────────────────────────────

    # Save as HuggingFace-compatible format
    model.encoder.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save task classifier weights separately
    torch.save(
        {
            "task_classifier": model.task_classifier.state_dict(),
            "dropout": model.dropout.state_dict(),
            "num_labels": model.num_labels,
            "num_domains": model.num_domains,
            "use_dann": model.use_dann,
            "model_name": BASE_MODEL,
        },
        os.path.join(OUTPUT_DIR, "classifier_head.pt"),
    )

    # Also save a simple AutoModelForSequenceClassification-compatible version
    # by loading the encoder and copying the classifier weights
    from transformers import AutoModelForSequenceClassification

    compat_model = AutoModelForSequenceClassification.from_pretrained(
        OUTPUT_DIR, num_labels=2
    )
    # Copy trained classifier weights
    with torch.no_grad():
        compat_model.classifier.weight.copy_(model.task_classifier.weight)
        compat_model.classifier.bias.copy_(model.task_classifier.bias)
    compat_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nModel saved to {OUTPUT_DIR}")
    print(f"  - HuggingFace compatible (AutoModelForSequenceClassification, 2 labels)")
    print(f"  - classifier_head.pt (DANN heads for further training)")

    # ── Sanity tests ──────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("Sanity Tests")
    print(f"{'='*70}")

    # Reload as standard model for sanity tests (simpler inference)
    test_model = AutoModelForSequenceClassification.from_pretrained(
        OUTPUT_DIR, num_labels=2
    ).to(device).float()
    test_model.eval()

    # Wrap to return dict format
    class SimpleWrapper(nn.Module):
        def __init__(self, hf_model):
            super().__init__()
            self.model = hf_model

        def forward(self, **kwargs):
            out = self.model(**kwargs)
            return {"logits": out.logits}

    wrapped = SimpleWrapper(test_model).to(device)
    wrapped.eval()
    run_sanity_tests(wrapped, tokenizer, device, temperatures=[1.0, 2.0])

    # ── Package ───────────────────────────────────────────────────────

    tar_path = os.path.join(WORKSPACE, "detector_v6.tar.gz")
    print(f"\nPackaging {OUTPUT_DIR} -> {tar_path}")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(OUTPUT_DIR, arcname="detector_v6")
    tar_size = os.path.getsize(tar_path) / (1024 * 1024 * 1024)
    print(f"  Archive size: {tar_size:.2f} GB")

    # ── Done ──────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"DONE. Model saved to {OUTPUT_DIR}")
    print(f"Archive: {tar_path}")
    print(f"Training time: {elapsed / 60:.1f} min")
    print(f"\nTo load in production:")
    print(f"  from transformers import AutoModelForSequenceClassification, AutoTokenizer")
    print(f"  model = AutoModelForSequenceClassification.from_pretrained('{OUTPUT_DIR}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{OUTPUT_DIR}')")
    print(f"  # Binary: label 0 = human, label 1 = AI")
    print(f"  # Use temperature=2.0 for production softmax")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
