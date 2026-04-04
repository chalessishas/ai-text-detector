#!/usr/bin/env python3
"""Expand OOD dataset for XGBoost meta-learner retraining.

Downloads diverse human + AI text samples from multiple HuggingFace datasets,
deduplicates against existing OOD data and training data (dataset_v4.jsonl),
and outputs in the same JSONL format.

Target: 250+ human + 250+ AI samples across diverse domains.

Datasets used:
  - Hello-SimpleAI/HC3: human expert answers vs ChatGPT (finance, medicine, QA, Reddit)
  - liamdugan/raid: 11 LLMs × 11 genres (news, books, Reddit, Wikipedia, recipes, etc.)
  - artem9k/ai-text-detection-pile: essays from human + GPT2/GPT3/ChatGPT/GPTJ
  - andythetechnerd03/AI-human-text: ~400k rows human vs AI text

Usage:
    pip install datasets
    python3 scripts/expand_ood_data.py [--output data_ood_xgboost.jsonl] [--dry-run]

Note: This script only downloads and formats text. It does NOT compute features.
      After running, use train_xgboost_fusion.py with the detection server to retrain.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent

# Minimum text length (chars) to keep — short texts have unreliable PPL/stats
MIN_TEXT_LENGTH = 200
# Maximum text length — truncate to avoid server timeout
MAX_TEXT_LENGTH = 5000

# Target counts per label
TARGET_HUMAN = 275
TARGET_AI = 275


def load_existing_hashes(paths: list[str]) -> set[str]:
    """Load text hashes from existing JSONL files to avoid duplicates."""
    hashes = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if text:
                        h = hashlib.md5(text[:500].strip().lower().encode()).hexdigest()
                        hashes.add(h)
                except json.JSONDecodeError:
                    continue
    return hashes


def text_hash(text: str) -> str:
    return hashlib.md5(text[:500].strip().lower().encode()).hexdigest()


def is_valid_text(text: str) -> bool:
    """Filter out garbage: too short, non-English, or mostly whitespace."""
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return False
    # Skip texts that are mostly non-ASCII (likely non-English)
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
    if ascii_ratio < 0.85:
        return False
    # Skip texts with too many repeated characters or lines
    lines = text.strip().split("\n")
    if len(lines) > 1:
        unique_lines = set(l.strip() for l in lines if l.strip())
        if len(unique_lines) < len(lines) * 0.5:
            return False
    return True


def truncate(text: str) -> str:
    """Truncate to MAX_TEXT_LENGTH at a sentence boundary if possible."""
    text = text.strip()
    if len(text) <= MAX_TEXT_LENGTH:
        return text
    truncated = text[:MAX_TEXT_LENGTH]
    # Try to cut at last sentence boundary
    for sep in [". ", ".\n", "! ", "? "]:
        idx = truncated.rfind(sep)
        if idx > MAX_TEXT_LENGTH * 0.7:
            return truncated[:idx + 1]
    return truncated


# ── Dataset loaders ──


def load_hc3(seen: set[str]) -> tuple[list[dict], list[dict]]:
    """HC3: Human vs ChatGPT comparison corpus.

    Columns: question, human_answers (list[str]), chatgpt_answers (list[str])
    Domains: finance, medicine, open_qa, wiki_csai, reddit_eli5
    """
    from datasets import load_dataset

    human_samples, ai_samples = [], []
    domains = ["finance", "medicine", "open_qa", "wiki_csai", "reddit_eli5"]

    for domain in domains:
        print(f"  HC3/{domain}...", end=" ", flush=True)
        try:
            ds = load_dataset("Hello-SimpleAI/HC3", domain, split="train", trust_remote_code=True)
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        count_h, count_a = 0, 0
        for row in ds:
            # Human answers
            for answer in (row.get("human_answers") or []):
                if not is_valid_text(answer):
                    continue
                h = text_hash(answer)
                if h in seen:
                    continue
                seen.add(h)
                human_samples.append({
                    "text": truncate(answer),
                    "label": 0,
                    "source": f"hf_hc3_{domain}",
                    "domain": domain,
                })
                count_h += 1

            # ChatGPT answers
            for answer in (row.get("chatgpt_answers") or []):
                if not is_valid_text(answer):
                    continue
                h = text_hash(answer)
                if h in seen:
                    continue
                seen.add(h)
                ai_samples.append({
                    "text": truncate(answer),
                    "label": 1,
                    "source": f"hf_hc3_{domain}",
                    "domain": domain,
                    "model": "chatgpt",
                })
                count_a += 1

        print(f"{count_h} human, {count_a} AI")

    return human_samples, ai_samples


def load_raid(seen: set[str]) -> tuple[list[dict], list[dict]]:
    """RAID: 11 LLMs x 11 genres, largest detection benchmark.

    Columns: id, generation, model, domain, attack, decoding, ...
    Human texts have model=None or model="human".
    """
    from datasets import load_dataset

    human_samples, ai_samples = [], []

    print("  RAID/train...", end=" ", flush=True)
    try:
        ds = load_dataset("liamdugan/raid", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"SKIP ({e})")
        return [], []

    # Shuffle and sample — RAID is huge (10M+), don't load all
    indices = list(range(len(ds)))
    random.shuffle(indices)

    # Take up to 5000 candidates to filter from
    count_h, count_a = 0, 0
    target_per_label = 80  # per-source cap to ensure diversity

    for idx in indices[:10000]:
        row = ds[idx]
        text = row.get("generation") or row.get("text") or ""
        if not is_valid_text(text):
            continue

        h = text_hash(text)
        if h in seen:
            continue

        model = row.get("model") or ""
        domain = row.get("domain") or "unknown"
        attack = row.get("attack") or ""

        # Skip adversarial attacks — we want clean OOD data
        if attack and attack != "none":
            continue

        seen.add(h)

        if model == "" or model.lower() == "human":
            if count_h >= target_per_label:
                continue
            human_samples.append({
                "text": truncate(text),
                "label": 0,
                "source": "hf_raid",
                "domain": f"raid_{domain}",
            })
            count_h += 1
        else:
            if count_a >= target_per_label:
                continue
            ai_samples.append({
                "text": truncate(text),
                "label": 1,
                "source": "hf_raid",
                "domain": f"raid_{domain}",
                "model": model,
            })
            count_a += 1

        if count_h >= target_per_label and count_a >= target_per_label:
            break

    print(f"{count_h} human, {count_a} AI")
    return human_samples, ai_samples


def load_detection_pile(seen: set[str]) -> tuple[list[dict], list[dict]]:
    """artem9k/ai-text-detection-pile: essays from human + GPT variants.

    Columns: text, label (0=human, 1=ai), source
    """
    from datasets import load_dataset

    human_samples, ai_samples = [], []

    print("  ai-text-detection-pile...", end=" ", flush=True)
    try:
        ds = load_dataset("artem9k/ai-text-detection-pile", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"SKIP ({e})")
        return [], []

    indices = list(range(len(ds)))
    random.shuffle(indices)

    count_h, count_a = 0, 0
    target_per_label = 60

    for idx in indices[:5000]:
        row = ds[idx]
        text = row.get("text") or ""
        if not is_valid_text(text):
            continue

        h = text_hash(text)
        if h in seen:
            continue
        seen.add(h)

        label = row.get("label")
        source_name = row.get("source") or "detection_pile"

        if label == 0 or (isinstance(label, str) and label.lower() == "human"):
            if count_h >= target_per_label:
                continue
            human_samples.append({
                "text": truncate(text),
                "label": 0,
                "source": f"hf_detection_pile_{source_name}",
                "domain": "essay",
            })
            count_h += 1
        elif label == 1 or (isinstance(label, str) and label.lower() in ("ai", "generated")):
            if count_a >= target_per_label:
                continue
            ai_samples.append({
                "text": truncate(text),
                "label": 1,
                "source": f"hf_detection_pile_{source_name}",
                "domain": "essay",
                "model": source_name if source_name != "human" else "unknown_ai",
            })
            count_a += 1

        if count_h >= target_per_label and count_a >= target_per_label:
            break

    print(f"{count_h} human, {count_a} AI")
    return human_samples, ai_samples


def load_ai_human_text(seen: set[str]) -> tuple[list[dict], list[dict]]:
    """andythetechnerd03/AI-human-text: ~400k rows, binary label.

    Columns: text, generated (0=human, 1=AI)
    """
    from datasets import load_dataset

    human_samples, ai_samples = [], []

    print("  AI-human-text...", end=" ", flush=True)
    try:
        ds = load_dataset("andythetechnerd03/AI-human-text", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"SKIP ({e})")
        return [], []

    indices = list(range(len(ds)))
    random.shuffle(indices)

    count_h, count_a = 0, 0
    target_per_label = 60

    for idx in indices[:8000]:
        row = ds[idx]
        text = row.get("text") or ""
        if not is_valid_text(text):
            continue

        h = text_hash(text)
        if h in seen:
            continue
        seen.add(h)

        # Column might be "generated" or "label"
        label = row.get("generated", row.get("label"))

        if label == 0 or label == "human":
            if count_h >= target_per_label:
                continue
            human_samples.append({
                "text": truncate(text),
                "label": 0,
                "source": "hf_ai_human_text",
                "domain": "mixed",
            })
            count_h += 1
        elif label == 1 or label == "ai":
            if count_a >= target_per_label:
                continue
            ai_samples.append({
                "text": truncate(text),
                "label": 1,
                "source": "hf_ai_human_text",
                "domain": "mixed",
            })
            count_a += 1

        if count_h >= target_per_label and count_a >= target_per_label:
            break

    print(f"{count_h} human, {count_a} AI")
    return human_samples, ai_samples


def load_dmitva(seen: set[str]) -> tuple[list[dict], list[dict]]:
    """dmitva/human_ai_generated_text: diverse human vs AI texts.

    Fallback dataset for additional diversity.
    """
    from datasets import load_dataset

    human_samples, ai_samples = [], []

    print("  dmitva/human_ai_generated_text...", end=" ", flush=True)
    try:
        ds = load_dataset("dmitva/human_ai_generated_text", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"SKIP ({e})")
        return [], []

    indices = list(range(len(ds)))
    random.shuffle(indices)

    count_h, count_a = 0, 0
    target_per_label = 40

    for idx in indices[:5000]:
        row = ds[idx]
        text = row.get("text") or ""
        if not is_valid_text(text):
            continue

        h = text_hash(text)
        if h in seen:
            continue
        seen.add(h)

        # Try common column names for labels
        label = row.get("label", row.get("generated", row.get("is_ai")))

        if label in (0, "human", "Human", False):
            if count_h >= target_per_label:
                continue
            human_samples.append({
                "text": truncate(text),
                "label": 0,
                "source": "hf_dmitva",
                "domain": "mixed",
            })
            count_h += 1
        elif label in (1, "ai", "AI", "generated", True):
            if count_a >= target_per_label:
                continue
            ai_samples.append({
                "text": truncate(text),
                "label": 1,
                "source": "hf_dmitva",
                "domain": "mixed",
            })
            count_a += 1

        if count_h >= target_per_label and count_a >= target_per_label:
            break

    print(f"{count_h} human, {count_a} AI")
    return human_samples, ai_samples


# ── Main ──


def main():
    parser = argparse.ArgumentParser(description="Expand OOD dataset from HuggingFace")
    parser.add_argument("--output", default=str(PROJECT_DIR / "data_ood_xgboost.jsonl"))
    parser.add_argument("--dry-run", action="store_true", help="Print stats but don't write")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("OOD Dataset Expansion for XGBoost Meta-Learner")
    print("=" * 60)

    # Load existing data hashes (OOD + training) to deduplicate
    existing_ood_path = args.output
    training_path = str(PROJECT_DIR / "dataset_v4.jsonl")

    print(f"\nLoading existing hashes for deduplication...")
    seen = load_existing_hashes([existing_ood_path, training_path])
    print(f"  {len(seen)} existing text hashes loaded")

    # Load existing OOD samples (we keep them all)
    existing_samples = []
    if os.path.exists(existing_ood_path):
        with open(existing_ood_path) as f:
            for line in f:
                try:
                    existing_samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    existing_h = sum(1 for s in existing_samples if s.get("label") == 0)
    existing_a = sum(1 for s in existing_samples if s.get("label") == 1)
    print(f"  Existing OOD: {existing_h} human + {existing_a} AI = {len(existing_samples)} total")

    # Collect new samples from each HuggingFace dataset
    print(f"\nDownloading from HuggingFace datasets...")
    all_human, all_ai = [], []

    loaders = [
        ("HC3", load_hc3),
        ("RAID", load_raid),
        ("Detection Pile", load_detection_pile),
        ("AI-Human-Text", load_ai_human_text),
        ("dmitva", load_dmitva),
    ]

    for name, loader in loaders:
        print(f"\n[{name}]")
        try:
            h, a = loader(seen)
            all_human.extend(h)
            all_ai.extend(a)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"New samples collected: {len(all_human)} human + {len(all_ai)} AI")

    # Calculate how many more we need
    need_human = max(0, TARGET_HUMAN - existing_h)
    need_ai = max(0, TARGET_AI - existing_a)

    # Shuffle and pick
    random.shuffle(all_human)
    random.shuffle(all_ai)

    new_human = all_human[:need_human]
    new_ai = all_ai[:need_ai]

    print(f"Adding: {len(new_human)} human + {len(new_ai)} AI")
    final = existing_samples + new_human + new_ai
    final_h = sum(1 for s in final if s.get("label") == 0)
    final_a = sum(1 for s in final if s.get("label") == 1)
    print(f"Final dataset: {final_h} human + {final_a} AI = {len(final)} total")

    # Domain distribution
    domain_dist: dict[str, int] = {}
    source_dist: dict[str, int] = {}
    for s in final:
        d = s.get("domain", "unknown")
        domain_dist[d] = domain_dist.get(d, 0) + 1
        src = s.get("source", "unknown")
        source_dist[src] = source_dist.get(src, 0) + 1

    print(f"\nSource distribution:")
    for src, count in sorted(source_dist.items(), key=lambda x: -x[1]):
        print(f"  {src:40s}: {count}")

    print(f"\nDomain distribution (top 20):")
    for dom, count in sorted(domain_dist.items(), key=lambda x: -x[1])[:20]:
        print(f"  {dom:30s}: {count}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Write output
    with open(args.output, "w") as f:
        for sample in final:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nWritten to {args.output}")
    print(f"Next step: run `python3 scripts/train_xgboost_fusion.py` with detection server running")


if __name__ == "__main__":
    main()
