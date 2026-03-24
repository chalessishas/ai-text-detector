#!/usr/bin/env python3
"""Merge RAID + MAGE + our dataset into a unified training corpus.

Handles label remapping, deduplication, length filtering, and stratified sampling.

Usage:
    /opt/anaconda3/bin/python3.13 scripts/merge_datasets.py
    /opt/anaconda3/bin/python3.13 scripts/merge_datasets.py --max-per-source 20000
    /opt/anaconda3/bin/python3.13 scripts/merge_datasets.py --output merged_dataset.jsonl

Output format (one JSON per line):
    {"text": "...", "label": 0-4, "source": "raid|mage|ours", "meta": {...}}
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
OUR_DATASET = PROJECT_DIR / "dataset.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "merged_dataset.jsonl"

# Label schema: ours
# 0=human, 1=ai, 2=ai_polished, 3=human_polished, 4=ai_humanized
LABEL_NAMES = {0: "human", 1: "ai", 2: "ai_polished", 3: "human_polished", 4: "ai_humanized"}

# Length filter (in tokens, approximated by whitespace split)
MIN_WORDS = 30
MAX_WORDS = 2000


def text_hash(text):
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


def load_our_dataset():
    """Load our existing dataset.jsonl."""
    if not OUR_DATASET.exists():
        print(f"  Warning: {OUR_DATASET} not found, skipping")
        return []

    samples = []
    with open(OUR_DATASET) as f:
        for line in f:
            d = json.loads(line)
            text = d.get("text", "").strip()
            if not text:
                continue
            samples.append({
                "text": text,
                "label": d["label"],
                "source": "ours",
                "meta": {
                    "model": d.get("model", ""),
                    "style": d.get("style", ""),
                    "topic": d.get("topic", ""),
                },
            })
    return samples


def load_mage(max_samples=50000):
    """Load MAGE dataset from HuggingFace. Label remap: MAGE 0=machine->1, 1=human->0."""
    from datasets import load_dataset

    print(f"  Loading MAGE (max {max_samples} per class)...")
    ds = load_dataset("yaful/MAGE", split="train", streaming=True)

    samples = []
    class_counts = Counter()

    for item in ds:
        # MAGE labels: 0=machine, 1=human (inverted from ours)
        mage_label = item["label"]
        our_label = 0 if mage_label == 1 else 1  # remap

        if class_counts[our_label] >= max_samples:
            if all(c >= max_samples for c in class_counts.values()):
                break
            continue

        text = item.get("text", "").strip()
        word_count = len(text.split())
        if word_count < MIN_WORDS or word_count > MAX_WORDS:
            continue

        samples.append({
            "text": text,
            "label": our_label,
            "source": "mage",
            "meta": {"src": item.get("src", "")},
        })
        class_counts[our_label] += 1

    return samples


def load_raid(max_samples=50000):
    """Load RAID dataset from HuggingFace. model='human' -> label 0, else -> label 1."""
    from datasets import load_dataset

    print(f"  Loading RAID (max {max_samples} per class, streaming)...")

    # RAID is huge — stream and sample strategically
    ds = load_dataset("liamdugan/raid", split="train", streaming=True)

    samples = []
    class_counts = Counter()
    domain_counts = Counter()
    attack_counts = Counter()

    for item in ds:
        is_human = item["model"] == "human"
        our_label = 0 if is_human else 1

        if class_counts[our_label] >= max_samples:
            if all(c >= max_samples for c in class_counts.values()):
                break
            continue

        text = item.get("generation", "").strip()
        word_count = len(text.split())
        if word_count < MIN_WORDS or word_count > MAX_WORDS:
            continue

        domain = item.get("domain", "unknown")
        attack = item.get("attack", "none")

        # Adversarial samples (attack != "none") go to label 4 (ai_humanized)
        if not is_human and attack != "none":
            our_label = 4

        samples.append({
            "text": text,
            "label": our_label,
            "source": "raid",
            "meta": {
                "model": item.get("model", ""),
                "domain": domain,
                "attack": attack,
                "decoding": item.get("decoding", ""),
            },
        })
        class_counts[our_label] += 1
        domain_counts[domain] += 1
        attack_counts[attack] += 1

    print(f"  RAID domains: {dict(domain_counts.most_common(10))}")
    print(f"  RAID attacks: {dict(attack_counts.most_common(10))}")
    return samples


def deduplicate(samples):
    """Remove exact duplicates by text hash."""
    seen = set()
    deduped = []
    for s in samples:
        h = text_hash(s["text"])
        if h not in seen:
            seen.add(h)
            deduped.append(s)
    removed = len(samples) - len(deduped)
    if removed:
        print(f"  Dedup removed {removed} duplicates ({removed/len(samples)*100:.1f}%)")
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Merge RAID + MAGE + our dataset")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-per-source", type=int, default=30000,
                        help="Max samples per class per source dataset")
    parser.add_argument("--skip-raid", action="store_true", help="Skip RAID (slow to download)")
    parser.add_argument("--skip-mage", action="store_true", help="Skip MAGE")
    args = parser.parse_args()

    print("=" * 60)
    print("Dataset Merger: RAID + MAGE + Ours")
    print("=" * 60)

    all_samples = []

    # 1. Our dataset
    print("\n[1/3] Loading our dataset...")
    ours = load_our_dataset()
    print(f"  Loaded {len(ours)} samples")
    all_samples.extend(ours)

    # 2. MAGE
    if not args.skip_mage:
        print("\n[2/3] Loading MAGE...")
        t0 = time.time()
        mage = load_mage(max_samples=args.max_per_source)
        print(f"  Loaded {len(mage)} samples in {time.time()-t0:.0f}s")
        all_samples.extend(mage)
    else:
        print("\n[2/3] Skipping MAGE")

    # 3. RAID
    if not args.skip_raid:
        print("\n[3/3] Loading RAID...")
        t0 = time.time()
        raid = load_raid(max_samples=args.max_per_source)
        print(f"  Loaded {len(raid)} samples in {time.time()-t0:.0f}s")
        all_samples.extend(raid)
    else:
        print("\n[3/3] Skipping RAID")

    # Deduplicate
    print(f"\n--- Deduplication ---")
    print(f"  Before: {len(all_samples)}")
    all_samples = deduplicate(all_samples)
    print(f"  After:  {len(all_samples)}")

    # Stats
    print(f"\n--- Final Distribution ---")
    label_counts = Counter(s["label"] for s in all_samples)
    source_counts = Counter(s["source"] for s in all_samples)
    for label_id in sorted(label_counts):
        name = LABEL_NAMES.get(label_id, f"unknown_{label_id}")
        print(f"  Label {label_id} ({name}): {label_counts[label_id]}")
    print()
    for source, count in source_counts.most_common():
        print(f"  Source {source}: {count}")

    # Write
    output_path = Path(args.output)
    print(f"\n--- Writing to {output_path} ---")
    with open(output_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Written {len(all_samples)} samples ({size_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
