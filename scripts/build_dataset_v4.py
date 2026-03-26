#!/usr/bin/env python3
"""Build balanced dataset_v4.jsonl from existing data.

Takes the best available data and creates a balanced 4-class dataset:
- Downsample over-represented human class from dataset_v3
- Ensure equal class distribution
- Add quality filters (min word count, dedup)

Run: python3 scripts/build_dataset_v4.py
"""

import json
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path

SEED = 42
MIN_WORDS = 200
MAX_WORDS = 2000
PROJECT_DIR = Path(__file__).parent.parent

# Sources in priority order
SOURCES = [
    PROJECT_DIR / "dataset_v3.jsonl",
    PROJECT_DIR / "dataset.jsonl",
    PROJECT_DIR / "dataset_augmented.jsonl",
    PROJECT_DIR / "dataset_raid_extract.jsonl",
]

OUTPUT = PROJECT_DIR / "dataset_v4.jsonl"


def text_hash(text):
    return hashlib.md5(text[:500].encode()).hexdigest()


def load_all_sources():
    """Load and deduplicate samples from all sources."""
    seen = set()
    by_label = defaultdict(list)

    for src in SOURCES:
        if not src.exists():
            print(f"  SKIP: {src.name} (not found)", file=sys.stderr)
            continue
        count = 0
        with open(src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                text = d.get("text", "")
                label = d.get("label")
                if label is None or label not in (0, 1, 2, 3):
                    continue

                word_count = len(text.split())
                if word_count < MIN_WORDS or word_count > MAX_WORDS:
                    continue

                h = text_hash(text)
                if h in seen:
                    continue
                seen.add(h)

                by_label[label].append(d)
                count += 1
        print(f"  {src.name}: +{count} unique samples", file=sys.stderr)

    return by_label


def main():
    random.seed(SEED)
    print("Building balanced dataset_v4.jsonl...", file=sys.stderr)

    by_label = load_all_sources()
    label_names = {0: "human", 1: "ai", 2: "ai_polished", 3: "human_polished"}

    print(f"\nRaw counts after dedup:", file=sys.stderr)
    for label in sorted(by_label):
        print(f"  {label_names[label]}: {len(by_label[label])}", file=sys.stderr)

    # Balance: target = min class size (so all classes equal)
    min_count = min(len(v) for v in by_label.values())
    target = min(min_count, 20000)  # cap at 20K per class

    print(f"\nTarget per class: {target}", file=sys.stderr)

    # Shuffle and sample
    output = []
    for label in sorted(by_label):
        samples = by_label[label]
        random.shuffle(samples)
        selected = samples[:target]
        output.extend(selected)
        print(f"  {label_names[label]}: {len(selected)}", file=sys.stderr)

    random.shuffle(output)

    # Write
    with open(OUTPUT, "w") as f:
        for entry in output:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total = len(output)
    print(f"\nWrote {total} samples to {OUTPUT}", file=sys.stderr)
    print(f"  ({target} × 4 classes = {target * 4})", file=sys.stderr)


if __name__ == "__main__":
    main()
