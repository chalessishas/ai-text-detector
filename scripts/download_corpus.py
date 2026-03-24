#!/usr/bin/env python3
"""Download human-written text at scale for the humanizer corpus.

Sources (all pre-2020, verified human-written):
  1. C4 (Colossal Clean Crawled Corpus) — diverse web text, April 2019
  2. CNN/DailyMail — formal news articles, pre-2015
  3. Wikipedia (wikitext-103) — encyclopedic text, 2017

Usage:
    pip install datasets
    python3 scripts/download_corpus.py [--target 10000000]
"""

import argparse
import os
import re
import sys

from datasets import load_dataset


def clean_sentences(text: str) -> list[str]:
    """Split and filter to clean, well-formed sentences."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'@-@', '-', text)
    text = re.sub(r'\s+', ' ', text).strip()

    raw = re.split(r'(?<=[.!?])\s+', text)
    out = []
    for s in raw:
        s = s.strip()
        words = s.split()
        if not (10 <= len(words) <= 40):
            continue
        if not s[0:1].isupper():
            continue
        if not re.search(r'[.!?]"?$', s):
            continue
        if any(c in s for c in '|\\@#<>{}'):
            continue
        if s.startswith('"') or s.startswith("'"):
            continue
        if not any(c.islower() for c in s[:20]):
            continue
        if re.search(r'https?://|www\.|\.com\b|\.org\b', s):
            continue
        out.append(s)
    return out


def download_c4(target: int, output_dir: str) -> int:
    """C4: diverse web text from April 2019 Common Crawl."""
    path = os.path.join(output_dir, 'c4.txt')
    print(f"[1/3] C4 web text (target: {target})...", file=sys.stderr)

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

    count = 0
    with open(path, 'w', encoding='utf-8') as f:
        for row in ds:
            sents = clean_sentences(row['text'])
            for s in sents:
                f.write(s + '\n')
                count += 1
                if count % 500_000 == 0:
                    print(f"       {count:,} sentences...", file=sys.stderr)
            if count >= target:
                break

    print(f"       Wrote {path} ({count:,} sentences)", file=sys.stderr)
    return count


def download_cnn(target: int, output_dir: str) -> int:
    """CNN/DailyMail: formal journalism."""
    path = os.path.join(output_dir, 'cnn_dailymail.txt')
    if os.path.exists(path):
        existing = sum(1 for _ in open(path))
        if existing >= target:
            print(f"[2/3] CNN/DailyMail: already have {existing:,} sentences, skipping", file=sys.stderr)
            return existing

    print(f"[2/3] CNN/DailyMail (target: {target})...", file=sys.stderr)
    count = 0
    with open(path, 'w', encoding='utf-8') as f:
        for split in ['train', 'validation', 'test']:
            ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=split)
            for row in ds:
                for s in clean_sentences(row['article']):
                    f.write(s + '\n')
                    count += 1
                if count >= target:
                    break
            if count >= target:
                break

    print(f"       Wrote {path} ({count:,} sentences)", file=sys.stderr)
    return count


def download_wikipedia(target: int, output_dir: str) -> int:
    """Wikipedia: encyclopedic writing."""
    path = os.path.join(output_dir, 'wikipedia.txt')
    print(f"[3/3] Wikipedia (target: {target})...", file=sys.stderr)

    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    count = 0
    with open(path, 'w', encoding='utf-8') as f:
        for row in ds:
            text = row['text'].strip()
            if not text or text.startswith('='):
                continue
            for s in clean_sentences(text):
                f.write(s + '\n')
                count += 1
            if count >= target:
                break

    print(f"       Wrote {path} ({count:,} sentences)", file=sys.stderr)
    return count


def main():
    parser = argparse.ArgumentParser(description="Download corpus data")
    parser.add_argument('--output-dir', default='corpus/raw')
    parser.add_argument('--target', type=int, default=10_000_000,
                        help='Target total sentences')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Remove old low-quality sources
    for old in ['ag_news.txt', 'imdb_reviews.txt', 'arxiv_abstracts.txt', 'eli5_reddit.txt']:
        old_path = os.path.join(args.output_dir, old)
        if os.path.exists(old_path):
            os.remove(old_path)
            print(f"Removed old source: {old}", file=sys.stderr)

    # Distribution: 70% C4 (diverse), 20% CNN (news), 10% Wikipedia (encyclopedic)
    c4_target = int(args.target * 0.7)
    cnn_target = int(args.target * 0.2)
    wiki_target = int(args.target * 0.1)

    total = 0
    total += download_c4(c4_target, args.output_dir)
    total += download_cnn(cnn_target, args.output_dir)
    total += download_wikipedia(wiki_target, args.output_dir)

    print(f"\nTotal: {total:,} sentences.", file=sys.stderr)
    print("Now run: python3 scripts/build_corpus.py", file=sys.stderr)


if __name__ == '__main__':
    main()
