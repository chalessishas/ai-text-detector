#!/usr/bin/env python3
"""Build a FAISS index from human-written text sources.

Usage:
    python3 scripts/build_corpus.py [--corpus-dir corpus]

Reads all .txt files from corpus/raw/, splits into sentences,
computes embeddings, and saves a FAISS IVF+PQ index + metadata.

For <1M sentences, uses flat index. For 1M+, uses IVF+PQ (~30x smaller).
"""

import argparse
import json
import os
import re
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def is_clean_sentence(s: str) -> bool:
    """Strict quality filter for corpus sentences."""
    s = s.strip()
    words = s.split()

    # Length: 10-40 words (not too short, not too long)
    if not (10 <= len(words) <= 40):
        return False

    # Must start with uppercase letter (real sentence, not fragment)
    if not s[0].isupper():
        return False

    # Must end with proper punctuation
    if not re.search(r'[.!?]"?$', s):
        return False

    # No markup/noise characters
    if any(c in s for c in '|\\@#<>{}'):
        return False

    # No URLs or emails
    if re.search(r'https?://|www\.|\.com|\.org|\S+@\S+', s):
        return False

    # Not too many CAPS (skip headlines)
    if sum(1 for c in s if c.isupper()) > len(s) * 0.3:
        return False

    # Max 2 parentheses (skip noisy citations/markup)
    if s.count('(') > 2 or s.count('[') > 1:
        return False

    # Must have lowercase words (not a headline)
    if not any(c.islower() for c in s[:20]):
        return False

    # Opening quote fragments: skip sentences like '"We are not going...'
    if s.startswith('"') or s.startswith("'"):
        return False

    # Must contain a verb (auxiliary or common verb form)
    verb_hints = {
        'is', 'are', 'was', 'were', 'has', 'have', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'can', 'may', 'might', 'been', 'being', 'became', 'become',
        'said', 'told', 'made', 'found', 'showed', 'led', 'took',
    }
    if not any(w.lower() in verb_hints for w in words):
        # Also accept words ending in -ed, -ing as verb indicators
        if not any(w.endswith(('ed', 'ing')) and len(w) > 4 for w in words):
            return False

    # Vocabulary diversity: at least 60% unique words
    unique_ratio = len(set(w.lower() for w in words)) / len(words)
    if unique_ratio < 0.5:
        return False

    # Must have at least 2 commas or be short — filters out run-on fragments
    # (long sentences without commas are usually fragments pasted together)
    if len(words) > 25 and s.count(',') == 0:
        return False

    return True


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping only clean ones."""
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if is_clean_sentence(s)]


def load_raw_texts(raw_dir: str) -> list[str]:
    """Load all .txt files from the raw directory."""
    if not os.path.exists(raw_dir):
        print(f"No raw text directory found at: {raw_dir}", file=sys.stderr)
        print("Create corpus/raw/ and add .txt files with human-written text.", file=sys.stderr)
        sys.exit(1)

    texts = []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith('.txt'):
            continue
        with open(os.path.join(raw_dir, fname), 'r', encoding='utf-8', errors='ignore') as f:
            texts.append(f.read())
        print(f"  Loaded {fname}", file=sys.stderr)

    if not texts:
        print("No .txt files found in corpus/raw/", file=sys.stderr)
        sys.exit(1)

    return texts


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build appropriate FAISS index based on corpus size."""
    n, dim = embeddings.shape

    if n < 500_000:
        # Small corpus: flat index (exact search)
        print(f"  Using flat index (exact search, {n} vectors)", file=sys.stderr)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    # Large corpus: IVF + PQ for memory efficiency
    # IVF partitions: sqrt(n) is a good heuristic, capped at 4096
    nlist = min(4096, int(np.sqrt(n)))
    # PQ subquantizers: dim must be divisible by m
    m = 48  # 384 / 48 = 8 bytes per sub-vector
    nbits = 8

    print(f"  Using IVF{nlist}+PQ{m} index ({n} vectors, ~{n * m // 1_000_000}MB)", file=sys.stderr)

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)

    # Train on a sample if corpus is very large
    train_size = min(n, 500_000)
    if train_size < n:
        indices = np.random.choice(n, train_size, replace=False)
        train_data = embeddings[indices]
    else:
        train_data = embeddings

    print(f"  Training index on {train_size} vectors...", file=sys.stderr)
    index.train(train_data)

    print(f"  Adding {n} vectors...", file=sys.stderr)
    # Add in batches to show progress
    batch = 100_000
    for i in range(0, n, batch):
        end = min(i + batch, n)
        index.add(embeddings[i:end])
        print(f"    {end}/{n}", file=sys.stderr)

    # Search more partitions for better recall
    index.nprobe = min(64, nlist // 4)

    return index


def main():
    parser = argparse.ArgumentParser(description="Build FAISS corpus index")
    parser.add_argument('--corpus-dir', default='corpus', help='Corpus directory')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Sentence transformer model')
    parser.add_argument('--batch-size', type=int, default=512, help='Encoding batch size')
    args = parser.parse_args()

    raw_dir = os.path.join(args.corpus_dir, 'raw')
    out_dir = args.corpus_dir

    print("Loading raw texts...", file=sys.stderr)
    texts = load_raw_texts(raw_dir)

    print("Splitting + filtering sentences...", file=sys.stderr)
    all_sentences = []
    for text in texts:
        all_sentences.extend(split_sentences(text))

    # Deduplicate
    all_sentences = list(dict.fromkeys(all_sentences))
    print(f"  {len(all_sentences)} unique clean sentences", file=sys.stderr)

    if len(all_sentences) == 0:
        print("No valid sentences found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {args.model}...", file=sys.stderr)
    model = SentenceTransformer(args.model)

    print("Computing embeddings...", file=sys.stderr)
    embeddings = model.encode(
        all_sentences,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    print(f"Building FAISS index ({embeddings.shape})...", file=sys.stderr)
    index = build_index(embeddings)

    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, 'sentences.faiss')
    meta_path = os.path.join(out_dir, 'sentences.jsonl')

    faiss.write_index(index, index_path)
    with open(meta_path, 'w', encoding='utf-8') as f:
        for s in all_sentences:
            f.write(json.dumps({"text": s}, ensure_ascii=False) + '\n')

    index_size = os.path.getsize(index_path) / (1024 * 1024)
    print(f"\nDone! Index: {index_path} ({len(all_sentences)} sentences, {index_size:.0f}MB)", file=sys.stderr)
    print(f"Metadata: {meta_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
