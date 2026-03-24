# -*- coding: utf-8 -*-
"""build_corpus_colab.py — Build 50M Sentence Corpus + FAISS Index on A100 GPU

Architecture: Incremental chunked pipeline
  - Downloads + encodes in 1M-sentence chunks
  - Each chunk is immediately saved to disk (sentences + embeddings)
  - Dedup via hash set (~400MB RAM for 50M hashes)
  - Peak RAM: ~3GB per chunk instead of ~75GB all at once
  - Crash-safe: restart picks up from last completed chunk
  - Final step: load all embeddings from disk, build FAISS index

Colab instructions:
  1. Runtime → Change runtime type → Select A100 GPU
  2. Runtime → Run all
  3. ~8 hours total
  4. Download sentences.faiss + sentences.jsonl
  5. Put them in ai-text-detector/corpus/
"""

# ============================================================
# Step 0a: Install dependencies
# ============================================================

import subprocess
import sys

def pip_install(*packages):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *packages])

pip_install('sentence-transformers', 'datasets')

# faiss-gpu is critical — CPU fallback adds 3-4 hours
try:
    import faiss
    if faiss.get_num_gpus() > 0:
        print('faiss-gpu already installed ✓')
    else:
        raise ImportError('no GPU support')
except (ImportError, Exception):
    print('Installing faiss-gpu...')
    candidates = ['faiss-gpu-cu12', 'faiss-gpu-cu11', 'faiss-gpu']
    installed = False
    for pkg in candidates:
        try:
            print(f'  Trying {pkg}...')
            pip_install(pkg)
            if 'faiss' in sys.modules:
                del sys.modules['faiss']
            import faiss
            if faiss.get_num_gpus() > 0:
                print(f'  {pkg} installed ✓ ({faiss.get_num_gpus()} GPU)')
                installed = True
                break
            else:
                print(f'  {pkg} installed but no GPU detected, trying next...')
                subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', pkg])
        except Exception as e:
            print(f'  {pkg} failed: {e}')
    if not installed:
        print('✗ All faiss-gpu variants FAILED.')
        raise RuntimeError('faiss-gpu is required for 50M corpus build on 8h budget')

# ============================================================
# Step 0b: Environment diagnostics
# ============================================================

import time
import os
import gc
import json
import re
import zlib
import shutil

import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def run_diagnostics():
    print('=' * 60)
    print('  ENVIRONMENT DIAGNOSTICS')
    print('=' * 60)

    print('\n[GPU]')
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free_mem, total = torch.cuda.mem_get_info(0)
            print(f'  Device:       {gpu_name}')
            print(f'  VRAM:         {total_mem:.1f} GB total, {free_mem/(1024**3):.1f} GB free')
            print(f'  CUDA:         {torch.version.cuda}')
            is_a100 = 'A100' in gpu_name
            print(f'  {"✓ A100 detected" if is_a100 else "⚠ Not A100 — may be slower"}')
        else:
            print('  ✗ No CUDA GPU detected!')
            return False
    except ImportError:
        print('  ✗ PyTorch not installed')
        return False

    print('\n[DISK]')
    _, _, free = shutil.disk_usage('/')
    free_gb = free / (1024**3)
    print(f'  Free: {free_gb:.0f} GB (need ~100GB for 50M chunked pipeline)')
    if free_gb < 100:
        print(f'  ⚠ Tight on disk. Monitor usage.')

    print('\n[RAM]')
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f'  Total: {ram.total/(1024**3):.1f} GB, Free: {ram.available/(1024**3):.1f} GB')
    except ImportError:
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith(('MemTotal', 'MemAvailable')):
                        parts = line.split()
                        print(f'  {parts[0].rstrip(":")}: {int(parts[1])/(1024*1024):.1f} GB')
        except:
            pass

    print(f'\n[FAISS]')
    print(f'  GPU support: {faiss.get_num_gpus()} GPU(s)')

    print('\n' + '=' * 60)
    print('  DIAGNOSTICS COMPLETE')
    print('=' * 60)
    return True


diagnostics_ok = run_diagnostics()
if not diagnostics_ok:
    raise RuntimeError('Environment check failed')


# ============================================================
# Step 1: Config
# ============================================================

TARGET = 50_000_000
MODEL_NAME = 'all-MiniLM-L6-v2'
BATCH_SIZE = 2048
CHUNK_SIZE = 1_000_000  # Process 1M sentences at a time
DIM = 384  # all-MiniLM-L6-v2 output dimension

# Mount Google Drive for persistent storage across runtime restarts.
# Colab's /content/ is ephemeral — if the runtime disconnects and
# a new one is allocated, all local files are lost.
try:
    from google.colab import drive
    drive.mount('/content/drive')
    WORK_DIR = '/content/drive/MyDrive/corpus_build'
    print(f'Using Google Drive for persistent storage: {WORK_DIR}')
except Exception:
    WORK_DIR = '/content/corpus_build'
    print(f'⚠ Google Drive not available, using local storage (not crash-safe)')
CHUNKS_DIR = os.path.join(WORK_DIR, 'chunks')
os.makedirs(CHUNKS_DIR, exist_ok=True)

t0 = time.time()


def elapsed():
    m, s = divmod(int(time.time() - t0), 60)
    h, m = divmod(m, 60)
    return f'{h}h{m:02d}m{s:02d}s'


# ============================================================
# Step 2: Sentence quality filter
# ============================================================

def clean_sentences(text: str) -> list[str]:
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
        if re.search(r'https?://|www\.|@', s):
            continue
        if len(set(w.lower() for w in words)) / len(words) < 0.5:
            continue
        out.append(s)
    return out


def sentence_hash(s: str) -> int:
    """Fast 64-bit stable hash for dedup. ~10x faster than MD5.
    Combines two CRC32 passes (forward + reversed) into a 64-bit int
    to avoid 32-bit birthday collisions at 50M scale."""
    b = s.encode()
    hi = zlib.crc32(b) & 0xFFFFFFFF
    lo = zlib.crc32(b[::-1]) & 0xFFFFFFFF
    return (hi << 32) | lo


# ============================================================
# Step 3: Chunked download + encode pipeline
# ============================================================
# Architecture:
#   - Stream sentences from all sources into a buffer
#   - When buffer hits CHUNK_SIZE (1M), encode + save to disk
#   - Dedup via hash set (persists across chunks)
#   - Each chunk produces: chunk_XX.npy (embeddings) + chunk_XX.jsonl (sentences)
#   - Crash recovery: skip already-completed chunks

print(f'\n{"="*60}')
print(f'  CHUNKED PIPELINE: {TARGET:,} sentences in {CHUNK_SIZE:,}-sentence chunks')
print(f'{"="*60}')

# Load model once, keep in VRAM for all chunks
print(f'\nLoading model: {MODEL_NAME}...')
model = SentenceTransformer(MODEL_NAME, device='cuda')

# Check how many chunks are already done (crash recovery)
existing_chunks = sorted([
    f for f in os.listdir(CHUNKS_DIR) if f.endswith('.npy')
])
start_chunk = len(existing_chunks)
total_done = 0

# Rebuild hash set from existing chunks for dedup
seen_hashes = set()
if start_chunk > 0:
    print(f'Found {start_chunk} completed chunks. Rebuilding dedup index...')
    for i in range(start_chunk):
        jsonl_path = os.path.join(CHUNKS_DIR, f'chunk_{i:04d}.jsonl')
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                s = json.loads(line)['text']
                seen_hashes.add(sentence_hash(s))
                total_done += 1
    print(f'  Resumed: {total_done:,} sentences, {start_chunk} chunks [{elapsed()}]')

# Define all data sources as generators
from datasets import load_dataset


def gen_c4():
    ds = load_dataset('allenai/c4', 'en', split='train', streaming=True)
    for row in ds:
        yield from clean_sentences(row['text'])


def gen_wiki():
    ds = load_dataset('wikitext', 'wikitext-103-v1', split='train')
    for row in ds:
        text = row['text'].strip()
        if text and not text.startswith('='):
            yield from clean_sentences(text)


def gen_ccnews():
    ds = load_dataset('cc_news', split='train', streaming=True)
    for row in ds:
        date = row.get('date', '') or ''
        # Exclude 2019+: GPT-2 released Feb 2019, so 2019 text may contain AI contamination
        if date and date[:4].isdigit() and int(date[:4]) >= 2019:
            continue
        yield from clean_sentences(row['text'])


def gen_cnn():
    for split in ['train', 'validation', 'test']:
        ds = load_dataset('abisee/cnn_dailymail', '3.0.0', split=split)
        for row in ds:
            yield from clean_sentences(row['article'])


def gen_gutenberg():
    ds = load_dataset('sedthh/gutenberg_english', split='train', streaming=True)
    for row in ds:
        text = row.get('TEXT', '') or row.get('text', '') or ''
        yield from clean_sentences(str(text))


# Chain all sources together with weights via round-robin-ish interleaving
# But simpler: just go through them sequentially. C4 is the backfill (infinite).
sources = [
    ('C4',          gen_c4,       int(TARGET * 0.30)),
    ('Wikipedia',   gen_wiki,     int(TARGET * 0.20)),
    ('CC-News',     gen_ccnews,   int(TARGET * 0.20)),
    ('CNN/DailyMail', gen_cnn,    int(TARGET * 0.15)),
    ('Gutenberg',   gen_gutenberg, int(TARGET * 0.15)),
]

# Main pipeline: collect sentences, encode in chunks, save to disk
# Recovery: hash set already contains all processed sentences.
# Streaming sources restart from the beginning, but hash check
# naturally skips everything already done — no skip counter needed.
buffer = []
chunk_idx = start_chunk
source_idx = 0
source_count = 0
current_gen = None
backfill_failures = 0

print(f'\nStarting pipeline from chunk {start_chunk} ({total_done:,} sentences done)...\n')

while total_done < TARGET:
    # Fill buffer from current source
    if current_gen is None:
        if source_idx >= len(sources):
            if backfill_failures >= 3:
                print(f'[ABORT] C4 backfill failed {backfill_failures} times. '
                      f'Proceeding with {total_done:,} sentences.')
                break
            print(f'[BACKFILL] All sources done at {total_done:,}. Using C4 to reach {TARGET:,}...')
            current_gen = gen_c4()
            source_name = 'C4-backfill'
            source_limit = TARGET
            source_count = 0
        else:
            source_name, source_factory, source_limit = sources[source_idx]
            print(f'[{source_idx+1}/5] {source_name} (target: {source_limit:,})...')
            try:
                current_gen = source_factory()
            except Exception as e:
                print(f'  {source_name} failed to start: {e}, skipping...')
                source_idx += 1
                continue
            source_count = 0

    try:
        for s in current_gen:
            h = sentence_hash(s)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            buffer.append(s)
            source_count += 1
            total_done += 1

            # Buffer full → encode + save chunk
            if len(buffer) >= CHUNK_SIZE:
                # Encode
                emb = model.encode(
                    buffer, batch_size=BATCH_SIZE,
                    show_progress_bar=True, normalize_embeddings=True, device='cuda',
                )
                emb = np.array(emb, dtype=np.float32)

                # Save
                npy_path = os.path.join(CHUNKS_DIR, f'chunk_{chunk_idx:04d}.npy')
                jsonl_path = os.path.join(CHUNKS_DIR, f'chunk_{chunk_idx:04d}.jsonl')
                np.save(npy_path, emb)
                with open(jsonl_path, 'w', encoding='utf-8') as f:
                    for sent in buffer:
                        f.write(json.dumps({'text': sent}, ensure_ascii=False) + '\n')

                mb = os.path.getsize(npy_path) / (1024*1024)
                print(f'  Chunk {chunk_idx:04d}: {len(buffer):,} sentences, '
                      f'{mb:.0f}MB embeddings | '
                      f'Total: {total_done:,}/{TARGET:,} | '
                      f'Source: {source_name} ({source_count:,}) | '
                      f'[{elapsed()}]')

                buffer = []
                chunk_idx += 1
                del emb
                gc.collect()

            # Source limit reached
            if source_count >= source_limit:
                break

            if total_done >= TARGET:
                break

    except Exception as e:
        print(f'  {source_name} error: {e}. Moving to next source...')
        if source_name == 'C4-backfill':
            backfill_failures += 1

    # Move to next source (backfill re-enters via source_idx >= len(sources))
    current_gen = None
    if source_name != 'C4-backfill':
        source_idx += 1

# Flush remaining buffer
if buffer:
    emb = model.encode(
        buffer, batch_size=BATCH_SIZE,
        show_progress_bar=True, normalize_embeddings=True, device='cuda',
    )
    emb = np.array(emb, dtype=np.float32)
    npy_path = os.path.join(CHUNKS_DIR, f'chunk_{chunk_idx:04d}.npy')
    jsonl_path = os.path.join(CHUNKS_DIR, f'chunk_{chunk_idx:04d}.jsonl')
    np.save(npy_path, emb)
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for sent in buffer:
            f.write(json.dumps({'text': sent}, ensure_ascii=False) + '\n')
    print(f'  Chunk {chunk_idx:04d} (final): {len(buffer):,} sentences [{elapsed()}]')
    chunk_idx += 1
    del emb, buffer
    gc.collect()

# Free model
del model
gc.collect()
torch.cuda.empty_cache()

total_chunks = chunk_idx
print(f'\nPipeline complete: {total_done:,} sentences in {total_chunks} chunks [{elapsed()}]')


# ============================================================
# Step 4: Build FAISS index from saved chunks
# ============================================================
# Uses memmap to avoid loading all 72GB embeddings into RAM at once.
# Only training sample (1M vectors = 1.4GB) and add batches (500K = 700MB)
# are in RAM at any time. Peak RAM: ~3GB instead of 72GB.

print(f'\n{"="*60}')
print(f'  BUILDING FAISS INDEX (chunk-streaming, zero extra disk)')
print(f'{"="*60}')

# Stream directly from chunk .npy files — no memmap needed.
# Peak disk: 0 extra bytes. Peak RAM: ~1.4GB (one chunk at a time).

# Count total vectors
chunk_sizes = []
total_n = 0
for i in range(total_chunks):
    npy_path = os.path.join(CHUNKS_DIR, f'chunk_{i:04d}.npy')
    n_chunk = np.load(npy_path, mmap_mode='r').shape[0]
    chunk_sizes.append(n_chunk)
    total_n += n_chunk

n = total_n
nlist = min(4096, int(np.sqrt(n)))
m = 48  # 384 / 48 = 8 sub-vectors

print(f'Total vectors: {total_n:,} across {total_chunks} chunks')
print(f'Building IVF{nlist}+PQ{m} index...')

res = faiss.StandardGpuResources()
res.setTempMemory(4 * 1024 * 1024 * 1024)

quantizer = faiss.IndexFlatIP(DIM)
index_cpu = faiss.IndexIVFPQ(quantizer, DIM, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)

# Train: sample 1M vectors from random chunks (no need to load all)
train_size = min(n, 1_000_000)
per_chunk = train_size // total_chunks + 1
train_samples = []
for i in range(total_chunks):
    npy_path = os.path.join(CHUNKS_DIR, f'chunk_{i:04d}.npy')
    chunk_emb = np.load(npy_path)
    n_sample = min(per_chunk, len(chunk_emb))
    idx = np.random.choice(len(chunk_emb), n_sample, replace=False)
    train_samples.append(chunk_emb[idx])
    del chunk_emb

train_data = np.concatenate(train_samples)[:train_size]
del train_samples
gc.collect()
print(f'Training on {len(train_data):,} vectors...')
index_gpu.train(train_data)
del train_data
gc.collect()
print(f'Training done [{elapsed()}]')

# Add: stream each chunk directly into the GPU index
print(f'Adding {n:,} vectors from {total_chunks} chunks...')
added = 0
for i in range(total_chunks):
    npy_path = os.path.join(CHUNKS_DIR, f'chunk_{i:04d}.npy')
    chunk_emb = np.load(npy_path)
    index_gpu.add(chunk_emb)
    added += len(chunk_emb)
    del chunk_emb
    gc.collect()
    print(f'  {added:,}/{n:,} (chunk {i+1}/{total_chunks}) [{elapsed()}]')

index_cpu = faiss.index_gpu_to_cpu(index_gpu)
index_cpu.nprobe = min(64, nlist // 4)

del index_gpu
gc.collect()
torch.cuda.empty_cache()
print(f'Index built [{elapsed()}]')


# ============================================================
# Step 5: Merge sentence files + save final output
# ============================================================

print(f'\nMerging {total_chunks} sentence files...')

# Write final output to WORK_DIR (Google Drive if available)
output_jsonl = os.path.join(WORK_DIR, 'sentences.jsonl')
output_faiss = os.path.join(WORK_DIR, 'sentences.faiss')

with open(output_jsonl, 'w', encoding='utf-8') as out:
    for i in range(total_chunks):
        jsonl_path = os.path.join(CHUNKS_DIR, f'chunk_{i:04d}.jsonl')
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                out.write(line)
        # Delete chunk jsonl after merging to free Drive space
        os.remove(jsonl_path)

faiss.write_index(index_cpu, output_faiss)

# Clean up empty chunks directory
if os.path.exists(CHUNKS_DIR):
    shutil.rmtree(CHUNKS_DIR, ignore_errors=True)
    print(f'Cleaned up {CHUNKS_DIR}')

faiss_size = os.path.getsize(output_faiss) / (1024 * 1024)
jsonl_size = os.path.getsize(output_jsonl) / (1024 * 1024)
print(f'\nsentences.faiss: {faiss_size:.0f} MB')
print(f'sentences.jsonl: {jsonl_size:.0f} MB')
print(f'Total sentences: {total_done:,}')
print(f'Total time: {elapsed()}')


# ============================================================
# Step 6: Download (Colab only)
# ============================================================

print(f'\nFinal output saved to: {WORK_DIR}/')
if 'drive' in WORK_DIR:
    print('Files are on Google Drive — download from Drive to: ai-text-detector/corpus/')
else:
    # Try Colab download for local storage
    try:
        from google.colab import files as colab_files
        colab_files.download(output_faiss)
        colab_files.download(output_jsonl)
    except (ImportError, Exception):
        print('Copy to: ai-text-detector/corpus/')
