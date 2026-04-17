#!/usr/bin/env python3
"""Build dataset_v6 supplement: Chinese multi-family AI samples (Path C).

Implements Path C from 2026-04-17 Research Loop 02:18
(docs/research/2026-04-17-research-loop-0218-deepseek-viability.md).

Target: ~24K samples each × 5 families = ~120K Chinese AI samples to
complement the existing DeepSeek-heavy dataset_v6.jsonl. After merge the
AI side becomes 60% DeepSeek + 40% other families — better cross-family
generalization for Chinese-market AI detection.

Cost estimate (500 words × 120K samples ≈ 60M tokens):
    gpt4o-mini     ~$10 @ $0.15/M in + $0.60/M out
    claude-haiku   ~$48 @ $1/M in + $5/M out
    qwen3-max      ~$2  @ ~$0.02/M (Alibaba)
    ernie-5        ~$4  @ $0.10/M (Baidu)
    kimi-k2.5      ~$5  @ $0.15/M (Moonshot)
    Total          ~$70

STATUS: SKELETON ONLY. Vendor client functions + per-sample generation
loop are TODO. Master to fill in API keys via .env.local and to verify
prompt templates from build_dataset_v6.py AI_PROMPT_STYLES apply cleanly
to non-DeepSeek APIs (some vendors tokenize system prompts differently).

Usage (after implementation):
    python3 scripts/build_dataset_v6_supplement.py --vendor gpt4o-mini --target 24000
    python3 scripts/build_dataset_v6_supplement.py --vendor claude-haiku --target 24000
    python3 scripts/build_dataset_v6_supplement.py --vendor qwen3-max --target 24000
    python3 scripts/build_dataset_v6_supplement.py --vendor ernie-5 --target 24000
    python3 scripts/build_dataset_v6_supplement.py --vendor kimi-k2.5 --target 24000
    python3 scripts/build_dataset_v6_supplement.py --merge  # append all into dataset_v6.jsonl

Requires: pip install openai python-dotenv tqdm
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Vendor configurations (OpenAI-compatible endpoints)
# ---------------------------------------------------------------------------

# Vendor registry. sdk_type distinguishes OpenAI-compatible endpoints
# (use `openai.OpenAI(base_url=..., api_key=...)`) from Anthropic's native
# SDK (use `anthropic.Anthropic(api_key=...)`, different method names +
# message format). Master: do NOT treat these uniformly in get_client().
#
# Quality caveat for Path C: Expect asymmetric Chinese generation
# quality across vendors — different training corpora, RL objectives,
# and token tokenizers produce stylistically distinct outputs even on
# identical prompts. No verified 2026 benchmark exists comparing these
# 5 vendors on the specific essay-style prompts used here, so do NOT
# skip the spot-check step. After generation, sample ~50 outputs per
# vendor. If any vendor produces obviously low-quality or translated-
# looking Chinese, exclude that shard or regenerate with a different
# model_id — mixing low-quality samples risks teaching the detector
# vendor-specific stylistic artifacts rather than generic AI-text signals.
VENDORS: Dict[str, Dict[str, Optional[str]]] = {
    "gpt4o-mini": {
        "sdk_type": "openai",
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "model_id": "gpt-4o-mini",
        "source_tag": "gpt4o_mini_chinese",
    },
    "claude-haiku": {
        "sdk_type": "anthropic",  # NOT OpenAI-compatible — use anthropic SDK
        "base_url": None,         # not used; anthropic SDK handles endpoint
        "env_key": "ANTHROPIC_API_KEY",
        "model_id": "claude-haiku-4-5-20251001",
        "source_tag": "claude_haiku_chinese",
    },
    "qwen3-max": {
        "sdk_type": "openai",  # Dashscope exposes OpenAI-compat mode
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "model_id": "qwen3-max",
        "source_tag": "qwen3_max_chinese",
    },
    "ernie-5": {
        "sdk_type": "openai",  # Baidu Qianfan v2 exposes OpenAI-compat
        "base_url": "https://qianfan.baidubce.com/v2",
        "env_key": "ERNIE_API_KEY",
        "model_id": "ernie-5-turbo",
        "source_tag": "ernie5_chinese",
    },
    "kimi-k2.5": {
        "sdk_type": "openai",  # Moonshot exposes OpenAI-compat
        "base_url": "https://api.moonshot.cn/v1",
        "env_key": "MOONSHOT_API_KEY",
        "model_id": "moonshot-v1-k2.5",
        "source_tag": "kimi_k25_chinese",
    },
}

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "supplement_v6_shards"
FINAL_DATASET = PROJECT_DIR / "dataset_v6.jsonl"


def get_client(vendor: str):
    """Return an OpenAI-compatible client for the chosen vendor. TODO: real impl.

    Claude requires the anthropic SDK or an anthropic-to-openai adapter.
    Other 4 vendors work with vanilla `openai.OpenAI(base_url=..., api_key=...)`.
    """
    raise NotImplementedError(
        f"Client factory for {vendor} not implemented. "
        f"Master: implement per VENDORS[{vendor!r}]['base_url'] pattern."
    )


def generate_sample(vendor: str, client, prompt: str, style: str) -> Optional[dict]:
    """Generate one AI sample for the given vendor. TODO: real impl.

    Should return dict matching dataset_v6.jsonl schema:
        {"text": str, "label": 1, "source": VENDORS[vendor]["source_tag"], "domain": str}

    Implementation notes:
    - Reuse prompt templates from build_dataset_v6.AI_PROMPT_STYLES
    - Validate: MIN_WORDS <= len(text.split()) <= MAX_WORDS (100, 500)
    - Skip and retry if validation fails (max 3 retries)
    """
    raise NotImplementedError


def run_vendor(vendor: str, target: int):
    """Generate `target` samples for the given vendor and write to a shard file."""
    cfg = VENDORS[vendor]
    env_key = cfg["env_key"]
    if not os.environ.get(env_key):
        print(f"ERROR: {env_key} not set in environment (.env.local).", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    shard_path = OUTPUT_DIR / f"{vendor}.jsonl"
    print(f"[{vendor}] target={target} output={shard_path}")

    # TODO: real generation loop
    # client = get_client(vendor)
    # with open(shard_path, "w") as f:
    #     for i in tqdm(range(target)):
    #         sample = generate_sample(vendor, client, prompt=..., style=...)
    #         if sample:
    #             f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[{vendor}] NOT IMPLEMENTED — see TODOs in generate_sample / get_client")
    sys.exit(2)


def merge_shards():
    """Append all supplement shards into dataset_v6.jsonl."""
    shards = list(OUTPUT_DIR.glob("*.jsonl")) if OUTPUT_DIR.exists() else []
    if not shards:
        print("ERROR: no shards found in supplement_v6_shards/", file=sys.stderr)
        sys.exit(1)

    print(f"Merging {len(shards)} shards into {FINAL_DATASET}")
    total = 0
    with open(FINAL_DATASET, "a") as out:
        for shard in shards:
            with open(shard) as inp:
                for line in inp:
                    out.write(line)
                    total += 1
    print(f"Appended {total} samples to {FINAL_DATASET}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", choices=list(VENDORS.keys()),
                        help="Which vendor to generate samples for")
    parser.add_argument("--target", type=int, default=24000,
                        help="Number of samples per vendor (default 24000)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge generated shards into dataset_v6.jsonl")
    args = parser.parse_args()

    if args.merge:
        merge_shards()
    elif args.vendor:
        run_vendor(args.vendor, args.target)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
