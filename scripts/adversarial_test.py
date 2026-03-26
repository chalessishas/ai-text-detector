#!/usr/bin/env python3
"""Adversarial Test Agent for AI Detector.

Generates AI texts across multiple categories, tests them against our detector,
reports failures, and iterates with new prompts until all pass.

Categories:
  1. Standard essays (baseline)
  2. Anti-detect / casual (hardest to detect)
  3. Professional domains (legal, medical, code, real estate)
  4. Creative writing (fiction, poetry, personal essay)
  5. Mimicking human flaws (bad grammar, disorganized, rambling)

Usage:
    python3 scripts/adversarial_test.py [--rounds 3] [--model deepseek]
"""

import argparse
import json
import math
import os
import sys
import time

import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perplexity import load_model, compute_features, compute_perplexity_score

# ── Prompt categories ────────────────────────────────────────────────────

CATEGORIES = {
    "standard": [
        "Explain the causes and consequences of the 2008 financial crisis in 500 words.",
        "Write a comprehensive overview of how CRISPR gene editing works and its potential applications in medicine.",
        "Discuss the pros and cons of universal basic income as a policy solution for technological unemployment.",
        "Write a detailed product description for a premium noise-cancelling headphone targeting audiophiles.",
        "Explain the difference between machine learning, deep learning, and artificial intelligence for a general audience.",
    ],
    "anti_detect": [
        "Write like you're a college student ranting on Reddit about your terrible roommate. Use slang, short sentences, and be emotional. Don't sound formal at all.",
        "Write a casual Twitter thread about your experience switching from iPhone to Android. Use 'honestly', 'ngl', 'lowkey', and other internet slang. Make it feel like a real person typing fast.",
        "Pretend you're texting your best friend about a movie you just watched. Use incomplete sentences, typos on purpose, and casual language.",
        "Write a stream-of-consciousness diary entry about a bad day at work. Jump between topics, use dashes, interrupt yourself mid-thought.",
        "Write a comment on a YouTube cooking video in the style of a real person. Keep it short, enthusiastic, mention something specific from the video.",
    ],
    "professional": [
        "Write a section of a legal contract regarding intellectual property assignment between an employer and employee.",
        "Write clinical notes for a patient presenting with chest pain in an emergency department, using standard medical documentation format.",
        "Write a code review comment explaining why a pull request has a race condition in the database connection pooling logic.",
        "Write a real estate listing description for a 3-bedroom craftsman bungalow in Portland, Oregon.",
        "Write internal meeting minutes from a product team discussing whether to delay a feature launch by two weeks.",
    ],
    "creative": [
        "Write the opening paragraph of a noir detective novel set in 1940s Shanghai.",
        "Write a poem about loneliness that doesn't use the word 'lonely' or any obvious synonyms.",
        "Write a short horror story in exactly 100 words about someone who realizes their reflection is moving independently.",
        "Write a personal essay about learning to cook from your grandmother, focusing on one specific dish and one specific memory.",
    ],
    "mimic_flaws": [
        "Write a college freshman's first essay about climate change. Make it slightly disorganized, with one paragraph that doesn't quite connect to the thesis. Include a weak conclusion that just restates the intro.",
        "Write an email from someone who is clearly not a native English speaker, asking their landlord to fix the heating. Include minor grammar mistakes that feel natural, not random.",
        "Write a Yelp review from an angry customer. Be specific but rambling, go off on a tangent about parking, then come back to the food.",
    ],
}


# ── AI text generation ───────────────────────────────────────────────────

def generate_ai_text(prompt, api_key, model="deepseek-chat", temperature=0.7):
    """Generate text using DeepSeek API."""
    try:
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": temperature,
            },
            timeout=30,
        )
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return None


# ── Detection ────────────────────────────────────────────────────────────

def detect_text(text, ppl_model, deb_tokenizer, deb_model):
    """Run all detection signals on a text. Returns dict of signals + final prediction."""
    result = {}

    # DeBERTa
    inputs = deb_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = deb_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        result["deberta_ai"] = round((probs[0][1] + probs[0][2]).item() * 100, 1)

    # PPL
    features = compute_features(ppl_model, text)
    tokens = features.get("tokens", [])
    if tokens and len(tokens) >= 10:
        score = compute_perplexity_score(tokens)
        result["ppl"] = score["perplexity"]
        result["min_ppl"] = score.get("min_window_ppl", score["perplexity"])
        result["top10"] = score["top10_pct"]
        result["entropy"] = score["mean_entropy"]
        result["lr_ai"] = score.get("lr_ai_probability", 50)
    else:
        result["ppl"] = 999
        result["min_ppl"] = 999
        result["top10"] = 50
        result["entropy"] = 3.0
        result["lr_ai"] = 50

    # Simple PPL threshold (best single signal at 80.4%)
    result["ppl_pred"] = "ai" if result["ppl"] < 15 else "human"

    # DeBERTa prediction
    result["deb_pred"] = "ai" if result["deberta_ai"] > 50 else "human"

    # Final: use PPL threshold as primary (highest accuracy)
    result["final_pred"] = result["ppl_pred"]

    return result


# ── Main loop ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES.keys()))
    args = parser.parse_args()

    # Load .env
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env.local")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not ds_key:
        print("ERROR: No DEEPSEEK_API_KEY in .env.local")
        sys.exit(1)

    # Load models
    print("Loading detection models...", file=sys.stderr)
    ppl_model = load_model()
    v1_path = os.path.join(os.path.dirname(__file__), "..", "models", "detector")
    deb_tok = AutoTokenizer.from_pretrained(v1_path)
    deb_mdl = AutoModelForSequenceClassification.from_pretrained(v1_path)
    deb_mdl.float()
    deb_mdl.eval()

    for round_num in range(1, args.rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}")
        print(f"{'='*60}")

        total = 0
        detected = 0
        failures = []

        for cat in args.categories:
            if cat not in CATEGORIES:
                continue
            prompts = CATEGORIES[cat]
            print(f"\n--- {cat} ({len(prompts)} prompts) ---")

            for prompt in prompts:
                # Generate AI text
                text = generate_ai_text(prompt, ds_key)
                if not text:
                    print(f"  SKIP: generation failed")
                    continue
                total += 1

                # Detect
                signals = detect_text(text, ppl_model, deb_tok, deb_mdl)
                pred = signals["final_pred"]
                is_correct = (pred == "ai")

                if is_correct:
                    detected += 1
                    status = "PASS"
                else:
                    status = "FAIL"
                    failures.append({
                        "category": cat,
                        "prompt": prompt[:80],
                        "text": text[:200],
                        "ppl": signals["ppl"],
                        "min_ppl": signals["min_ppl"],
                        "top10": signals["top10"],
                        "deberta": signals["deberta_ai"],
                        "lr": signals["lr_ai"],
                    })

                print(f"  [{status}] PPL={signals['ppl']:.1f} Top10={signals['top10']:.0f}% "
                      f"DeBERTa={signals['deberta_ai']:.0f}% | {prompt[:60]}...")
                time.sleep(0.5)

        # Summary
        print(f"\n{'='*60}")
        print(f"ROUND {round_num} RESULTS: {detected}/{total} detected ({detected/max(total,1)*100:.0f}%)")
        print(f"{'='*60}")

        if failures:
            print(f"\nFAILURES ({len(failures)}):")
            for f in failures:
                print(f"  [{f['category']}] PPL={f['ppl']:.1f} MinPPL={f['min_ppl']:.1f} "
                      f"Top10={f['top10']:.0f}% DeBERTa={f['deberta']:.0f}% LR={f['lr']:.0f}%")
                print(f"    Prompt: {f['prompt']}")
                print(f"    Text: {f['text'][:100]}...")

            # Save failures for analysis
            fail_path = os.path.join(os.path.dirname(__file__), "data",
                                     f"adversarial_failures_r{round_num}.jsonl")
            os.makedirs(os.path.dirname(fail_path), exist_ok=True)
            with open(fail_path, "w") as fp:
                for f in failures:
                    fp.write(json.dumps(f, ensure_ascii=False) + "\n")
            print(f"\nFailures saved to {fail_path}")
        else:
            print("\nALL PASSED!")

        if detected == total:
            print("All texts detected. Stopping.")
            break


if __name__ == "__main__":
    main()
