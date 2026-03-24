#!/usr/bin/env python3
"""Validate CoPA humanizer output against the retrained DeBERTa (98.5%).

Loads DeBERTa directly (no HTTP server needed), runs inference on CoPA outputs,
scores each with DeBERTa ai_score.

Run: /opt/anaconda3/bin/python3.13 scripts/validate_copa.py
"""

import json
import os
import time
from datetime import datetime


LABEL_NAMES = ["human", "ai", "ai_polished", "human_polished"]


def load_deberta():
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "detector")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.float()
    model.requires_grad_(False)
    return tokenizer, model


def deberta_score(tokenizer, model, text):
    import torch

    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    ai_score = float(probs[1] + probs[2]) * 100
    human_score = float(probs[0] + probs[3]) * 100

    return {
        "prediction": "AI" if ai_score > 50 else "HUMAN",
        "ai_score": round(ai_score, 1),
        "human_score": round(human_score, 1),
        "detail": {name: round(float(probs[i]) * 100, 1) for i, name in enumerate(LABEL_NAMES)},
    }


ORIGINAL_TEXTS = {
    "academic": (
        "Artificial intelligence has fundamentally transformed the landscape "
        "of modern education. Furthermore, the integration of machine learning "
        "algorithms into educational platforms has enabled personalized learning "
        "experiences that cater to individual student needs. Moreover, these "
        "technological advancements have facilitated the development of adaptive "
        "assessment tools that provide real-time feedback to both educators and "
        "learners. Consequently, the educational sector has witnessed a paradigm "
        "shift in pedagogical approaches."
    ),
    "blog": (
        "Climate change is one of the most pressing challenges facing humanity "
        "today. The rising global temperatures have led to unprecedented weather "
        "events, including devastating hurricanes, prolonged droughts, and massive "
        "wildfires. Scientists have consistently warned that without immediate and "
        "decisive action, the consequences will be catastrophic and irreversible. "
        "It is imperative that governments, corporations, and individuals work "
        "together to reduce carbon emissions and transition to renewable energy."
    ),
    "technical": (
        "The transformer architecture has revolutionized natural language processing "
        "by introducing the self-attention mechanism. Unlike recurrent neural networks, "
        "transformers can process entire sequences in parallel, significantly reducing "
        "training time. The key innovation lies in the multi-head attention mechanism, "
        "which allows the model to attend to different positions simultaneously. "
        "This architecture has become the foundation for large language models such "
        "as GPT and BERT, achieving state-of-the-art results across numerous benchmarks."
    ),
}

COPA_OUTPUTS = {
    "copa_academic_1": (
        "AI changed edtech big time. ML algos = personalized learning "
        "(tailored to individual needs). Also: adaptive assessments give "
        "real-time feedback to teachers & students. Result? Total paradigm "
        "shift in pedagogy. Big one."
    ),
    "copa_academic_2": (
        "AI changed everything. Machine learning takes it even further. "
        "Platforms now crank out personalized lessons tailored just for your "
        "kid. No more one-size-fits-all drills. New adaptive tools? Real-time "
        "feedback for teachers and students. It's instant. It's direct."
    ),
    "copa_academic_3": (
        "AI changed everything. Old rules just don't apply anymore. Throw "
        "machine learning into the mix, and suddenly, lessons actually fit "
        "your student. Needs vanish into the background; growth stays front "
        "and center. Get instant, real-time feedback on assignments? No more "
        "weeks of guessing. Teachers get it too."
    ),
    "copa_blog_1": (
        "Key points: Climate change is a top threat. Rising temps lead to bad "
        "weather including hurricanes, drought, wildfires. Scientists say we "
        "need action now or it's catastrophic and irreversible. Must act: "
        "governments, corporations, and individuals. Goals: cut carbon, "
        "switch to renewables."
    ),
    "copa_technical_1": (
        "Transformers flipped natural language processing. They kill the slow "
        "sequential slowness of old RNNs. Entire sequences process in parallel. "
        "Training skyrockets in speed. That's the multi-head attention magic. "
        "You can track multiple spots at once."
    ),
    "copa_technical_2": (
        "Transformers changed everything. They brought self-attention to NLP. "
        "Forget slow recurrence. Transformers crunch whole sequences in parallel. "
        "Training days turn into minutes. That's the trick: multi-head attention. "
        "Models spot patterns everywhere at once. Now, giants like GPT and BERT rule."
    ),
}


def main():
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] CoPA Humanizer vs DeBERTa 98.5% — Validation")
    print("=" * 70)

    print(f"\n[{ts}] Loading DeBERTa...")
    t0 = time.time()
    tok, model = load_deberta()
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] DeBERTa loaded in {time.time()-t0:.1f}s")

    print(f"\n{'─'*70}")
    print("ORIGINAL AI TEXTS (should be detected as AI)")
    print(f"{'─'*70}")
    for name, text in ORIGINAL_TEXTS.items():
        r = deberta_score(tok, model, text)
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] {name:15s} -> {r['prediction']:5s}  AI={r['ai_score']:5.1f}%  "
              f"(h={r['detail']['human']:.1f}% ai={r['detail']['ai']:.1f}% "
              f"ap={r['detail']['ai_polished']:.1f}% hp={r['detail']['human_polished']:.1f}%)")

    print(f"\n{'─'*70}")
    print("COPA v2 OUTPUTS (goal: detected as HUMAN)")
    print(f"{'─'*70}")

    passed = 0
    total = 0
    for name, text in COPA_OUTPUTS.items():
        r = deberta_score(tok, model, text)
        ts = datetime.now().strftime("%H:%M:%S")
        total += 1
        is_pass = r["prediction"] == "HUMAN"
        if is_pass:
            passed += 1
        mark = "PASS" if is_pass else "FAIL"
        print(f"  [{ts}] {name:20s} -> {r['prediction']:5s}  AI={r['ai_score']:5.1f}%  [{mark}]  "
              f"(h={r['detail']['human']:.1f}% ai={r['detail']['ai']:.1f}% "
              f"ap={r['detail']['ai_polished']:.1f}% hp={r['detail']['human_polished']:.1f}%)")

    print(f"\n{'─'*70}")
    print(f"RESULT: {passed}/{total} CoPA outputs passed DeBERTa ({passed/total*100:.0f}%)")
    if passed == total:
        print("ALL PASSED")
    elif passed > total * 0.5:
        print("PARTIAL PASS — needs tuning")
    else:
        print("FAILED — CoPA outputs still detected as AI")
    print(f"{'─'*70}")


if __name__ == "__main__":
    main()
