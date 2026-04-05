#!/usr/bin/env python3
"""StealthRL — GRPO LoRA training on RunPod A100.

RL fine-tunes Qwen2.5-3B to paraphrase AI text so it fools DeBERTa detector
while preserving original meaning (E5 cosine similarity).

Based on: StealthRL (arXiv 2602.08934), 97-99% evasion rate reported.

Setup on RunPod (A100 80GB recommended):
  pip install unsloth trl transformers datasets sentence-transformers safetensors vllm
  # Upload: dataset_v4.jsonl + detector_v5/ directory
  python3 train_stealth_runpod.py

Expected: ~1-2 hours on A100 80GB, ~$2-3 total.
"""

import json
import os
import random
import sys
import time

import numpy as np
import torch

WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
DETECTOR_DIR = os.path.join(WORKSPACE, "detector_v5")
DATASET_FILE = os.path.join(WORKSPACE, "dataset_v4.jsonl")
OUTPUT_DIR = os.path.join(WORKSPACE, "stealth_lora")

LORA_RANK = 32
MAX_SEQ = 1024
N_TRAIN = 5000
MAX_STEPS = 500
BETA = 0.3  # semantic preservation weight
SIM_GATE = 0.5
MIN_WORDS = 20

SYSTEM_MSG = "You are a writing assistant. Paraphrase the following text while preserving its meaning. Write naturally."


def main():
    print("=" * 70)
    print("StealthRL — GRPO LoRA Training")
    print("=" * 70)

    assert torch.cuda.is_available(), "CUDA required"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")

    for f in [DETECTOR_DIR, DATASET_FILE]:
        assert os.path.exists(f), f"Missing: {f}"
    print(f"Detector: {DETECTOR_DIR}")
    print(f"Dataset: {DATASET_FILE}")

    # ── Load paraphraser model + LoRA ──
    print("\nLoading Qwen2.5-3B-Instruct + LoRA...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-3B-Instruct",
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.6,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    print(f"Paraphraser ready: LoRA r={LORA_RANK}")

    # ── Load DeBERTa detector ──
    print("\nLoading DeBERTa v5 detector...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    det_tokenizer = AutoTokenizer.from_pretrained(DETECTOR_DIR)
    det_model = AutoModelForSequenceClassification.from_pretrained(DETECTOR_DIR)
    det_model.float().requires_grad_(False).cuda()

    test_input = det_tokenizer("Test.", return_tensors="pt", truncation=True, max_length=512)
    test_input = {k: v.cuda() for k, v in test_input.items()}
    with torch.no_grad():
        test_probs = torch.softmax(det_model(**test_input).logits, dim=-1).cpu().numpy()[0]
    print(f"DeBERTa OK: human={test_probs[0]:.3f} ai={test_probs[1]:.3f}")

    # ── Load E5 similarity model ──
    print("Loading E5 similarity model...")
    from sentence_transformers import SentenceTransformer
    e5_model = SentenceTransformer("intfloat/e5-base-v2").to("cuda")
    print(f"E5 OK: {e5_model.get_sentence_embedding_dimension()}d")

    # ── Prepare dataset ──
    print(f"\nLoading AI texts from {DATASET_FILE}...")
    random.seed(42)
    ai_texts = []
    with open(DATASET_FILE) as f:
        for line in f:
            d = json.loads(line)
            if d["label"] in (1, 2):
                text = d["text"].strip()
                word_count = len(text.split())
                if 30 <= word_count <= 300:
                    ai_texts.append(text)

    random.shuffle(ai_texts)
    ai_texts = ai_texts[:N_TRAIN]
    print(f"Training samples: {len(ai_texts)}, avg {np.mean([len(t.split()) for t in ai_texts]):.0f} words")

    from datasets import Dataset

    train_data = []
    for text in ai_texts:
        train_data.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": text},
            ],
            "original_text": text,
        })
    train_dataset = Dataset.from_list(train_data)

    # ── Reward function ──
    def stealth_reward(prompts, completions, original_text, **kwargs):
        rewards = []
        if isinstance(completions[0], list):
            comp_texts = [c[-1]["content"] if c else "" for c in completions]
        else:
            comp_texts = [str(c) for c in completions]

        if isinstance(original_text, list):
            orig_texts = original_text
        else:
            orig_texts = [original_text] * len(comp_texts)

        with torch.no_grad():
            inputs = det_tokenizer(
                comp_texts, truncation=True, max_length=512,
                padding=True, return_tensors="pt"
            )
            inputs = {k: v.cuda() for k, v in inputs.items()}
            logits = det_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        orig_embs = e5_model.encode(
            ["query: " + t for t in orig_texts],
            normalize_embeddings=True, show_progress_bar=False
        )
        comp_embs = e5_model.encode(
            ["query: " + t for t in comp_texts],
            normalize_embeddings=True, show_progress_bar=False
        )

        for i in range(len(comp_texts)):
            if len(comp_texts[i].split()) < MIN_WORDS:
                rewards.append(-0.5)
                continue
            r_sem = float(np.dot(orig_embs[i], comp_embs[i]))
            if r_sem < SIM_GATE:
                rewards.append(-1.0)
                continue
            p_ai = float(probs[i][1] + probs[i][2])
            if p_ai < 0.3:
                r_det = 1.0
            elif p_ai < 0.5:
                r_det = 0.6
            elif p_ai < 0.7:
                r_det = 0.3
            elif p_ai < 0.9:
                r_det = 0.1
            else:
                r_det = 0.0
            rewards.append(r_det + BETA * r_sem)
        return rewards

    # ── GRPO Training ──
    from trl import GRPOConfig, GRPOTrainer

    if gpu_mem > 60:
        num_gens, max_comp, max_prompt, batch, grad_accum = 8, 512, 400, 4, 1
    elif gpu_mem > 30:
        num_gens, max_comp, max_prompt, batch, grad_accum = 6, 400, 300, 2, 2
    else:
        num_gens, max_comp, max_prompt, batch, grad_accum = 2, 400, 300, 1, 4

    training_args = GRPOConfig(
        use_vllm=True if gpu_mem > 30 else False,
        learning_rate=2e-5,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=5,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_gens,
        max_prompt_length=max_prompt,
        max_completion_length=max_comp,
        max_steps=MAX_STEPS,
        save_steps=100,
        max_grad_norm=0.1,
        beta=0.05,
        temperature=1.0,
        report_to="none",
        output_dir=os.path.join(WORKSPACE, "stealth_checkpoints"),
    )

    print(f"\nTraining config:")
    print(f"  gens={num_gens}, comp={max_comp}, prompt={max_prompt}")
    print(f"  batch={batch}, accum={grad_accum}, effective={batch * grad_accum}")
    print(f"  Steps: {MAX_STEPS}, vLLM: {training_args.use_vllm}")

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[stealth_reward],
        processing_class=tokenizer,
    )

    print("\nStarting GRPO training...")
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Loss: {train_result.training_loss:.4f}")

    # ── Evaluate ──
    print("\n=== Post-training evaluation ===")
    test_texts = [
        "Artificial intelligence has fundamentally transformed the landscape of modern education. The integration of machine learning algorithms into educational platforms has enabled personalized learning experiences.",
        "Climate change is one of the most pressing challenges facing humanity today. Rising global temperatures have led to unprecedented weather events including devastating hurricanes.",
        "The transformer architecture has revolutionized natural language processing by introducing the self-attention mechanism. Unlike recurrent neural networks, transformers process sequences in parallel.",
    ]

    FastLanguageModel.for_inference(model)
    passed = 0
    for i, orig in enumerate(test_texts):
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": orig},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256, temperature=1.0, top_p=0.9, do_sample=True)
        paraphrase = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        det_input = det_tokenizer(paraphrase, truncation=True, max_length=512, return_tensors="pt")
        det_input = {k: v.cuda() for k, v in det_input.items()}
        with torch.no_grad():
            probs = torch.softmax(det_model(**det_input).logits, dim=-1).cpu().numpy()[0]
        ai_score = (probs[1] + probs[2]) * 100

        orig_emb = e5_model.encode(["query: " + orig], normalize_embeddings=True)
        para_emb = e5_model.encode(["query: " + paraphrase], normalize_embeddings=True)
        sim = float(np.dot(orig_emb[0], para_emb[0]))

        status = "PASS" if ai_score < 50 else "FAIL"
        if ai_score < 50:
            passed += 1
        print(f"  Text {i+1}: AI={ai_score:.1f}% [{status}] Sim={sim:.3f}")
        print(f"    Original:   {orig[:80]}...")
        print(f"    Paraphrase: {paraphrase[:80]}...")

    print(f"\nEvasion rate: {passed}/{len(test_texts)}")

    # ── Save ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    import subprocess
    subprocess.run(["tar", "-czf", f"{WORKSPACE}/stealth_lora.tar.gz", "-C", WORKSPACE, "stealth_lora/"], check=True)
    size = os.path.getsize(f"{WORKSPACE}/stealth_lora.tar.gz") / (1024*1024)
    print(f"\nLoRA adapter saved: {OUTPUT_DIR}/ ({size:.0f} MB)")
    print(f"Download: stealth_lora.tar.gz")
    print("=" * 70)


if __name__ == "__main__":
    main()
