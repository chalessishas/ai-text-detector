#!/usr/bin/env python3
"""Test the Desklib RAID-winning AI detector model.

Downloads and runs desklib/ai-text-detector-v1.01 (DeBERTa-v3-large, 400M params)
on our test suite of different text genres and compares with our DeBERTa-base.

Run: python3 scripts/test_desklib.py
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig


class DesklibAIDetectionModel(nn.Module):
    """Desklib model: DeBERTa-v3-large + mean pooling + linear classifier."""

    def __init__(self, config):
        super().__init__()
        from transformers import DebertaV2Model
        self.model = DebertaV2Model(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        # Mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden_state * mask).sum(1) / mask.sum(1)
        logits = self.classifier(pooled)
        return logits

    @classmethod
    def from_pretrained(cls, model_name):
        from safetensors.torch import load_file
        config = AutoConfig.from_pretrained(model_name)
        model = cls(config)
        # Load safetensors weights directly
        from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download(model_name, "model.safetensors")
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=True)
        return model


def predict(text, model, tokenizer, device="cpu"):
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    # Remove token_type_ids (DeBERTa-v2 doesn't use them)
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs)
    prob = torch.sigmoid(logits).item()
    label = "AI" if prob > 0.5 else "Human"
    return prob, label


def main():
    MODEL_NAME = "desklib/ai-text-detector-v1.01"
    print(f"Loading {MODEL_NAME}...")

    try:
        model = DesklibAIDetectionModel.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Try: pip install transformers torch")
        return

    model.float()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"  Loaded on {device}")

    # Test suite: same texts we tested with our detector
    tests = {
        # AI-generated texts (should detect as AI)
        "AI: tech_doc": (
            "Kubernetes orchestrates containerized applications across clusters of machines. "
            "The control plane consists of the API server, etcd, the scheduler, and controller "
            "managers. When you deploy a Pod, the scheduler examines node resources, affinity "
            "rules, and taints to select optimal placement."
        ),
        "AI: news": (
            "WASHINGTON - The Federal Reserve held interest rates steady on Wednesday, signaling "
            "officials remain cautious about cutting borrowing costs despite mounting evidence "
            "that inflation is cooling. Chair Jerome Powell said the central bank needs more "
            "confidence that price increases are sustainably moving toward the 2 percent target."
        ),
        "AI: business_email": (
            "Hi Sarah, Following up on our Q3 planning session yesterday. I have put together "
            "a summary of the key action items we discussed. First, the marketing budget needs "
            "to be reallocated by end of month. Second, the product roadmap review is scheduled "
            "for March 15th."
        ),
        "AI: deep_research": (
            "Recent advances in protein structure prediction have been catalyzed by deep learning "
            "architectures. AlphaFold2 achieved median GDT-TS scores of 92.4 on CASP14 targets. "
            "The model leverages an Evoformer module that processes multiple sequence alignments "
            "through axial attention."
        ),
        "AI: product_review": (
            "I have been using the Sony WH-1000XM5 headphones for about three months now, and "
            "they are genuinely excellent for noise cancellation. The ANC blocks out airplane "
            "engine noise almost completely. Battery life is solid at around 30 hours."
        ),
        "AI: how_to_guide": (
            "To set up a home composting system, start by choosing a location that gets partial "
            "sunlight and has good drainage. Layer your materials: start with coarse browns, add "
            "kitchen scraps, then cover with dry leaves. Turn the pile every two weeks."
        ),
        "AI: literary_analysis": (
            "Toni Morrison employs a non-linear narrative structure in Beloved that mirrors the "
            "fragmented psychology of trauma survivors. The novel opens in medias res with the "
            "haunting of 124 Bluestone Road."
        ),
        "AI: opinion_editorial": (
            "The push to ban smartphones in schools misses the point entirely. Yes, students are "
            "distracted but the solution is not prohibition, it is integration. Schools should be "
            "teaching digital literacy."
        ),
        # Human-written texts (should detect as Human)
        "Human: casual": (
            "So I was walking my dog yesterday and he just stops dead in his tracks right in the "
            "middle of the sidewalk. Like completely frozen. Turns out there was this tiny little "
            "frog just sitting there. My 80 pound lab was terrified of a frog."
        ),
        "Human: reddit": (
            "honestly the worst part about cooking is not the cooking itself its the cleanup. "
            "like I just made this amazing pasta from scratch, homemade sauce and everything, "
            "and now I have to deal with the aftermath."
        ),
        "Human: email": (
            "Hi team, Quick update on the Henderson account. I spoke with their VP of marketing "
            "this morning and she mentioned they are considering expanding the contract but want "
            "to see Q2 numbers first."
        ),
    }

    print(f"\n{'Text':<25} {'Prob':>6} {'Label':>7} {'Correct':>8}")
    print("-" * 55)

    correct = 0
    total = 0
    for name, text in tests.items():
        expected = "AI" if name.startswith("AI:") else "Human"
        prob, label = predict(text, model, tokenizer, device)
        is_correct = label == expected
        correct += is_correct
        total += 1
        mark = "OK" if is_correct else "MISS"
        print(f"{name:<25} {prob:>5.1%} {label:>7} {mark:>8}")

    print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.0f}%)")


if __name__ == "__main__":
    main()
