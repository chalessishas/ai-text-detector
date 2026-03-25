#!/usr/bin/env python3
"""Generate cross-domain AI text samples for DeBERTa retraining.

Uses DeepSeek V3 to generate AI text in domains where DeBERTa is blind:
creative writing, legal, code reviews, medical, technical, business.

Output: domain_ai_samples.jsonl — each line has {text, domain, model, prompt_strategy}
"""

import json
import os
import sys
import time
import random
from pathlib import Path

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    # Try .env.local
    env_path = Path(__file__).parent.parent / ".env.local"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                API_KEY = line.split("=", 1)[1].strip()

OUTPUT = Path(__file__).parent.parent / "scripts" / "data" / "domain_ai_samples.jsonl"

DOMAINS = {
    "creative_fiction": [
        "Write a short literary fiction passage (200-300 words) about {topic}. Use vivid imagery and metaphor.",
        "Write the opening scene of a short story about {topic}. Show, don't tell. 200-300 words.",
        "Write a descriptive passage about {topic} in the style of literary fiction. Focus on sensory details. 200-300 words.",
    ],
    "legal": [
        "Write a legal memorandum paragraph (200-300 words) analyzing {topic}.",
        "Draft a contract clause (200-300 words) addressing {topic}.",
        "Write a legal brief excerpt (200-300 words) arguing about {topic}.",
    ],
    "code_review": [
        "Write a detailed code review comment (200-300 words) for a pull request about {topic}.",
        "Write technical documentation (200-300 words) explaining {topic}.",
        "Write a Stack Overflow answer (200-300 words) explaining how to solve {topic}.",
    ],
    "medical": [
        "Write a clinical case summary (200-300 words) about a patient with {topic}.",
        "Write a medical research abstract (200-300 words) about {topic}.",
        "Write patient education material (200-300 words) explaining {topic}.",
    ],
    "business": [
        "Write a business report paragraph (200-300 words) analyzing {topic}.",
        "Write a consulting recommendation (200-300 words) for {topic}.",
        "Write an executive summary (200-300 words) about {topic}.",
    ],
    "journalism": [
        "Write a news article lead and body (200-300 words) about {topic}.",
        "Write an opinion editorial (200-300 words) arguing about {topic}.",
        "Write an investigative journalism paragraph (200-300 words) about {topic}.",
    ],
}

TOPICS = {
    "creative_fiction": [
        "a lighthouse keeper's last night on duty",
        "a child discovering a hidden garden",
        "two strangers meeting on a train during a storm",
        "an astronaut's first morning on Mars",
        "a musician losing their hearing",
        "a letter that arrives 50 years late",
        "the last bookstore in a digital world",
        "a grandmother teaching her grandchild to cook",
    ],
    "legal": [
        "intellectual property infringement in AI-generated art",
        "employment discrimination based on social media activity",
        "data privacy violations under GDPR",
        "breach of fiduciary duty in corporate governance",
        "environmental liability for industrial contamination",
        "tenant rights in commercial lease disputes",
        "product liability for autonomous vehicle accidents",
        "non-compete clause enforceability across state lines",
    ],
    "code_review": [
        "a React component with excessive re-renders",
        "SQL injection vulnerability in a user login system",
        "implementing rate limiting for a REST API",
        "memory leak in a Node.js WebSocket server",
        "optimizing database queries with N+1 problem",
        "implementing JWT authentication middleware",
        "handling concurrent writes in a distributed system",
        "migrating from REST to GraphQL",
    ],
    "medical": [
        "treatment-resistant depression in adolescents",
        "post-surgical complications after knee replacement",
        "managing Type 2 diabetes with comorbid hypertension",
        "early detection of pancreatic cancer biomarkers",
        "antibiotic resistance in hospital-acquired infections",
        "long COVID neurological symptoms",
        "prenatal screening for genetic disorders",
        "palliative care decision-making for terminal patients",
    ],
    "business": [
        "market entry strategy for Southeast Asian e-commerce",
        "remote work policy impact on employee retention",
        "supply chain resilience after pandemic disruptions",
        "ESG reporting requirements for public companies",
        "AI adoption ROI in manufacturing operations",
        "customer acquisition cost optimization for SaaS startups",
        "mergers and acquisitions due diligence process",
        "pricing strategy for subscription-based services",
    ],
    "journalism": [
        "rising housing costs in American college towns",
        "the impact of social media on local elections",
        "water scarcity in the American Southwest",
        "gig economy workers seeking labor protections",
        "the decline of rural hospitals",
        "cryptocurrency regulation debates in Congress",
        "school meal program funding cuts",
        "wildfire prevention and forest management policy",
    ],
}


def call_deepseek(prompt, temperature=0.7):
    import urllib.request
    body = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 500,
    }).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing file
    existing = set()
    if OUTPUT.exists():
        for line in OUTPUT.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                existing.add(f"{obj['domain']}_{obj['topic'][:30]}_{obj['prompt_idx']}")

    total = sum(len(TOPICS[d]) * len(DOMAINS[d]) for d in DOMAINS)
    done = len(existing)
    print(f"Generating {total} samples across {len(DOMAINS)} domains ({done} already done)")

    with open(OUTPUT, "a") as f:
        for domain, templates in DOMAINS.items():
            topics = TOPICS[domain]
            for ti, topic in enumerate(topics):
                for pi, template in enumerate(templates):
                    key = f"{domain}_{topic[:30]}_{pi}"
                    if key in existing:
                        continue

                    prompt = template.format(topic=topic)
                    temp = random.choice([0.5, 0.7, 0.9])  # vary temperature

                    try:
                        text = call_deepseek(prompt, temperature=temp)
                        record = {
                            "text": text,
                            "label": "ai",
                            "domain": domain,
                            "topic": topic,
                            "prompt_idx": pi,
                            "model": "deepseek-chat",
                            "temperature": temp,
                        }
                        f.write(json.dumps(record) + "\n")
                        f.flush()
                        done += 1
                        wc = len(text.split())
                        print(f"  [{done}/{total}] {domain}/{topic[:30]}... ({wc} words)")
                        time.sleep(0.3)  # rate limit
                    except Exception as e:
                        print(f"  ERROR: {domain}/{topic[:30]}: {e}", file=sys.stderr)
                        time.sleep(1)

    print(f"\nDone. {done} samples in {OUTPUT}")


if __name__ == "__main__":
    main()
