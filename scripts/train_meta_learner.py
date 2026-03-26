"""Train a stacking meta-learner to replace 140-line hand-tuned fusion logic.

Collects 4-signal scores (DeBERTa, PPL, LR, Stat) from the running detection
server on labeled samples, then trains a LogisticRegression meta-model.
Saves to models/meta_fusion.pkl.

This replaces perplexity.py's complex if-else tree with:
  meta_model.predict_proba([[deb, ppl, lr, stat]])[0][1] * 100

Usage: python3.11 scripts/train_meta_learner.py
Requires: detection server running on port 5001
"""
import json
import os
import pickle
import sys
import urllib.request
import numpy as np

SERVER = "http://127.0.0.1:5001/analyze"


def get_signals(text: str) -> dict | None:
    """Get all 4 signal scores from detection server."""
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request(SERVER, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return None

    fused = result.get("fused", {})
    if not fused or "signal_source" not in fused:
        return None

    # Extract the 4 signal scores from signal_source string
    sig = fused.get("signal_source", "")
    clf = result.get("classification", {})
    ps = result.get("perplexity_stats", {})

    deb = clf.get("ai_score", 50)
    lr = ps.get("lr_ai_probability", 50)
    ppl = ps.get("perplexity", 20)
    top10 = ps.get("top10_pct", 80)

    return {
        "deb_ai": deb,
        "ppl": ppl,
        "top10": top10,
        "lr": lr,
        "fused_ai": fused.get("ai_score", 50),
    }


# Labeled samples (same as train_lr_local.py + adversarial)
SAMPLES = [
    # Human samples (label=0)
    (0, "I honestly had no idea what I was doing when I first tried to bake bread. The dough stuck to everything, my kitchen looked like a flour bomb went off, and the end result was basically a brick. But you know what? I kept trying, and eventually something clicked. Now I make a decent loaf every Sunday."),
    (0, "Just got back from the dentist and honestly it wasnt as bad as I thought. They said I need a crown on one of my molars which sucks but at least no root canal. The hygienist was super nice and even gave me extra numbing when I asked. Pro tip: always ask for more novocaine."),
    (0, "My dog ate an entire bag of chocolate chips last night and I freaked out. Called the emergency vet at like 2am, they said to watch for vomiting. He threw up twice and then went back to sleep like nothing happened. $200 vet bill for him to literally sleep it off."),
    (0, "So my neighbor has been playing drums at midnight for the past three weeks. I've asked nicely, left notes, even brought cookies. Nothing works. Last night I snapped and started playing the tuba at 6am. He hasn't played since. Problem solved I guess."),
    (0, "Started a garden last spring on a total whim. Bought some tomato plants and basil from Home Depot, stuck them in the ground, and kinda forgot about them for two weeks. When I came back out, the tomatoes were going crazy but something had eaten all my basil. Turns out rabbits love basil. Who knew?"),
    (0, "Tried to fix my own plumbing yesterday. YouTube made it look so easy. Four hours later I'm standing in two inches of water calling a real plumber. The guy charged me double because of the mess I made. Lesson learned honestly."),
    (0, "The patient presented with acute onset of chest pain radiating to the left arm, accompanied by diaphoresis and nausea. Initial troponin levels were elevated at 0.45 ng/mL. An ECG revealed ST-segment elevation in leads II, III, and aVF consistent with an inferior STEMI."),
    (0, "Can someone PLEASE explain to me why my ISP thinks its acceptable to charge me 80 bucks a month for speeds that barely hit 10mbps?? I ran a speed test at 3pm on a TUESDAY and got 8.2 download. Eight. Point. Two. I swear these companies just dont care."),
    (0, "Q3 revenue came in at 14.2M, up 8% YoY but below our 15M target. The miss was primarily driven by delayed enterprise deals in EMEA — two contracts worth a combined 1.8M slipped to Q4. Pipeline for Q4 looks strong with 23M in qualified opportunities."),
    (0, "After three iterations we finally got the API latency down to under 50ms p99. The bottleneck was the N+1 query in the user permissions check — switched it to a single JOIN with a materialized view and it dropped from 200ms to 12ms."),
    (0, "Lost my wallet at target yesterday and some random dude found it and turned it in with all my cash still inside. Theres like 200 bucks in there too. Faith in humanity restored honestly. Whoever you are, thank you."),
    (0, "Burned my tongue on hot pizza for the third time this week because I have zero patience. The roof of my mouth is basically sandpaper at this point. My wife says just wait five minutes but who has that kind of self control when theres fresh pizza sitting right there."),
    (0, "My 4 year old asked me why the sky is blue and I said something about light scattering and he just stared at me and said no daddy its because God painted it and honestly his answer was better than mine."),
    (0, "The rain came down in sheets, turning the parking lot into a shallow lake. Sarah sat in her car, watching the windshield wipers fight a losing battle. She was twenty minutes early for the interview, which meant she had twenty minutes to talk herself out of going in."),
    (0, "Finally finished painting the bedroom and I gotta say it looks amazing. Went with this sage green color my wife picked and I was skeptical at first but she was totally right. Only took 6 hours, two trips to Lowes for more tape, and one near-disaster when I knocked a full can off the ladder."),
    # AI samples (label=1)
    (1, "The rapid advancement of artificial intelligence has fundamentally transformed how we approach complex problem-solving in modern society. Machine learning algorithms now process vast amounts of data with unprecedented efficiency, enabling breakthroughs in healthcare, finance, and scientific research. Furthermore, the integration of deep learning models has revolutionized NLP."),
    (1, "Climate change represents one of the most pressing challenges facing humanity in the 21st century. Rising global temperatures, driven primarily by anthropogenic greenhouse gas emissions, have led to significant alterations in weather patterns, sea levels, and biodiversity across the planet."),
    (1, "The importance of mental health awareness in modern society cannot be overstated. As our understanding of psychological well-being continues to evolve, it becomes increasingly clear that mental health is just as crucial as physical health in determining overall quality of life."),
    (1, "Education serves as the cornerstone of societal progress and individual empowerment. Through the acquisition of knowledge and the development of critical thinking skills, individuals are better equipped to navigate the complexities of the modern world and contribute meaningfully."),
    (1, "Sustainable development has emerged as a critical paradigm for addressing the complex interplay between economic growth, social equity, and environmental preservation. As the world grapples with resource constraints, the need for sustainable solutions becomes increasingly urgent."),
    (1, "The evolution of social media has profoundly impacted how individuals communicate, consume information, and form social connections. These digital platforms have democratized content creation while simultaneously raising concerns about privacy, misinformation, and mental health."),
    (1, "Blockchain technology represents a paradigm shift in how we conceptualize trust, transparency, and decentralized governance. By leveraging cryptographic principles and distributed consensus mechanisms, blockchain has the potential to revolutionize industries ranging from finance to supply chain."),
    (1, "The healthcare industry is experiencing a period of unprecedented transformation, driven by technological innovation, changing patient expectations, and evolving regulatory frameworks. From telemedicine to precision medicine, new approaches are reshaping how healthcare is delivered."),
    (1, "The intersection of ethics and artificial intelligence presents complex challenges that require careful consideration from policymakers, technologists, and society at large. As AI systems become more sophisticated, questions about accountability, bias, and oversight become critical."),
    (1, "It is imperative that governments around the world take decisive action to combat climate change before it is too late. The scientific evidence is overwhelming and irrefutable: greenhouse gas emissions from human activities are driving unprecedented changes in our climate system."),
    (1, "The field of renewable energy has witnessed remarkable advancements in recent years, with solar and wind technologies achieving cost parity with traditional fossil fuels in many markets. This transformation is driven by technological innovation and supportive policy frameworks."),
    (1, "The integration of blockchain technology with existing financial infrastructure presents both significant opportunities and notable challenges for the global banking sector. Distributed ledger systems offer enhanced transparency, reduced transaction costs, and improved security."),
    (1, "Natural language processing has undergone a paradigm shift with the introduction of transformer-based architectures. These models, characterized by self-attention mechanisms and parallel processing capabilities, have achieved state-of-the-art results across virtually every NLP benchmark."),
    (1, "The convergence of artificial intelligence and biotechnology represents a transformative paradigm in modern healthcare. Advanced machine learning algorithms can now analyze genomic data with unprecedented precision, enabling personalized treatment strategies that were previously inconceivable."),
    (1, "Federated learning has emerged as a privacy-preserving approach to training machine learning models across distributed datasets without centralizing sensitive information. By keeping data on local devices and sharing only model updates, this paradigm addresses growing privacy concerns."),
]


def main():
    print(f"Collecting 4-signal scores from {len(SAMPLES)} samples...")
    X = []  # [deb, ppl, top10, lr]
    y = []

    for label, text in SAMPLES:
        tag = "human" if label == 0 else "AI"
        signals = get_signals(text)
        if signals is None:
            print(f"  SKIP ({tag})")
            continue
        X.append([signals["deb_ai"], signals["ppl"], signals["top10"], signals["lr"]])
        y.append(label)
        print(f"  {tag}: deb={signals['deb_ai']:.0f} ppl={signals['ppl']:.1f} top10={signals['top10']:.0f} lr={signals['lr']:.0f} → fused={signals['fused_ai']:.0f}")

    X = np.array(X)
    y = np.array(y)
    print(f"\nCollected {len(X)} samples ({sum(y==0)} human, {sum(y==1)} AI)")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])

    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"Meta-learner CV accuracy: {scores.mean():.1%} (+/- {scores.std():.1%})")

    pipeline.fit(X, y)

    # Show predictions
    probs = pipeline.predict_proba(X)
    print("\nPer-sample meta-learner predictions:")
    correct = 0
    for i, (label, text) in enumerate(SAMPLES):
        if i >= len(probs):
            break
        tag = "human" if label == 0 else "AI"
        ai_prob = probs[i][1] * 100
        pred = "AI" if ai_prob > 50 else "human"
        ok = pred.lower() == tag.lower()
        correct += ok
        mark = "OK" if ok else "MISS"
        print(f"  [{mark}] {tag:>5} → ai_prob={ai_prob:5.1f}% {text[:50]}...")
    print(f"\nAccuracy: {correct}/{len(probs)} ({correct/len(probs)*100:.1f}%)")

    # Feature importance
    coefs = pipeline.named_steps["lr"].coef_[0]
    feature_names = ["DeBERTa", "PPL", "Top10%", "LR"]
    print("\nFeature weights (meta-learner):")
    for name, coef in zip(feature_names, coefs):
        print(f"  {name}: {coef:+.3f}")

    # Save
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "meta_fusion.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "model": pipeline,
            "features": ["deb_ai", "ppl", "top10", "lr"],
            "version": "v1",
            "n_train": len(X),
        }, f)
    print(f"\nSaved to {out_path}")
    print("This can replace 140 lines of if-else in perplexity.py fusion logic.")


if __name__ == "__main__":
    main()
