"""Train a local LR model using the running detection server (port 5001).

Sends diverse human + AI text samples, collects PPL features, trains sklearn LR,
saves to models/perplexity_lr_v2.pkl.

Usage: python3.11 scripts/train_lr_local.py
"""
import json, os, pickle, sys
import urllib.request
import numpy as np

SERVER = "http://127.0.0.1:5001/analyze"

# --- Diverse training samples ---
# Label: 0 = human, 1 = AI

SAMPLES = [
    # === HUMAN samples ===
    (0, "I honestly had no idea what I was doing when I first tried to bake bread. The dough stuck to everything, my kitchen looked like a flour bomb went off, and the end result was basically a brick. But you know what? I kept trying, and eventually something clicked."),
    (0, "Just got back from the dentist and honestly it wasnt as bad as I thought. They said I need a crown on one of my molars which sucks but at least no root canal. The hygienist was super nice and even gave me extra numbing when I asked. Pro tip: always ask for more novocaine, they literally dont care."),
    (0, "My dog ate an entire bag of chocolate chips last night and I freaked out. Called the emergency vet at like 2am, they said to watch for vomiting. He threw up twice and then went back to sleep like nothing happened. $200 vet bill for him to literally sleep it off."),
    (0, "The thing about working from home that nobody tells you is how lonely it gets after a few months. You start talking to your plants and having full conversations with the cat. I've started going to coffee shops just to hear other humans exist."),
    (0, "So my neighbor has been playing drums at midnight for the past three weeks. I've asked nicely, left notes, even brought cookies. Nothing works. Last night I snapped and started playing the tuba at 6am. He hasn't played since."),
    (0, "We hiked to the summit and the view was absolutely breathtaking. You could see three different mountain ranges and a tiny lake that looked like a sapphire. My legs were jello by the time we got back but totally worth every step."),
    (0, "I've been teaching for 15 years and I still get nervous on the first day of school. The kids can smell fear, I swear. But once we get past the awkward introductions and someone cracks a joke, it all clicks into place."),
    (0, "Tried to fix my own plumbing yesterday. YouTube made it look so easy. Four hours later I'm standing in two inches of water calling a real plumber. The guy charged me double because of the mess I made. Lesson learned."),
    (0, "While the study had several limitations including a small sample size and the lack of a control group, the findings suggest a correlation between sleep duration and cognitive performance in elderly adults. More research is needed, particularly longitudinal studies that track participants over multiple years."),
    (0, "The patient presented with acute onset of chest pain radiating to the left arm, accompanied by diaphoresis and nausea. Initial troponin levels were elevated at 0.45 ng/mL. An ECG revealed ST-segment elevation in leads II, III, and aVF consistent with an inferior STEMI."),
    (0, "Look I know everyone says they hate meetings but some of them are actually useful. The problem is when you have eight people in a room and only two of them need to be there. Just send an email for gods sake."),
    (0, "Made my grandmas pasta recipe last night — the one she refuses to write down. I watched her make it once and took secret notes on my phone. Pretty sure she saw me but pretended not to. The sauce came out almost as good as hers. Almost."),
    (0, "Been sober for 6 months now and honestly the hardest part isn't the cravings. It's explaining to people why you're not drinking at a party. No, I don't want a mocktail. No, I'm not pregnant. I just don't drink anymore, Karen."),
    (0, "The dataset contains approximately 2.3 million records spanning fiscal years 2018 through 2023. After removing duplicates and entries with missing key fields, the clean dataset comprises 1.87 million unique records suitable for analysis."),
    (0, "I grew up in a small town in Iowa where everyone knew everyone. The kind of place where you couldn't speed because someone would call your mom before you got home. I moved to Chicago at 18 and the anonymity was both terrifying and freeing."),
    (0, "My 4 year old asked me why the sky is blue and I said something about light scattering and he just stared at me and said 'no daddy its because God painted it' and honestly his answer was better."),
    (0, "The merger between the two companies was finalized on March 15th after eighteen months of negotiations. Regulatory concerns delayed the process by approximately six months, primarily due to antitrust considerations in the European market."),
    (0, "Thanksgiving dinner was a disaster this year. Uncle Jim brought politics to the table again, mom burned the turkey because she was on the phone, and the dog ate an entire pie off the counter. We ended up ordering pizza. Best Thanksgiving ever honestly."),
    (0, "I run a small bakery in Portland and let me tell you, the gluten-free trend has been both a blessing and a curse. Revenue is up 30% but my ingredient costs have tripled. Almond flour ain't cheap people."),
    (0, "The interview went terribly. I showed up 10 minutes late because Google Maps sent me to the wrong building, my shirt had a coffee stain I didn't notice until I sat down, and I blanked on the most basic question about my own resume."),

    # === AI samples ===
    (1, "The rapid advancement of artificial intelligence has fundamentally transformed how we approach complex problem-solving in modern society. Machine learning algorithms now process vast amounts of data with unprecedented efficiency, enabling breakthroughs in healthcare, finance, and scientific research."),
    (1, "In the contemporary landscape of technological innovation, artificial intelligence stands as a transformative force reshaping industries and redefining human capabilities. The convergence of advanced algorithms, massive datasets, and computational power has enabled unprecedented progress."),
    (1, "Climate change represents one of the most pressing challenges facing humanity in the 21st century. Rising global temperatures, driven primarily by anthropogenic greenhouse gas emissions, have led to significant alterations in weather patterns, sea levels, and biodiversity across the planet."),
    (1, "The importance of mental health awareness in modern society cannot be overstated. As our understanding of psychological well-being continues to evolve, it becomes increasingly clear that mental health is just as crucial as physical health in determining overall quality of life."),
    (1, "Education serves as the cornerstone of societal progress and individual empowerment. Through the acquisition of knowledge and the development of critical thinking skills, individuals are better equipped to navigate the complexities of the modern world and contribute meaningfully to their communities."),
    (1, "The global economy has undergone a remarkable transformation in recent decades, driven by technological innovation, globalization, and shifting demographic patterns. These interconnected forces have created both unprecedented opportunities and significant challenges for nations worldwide."),
    (1, "Sustainable development has emerged as a critical paradigm for addressing the complex interplay between economic growth, social equity, and environmental preservation. As the world grapples with resource constraints and ecological degradation, the need for sustainable solutions becomes increasingly urgent."),
    (1, "The evolution of social media has profoundly impacted how individuals communicate, consume information, and form social connections. These digital platforms have democratized content creation while simultaneously raising concerns about privacy, misinformation, and mental health."),
    (1, "Blockchain technology represents a paradigm shift in how we conceptualize trust, transparency, and decentralized governance. By leveraging cryptographic principles and distributed consensus mechanisms, blockchain has the potential to revolutionize industries ranging from finance to supply chain management."),
    (1, "The intersection of ethics and artificial intelligence presents complex challenges that require careful consideration from policymakers, technologists, and society at large. As AI systems become more sophisticated and autonomous, questions about accountability, bias, and human oversight become increasingly critical."),
    (1, "Remote work has become an increasingly prevalent feature of the modern workplace, accelerated by the global pandemic and enabled by advances in communication technology. This shift has fundamentally altered the traditional employer-employee relationship and raised important questions about productivity, work-life balance, and organizational culture."),
    (1, "The healthcare industry is experiencing a period of unprecedented transformation, driven by technological innovation, changing patient expectations, and evolving regulatory frameworks. From telemedicine to precision medicine, new approaches are reshaping how healthcare is delivered and experienced."),
    (1, "Urban planning in the 21st century faces the dual challenge of accommodating population growth while ensuring environmental sustainability. Smart city initiatives leverage technology and data-driven approaches to optimize resource allocation, improve public services, and enhance the overall quality of urban life."),
    (1, "The field of renewable energy has witnessed remarkable advancements in recent years, with solar and wind technologies achieving cost parity with traditional fossil fuels in many markets. This transformation is driven by technological innovation, supportive policy frameworks, and growing awareness of climate change impacts."),
    (1, "Digital literacy has become an essential skill in the modern world, encompassing the ability to effectively navigate, evaluate, and create information using digital technologies. As our reliance on digital platforms continues to grow, the importance of developing comprehensive digital literacy programs becomes increasingly apparent."),
    (1, "The concept of emotional intelligence has gained significant traction in both academic and professional settings, reflecting a growing recognition that cognitive abilities alone are insufficient for personal and professional success. Understanding and managing emotions effectively has been linked to improved leadership, teamwork, and interpersonal relationships."),
    (1, "Space exploration continues to captivate the human imagination and drive scientific discovery. Recent missions to Mars, advances in satellite technology, and the emergence of private space companies have ushered in a new era of space exploration that promises to expand our understanding of the universe."),
    (1, "The role of government in regulating emerging technologies has become a topic of intense debate among policymakers, industry leaders, and civil society organizations. Striking the right balance between fostering innovation and protecting public interests requires nuanced approaches that adapt to the rapidly evolving technological landscape."),
    (1, "Artificial neural networks, inspired by the biological structure of the human brain, have demonstrated remarkable capabilities in pattern recognition, natural language processing, and decision-making tasks. These computational models continue to evolve, pushing the boundaries of what machines can accomplish."),
    (1, "The pharmaceutical industry plays a crucial role in advancing human health through the development of innovative therapies and preventive measures. However, the industry also faces significant challenges related to drug pricing, regulatory compliance, and the need to address emerging health threats."),

    # === BATCH 2: More diverse samples ===

    # Human: social media, forum, conversational
    (0, "Can someone PLEASE explain to me why my ISP thinks its acceptable to charge me 80 bucks a month for speeds that barely hit 10mbps?? I ran a speed test at 3pm on a TUESDAY and got 8.2 download. Eight. Point. Two."),
    (0, "The summer I turned sixteen, my dad lost his job and everything changed. We went from eating out twice a week to counting pennies for groceries. I had to drop out of soccer because we couldnt afford the cleats."),
    (0, "Started a garden last spring on a total whim. Bought some tomato plants and basil from Home Depot, stuck them in the ground, and kinda forgot about them for two weeks. When I came back out, the tomatoes were going crazy but something had eaten all my basil."),
    (0, "hey so i was thinking about what u said yesterday and ur totally right we should just go for it. worst case scenario we lose like 200 bucks which whatever thats like two dinners out. ive been looking at flights and if we book by friday theres some decent deals."),
    (0, "Went here for my anniversary and wow. The steak was cooked perfectly medium rare, the mashed potatoes were creamy and had just the right amount of garlic. Service was a bit slow at first but our waiter Carlos made up for it."),
    (0, "My 3 year old just told me she wants to be a dinosaur when she grows up and I dont have the heart to tell her thats not a career option. She also insists that broccoli is tiny trees and honestly shes not wrong."),
    (0, "PSA for anyone with a Samsung washer: do NOT ignore the recall notice. Mine started smoking last week while I was at work. If my neighbor hadnt smelled it and called the fire department, my whole apartment building could have gone up."),
    (0, "Anyone else feel like grocery prices are completely out of control? I spent 87 dollars today and came home with like two bags. Eggs are 6 bucks. SIX. I remember when they were like a dollar fifty. What is happening."),
    (0, "Okay hot take but pineapple on pizza is actually good and I will die on this hill. The sweetness of the pineapple with the saltiness of the ham and cheese is literally perfect. Fight me."),
    (0, "My cat has decided that 4am is the perfect time to practice his opera singing. Every. Single. Morning. I've tried everything — feeding him before bed, playing with him until he's tired, even those calming treats. Nothing works. He just really loves 4am apparently."),

    # Human: professional, technical, academic
    (0, "The patient presented with acute onset of chest pain radiating to the left arm, accompanied by diaphoresis and nausea. Initial troponin levels were elevated at 0.45 ng/mL. An ECG revealed ST-segment elevation in leads II, III, and aVF."),
    (0, "Q3 revenue came in at 14.2M, up 8% YoY but below our 15M target. The miss was primarily driven by delayed enterprise deals in EMEA — two contracts worth a combined 1.8M slipped to Q4. Pipeline for Q4 looks strong with 23M in qualified opportunities."),
    (0, "The defendant's motion to dismiss is hereby denied. The plaintiff has demonstrated sufficient facts to state a claim under Section 1983. The case will proceed to discovery. Counsel for both parties shall submit a proposed scheduling order within 14 days."),
    (0, "After three iterations we finally got the API latency down to under 50ms p99. The bottleneck was the N+1 query in the user permissions check — switched it to a single JOIN with a materialized view and it dropped from 200ms to 12ms. Deployed to staging, monitoring overnight."),
    (0, "The absorption spectrum showed a characteristic peak at 420nm consistent with the formation of the gold nanoparticle conjugate. TEM imaging confirmed an average particle diameter of 15.3 ± 2.1nm with a narrow size distribution."),
    (0, "We need to talk about the deployment pipeline. Three times this month we've had hotfixes that bypassed staging entirely because 'it was urgent.' I get it, stuff breaks, but we can't keep doing cowboy deploys to prod. I'm proposing a 30-minute fast-track staging gate."),
    (0, "The audit identified 47 instances of non-compliance with SOC 2 Type II controls, primarily concentrated in access management and change control processes. Of these, 12 were classified as high severity requiring immediate remediation."),
    (0, "Honestly the hardest part of being a freelance designer isn't the work itself, it's chasing invoices. I have one client who is currently 90 days past due on a $4,500 invoice and just keeps saying 'we'll get to it.' Meanwhile I still have rent to pay."),
    (0, "The focus group results were mixed. Participants aged 25-34 responded positively to the new branding (NPS +22) but the 45-54 cohort found it 'too modern' and 'hard to read.' We're recommending a hybrid approach that retains the new color palette but uses the original serif typeface for body copy."),
    (0, "My dissertation defense is in two weeks and I still haven't finished chapter 4. My advisor keeps telling me it's fine and that nobody ever feels ready, but the panic is real. I've been living on coffee and imposter syndrome for about three months now."),

    # AI: various styles
    (1, "The advancement of renewable energy technologies has created unprecedented opportunities for sustainable development. Solar panel efficiency has increased dramatically while costs have decreased substantially, making solar energy an increasingly viable alternative to traditional fossil fuels."),
    (1, "In today's rapidly evolving digital landscape, cybersecurity has emerged as a critical concern for organizations of all sizes. The increasing sophistication of cyber threats, coupled with the expanding attack surface created by remote work and cloud adoption, necessitates a comprehensive approach to security."),
    (1, "Water scarcity represents a growing global challenge that affects billions of people across developing and developed nations alike. Climate change, population growth, and inefficient water management practices have combined to create a crisis that demands immediate attention and innovative solutions."),
    (1, "The evolution of transportation systems reflects broader societal transformations in technology, economics, and urban planning. From horse-drawn carriages to autonomous vehicles, each innovation has reshaped how people and goods move through physical space, fundamentally altering patterns of settlement and commerce."),
    (1, "Effective leadership in the modern workplace requires a multifaceted approach that balances strategic vision with emotional intelligence. Leaders who demonstrate empathy, adaptability, and transparent communication are better positioned to navigate complex organizational challenges and inspire high-performing teams."),
    (1, "The impact of social media on political discourse has been profound and multifaceted. While these platforms have democratized access to information and enabled new forms of civic engagement, they have also facilitated the spread of misinformation and contributed to political polarization."),
    (1, "Genetic engineering has opened new frontiers in medicine, agriculture, and environmental science. The development of CRISPR-Cas9 technology has made gene editing more accessible and precise, enabling researchers to target specific genetic sequences with unprecedented accuracy and efficiency."),
    (1, "The concept of work-life balance has undergone significant transformation in recent years, particularly in the wake of the global pandemic. As remote and hybrid work arrangements become increasingly normalized, both employers and employees are redefining what constitutes a healthy and productive work environment."),
    (1, "Artificial intelligence in healthcare presents both tremendous opportunities and significant ethical challenges. Machine learning algorithms can analyze medical imaging with remarkable accuracy, potentially detecting diseases earlier than traditional methods. However, concerns about bias, privacy, and the appropriate role of AI in clinical decision-making remain."),
    (1, "The circular economy represents a fundamental shift from the traditional linear model of production and consumption. By designing products for longevity, reuse, and recyclability, businesses can reduce waste, conserve resources, and create new economic opportunities while minimizing environmental impact."),

    # AI: listicle and structured
    (1, "Here are five key benefits of regular exercise. First, physical activity helps maintain a healthy weight by burning calories and boosting metabolism. Second, exercise strengthens the cardiovascular system, reducing the risk of heart disease. Third, regular movement improves mental health by releasing endorphins."),
    (1, "There are several important factors to consider when choosing a college. Academic reputation plays a crucial role in ensuring quality education. Campus culture and social environment significantly impact the overall student experience. Financial considerations, including tuition costs and available scholarships, often determine feasibility."),

    # AI: email and formal
    (1, "Dear Team, I am writing to inform you about the significant developments in our artificial intelligence initiative. Our machine learning models have demonstrated exceptional performance in processing large-scale datasets, achieving unprecedented accuracy rates across multiple benchmarks."),
    (1, "Thank you for your inquiry regarding our services. We are pleased to provide you with a comprehensive overview of our offerings. Our company specializes in delivering innovative solutions that leverage cutting-edge technology to address complex business challenges. We would be happy to schedule a consultation at your earliest convenience."),

    # AI: Wikipedia-like (important to include — DeBERTa struggles with these)
    (1, "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations. The devices that perform quantum computations are known as quantum computers."),
    (1, "The French Revolution was a period of radical political and societal change in France that began with the Estates General of 1789 and ended with the formation of the French Consulate in November 1799. Many of its ideas are considered fundamental principles of liberal democracy."),
    (1, "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. The process begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data."),
    (1, "The Internet of Things refers to the interconnected network of physical devices, vehicles, home appliances, and other items embedded with electronics, software, sensors, and connectivity which enables these objects to connect and exchange data."),

    # Human: creative writing
    (0, "The rain came down in sheets, turning the parking lot into a shallow lake. Sarah sat in her car, watching the windshield wipers fight a losing battle. She was twenty minutes early for the interview, which meant she had twenty minutes to talk herself out of going in. The building looked exactly like every other corporate building she'd ever seen — all glass and ambition."),
    (0, "He found the letter in a shoebox under his mother's bed, three weeks after the funeral. The handwriting was shaky, the ink faded to a pale blue, but he recognized it immediately. It was from his father. A man who, as far as he'd known his whole life, had died before he was born. The letter was dated two years after his birth."),
]

def get_features(text):
    """Send text to detection server and extract PPL features."""
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request(SERVER, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return None

    ps = result.get("perplexity_stats")
    if not ps or "perplexity" not in ps:
        return None

    import math
    diveye = ps.get("diveye", {})
    return [
        math.log(max(ps["perplexity"], 1e-5)),
        ps.get("top10_pct", 80),
        ps.get("mean_entropy", 2.5),
        ps.get("top1_pct", 50),
        ps.get("entropy_std", 2.0),
        diveye.get("surprisal_mean", 0),
        diveye.get("surprisal_std", 0),
        diveye.get("surprisal_var", 0),
        diveye.get("surprisal_skew", 0),
        diveye.get("surprisal_kurtosis", 0),
        diveye.get("diff1_mean", 0),
        diveye.get("diff1_std", 0),
        diveye.get("diff2_var", 0),
        diveye.get("diff2_entropy", 0),
        diveye.get("diff2_autocorr", 0),
        ps.get("specdetect_energy", 0),
    ]


def main():
    print(f"Collecting features from {len(SAMPLES)} samples...")
    X, y = [], []
    for label, text in SAMPLES:
        tag = "human" if label == 0 else "AI"
        features = get_features(text)
        if features is None:
            print(f"  SKIP ({tag}): no PPL data")
            continue
        X.append(features)
        y.append(label)
        ppl = np.exp(features[0])
        print(f"  {tag}: ppl={ppl:.1f} top10={features[1]:.1f} ent={features[2]:.2f}")

    X = np.array(X)
    y = np.array(y)
    print(f"\nCollected {len(X)} samples ({sum(y==0)} human, {sum(y==1)} AI)")

    if len(X) < 10:
        print("ERROR: Not enough samples", file=sys.stderr)
        sys.exit(1)

    # Train LR with StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])

    # Cross-validation
    scores = cross_val_score(pipeline, X, y, cv=min(5, len(X)//4), scoring="accuracy")
    print(f"Cross-val accuracy: {scores.mean():.1%} (+/- {scores.std():.1%})")

    # Train on full data
    pipeline.fit(X, y)

    # Show predictions on training data
    probs = pipeline.predict_proba(X)
    print("\nPer-sample predictions:")
    for i, (label, text) in enumerate(SAMPLES):
        if i >= len(probs):
            break
        tag = "human" if label == 0 else "AI"
        ai_prob = probs[i][1] * 100
        pred = "AI" if ai_prob > 50 else "human"
        ok = "OK" if (pred == tag or (tag == "human" and pred == "human") or (tag == "AI" and pred == "AI")) else "MISS"
        print(f"  [{ok}] {tag:>5} → ai_prob={ai_prob:5.1f}% {text[:50]}...")

    # Save
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "perplexity_lr_v2.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"model": pipeline, "features": 16, "version": "v2_local"}, f)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
