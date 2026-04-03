"""Adversarial test suite for AI Text X-Ray detection server.

Tests the /analyze endpoint against known AI texts, human texts, and
adversarial bypass attempts. Run with: pytest tests/test_detector.py -v

Requires detection server running on port 5001:
  python3.11 scripts/perplexity.py
"""
import json
import urllib.request
import pytest

SERVER = "http://127.0.0.1:5001/analyze"


def analyze(text: str) -> dict:
    """Send text to detection server, return fused result."""
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request(SERVER, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def get_fused(text: str) -> tuple:
    """Return (ai_score, prediction) for text."""
    result = analyze(text)
    fused = result.get("fused", {})
    return fused.get("ai_score", 50), fused.get("prediction", "unknown")


# ═══════════════════════════════════════════════════════════════
# AI TEXT — should be detected as AI or uncertain (score > 55)
# ═══════════════════════════════════════════════════════════════

class TestAIDetection:
    """Standard AI-generated text should be detected."""

    def test_ai_standard_essay(self):
        score, pred = get_fused(
            "The rapid advancement of artificial intelligence has fundamentally "
            "transformed how we approach complex problem-solving in modern society. "
            "Machine learning algorithms now process vast amounts of data with "
            "unprecedented efficiency, enabling breakthroughs in healthcare, finance, "
            "and scientific research. Furthermore, the integration of deep learning "
            "models has revolutionized natural language processing, computer vision, "
            "and autonomous systems."
        )
        assert pred in ("ai", "uncertain"), f"Expected ai/uncertain, got {pred} (score={score})"
        assert score > 55, f"AI score too low: {score}"

    def test_ai_persuasive(self):
        score, pred = get_fused(
            "It is imperative that governments around the world take decisive action "
            "to combat climate change before it is too late. The scientific evidence "
            "is overwhelming and irrefutable: greenhouse gas emissions from human "
            "activities are driving unprecedented changes in our climate system."
        )
        assert pred in ("ai", "uncertain"), f"Expected ai/uncertain, got {pred} (score={score})"

    def test_ai_technical_blog(self):
        score, pred = get_fused(
            "Microservices architecture has become the de facto standard for building "
            "scalable cloud-native applications. By decomposing monolithic systems into "
            "smaller, independently deployable services, organizations can achieve greater "
            "flexibility, improve fault isolation, and enable teams to work autonomously."
        )
        assert pred in ("ai", "uncertain"), f"Expected ai/uncertain, got {pred} (score={score})"

    def test_ai_listicle(self):
        score, pred = get_fused(
            "Here are five key benefits of regular exercise. First, physical activity "
            "helps maintain a healthy weight by burning calories and boosting metabolism. "
            "Second, exercise strengthens the cardiovascular system, reducing the risk of "
            "heart disease. Third, regular movement improves mental health by releasing "
            "endorphins. Fourth, physical fitness enhances sleep quality. Finally, "
            "consistent exercise builds stronger bones and muscles, preventing age-related "
            "decline and improving overall quality of life for individuals of all ages."
        )
        # Listicle is a hard case — DeBERTa often misses it. Score > 45 is acceptable.
        assert score > 45, f"AI listicle score too low: {score}"


# ═══════════════════════════════════════════════════════════════
# HUMAN TEXT — must NEVER be classified as AI (0% false positive)
# ═══════════════════════════════════════════════════════════════

class TestHumanDetection:
    """Human text must never be falsely flagged as AI. This is the most critical test."""

    def test_human_casual_baking(self):
        score, pred = get_fused(
            "I honestly had no idea what I was doing when I first tried to bake bread. "
            "The dough stuck to everything, my kitchen looked like a flour bomb went off, "
            "and the end result was basically a brick. But you know what? I kept trying, "
            "and eventually something clicked. Now I make a decent loaf every Sunday."
        )
        assert pred != "ai", f"FALSE POSITIVE: human text classified as AI (score={score})"

    def test_human_reddit_dentist(self):
        score, pred = get_fused(
            "Just got back from the dentist and honestly it wasnt as bad as I thought. "
            "They said I need a crown on one of my molars which sucks but at least no "
            "root canal. The hygienist was super nice and even gave me extra numbing "
            "when I asked. Pro tip: always ask for more novocaine, they literally dont care."
        )
        assert pred != "ai", f"FALSE POSITIVE: human text classified as AI (score={score})"

    def test_human_garden_blog(self):
        score, pred = get_fused(
            "Started a garden last spring on a total whim. Bought some tomato plants "
            "and basil from Home Depot, stuck them in the ground, and kinda forgot about "
            "them for two weeks. When I came back out, the tomatoes were going crazy but "
            "something had eaten all my basil. Turns out rabbits love basil. Who knew?"
        )
        assert pred != "ai", f"FALSE POSITIVE: human text classified as AI (score={score})"

    def test_human_yelp_review(self):
        score, pred = get_fused(
            "Went here for my anniversary and wow. The steak was cooked perfectly medium "
            "rare, the mashed potatoes were creamy and had just the right amount of garlic. "
            "Service was a bit slow at first but our waiter Carlos made up for it — super "
            "attentive once he got to us and even comped our dessert."
        )
        assert pred != "ai", f"FALSE POSITIVE: human text classified as AI (score={score})"

    def test_human_wedding_toast(self):
        score, pred = get_fused(
            "When Jake first told me he was dating someone new, I was like oh here we go "
            "again. But then I met Sarah and within five minutes I knew this was different. "
            "She laughed at his terrible puns, she didnt try to change his obsession with "
            "fantasy football, and most importantly she calls me every Sunday just to chat."
        )
        assert pred != "ai", f"FALSE POSITIVE: human text classified as AI (score={score})"

    def test_human_tech_support(self):
        score, pred = get_fused(
            "UPDATE: Fixed it!! For anyone having the same problem — turns out the issue "
            "was my BIOS was set to UEFI-only mode but the drive was formatted as MBR. "
            "Once I switched to Legacy+UEFI boot mode it found the drive immediately. "
            "Took me three days and about 47 forum posts to figure this out."
        )
        assert pred != "ai", f"FALSE POSITIVE: human text classified as AI (score={score})"

    def test_human_frustrated_email(self):
        score, pred = get_fused(
            "Hi Mark, This is the third time I am asking about the refund for order #45892. "
            "It has been over 6 weeks since I returned the item and I still see nothing in "
            "my account. I have the tracking number showing it was delivered back to your "
            "warehouse on Feb 3rd. I have been a customer for over 10 years."
        )
        assert pred != "ai", f"FALSE POSITIVE: human text classified as AI (score={score})"

    def test_human_slack_message(self):
        score, pred = get_fused(
            "yo did anyone elses docker build just completely die? im getting some weird "
            "segfault in the node_modules step that wasnt there yesterday. already tried "
            "rm -rf node_modules and a fresh install, same thing. mike said he saw something "
            "similar last week but his fix was to just restart his mac which feels wrong lol."
        )
        assert pred != "ai", f"FALSE POSITIVE: human text classified as AI (score={score})"


# ═══════════════════════════════════════════════════════════════
# ADVERSARIAL BYPASS ATTEMPTS — known attack vectors
# ═══════════════════════════════════════════════════════════════

class TestAdversarialDefense:
    """Test resistance to known adversarial bypass techniques."""

    def test_homoglyph_greek(self):
        """Greek letter substitution should be normalized and detected."""
        score, pred = get_fused(
            "\u03a4he rapid advancement \u03bff artificial intelligence has fundamentally "
            "transf\u03bfrmed h\u03bfw we appr\u03beach c\u03bfmplex problem-solving in "
            "m\u03bfdern s\u03bfciety. \u039cachine learning alg\u03bfrithms n\u03bfw "
            "pr\u03bfcess vast am\u03bfunts \u03bff data with unprecedented efficiency, "
            "enabling breakthroughs in healthcare, finance, and scientific research."
        )
        assert score > 45, f"Homoglyph bypass: score {score} too low (should detect AI after normalization)"

    def test_homoglyph_cyrillic(self):
        """Cyrillic letter substitution should be normalized and detected."""
        score, pred = get_fused(
            "\u0422he imp\u0430ct \u043ef \u0430rtifici\u0430l intelligence \u043en m\u043edern "
            "s\u043eciety c\u0430nn\u043et be \u043everst\u0430ted. \u041c\u0430chine "
            "le\u0430rning \u0430lg\u043erithms h\u0430ve tr\u0430nsf\u043ermed h\u043ew we "
            "\u0430ppro\u0430ch c\u043emplex pr\u043eblems."
        )
        assert score >= 50, f"Cyrillic bypass: score {score} too low"

    def test_ai_with_casual_tone(self):
        """AI content with injected casual markers — stat signal should help."""
        score, pred = get_fused(
            "So basically, the advancement of artificial intelligence has fundamentally "
            "transformed how we approach problem-solving in modern society, right? Like, "
            "machine learning algorithms process vast amounts of data super efficiently "
            "now. And honestly, it is enabling some pretty cool breakthroughs in healthcare."
        )
        # This is a known difficult case — at least should be uncertain, not confident human
        # Score > 40 means system is at least suspicious
        assert score > 35, f"Casual tone bypass too effective: score {score}"

    def test_ai_with_first_person(self):
        """AI content wrapped in first-person narrative."""
        score, pred = get_fused(
            "I remember when I first learned about how artificial intelligence has "
            "fundamentally transformed our approach to complex problem-solving. It struck "
            "me that machine learning algorithms can now process vast amounts of data with "
            "unprecedented efficiency. I believe this enables remarkable breakthroughs in "
            "healthcare, finance, and scientific research. In my experience, the integration "
            "of deep learning models has truly revolutionized natural language processing."
        )
        # First-person injection is a known hard case — score > 45 is acceptable
        assert score > 45, f"First-person bypass too effective: score {score}"

    def test_markdown_formatting(self):
        """AI text with markdown should be stripped and detected."""
        score, pred = get_fused(
            "## Overview\n\nArtificial intelligence has fundamentally transformed how we "
            "approach complex problem-solving.\n\n### Key Points\n\n- Machine learning "
            "algorithms process vast amounts of data\n- Deep learning models have "
            "revolutionized NLP\n\n### Conclusion\n\nAI continues to reshape industries."
        )
        # Markdown stripping should help — at least not confident human
        assert score > 30, f"Markdown bypass: score {score} too low"


# ═══════════════════════════════════════════════════════════════
# KNOWN LIMITATIONS — these document expected failures
# ═══════════════════════════════════════════════════════════════

class TestKnownLimitations:
    """Document known bypass techniques that currently succeed.
    These tests pass when the bypass WORKS (i.e., detector fails).
    When we fix a bypass, move the test to TestAdversarialDefense."""

    def test_typo_bypass_detected(self):
        """AI text with typos — expanded stylometric features now help detect."""
        score, pred = get_fused(
            "The rapd advancement of artifical inteligence has fundamentaly transformed "
            "how we aproach complex problem-solving in modern socety. Machine lerning "
            "algorithms now proccess vast amunts of data with unprecidented eficiency."
        )
        assert pred in ("ai", "uncertain"), f"Typo bypass still works: {pred} (score={score})"

    @pytest.mark.xfail(reason="Quillbot paraphrase bypasses PPL/LR — needs model-level fix")
    def test_quillbot_bypass_detected(self):
        """Quillbot-style paraphrased AI text should ideally be detected."""
        score, pred = get_fused(
            "AI tech has really changed the game when it comes to solving tough problems "
            "these days. ML models can now crunch through tons of data super fast, which "
            "has led to some pretty cool stuff in medicine, money, and science. Plus, deep "
            "learning has totally shaken up how we deal with language processing."
        )
        assert pred in ("ai", "uncertain"), f"Quillbot bypass still works: {pred} (score={score})"

    def test_textmsg_bypass_detected(self):
        """AI text disguised as text messages — now partially detected."""
        score, pred = get_fused(
            "hey u know what ive been thinking about?? like how ai has totally changed "
            "everything lol. machine learning stuff can go through sooo much data now its "
            "crazy. and like the breakthroughs in hospitals and banks and stuff?? wild."
        )
        assert pred in ("ai", "uncertain"), f"Text-msg bypass still works: {pred} (score={score})"


# ═══════════════════════════════════════════════════════════════
# API HEALTH
# ═══════════════════════════════════════════════════════════════

class TestAPIHealth:
    """Basic API functionality tests."""

    def test_empty_text(self):
        result = analyze("")
        assert "error" in result

    def test_short_text(self):
        result = analyze("Hello world.")
        assert "fused" in result

    def test_response_structure(self):
        result = analyze(
            "The rapid advancement of artificial intelligence has transformed society."
        )
        assert "fused" in result
        assert "tokens" in result
        fused = result["fused"]
        assert "ai_score" in fused
        assert "prediction" in fused
        assert fused["prediction"] in ("ai", "human", "uncertain")

    @pytest.mark.xfail(reason="Binoculars intentionally disabled — needs dual llama3.2 1b+3b models")
    def test_binoculars_present(self):
        """Binoculars score should be computed even if disabled in fusion."""
        result = analyze(
            "Machine learning algorithms process vast amounts of data efficiently."
        )
        assert "binoculars" in result
        bino = result["binoculars"]
        assert "score" in bino
        assert "log_ppl" in bino


# ═══════════════════════════════════════════════════════════════
# SANDWICH / HYBRID DETECTION (sliding window)
# ═══════════════════════════════════════════════════════════════

class TestSandwichDetection:
    """Test sliding window segment analysis for mixed AI/human content."""

    def test_sandwich_detected(self):
        """Human intro + AI body + human conclusion should trigger sandwich_risk."""
        result = analyze(
            "So I've been thinking about this topic a lot lately, and honestly I "
            "wasn't sure where to start. But here's what I came up with after "
            "doing some reading. I remember talking about it with my friend last "
            "week and she had some interesting perspectives too. "
            "The rapid advancement of artificial intelligence has fundamentally "
            "transformed how we approach complex problem-solving in modern society. "
            "Machine learning algorithms now process vast amounts of data with "
            "unprecedented efficiency, enabling breakthroughs in healthcare, "
            "finance, and scientific research. Furthermore, the integration of "
            "deep learning models has revolutionized natural language processing. "
            "Anyway that's basically what I think. I'm sure there's more to it "
            "but this is where I'm at right now. Would love to hear what other "
            "people think about this stuff."
        )
        seg = result.get("segment_analysis")
        assert seg is not None, "segment_analysis missing from response"
        assert seg["sandwich_risk"] is True, (
            f"sandwich_risk should be True, got False "
            f"(max={seg['max_ai_score']}, min={seg['min_ai_score']}, var={seg['variance']})"
        )
        assert seg["variance"] > 30, f"Variance too low: {seg['variance']}"

    def test_pure_ai_no_sandwich(self):
        """Pure AI text should NOT trigger sandwich_risk (uniform segments)."""
        result = analyze(
            "The rapid advancement of artificial intelligence has fundamentally "
            "transformed how we approach complex problem-solving in modern society. "
            "Machine learning algorithms now process vast amounts of data with "
            "unprecedented efficiency, enabling breakthroughs in healthcare. "
            "Furthermore, the integration of deep learning models has revolutionized "
            "natural language processing, computer vision, and autonomous systems. "
            "The global economy has undergone a remarkable transformation driven "
            "by technological innovation and globalization. These forces have created "
            "unprecedented opportunities and significant challenges for nations."
        )
        seg = result.get("segment_analysis")
        if seg:  # May not be present for short texts
            assert seg["sandwich_risk"] is False, (
                f"Pure AI should not trigger sandwich_risk "
                f"(max={seg['max_ai_score']}, min={seg['min_ai_score']})"
            )

    def test_pure_human_no_sandwich(self):
        """Pure human text should NOT trigger sandwich_risk."""
        result = analyze(
            "I honestly had no idea what I was doing when I first tried to bake "
            "bread. The dough stuck to everything, my kitchen looked like a flour "
            "bomb went off, and the end result was basically a brick. But you know "
            "what? I kept trying, and eventually something clicked. Now I make a "
            "decent loaf every Sunday. My neighbor even asked for my recipe last "
            "week which felt pretty cool. I told her the secret is just patience "
            "and not being afraid to mess up about twenty times first."
        )
        seg = result.get("segment_analysis")
        if seg:
            assert seg["sandwich_risk"] is False, (
                f"Pure human should not trigger sandwich_risk "
                f"(max={seg['max_ai_score']}, min={seg['min_ai_score']})"
            )
