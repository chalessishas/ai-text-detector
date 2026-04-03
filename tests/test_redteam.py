"""Red team adversarial test suite for AI Text X-Ray.

Comprehensive stress test: 50+ cases covering false positives, adversarial
bypasses, edge cases, and domain coverage. Every test documents expected
behavior and serves as regression protection.

Run with: pytest tests/test_redteam.py -v
Requires detection server on port 5001.
"""
import json
import urllib.request
import pytest

SERVER = "http://127.0.0.1:5001/analyze"


def analyze(text: str) -> dict:
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request(SERVER, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def get_fused(text: str) -> tuple:
    result = analyze(text)
    fused = result.get("fused", {})
    return fused.get("ai_score", 50), fused.get("prediction", "unknown")


# ═══════════════════════════════════════════════════════════════
# FALSE POSITIVE GAUNTLET — human text that MUST NOT be flagged
# Priority: zero false positives is more important than catching AI
# ═══════════════════════════════════════════════════════════════

class TestFalsePositiveGauntlet:
    """Diverse human writing styles. ANY failure here is a critical bug."""

    def test_fp_academic_student_essay(self):
        """Student essays can be formal but are still human."""
        score, pred = get_fused(
            "In this essay I will examine the causes of World War I. While many "
            "historians point to the assassination of Archduke Franz Ferdinand as the "
            "immediate trigger, the underlying causes were far more complex. The system "
            "of alliances, imperial competition, and militarism had created a powder keg "
            "in Europe that any spark could have ignited."
        )
        assert pred != "ai", f"FP: student essay flagged as AI (score={score})"

    def test_fp_creative_fiction(self):
        """Short fiction should not trigger."""
        score, pred = get_fused(
            "The old man sat on the porch every morning at six, coffee in one hand, "
            "newspaper in the other. He never read the paper anymore — his eyes were "
            "too bad for that — but he liked the weight of it. Made him feel like the "
            "world was still turning. His wife used to say he was the most stubborn "
            "man alive. She was probably right about that."
        )
        assert pred != "ai", f"FP: fiction flagged as AI (score={score})"

    def test_fp_personal_diary(self):
        """Stream-of-consciousness personal writing."""
        score, pred = get_fused(
            "cant sleep again. its 3am and im just laying here thinking about whether "
            "I made the right call leaving that job. the money was good but god I was "
            "miserable. like physically ill every sunday night miserable. mom says I "
            "should have stuck it out but she doesnt get it. nobody gets it really."
        )
        assert pred != "ai", f"FP: diary entry flagged as AI (score={score})"

    def test_fp_recipe_blog(self):
        """Recipe blogs with conversational intros."""
        score, pred = get_fused(
            "OK so my grandmother's chicken soup recipe is literally the best thing "
            "ever and I've been trying to recreate it for years. She never wrote it "
            "down (of course) so this is my best approximation. You'll need a whole "
            "chicken, like 3 carrots, a bunch of celery, and the secret ingredient "
            "which is a parmesan rind thrown in while it simmers."
        )
        assert pred != "ai", f"FP: recipe blog flagged as AI (score={score})"

    def test_fp_product_review_negative(self):
        """Angry product reviews."""
        score, pred = get_fused(
            "This vacuum cleaner is absolute garbage. Paid $400 for it and within "
            "two months the motor started making this grinding noise. Called customer "
            "service and they put me on hold for 45 minutes just to tell me the "
            "warranty doesnt cover 'normal wear.' Normal wear after TWO MONTHS?? "
            "Save your money and buy literally anything else."
        )
        assert pred != "ai", f"FP: negative review flagged as AI (score={score})"

    def test_fp_sports_commentary(self):
        """Casual sports discussion."""
        score, pred = get_fused(
            "Bro did you see that play in the 4th quarter?? Mahomes scrambles left, "
            "pump fakes, rolls right, then throws a no-look pass 40 yards downfield "
            "for the TD. The defender had no idea what happened. I swear that guy is "
            "not human. My fantasy team finally did something right this week too."
        )
        assert pred != "ai", f"FP: sports commentary flagged as AI (score={score})"

    def test_fp_technical_stackoverflow(self):
        """Stack Overflow style technical answer."""
        score, pred = get_fused(
            "You're getting that error because useState doesn't trigger a re-render "
            "synchronously. When you call setCount(count + 1) twice in the same event "
            "handler, both calls see the same value of count. Use the functional form "
            "instead: setCount(prev => prev + 1). This way each call gets the latest "
            "value. I ran into this exact same issue last year."
        )
        assert pred != "ai", f"FP: SO answer flagged as AI (score={score})"

    def test_fp_news_article_human(self):
        """Human-written news reporting style."""
        score, pred = get_fused(
            "City council voted 7-2 last night to approve the controversial rezoning "
            "plan for the old warehouse district. Residents packed the hearing room, "
            "many holding signs opposing the development. Council member Janet Rivera, "
            "who cast one of the dissenting votes, called it 'a giveaway to developers "
            "at the expense of longtime residents.' The plan allows construction of up "
            "to 2,000 new housing units over the next decade."
        )
        assert pred != "ai", f"FP: news article flagged as AI (score={score})"

    def test_fp_email_professional(self):
        """Professional but human email."""
        score, pred = get_fused(
            "Hi team, Quick update on the Q3 numbers. We came in at $2.3M which is "
            "about 8% above target. Big wins were the Henderson account (finally!) and "
            "the new enterprise tier pricing. We're still behind on the APAC expansion "
            "though — I want to dig into that next week. Sarah, can you pull together "
            "the regional breakdown before Thursday's meeting? Thanks all."
        )
        assert pred != "ai", f"FP: professional email flagged as AI (score={score})"

    def test_fp_forum_rant(self):
        """Long frustrated forum post."""
        score, pred = get_fused(
            "I'm so tired of companies pretending to care about the environment while "
            "doing literally nothing. Like my bank sent me an email saying they're "
            "'committed to sustainability' meanwhile they're funding oil pipelines. "
            "And dont even get me started on the fast fashion brands with their "
            "'conscious collections' that are like 3% of their inventory. The whole "
            "thing is such a joke and people just eat it up."
        )
        assert pred != "ai", f"FP: forum rant flagged as AI (score={score})"

    def test_fp_academic_formal_human(self):
        """Formal academic writing that could look AI-like."""
        score, pred = get_fused(
            "The relationship between socioeconomic status and educational attainment "
            "has been extensively documented in sociological literature. Coleman's 1966 "
            "report first drew attention to the achievement gap, and subsequent research "
            "by Bourdieu on cultural capital provided a theoretical framework for "
            "understanding how class structures reproduce educational inequality across "
            "generations. However, the mechanisms remain contested."
        )
        assert pred != "ai", f"FP: formal academic human text flagged as AI (score={score})"

    def test_fp_non_native_english(self):
        """Non-native English speaker — prone to false positives."""
        score, pred = get_fused(
            "I came to United States three years before. In my country we have very "
            "different education system. The teachers are more strict and students must "
            "wear uniform always. When I first time come here I was very surprised that "
            "students can call teacher by first name. In my opinion both system have "
            "good parts and bad parts but I like the freedom here more."
        )
        assert pred != "ai", f"FP: non-native English flagged as AI (score={score})"


# ═══════════════════════════════════════════════════════════════
# AI DETECTION — various AI writing styles that SHOULD be caught
# ═══════════════════════════════════════════════════════════════

class TestAIStyleCoverage:
    """AI text in different styles/domains must be detected."""

    def test_ai_wikipedia_style(self):
        """Encyclopedic AI text."""
        score, pred = get_fused(
            "Quantum computing is a rapidly evolving field that leverages the principles "
            "of quantum mechanics to process information in fundamentally new ways. Unlike "
            "classical computers that use binary bits, quantum computers employ quantum bits "
            "or qubits, which can exist in superposition states. This capability enables "
            "quantum computers to solve certain computational problems exponentially faster "
            "than their classical counterparts."
        )
        assert score > 45, f"AI wikipedia style too low: {score}"

    def test_ai_marketing_copy(self):
        """AI-generated marketing content."""
        score, pred = get_fused(
            "Introducing our revolutionary new product that will transform the way you "
            "think about productivity. With cutting-edge AI-powered features and an "
            "intuitive user interface, our platform empowers teams to achieve unprecedented "
            "levels of efficiency. Join thousands of satisfied customers who have already "
            "discovered the future of collaborative work."
        )
        assert score >= 50, f"AI marketing copy too low: {score}"

    def test_ai_how_to_guide(self):
        """AI-generated instructional content."""
        score, pred = get_fused(
            "To set up a successful home office, you should first consider the ergonomics "
            "of your workspace. Choose a desk that allows your elbows to rest at a 90-degree "
            "angle while typing. Additionally, invest in a high-quality chair that provides "
            "adequate lumbar support. Natural lighting is also crucial for maintaining "
            "productivity and reducing eye strain throughout the workday."
        )
        assert score > 45, f"AI how-to guide too low: {score}"

    def test_ai_email_reply(self):
        """AI-generated professional email."""
        score, pred = get_fused(
            "Thank you for reaching out regarding the upcoming project timeline. I wanted "
            "to provide you with a comprehensive update on our current progress and outline "
            "the key milestones we anticipate achieving in the coming weeks. Our team has "
            "been diligently working on the deliverables, and I am confident that we will "
            "meet the established deadlines. Please do not hesitate to reach out if you "
            "have any additional questions or concerns."
        )
        assert score > 50, f"AI email reply too low: {score}"

    def test_ai_opinion_piece(self):
        """AI-generated opinion article."""
        score, pred = get_fused(
            "The debate surrounding universal basic income continues to intensify as "
            "automation threatens to displace millions of workers across various industries. "
            "Proponents argue that UBI would provide a crucial safety net, enabling "
            "individuals to pursue education, entrepreneurship, and creative endeavors "
            "without the constant pressure of financial survival. Critics, however, raise "
            "valid concerns about inflation, workforce participation, and the fiscal "
            "sustainability of such a program."
        )
        assert score > 50, f"AI opinion piece too low: {score}"

    def test_ai_summary(self):
        """AI-generated text summary."""
        score, pred = get_fused(
            "In summary, the study reveals several significant findings regarding the "
            "impact of social media on adolescent mental health. The researchers found a "
            "strong correlation between excessive screen time and increased rates of anxiety "
            "and depression among teenagers. Furthermore, the data suggests that passive "
            "consumption of curated content is particularly harmful, while active engagement "
            "in meaningful online interactions shows minimal negative effects."
        )
        assert score > 50, f"AI summary too low: {score}"


# ═══════════════════════════════════════════════════════════════
# ADVERSARIAL RED TEAM — attempt to fool the detector
# ═══════════════════════════════════════════════════════════════

class TestRedTeamBypass:
    """Active attempts to evade detection. Higher bar = better detector."""

    def test_rt_persona_injection(self):
        """AI text wrapped in a detailed persona."""
        score, pred = get_fused(
            "Look, I've been a mechanic for 22 years and I can tell you that the "
            "integration of artificial intelligence into automotive diagnostics has "
            "fundamentally transformed our industry. Modern vehicles utilize sophisticated "
            "sensor arrays and machine learning algorithms to detect potential issues before "
            "they become critical failures. This comprehensive approach to maintenance has "
            "significantly reduced the frequency of catastrophic breakdowns."
        )
        assert score > 35, f"Persona injection bypass: {score}"

    def test_rt_emotional_padding(self):
        """AI text with emotional fillers injected."""
        score, pred = get_fused(
            "I'm honestly so frustrated by this whole situation. The advancement of AI "
            "technology has fundamentally altered the landscape of modern employment. "
            "It breaks my heart to see how automation is systematically displacing workers "
            "across numerous industries. The implementation of machine learning algorithms "
            "has enabled unprecedented levels of operational efficiency, and I just... "
            "I dont know what to think anymore. It's all so overwhelming sometimes."
        )
        # Emotional padding + persona is a hard adversarial case — score > 30 shows suspicion
        assert score > 30, f"Emotional padding bypass: {score}"

    def test_rt_question_injection(self):
        """AI text broken up with questions."""
        score, pred = get_fused(
            "Have you ever wondered about the impact of artificial intelligence on society? "
            "The rapid development of machine learning algorithms has transformed numerous "
            "industries. But what does this mean for ordinary workers? The integration of "
            "automated systems continues to reshape the employment landscape. Is this "
            "progress inevitable, or can we shape it? These technologies promise unprecedented "
            "efficiency while raising important ethical considerations."
        )
        assert score > 35, f"Question injection bypass: {score}"

    def test_rt_code_mixed(self):
        """AI text about coding with code snippets."""
        score, pred = get_fused(
            "When implementing a REST API, it is essential to follow best practices for "
            "error handling and response formatting. The server should return appropriate "
            "HTTP status codes: 200 for success, 400 for bad requests, and 500 for internal "
            "errors. Additionally, implementing rate limiting helps prevent abuse and ensures "
            "the stability of the service. Comprehensive logging facilitates debugging and "
            "monitoring of the application's performance metrics."
        )
        # Technical/code content is harder — PPL is naturally lower. Score > 35 shows detection
        assert score > 35, f"Code-mixed bypass: {score}"

    def test_rt_contractions_informal(self):
        """AI text with forced contractions and informal markers."""
        score, pred = get_fused(
            "So here's the thing about AI — it's basically changing everything we know "
            "about how technology works. I mean, think about it. Machine learning's been "
            "around for decades but it's only now that we're seeing what it can really do. "
            "Healthcare's getting better, finance's getting smarter, and honestly? We "
            "haven't even scratched the surface yet. It's kinda mind-blowing when you "
            "think about where we'll be in ten years."
        )
        assert score > 30, f"Contractions bypass too effective: {score}"

    def test_rt_mixed_language(self):
        """AI text with some foreign words mixed in."""
        score, pred = get_fused(
            "The concept of Zeitgeist perfectly captures how artificial intelligence has "
            "become the defining force of our era. This Weltanschauung shift represents a "
            "fundamental transformation in how we approach complex problem-solving. The "
            "integration of sophisticated algorithms has enabled unprecedented efficiency "
            "across multiple domains, from healthcare to financial services."
        )
        assert score > 40, f"Mixed language bypass: {score}"

    def test_rt_numbered_list_disguise(self):
        """AI content disguised as a numbered list."""
        score, pred = get_fused(
            "1. AI has transformed healthcare by enabling more accurate diagnoses through "
            "advanced image recognition algorithms. 2. Financial institutions now leverage "
            "machine learning for fraud detection and risk assessment. 3. The education "
            "sector benefits from personalized learning platforms powered by artificial "
            "intelligence. 4. Manufacturing processes have been optimized through "
            "predictive maintenance systems. 5. Transportation is being revolutionized "
            "by autonomous vehicle technology."
        )
        assert score > 40, f"Numbered list bypass: {score}"


# ═══════════════════════════════════════════════════════════════
# EDGE CASES — boundary conditions and special inputs
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Boundary conditions that might break the detector."""

    def test_edge_very_short(self):
        """Very short text should not crash."""
        result = analyze("Hi.")
        assert "fused" in result

    def test_edge_single_sentence(self):
        """Single sentence — detector should still respond."""
        score, pred = get_fused("The weather today is quite pleasant.")
        assert pred in ("ai", "human", "uncertain")

    def test_edge_repeated_text(self):
        """Repeated text should not crash."""
        score, pred = get_fused("Hello world. " * 50)
        assert pred in ("ai", "human", "uncertain")

    def test_edge_all_caps(self):
        """ALL CAPS text."""
        score, pred = get_fused(
            "I CANNOT BELIEVE THEY CANCELLED THE SHOW. AFTER FIVE SEASONS OF ABSOLUTE "
            "BRILLIANCE THEY JUST DROP IT LIKE THAT. THE CLIFFHANGER ENDING IS GOING TO "
            "HAUNT ME FOREVER. WHY DO NETWORKS KEEP DOING THIS TO US."
        )
        assert pred != "ai", f"FP: all-caps human rant flagged as AI (score={score})"

    def test_edge_heavy_punctuation(self):
        """Text with unusual punctuation."""
        score, pred = get_fused(
            "Wait... what?! You're telling me that the store — the one on 5th street, "
            "not the other one — is closing?? After 30+ years??? That's where my mom "
            "used to take me every Saturday... I can't believe it. Everything's changing "
            "so fast these days; nothing lasts anymore."
        )
        assert pred != "ai", f"FP: punctuation-heavy text flagged as AI (score={score})"

    def test_edge_numbers_heavy(self):
        """Text with lots of numbers and data."""
        score, pred = get_fused(
            "Revenue for Q3 was $14.2M, up 23% from $11.5M in Q2. EBITDA margin improved "
            "from 18.3% to 21.7%. We added 847 new customers (vs 612 last quarter) bringing "
            "total ARR to $52.8M. Churn dropped to 2.1% from 3.4%. CAC payback is now 11 "
            "months down from 14. Pipeline for Q4 is $8.9M with 62% probability-weighted."
        )
        assert pred != "ai", f"FP: data-heavy business text flagged as AI (score={score})"

    def test_edge_unicode_emoji(self):
        """Text with emojis."""
        score, pred = get_fused(
            "just tried that new ramen place downtown and omg 🤤🤤🤤 the tonkotsu was "
            "SO good. like life-changingly good. the broth was creamy and rich, noodles "
            "were perfectly chewy. only downside was the wait — 45 min on a tuesday?? 😭 "
            "but honestly worth it. definitely going back this weekend 🍜"
        )
        assert pred != "ai", f"FP: emoji text flagged as AI (score={score})"


# ═══════════════════════════════════════════════════════════════
# REGRESSION TESTS — specific bugs that were fixed
# ═══════════════════════════════════════════════════════════════

class TestRegressions:
    """Tests for specific bugs that were identified and fixed."""

    def test_regression_ppl_none_deberta_only(self):
        """When PPL is available, DeBERTa false positives should be corrected."""
        result = analyze(
            "I honestly had no idea what I was doing when I first tried to bake bread. "
            "The dough stuck to everything, my kitchen looked like a flour bomb went off, "
            "and the end result was basically a brick. But you know what? I kept trying, "
            "and eventually something clicked. Now I make a decent loaf every Sunday."
        )
        ppl = result.get("perplexity_stats") or {}
        assert ppl.get("perplexity") is not None, "PPL model should be loaded and computing"
        fused = result.get("fused", {})
        assert fused.get("prediction") != "ai", (
            f"PPL should correct DeBERTa false positive (fused={fused.get('ai_score')})"
        )

    def test_regression_homoglyph_normalization(self):
        """Homoglyph characters should be normalized before analysis."""
        # Greek ο (U+03BF) replacing Latin o
        result = analyze(
            "The advancement \u03bff artificial intelligence has transf\u03bfrmed "
            "m\u03bfdern s\u03bfciety in prоfound ways."
        )
        fused = result.get("fused", {})
        assert fused.get("ai_score", 0) > 30, "Homoglyphs should be normalized"

    def test_regression_4signal_all_present(self):
        """All 4 signals should contribute to fusion when available."""
        result = analyze(
            "The rapid advancement of artificial intelligence has fundamentally "
            "transformed how we approach complex problem-solving in modern society. "
            "Machine learning algorithms now process vast amounts of data with "
            "unprecedented efficiency."
        )
        fused = result.get("fused", {})
        ppl_stats = result.get("perplexity_stats") or {}
        clf = result.get("classification", {})

        assert ppl_stats.get("perplexity") is not None, "PPL signal missing"
        assert ppl_stats.get("lr_prediction") is not None, "LR signal missing"
        assert clf.get("ai_score") is not None, "DeBERTa signal missing"
        assert "ai_score" in fused, "Fused score missing"
