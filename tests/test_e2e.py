"""End-to-end smoke tests for AI Text X-Ray.

Tests the full stack: Next.js frontend → API routes → Python backends.
Requires all services running:
  - Next.js dev server on port 3000
  - Detection server on port 5001
  - Humanizer server on port 5002 (optional, tests skip if unavailable)

Run with: pytest tests/test_e2e.py -v
"""
import json
import urllib.request
import pytest

FRONTEND = "http://localhost:3000"
DETECTOR = "http://127.0.0.1:5001"
HUMANIZER = "http://127.0.0.1:5002"


def _get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except Exception as e:
        return None, str(e)


def _post(url, data, timeout=30):
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except Exception as e:
        return None, str(e)


# ═══════════════════════════════════════════════════════════════
# FRONTEND PAGES
# ═══════════════════════════════════════════════════════════════

class TestFrontendPages:
    """Frontend pages should load without errors."""

    def test_landing_page(self):
        status, body = _get(FRONTEND)
        assert status == 200, f"Landing page failed: {body[:100]}"
        assert "AI Text X-Ray" in body

    def test_app_page(self):
        status, body = _get(f"{FRONTEND}/app")
        assert status == 200, f"App page failed: {body[:100]}"
        assert "Analyze" in body or "Detector" in body

    def test_blog_page(self):
        status, body = _get(f"{FRONTEND}/blog")
        assert status == 200, f"Blog page failed: {body[:100]}"
        assert "Dev Log" in body


# ═══════════════════════════════════════════════════════════════
# DETECTION API (via Next.js proxy)
# ═══════════════════════════════════════════════════════════════

class TestDetectionAPI:
    """Detection should work end-to-end through Next.js API route."""

    def test_analyze_ai_text(self):
        status, result = _post(f"{FRONTEND}/api/analyze", {
            "text": "The rapid advancement of artificial intelligence has fundamentally "
                    "transformed how we approach complex problem-solving in modern society. "
                    "Machine learning algorithms now process vast amounts of data with "
                    "unprecedented efficiency, enabling breakthroughs in healthcare."
        })
        assert status == 200, f"API failed: {result}"
        assert "fused" in result
        fused = result["fused"]
        assert fused["prediction"] in ("ai", "uncertain"), (
            f"Expected AI detection, got {fused['prediction']} (score={fused['ai_score']})"
        )

    def test_analyze_human_text(self):
        status, result = _post(f"{FRONTEND}/api/analyze", {
            "text": "I honestly had no idea what I was doing when I first tried to bake "
                    "bread. The dough stuck to everything, my kitchen looked like a flour "
                    "bomb went off, and the end result was basically a brick. But you know "
                    "what? I kept trying, and eventually something clicked."
        })
        assert status == 200, f"API failed: {result}"
        fused = result["fused"]
        assert fused["prediction"] != "ai", (
            f"FALSE POSITIVE: human text flagged as AI (score={fused['ai_score']})"
        )


# ═══════════════════════════════════════════════════════════════
# WRITING CENTER API
# ═══════════════════════════════════════════════════════════════

class TestWritingCenterAPI:
    """Writing Center API should respond to all actions."""

    def test_daily_tip(self):
        status, result = _post(f"{FRONTEND}/api/writing-assist", {
            "action": "daily-tip", "text": "", "context": {}
        })
        assert status == 200, f"daily-tip failed: {result}"
        assert "tip" in result

    def test_guide(self):
        status, result = _post(f"{FRONTEND}/api/writing-assist", {
            "action": "guide",
            "text": "I want to write about climate change",
            "context": {"messages": []}
        })
        assert status == 200, f"guide failed: {result}"
        assert "cards" in result or "type" in result

    def test_expand(self):
        status, result = _post(f"{FRONTEND}/api/writing-assist", {
            "action": "expand",
            "text": "Climate change is a problem",
            "document": "Climate change is a problem. We need solutions.",
            "context": {}
        })
        assert status == 200, f"expand failed: {result}"


# ═══════════════════════════════════════════════════════════════
# HUMANIZER API (optional — server may not be running)
# ═══════════════════════════════════════════════════════════════

class TestHumanizerAPI:
    """Humanizer API should work when server is available."""

    @pytest.fixture(autouse=True)
    def check_humanizer(self):
        """Skip all tests if humanizer server is not running."""
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(2)
            s.connect(("127.0.0.1", 5002))
            s.close()
        except (ConnectionRefusedError, socket.timeout, OSError):
            pytest.skip("Humanizer server not running on port 5002")

    def test_humanize_via_proxy(self):
        status, result = _post(f"{FRONTEND}/api/humanize", {
            "text": "The advancement of AI has transformed modern society significantly."
        }, timeout=60)
        assert status == 200, f"Humanize failed: {result}"

    def test_humanize_direct(self):
        status, result = _post(f"{HUMANIZER}/humanize", {
            "text": "Machine learning enables breakthroughs in healthcare and science."
        }, timeout=60)
        assert status == 200, f"Direct humanize failed: {result}"
        assert "details" in result or "humanized" in result


# ═══════════════════════════════════════════════════════════════
# BACKEND HEALTH
# ═══════════════════════════════════════════════════════════════

class TestBackendHealth:
    """Backend servers should be responsive and healthy."""

    def test_detector_responds(self):
        status, result = _post(f"{DETECTOR}/analyze", {"text": "hello"})
        assert status == 200, f"Detector not responding: {result}"

    def test_detector_has_ppl(self):
        """PPL model must be loaded for reliable detection."""
        status, result = _post(f"{DETECTOR}/analyze", {
            "text": "Testing if perplexity model is loaded and computing token features."
        })
        assert status == 200
        ppl = result.get("perplexity_stats") or {}
        assert ppl.get("perplexity") is not None, (
            "PPL model not loaded — detection will be unreliable (DeBERTa-only)"
        )
