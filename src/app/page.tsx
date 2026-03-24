"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

// ── X-Ray Scanner Animation ──

const SAMPLE_TEXT =
  "The rapid advancement of artificial intelligence has fundamentally transformed the way modern organizations approach complex problem-solving and decision-making processes across virtually every industry sector.";

const WORD_SCORES = [
  0.9, 0.85, 0.7, 0.3, 0.95, 0.92, 0.88, 0.4, 0.93, 0.85, 0.2, 0.91, 0.87,
  0.95, 0.6, 0.89, 0.93, 0.88, 0.3, 0.91, 0.85, 0.92, 0.45, 0.88, 0.93,
  0.91, 0.87, 0.55, 0.92, 0.89,
];

function scoreToColor(score: number, revealed: boolean): string {
  if (!revealed) return "inherit";
  if (score > 0.8) return "#c96442";
  if (score > 0.6) return "#d4956b";
  return "#5a8a6a";
}

function HeroScanner() {
  const [scanPos, setScanPos] = useState(-1);
  const [revealed, setRevealed] = useState<boolean[]>([]);
  const words = SAMPLE_TEXT.split(" ");

  useEffect(() => {
    setRevealed(new Array(words.length).fill(false));
    const timer = setTimeout(() => {
      let i = 0;
      const interval = setInterval(() => {
        if (i >= words.length) {
          clearInterval(interval);
          return;
        }
        setScanPos(i);
        setRevealed((prev) => {
          const next = [...prev];
          next[i] = true;
          return next;
        });
        i++;
      }, 120);
      return () => clearInterval(interval);
    }, 1500);
    return () => clearTimeout(timer);
  }, [words.length]);

  return (
    <div className="relative max-w-2xl mx-auto">
      {/* Scan frame */}
      <div className="relative bg-white/60 backdrop-blur-sm border border-[var(--card-border)] rounded-2xl p-8 overflow-hidden">
        {/* Scan line */}
        {scanPos >= 0 && scanPos < words.length && (
          <div
            className="absolute top-0 bottom-0 w-[2px] bg-[var(--accent)] opacity-60 z-10 transition-all duration-100"
            style={{
              left: `${((scanPos + 1) / words.length) * 100}%`,
              boxShadow: "0 0 20px 4px rgba(201, 100, 66, 0.3)",
            }}
          />
        )}

        {/* Corner brackets */}
        <div className="absolute top-3 left-3 w-5 h-5 border-t-2 border-l-2 border-[var(--accent)]/40 rounded-tl" />
        <div className="absolute top-3 right-3 w-5 h-5 border-t-2 border-r-2 border-[var(--accent)]/40 rounded-tr" />
        <div className="absolute bottom-3 left-3 w-5 h-5 border-b-2 border-l-2 border-[var(--accent)]/40 rounded-bl" />
        <div className="absolute bottom-3 right-3 w-5 h-5 border-b-2 border-r-2 border-[var(--accent)]/40 rounded-br" />

        {/* Label */}
        <div className="flex items-center gap-2 mb-4">
          <div className="w-2 h-2 rounded-full bg-[var(--accent)] animate-pulse" />
          <span
            className="text-[10px] tracking-[0.2em] uppercase text-[var(--muted)]"
            style={{ fontFamily: "var(--font-geist-mono)" }}
          >
            Scanning for AI patterns
          </span>
        </div>

        {/* Text with word-level coloring */}
        <p className="text-base leading-relaxed" style={{ fontFamily: "var(--font-geist-sans)" }}>
          {words.map((word, i) => (
            <span key={i}>
              <span
                className="transition-colors duration-300"
                style={{
                  color: scoreToColor(WORD_SCORES[i] ?? 0.5, revealed[i] ?? false),
                  fontWeight: revealed[i] && (WORD_SCORES[i] ?? 0) > 0.8 ? 500 : 400,
                }}
              >
                {word}
              </span>
              {i < words.length - 1 ? " " : ""}
            </span>
          ))}
        </p>

        {/* Legend */}
        <div className="flex items-center gap-4 mt-5 pt-4 border-t border-[var(--card-border)]">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-1.5 rounded-full bg-[#c96442]" />
            <span className="text-[10px] text-[var(--muted)]">High AI probability</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-1.5 rounded-full bg-[#d4956b]" />
            <span className="text-[10px] text-[var(--muted)]">Moderate</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-1.5 rounded-full bg-[#5a8a6a]" />
            <span className="text-[10px] text-[var(--muted)]">Likely human</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Animated counter ──

function AnimatedNumber({ value, suffix = "" }: { value: number; suffix?: string }) {
  const [display, setDisplay] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const hasAnimated = useRef(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting && !hasAnimated.current) {
          hasAnimated.current = true;
          let start = 0;
          const step = value / 40;
          const interval = setInterval(() => {
            start += step;
            if (start >= value) {
              setDisplay(value);
              clearInterval(interval);
            } else {
              setDisplay(Math.floor(start));
            }
          }, 30);
        }
      },
      { threshold: 0.5 }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [value]);

  return (
    <span ref={ref}>
      {display}
      {suffix}
    </span>
  );
}

// ── Main Page ──

export default function LandingPage() {
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const handler = () => setScrollY(window.scrollY);
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  return (
    <div className="min-h-screen bg-[var(--background)] overflow-x-hidden">
      {/* ── Nav ── */}
      <nav
        className="fixed top-0 left-0 right-0 z-50 transition-all duration-300"
        style={{
          backgroundColor: scrollY > 20 ? "rgba(249, 245, 239, 0.85)" : "transparent",
          backdropFilter: scrollY > 20 ? "blur(12px)" : "none",
          borderBottom: scrollY > 20 ? "1px solid var(--card-border)" : "1px solid transparent",
        }}
      >
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="w-8 h-8 rounded-lg bg-[var(--foreground)] flex items-center justify-center group-hover:bg-[var(--accent)] transition-colors">
              <span className="text-white text-xs font-bold tracking-tight">X</span>
            </div>
            <span className="text-sm font-medium text-[var(--foreground)]">
              AI Text X-Ray
            </span>
          </Link>
          <div className="flex items-center gap-6">
            <a
              href="#tools"
              className="text-[13px] text-[var(--muted)] hover:text-[var(--foreground)] transition-colors hidden sm:block"
            >
              Tools
            </a>
            <a
              href="#how-it-works"
              className="text-[13px] text-[var(--muted)] hover:text-[var(--foreground)] transition-colors hidden sm:block"
            >
              How it works
            </a>
            <Link
              href="/blog"
              className="text-[13px] text-[var(--muted)] hover:text-[var(--foreground)] transition-colors hidden sm:block"
            >
              Blog
            </Link>
            <Link
              href="/app"
              className="px-5 py-2 text-[13px] font-medium bg-[var(--foreground)] text-[var(--background)] rounded-full hover:opacity-90 transition-opacity"
            >
              Open App
            </Link>
          </div>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="pt-36 pb-20 px-6">
        <div className="max-w-4xl mx-auto text-center space-y-8">
          {/* Tagline */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[var(--accent)]/8 border border-[var(--accent)]/15">
            <div className="w-1.5 h-1.5 rounded-full bg-[var(--accent)]" />
            <span className="text-[11px] font-medium text-[var(--accent)] tracking-wide">
              Free &amp; open — no signup required
            </span>
          </div>

          {/* Headline */}
          <h1
            className="text-[clamp(2.5rem,6vw,4.5rem)] leading-[1.1] tracking-tight text-[var(--foreground)]"
            style={{ fontWeight: 300, letterSpacing: "-0.03em" }}
          >
            See through
            <br />
            AI-generated text
          </h1>

          {/* Subheadline */}
          <p className="text-lg text-[var(--muted)] max-w-lg mx-auto leading-relaxed">
            Five scientific signals. Word-level precision.
            <br />
            Not just a score — the <em>evidence</em> behind it.
          </p>

          {/* CTAs */}
          <div className="flex items-center justify-center gap-4 pt-2">
            <Link
              href="/app"
              className="group px-7 py-3 bg-[var(--foreground)] text-[var(--background)] text-sm font-medium rounded-full hover:opacity-90 transition-all inline-flex items-center gap-2"
            >
              Try it now
              <span className="group-hover:translate-x-0.5 transition-transform">→</span>
            </Link>
            <a
              href="#how-it-works"
              className="px-7 py-3 text-sm font-medium text-[var(--foreground)] rounded-full border border-[var(--card-border)] hover:border-[var(--foreground)]/30 transition-colors"
            >
              See how it works
            </a>
          </div>
        </div>

        {/* Scanner demo */}
        <div className="mt-16 max-w-4xl mx-auto">
          <HeroScanner />
        </div>
      </section>

      {/* ── Tools ── */}
      <section id="tools" className="py-24 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-14">
            <p
              className="text-[11px] tracking-[0.25em] uppercase text-[var(--accent)] font-medium mb-3"
              style={{ fontFamily: "var(--font-geist-mono)" }}
            >
              Three tools
            </p>
            <h2
              className="text-3xl tracking-tight text-[var(--foreground)]"
              style={{ fontWeight: 300, letterSpacing: "-0.03em" }}
            >
              Detect. Rewrite. Learn.
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                num: "01",
                title: "AI Detector",
                description:
                  "Paste text. See perplexity curves, GLTR rank distribution, entropy patterns, and sentence-level scores. Every result is explainable.",
                accent: "#3b82f6",
                href: "/app",
              },
              {
                num: "02",
                title: "Humanizer",
                description:
                  "50 million real human sentences. 11 replacement strategies. From full sentence swaps to subtle phrase adjustments. You choose the approach.",
                accent: "#22c55e",
                href: "/app",
              },
              {
                num: "03",
                title: "Writing Center",
                description:
                  "An AI writing coach that guides you through brainstorming, drafting, and revising. Feedback based on the 6+1 Traits framework.",
                accent: "#a855f7",
                href: "/app",
              },
            ].map((tool) => (
              <Link
                key={tool.num}
                href={tool.href}
                className="group relative bg-white rounded-2xl border border-[var(--card-border)] p-7 hover:shadow-lg hover:shadow-black/[0.03] transition-all duration-300 overflow-hidden"
              >
                {/* Top accent line */}
                <div
                  className="absolute top-0 left-0 right-0 h-[2px] opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ backgroundColor: tool.accent }}
                />

                <span
                  className="text-[11px] font-medium tracking-wider"
                  style={{ color: tool.accent, fontFamily: "var(--font-geist-mono)" }}
                >
                  {tool.num}
                </span>

                <h3
                  className="text-lg mt-3 mb-3 text-[var(--foreground)] tracking-tight"
                  style={{ fontWeight: 300, letterSpacing: "-0.03em" }}
                >
                  {tool.title}
                </h3>

                <p className="text-[13px] text-[var(--muted)] leading-relaxed">
                  {tool.description}
                </p>

                <span
                  className="inline-block mt-5 text-[12px] font-medium opacity-0 group-hover:opacity-100 -translate-x-2 group-hover:translate-x-0 transition-all duration-300"
                  style={{ color: tool.accent }}
                >
                  Open tool →
                </span>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* ── How It Works ── */}
      <section id="how-it-works" className="py-24 px-6 bg-[var(--foreground)]">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-14">
            <p
              className="text-[11px] tracking-[0.25em] uppercase text-[var(--accent)] font-medium mb-3"
              style={{ fontFamily: "var(--font-geist-mono)" }}
            >
              The science
            </p>
            <h2
              className="text-3xl tracking-tight text-[var(--background)]"
              style={{ fontWeight: 300, letterSpacing: "-0.03em" }}
            >
              Five signals. One verdict.
            </h2>
            <p className="text-sm text-[var(--background)]/50 mt-3 max-w-md mx-auto">
              Any single signal can be fooled. Defeating all five simultaneously is
              a different problem entirely.
            </p>
          </div>

          <div className="space-y-3">
            {[
              {
                name: "Perplexity",
                ai: "3–8",
                human: "20–50",
                desc: "How surprised is the model by each word?",
              },
              {
                name: "Token Rank",
                ai: ">90% top-10",
                human: "<75% top-10",
                desc: "Does the model's top prediction always match?",
              },
              {
                name: "Entropy",
                ai: "Low (1.0–2.0)",
                human: "High (2.5–3.5)",
                desc: "How uncertain is the model at each position?",
              },
              {
                name: "Burstiness",
                ai: "0.10–0.20",
                human: "0.35–0.65",
                desc: "Do sentence lengths vary?",
              },
              {
                name: "Vocabulary",
                ai: "Narrow",
                human: "Diverse",
                desc: "How rich is the word choice?",
              },
            ].map((s, i) => (
              <div
                key={s.name}
                className="flex items-center gap-5 bg-white/[0.05] border border-white/[0.08] rounded-xl px-6 py-4 backdrop-blur-sm"
              >
                <span
                  className="text-[10px] text-[var(--accent)] font-medium w-5"
                  style={{ fontFamily: "var(--font-geist-mono)" }}
                >
                  {String(i + 1).padStart(2, "0")}
                </span>
                <div className="w-28 shrink-0">
                  <div className="text-sm font-medium text-[var(--background)]">
                    {s.name}
                  </div>
                  <div className="text-[10px] text-[var(--background)]/40 mt-0.5">
                    {s.desc}
                  </div>
                </div>
                <div className="flex-1 flex items-center gap-3">
                  <div className="flex-1 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2.5 text-center">
                    <div className="text-[9px] text-red-400/70 font-medium uppercase tracking-wider">
                      AI
                    </div>
                    <div className="text-xs text-red-400 font-semibold mt-0.5">
                      {s.ai}
                    </div>
                  </div>
                  <div className="flex-1 bg-emerald-500/10 border border-emerald-500/20 rounded-lg px-3 py-2.5 text-center">
                    <div className="text-[9px] text-emerald-400/70 font-medium uppercase tracking-wider">
                      Human
                    </div>
                    <div className="text-xs text-emerald-400 font-semibold mt-0.5">
                      {s.human}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Numbers ── */}
      <section className="py-24 px-6">
        <div className="max-w-4xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {[
              { value: 50, suffix: "M", label: "Human sentences in corpus" },
              { value: 5, suffix: "", label: "Detection signals" },
              { value: 11, suffix: "", label: "Humanization strategies" },
              { value: 7, suffix: "", label: "Writing trait dimensions" },
            ].map((s) => (
              <div key={s.label} className="text-center">
                <div
                  className="text-4xl font-light text-[var(--foreground)] tracking-tight"
                  style={{ fontWeight: 300, letterSpacing: "-0.03em" }}
                >
                  <AnimatedNumber value={s.value} suffix={s.suffix} />
                </div>
                <div className="text-[11px] text-[var(--muted)] mt-2 leading-snug">
                  {s.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── What makes us different ── */}
      <section className="py-24 px-6">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-14">
            <h2
              className="text-3xl tracking-tight text-[var(--foreground)]"
              style={{ fontWeight: 300, letterSpacing: "-0.03em" }}
            >
              Not another black box
            </h2>
          </div>

          <div className="grid md:grid-cols-2 gap-5">
            {[
              {
                title: "We show the evidence",
                body: "Other detectors give a percentage. We give interactive charts — perplexity curves, token rank heatmaps, entropy plots. You see exactly which sentences triggered detection.",
              },
              {
                title: "Corpus, not paraphrase",
                body: "Our humanizer doesn't use AI to rewrite AI text (that's detectable too). We match against 50 million real human sentences from pre-2019 sources. Zero AI contamination.",
              },
              {
                title: "Education, not evasion",
                body: "The Writing Center teaches the 6+1 Traits framework. We help you understand what makes writing human — varied sentence length, specific detail, authentic voice.",
              },
              {
                title: "Free and transparent",
                body: "No hidden algorithms. No paywall for core features. We believe AI detection should be explainable, accessible, and honest about its limitations.",
              },
            ].map((p) => (
              <div
                key={p.title}
                className="bg-white rounded-2xl border border-[var(--card-border)] p-7 hover:shadow-lg hover:shadow-black/[0.02] transition-shadow duration-300"
              >
                <h3
                  className="text-base text-[var(--foreground)] tracking-tight mb-2"
                  style={{ fontWeight: 300, letterSpacing: "-0.03em" }}
                >
                  {p.title}
                </h3>
                <p className="text-[13px] text-[var(--muted)] leading-relaxed">
                  {p.body}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="py-24 px-6">
        <div className="max-w-2xl mx-auto text-center space-y-6">
          <h2
            className="text-3xl tracking-tight text-[var(--foreground)]"
            style={{ fontWeight: 300, letterSpacing: "-0.03em" }}
          >
            Ready to see through the text?
          </h2>
          <p className="text-sm text-[var(--muted)]">
            No signup. No paywall. Just paste and analyze.
          </p>
          <Link
            href="/app"
            className="group inline-flex items-center gap-2 px-8 py-3.5 bg-[var(--foreground)] text-[var(--background)] text-sm font-medium rounded-full hover:opacity-90 transition-all"
          >
            Open AI Text X-Ray
            <span className="group-hover:translate-x-0.5 transition-transform">→</span>
          </Link>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="py-10 px-6 border-t border-[var(--card-border)]">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded bg-[var(--foreground)] flex items-center justify-center">
              <span className="text-[var(--background)] text-[8px] font-bold">X</span>
            </div>
            <span className="text-[11px] text-[var(--muted)]">
              AI Text X-Ray
            </span>
          </div>
          <div className="flex items-center gap-5">
            <Link
              href="/app"
              className="text-[11px] text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
            >
              App
            </Link>
            <Link
              href="/blog"
              className="text-[11px] text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
            >
              Blog
            </Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
