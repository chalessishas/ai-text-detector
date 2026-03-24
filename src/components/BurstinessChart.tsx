"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { SentenceData } from "@/lib/analysis";

interface Props {
  sentences: SentenceData[];
}

export default function BurstinessChart({ sentences }: Props) {
  const wordCounts = sentences.map((s) => s.wordCount);
  const mean = wordCounts.length > 0
    ? wordCounts.reduce((a, b) => a + b, 0) / wordCounts.length
    : 0;
  const variance = wordCounts.length > 0
    ? wordCounts.reduce((sum, w) => sum + (w - mean) ** 2, 0) / wordCounts.length
    : 0;
  const std = Math.sqrt(variance);
  const cv = mean > 0 ? std / mean : 0;

  const cvLabel =
    cv < 0.3 ? "Low variation" : cv < 0.5 ? "Moderate" : "High variation";
  const cvColor =
    cv < 0.3
      ? "text-[#c44] bg-red-50"
      : cv < 0.5
        ? "text-[var(--accent)] bg-orange-50"
        : "text-[#4a8] bg-emerald-50";

  return (
    <div className="bg-white rounded-2xl p-6 border border-[var(--card-border)] shadow-sm">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-sm font-semibold text-[var(--foreground)]">
          Burstiness Distribution
        </h3>
        <span
          className={`text-xs px-2 py-0.5 rounded-full ${cvColor}`}
        >
          CV: {cv.toFixed(2)} &middot; {cvLabel}
        </span>
      </div>
      <p className="text-xs text-[var(--muted)] mb-5">
        AI sentences cluster in a narrow range. Human writing varies more.
      </p>
      <div style={{ width: "100%", height: 220 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={sentences}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e8e2d9" />
            <XAxis
              dataKey="index"
              stroke="#c4bfb7"
              tick={{ fontSize: 10, fill: "#8b8580" }}
            />
            <YAxis
              stroke="#c4bfb7"
              tick={{ fontSize: 10, fill: "#8b8580" }}
            />
            <Tooltip
              contentStyle={{
                background: "#fff",
                border: "1px solid #e8e2d9",
                borderRadius: 12,
                fontSize: 12,
                boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
              }}
              formatter={(value) => [value, "Words"]}
              labelFormatter={(_label, payload) => {
                const item = payload?.[0]?.payload as SentenceData;
                if (!item) return "";
                const preview =
                  item.text.length > 40 ? item.text.slice(0, 40) + "…" : item.text;
                return `"${preview}"`;
              }}
            />
            <Bar dataKey="wordCount" fill="#c96442" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
