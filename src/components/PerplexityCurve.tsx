"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import type { TokenData } from "@/lib/analysis";

interface Props {
  tokens: TokenData[];
}

export default function PerplexityCurve({ tokens }: Props) {
  const avgPerplexity =
    tokens.length > 0
      ? tokens.reduce((sum, t) => sum + t.perplexity, 0) / tokens.length
      : 0;

  return (
    <div className="bg-white rounded-2xl p-6 border border-[var(--card-border)] shadow-sm">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-sm font-semibold text-[var(--foreground)]">
          Perplexity Curve
        </h3>
        <span className="text-xs text-[var(--muted)] bg-[var(--background)] px-2 py-0.5 rounded-full">
          avg {avgPerplexity.toFixed(2)}
        </span>
      </div>
      <p className="text-xs text-[var(--muted)] mb-5">
        Flat line = predictable (AI-like). Spikes = surprising (human-like).
      </p>
      <div style={{ width: "100%", height: 220 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={tokens}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e8e2d9" />
            <XAxis
              dataKey="position"
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
              formatter={(value) => [
                Number(value).toFixed(2),
                "Perplexity",
              ]}
              labelFormatter={(_label, payload) => {
                const item = payload?.[0]?.payload as TokenData;
                return item ? `Token ${item.position}: "${item.token}"` : "";
              }}
            />
            <ReferenceLine
              y={avgPerplexity}
              stroke="#c4bfb7"
              strokeDasharray="5 5"
            />
            <Line
              type="monotone"
              dataKey="perplexity"
              stroke="#c96442"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: "#c96442", stroke: "#fff", strokeWidth: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
