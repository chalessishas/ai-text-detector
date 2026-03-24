"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import type { SlidingWindowData } from "@/lib/analysis";

interface Props {
  data: SlidingWindowData[];
}

export default function SlidingWindowChart({ data }: Props) {
  return (
    <div className="bg-white rounded-2xl p-6 border border-[var(--card-border)] shadow-sm">
      <h3 className="text-sm font-semibold text-[var(--foreground)] mb-1">
        Sliding Window AI Score
      </h3>
      <p className="text-xs text-[var(--muted)] mb-5">
        Pinpoint which sections push the overall score up.
      </p>
      <div style={{ width: "100%", height: 220 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#c96442" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#c96442" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#e8e2d9" />
            <XAxis
              dataKey="position"
              stroke="#c4bfb7"
              tick={{ fontSize: 10, fill: "#8b8580" }}
            />
            <YAxis
              stroke="#c4bfb7"
              tick={{ fontSize: 10, fill: "#8b8580" }}
              domain={[0, 100]}
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
                `${Number(value).toFixed(1)}%`,
                "AI Score",
              ]}
              labelFormatter={(_label, payload) => {
                const item = payload?.[0]?.payload as SlidingWindowData;
                return item?.label || "";
              }}
            />
            <ReferenceLine
              y={50}
              stroke="#c4bfb7"
              strokeDasharray="5 5"
            />
            <Area
              type="monotone"
              dataKey="score"
              stroke="#c96442"
              strokeWidth={2}
              fill="url(#scoreGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
