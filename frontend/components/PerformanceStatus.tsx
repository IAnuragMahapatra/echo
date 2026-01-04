"use client";

import { Activity, Clock, Zap } from "lucide-react";

type Props = {
  latencyMs: number;
  throughputMps: number;
  totalMessages: number;
  warningsCount?: number;
};

export default function PerformanceStatus({
  latencyMs,
  throughputMps,
  totalMessages,
  warningsCount = 0,
}: Props) {
  // Determine latency status color
  const latencyColor =
    latencyMs < 1000
      ? "text-emerald-400"
      : latencyMs < 3000
      ? "text-yellow-400"
      : "text-red-400";

  const latencyLabel =
    latencyMs < 1000
      ? "Excellent"
      : latencyMs < 3000
      ? "Good"
      : latencyMs < 5000
      ? "Slow"
      : "Warning";

  return (
    <div className="glass-panel rounded-2xl p-4 flex flex-col justify-between">
      <div className="text-xs font-bold text-foreground/60 uppercase tracking-widest mb-3 flex items-center gap-2">
        <Activity size={14} />
        Performance
      </div>

      <div className="space-y-3">
        {/* Latency */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs text-foreground/60">
            <Clock size={12} />
            Latency
          </div>
          <div className={`text-sm font-mono font-semibold ${latencyColor}`}>
            {latencyMs.toFixed(0)}ms
          </div>
        </div>

        {/* Throughput */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs text-foreground/60">
            <Zap size={12} />
            Throughput
          </div>
          <div className="text-sm font-mono font-semibold text-softBlue">
            {throughputMps.toFixed(1)} msg/s
          </div>
        </div>

        {/* Total Messages */}
        <div className="flex items-center justify-between">
          <div className="text-xs text-foreground/60">Messages</div>
          <div className="text-sm font-mono font-semibold text-foreground">
            {totalMessages.toLocaleString()}
          </div>
        </div>

        {/* Status indicator */}
        <div className="pt-2 border-t border-white/10">
          <div className={`text-xs font-semibold uppercase tracking-wider ${latencyColor}`}>
            {latencyLabel}
            {warningsCount > 0 && (
              <span className="ml-2 text-red-400">({warningsCount} warnings)</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
