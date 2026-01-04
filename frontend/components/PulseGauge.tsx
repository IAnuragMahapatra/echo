"use client";

import React from "react";

type Props = {
  score: number;
};

const PulseGauge = React.memo(function PulseGauge({ score }: Props) {
  // Clamp score to 1-10 range
  const clampedScore = Math.max(1, Math.min(10, score));

  // Calculate color based on score
  const getScoreColor = (s: number) => {
    if (s >= 7) return "text-emerald-400";
    if (s >= 4) return "text-yellow-400";
    return "text-red-400";
  };

  const getScoreLabel = (s: number) => {
    if (s >= 8) return "🚀 Very Bullish";
    if (s >= 7) return "📈 Bullish";
    if (s >= 5) return "⚖️ Neutral";
    if (s >= 3) return "📉 Bearish";
    return "❄️ Very Bearish";
  };

  const getGlowColor = (s: number) => {
    if (s >= 7) return "shadow-emerald-400/30";
    if (s >= 4) return "shadow-yellow-400/30";
    return "shadow-red-400/30";
  };

  // Calculate gauge fill percentage (1-10 maps to 0-100%)
  const fillPercent = ((clampedScore - 1) / 9) * 100;

  return (
    <div className={`glass-panel rounded-2xl p-6 h-full flex flex-col justify-center items-center shadow-lg ${getGlowColor(clampedScore)}`}>
      <div className="text-xs font-bold text-foreground/60 uppercase tracking-widest mb-4">
        Pulse Score
      </div>

      {/* Score Display */}
      <div className={`text-6xl font-extrabold ${getScoreColor(clampedScore)} mb-2`}>
        {clampedScore.toFixed(1)}
      </div>

      {/* Score Bar */}
      <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden mb-3">
        <div
          className={`h-full transition-all duration-500 rounded-full ${
            clampedScore >= 7 ? "bg-emerald-400" :
            clampedScore >= 4 ? "bg-yellow-400" :
            "bg-red-400"
          }`}
          style={{ width: `${fillPercent}%` }}
        />
      </div>

      {/* Label */}
      <div className="text-sm text-foreground/80">
        {getScoreLabel(clampedScore)}
      </div>
    </div>
  );
});

export default PulseGauge;
