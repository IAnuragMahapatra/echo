"use client";

import React from "react";
import { TrendingUp } from "lucide-react";

type Props = {
  phrases: string[];
};

const TrendingPhrases = React.memo(function TrendingPhrases({ phrases }: Props) {
  const displayPhrases = phrases.slice(0, 5);

  return (
    <div className="glass-panel rounded-2xl p-4 flex flex-col">
      <div className="text-xs font-bold text-foreground/60 uppercase tracking-widest mb-3 flex items-center gap-2">
        <TrendingUp size={14} className="text-emerald-400" />
        Trending
      </div>

      <div className="flex-1 flex flex-col gap-2">
        {displayPhrases.length > 0 ? (
          displayPhrases.map((phrase, idx) => (
            <div
              key={`${phrase}-${idx}`}
              className="flex items-center gap-2 text-sm"
            >
              <span className="text-emerald-400 font-mono text-xs">#{idx + 1}</span>
              <span className="text-foreground/90 truncate">{phrase}</span>
            </div>
          ))
        ) : (
          <div className="text-foreground/40 text-sm italic">
            Waiting for data...
          </div>
        )}
      </div>
    </div>
  );
});

export default TrendingPhrases;
