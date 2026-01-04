"use client";

import { useEffect, useState } from "react";
import { Users, TrendingUp, TrendingDown, Minus, Crown, Star } from "lucide-react";

interface Influencer {
  author_id: string;
  followers: number;
  engagement: number;
  influence_score: number;
  sentiment: number;
  message_count: number;
  last_updated: string;
}

interface InfluencerLeaderboardProps {
  limit?: number;
  refreshInterval?: number;
}

export default function InfluencerLeaderboard({
  limit = 10,
  refreshInterval = 5000,
}: InfluencerLeaderboardProps) {
  const [influencers, setInfluencers] = useState<Influencer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchInfluencers = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);

        const response = await fetch(`${apiUrl}/api/influencers?limit=${limit}`, {
          signal: controller.signal,
        });
        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        setInfluencers(data.influencers || []);
        setError(null);
      } catch {
        // Silently fall back to simulated data when backend unavailable
        setError("Demo Data");
        setInfluencers(getSimulatedInfluencers(limit));
      } finally {
        setLoading(false);
      }
    };

    fetchInfluencers();
    const interval = setInterval(fetchInfluencers, refreshInterval);
    return () => clearInterval(interval);
  }, [limit, refreshInterval]);

  const getSentimentIcon = (sentiment: number) => {
    if (sentiment > 0.3) return <TrendingUp className="w-4 h-4 text-emerald-400" />;
    if (sentiment < -0.3) return <TrendingDown className="w-4 h-4 text-red-400" />;
    return <Minus className="w-4 h-4 text-yellow-400" />;
  };

  const getSentimentColor = (sentiment: number) => {
    if (sentiment > 0.3) return "text-emerald-400";
    if (sentiment < -0.3) return "text-red-400";
    return "text-yellow-400";
  };

  const getSentimentLabel = (sentiment: number) => {
    if (sentiment > 0.5) return "Very Bullish";
    if (sentiment > 0.3) return "Bullish";
    if (sentiment < -0.5) return "Very Bearish";
    if (sentiment < -0.3) return "Bearish";
    return "Neutral";
  };

  const formatFollowers = (count: number) => {
    if (count >= 1000000) return `${(count / 1000000).toFixed(1)}M`;
    if (count >= 1000) return `${(count / 1000).toFixed(1)}K`;
    return count.toString();
  };

  const getRankIcon = (index: number) => {
    if (index === 0) return <Crown className="w-5 h-5 text-yellow-400" />;
    if (index === 1) return <Star className="w-5 h-5 text-gray-300" />;
    if (index === 2) return <Star className="w-5 h-5 text-amber-600" />;
    return <span className="w-5 h-5 text-center text-sm text-muted-foreground">{index + 1}</span>;
  };

  if (loading) {
    return (
      <div className="glass-panel rounded-2xl p-6 animate-pulse">
        <div className="flex items-center gap-2 mb-4">
          <Users className="w-5 h-5 text-softBlue" />
          <h3 className="text-xs font-bold uppercase tracking-widest text-foreground/60">
            Influencer Leaderboard
          </h3>
        </div>
        <div className="space-y-3">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-12 bg-white/5 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="glass-panel rounded-2xl p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Users className="w-5 h-5 text-softBlue" />
          <h3 className="text-xs font-bold uppercase tracking-widest text-foreground/60">
            Influencer Leaderboard
          </h3>
        </div>
        {error && (
          <span className="text-xs text-yellow-400/60">Demo Data</span>
        )}
      </div>

      <div className="space-y-2">
        {influencers.map((influencer, index) => (
          <div
            key={influencer.author_id}
            className="flex items-center gap-3 p-3 rounded-xl bg-white/5 hover:bg-white/10 transition-colors"
          >
            {/* Rank */}
            <div className="flex-shrink-0 w-6 flex justify-center">
              {getRankIcon(index)}
            </div>

            {/* Author Info */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-sm truncate">
                  @{influencer.author_id}
                </span>
                {index < 3 && (
                  <span className="text-xs px-1.5 py-0.5 rounded bg-softBlue/20 text-softBlue">
                    Top {index + 1}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-3 text-xs text-muted-foreground mt-0.5">
                <span>{formatFollowers(influencer.followers)} followers</span>
                <span>•</span>
                <span>{influencer.message_count} posts</span>
              </div>
            </div>

            {/* Sentiment */}
            <div className="flex items-center gap-2">
              {getSentimentIcon(influencer.sentiment)}
              <div className="text-right">
                <div className={`text-sm font-semibold ${getSentimentColor(influencer.sentiment)}`}>
                  {(influencer.sentiment * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-muted-foreground">
                  {getSentimentLabel(influencer.sentiment)}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {influencers.length === 0 && !loading && (
        <div className="text-center py-8 text-muted-foreground">
          <Users className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">No influencer data available</p>
        </div>
      )}
    </div>
  );
}

// Simulated data for demo/fallback
function getSimulatedInfluencers(limit: number): Influencer[] {
  const influencers = [
    { author_id: "crypto_whale_1", followers: 500000, base_engagement: 15000 },
    { author_id: "degen_trader_2", followers: 250000, base_engagement: 8000 },
    { author_id: "nft_guru_3", followers: 150000, base_engagement: 5000 },
    { author_id: "defi_master_4", followers: 120000, base_engagement: 4000 },
    { author_id: "moon_hunter_5", followers: 100000, base_engagement: 3500 },
    { author_id: "alpha_seeker_6", followers: 80000, base_engagement: 2800 },
    { author_id: "chart_wizard_7", followers: 75000, base_engagement: 2500 },
    { author_id: "token_analyst_8", followers: 60000, base_engagement: 2000 },
    { author_id: "yield_farmer_9", followers: 50000, base_engagement: 1800 },
    { author_id: "gem_finder_10", followers: 45000, base_engagement: 1500 },
  ];

  return influencers.slice(0, limit).map((inf) => {
    const engagement = inf.base_engagement + Math.floor(Math.random() * 1000 - 500);
    const influence_score = inf.followers * 0.6 + engagement * 0.4;
    const sentiment = Math.random() * 1.3 - 0.5; // Range: -0.5 to 0.8

    return {
      author_id: inf.author_id,
      followers: inf.followers,
      engagement,
      influence_score,
      sentiment,
      message_count: Math.floor(Math.random() * 45) + 5,
      last_updated: new Date().toISOString(),
    };
  });
}
