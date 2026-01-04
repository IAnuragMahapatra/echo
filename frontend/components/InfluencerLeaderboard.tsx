"use client";

import { useEffect, useState, useMemo } from "react";
import { Users, TrendingUp, TrendingDown, Minus, Crown, Star, Activity } from "lucide-react";

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
  pulseScore?: number;  // Connect to dashboard pulse score
  onSentimentChange?: (avgSentiment: number, bullishCount: number, bearishCount: number) => void;
}

export default function InfluencerLeaderboard({
  limit = 10,
  refreshInterval = 5000,
  pulseScore = 5.0,
  onSentimentChange,
}: InfluencerLeaderboardProps) {
  const [influencers, setInfluencers] = useState<Influencer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Calculate aggregate sentiment metrics
  const sentimentMetrics = useMemo(() => {
    if (influencers.length === 0) return { avg: 0, bullish: 0, bearish: 0, neutral: 0 };

    const bullish = influencers.filter(i => i.sentiment > 0.3).length;
    const bearish = influencers.filter(i => i.sentiment < -0.3).length;
    const neutral = influencers.length - bullish - bearish;
    const avg = influencers.reduce((sum, i) => sum + i.sentiment, 0) / influencers.length;

    return { avg, bullish, bearish, neutral };
  }, [influencers]);

  // Notify parent of sentiment changes
  useEffect(() => {
    if (onSentimentChange && influencers.length > 0) {
      onSentimentChange(sentimentMetrics.avg, sentimentMetrics.bullish, sentimentMetrics.bearish);
    }
  }, [sentimentMetrics, onSentimentChange, influencers.length]);

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
        // Fall back to simulated data that correlates with pulse score
        setError("Demo Data");
        setInfluencers(getSimulatedInfluencers(limit, pulseScore));
      } finally {
        setLoading(false);
      }
    };

    fetchInfluencers();
    const interval = setInterval(fetchInfluencers, refreshInterval);
    return () => clearInterval(interval);
  }, [limit, refreshInterval, pulseScore]);

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

  // Determine consensus alignment with pulse score
  const consensusAlignment = useMemo(() => {
    const pulseBullish = pulseScore > 6.5;
    const pulseBearish = pulseScore < 3.5;
    const influencerBullish = sentimentMetrics.avg > 0.2;
    const influencerBearish = sentimentMetrics.avg < -0.2;

    if ((pulseBullish && influencerBullish) || (pulseBearish && influencerBearish)) {
      return "aligned";
    } else if ((pulseBullish && influencerBearish) || (pulseBearish && influencerBullish)) {
      return "divergent";
    }
    return "neutral";
  }, [pulseScore, sentimentMetrics.avg]);

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
        <div className="flex items-center gap-2">
          {/* Sentiment summary badge */}
          <div className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs ${
            consensusAlignment === "aligned" ? "bg-emerald-400/20 text-emerald-400" :
            consensusAlignment === "divergent" ? "bg-red-400/20 text-red-400" :
            "bg-yellow-400/20 text-yellow-400"
          }`}>
            <Activity className="w-3 h-3" />
            {consensusAlignment === "aligned" ? "Aligned" :
             consensusAlignment === "divergent" ? "Divergent" : "Mixed"}
          </div>
          {error && (
            <span className="text-xs text-yellow-400/60">Demo</span>
          )}
        </div>
      </div>

      {/* Sentiment breakdown bar */}
      <div className="mb-4">
        <div className="flex h-2 rounded-full overflow-hidden bg-white/5">
          <div
            className="bg-emerald-400 transition-all duration-500"
            style={{ width: `${(sentimentMetrics.bullish / Math.max(influencers.length, 1)) * 100}%` }}
          />
          <div
            className="bg-yellow-400 transition-all duration-500"
            style={{ width: `${(sentimentMetrics.neutral / Math.max(influencers.length, 1)) * 100}%` }}
          />
          <div
            className="bg-red-400 transition-all duration-500"
            style={{ width: `${(sentimentMetrics.bearish / Math.max(influencers.length, 1)) * 100}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-muted-foreground mt-1">
          <span className="text-emerald-400">{sentimentMetrics.bullish} Bullish</span>
          <span className="text-yellow-400">{sentimentMetrics.neutral} Neutral</span>
          <span className="text-red-400">{sentimentMetrics.bearish} Bearish</span>
        </div>
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

// Simulated data for demo/fallback - correlates with pulse score
function getSimulatedInfluencers(limit: number, pulseScore: number = 5.0): Influencer[] {
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

  // Bias sentiment based on pulse score (higher pulse = more bullish influencers)
  const sentimentBias = (pulseScore - 5) / 10; // -0.5 to +0.5 bias

  return influencers.slice(0, limit).map((inf, index) => {
    const engagement = inf.base_engagement + Math.floor(Math.random() * 1000 - 500);
    const influence_score = inf.followers * 0.6 + engagement * 0.4;

    // Top influencers tend to align with market sentiment more
    const alignmentFactor = index < 3 ? 0.6 : 0.3;
    const baseSentiment = Math.random() * 1.3 - 0.5; // Range: -0.5 to 0.8
    const sentiment = Math.max(-1, Math.min(1, baseSentiment + sentimentBias * alignmentFactor));

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
