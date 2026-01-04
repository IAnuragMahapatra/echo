# Jury Questions - Crypto Narrative Pulse Tracker

## Most Probable Questions

### 1. What problem does this solve?

**Answer:** Crypto traders miss momentum shifts because they can't monitor thousands of social media posts in real-time. Our system aggregates social sentiment, detects trending narratives, and provides a single "Pulse Score" (1-10) so traders can quickly assess market momentum without manually scrolling through Twitter/Discord.

### 2. How does the Pulse Score work?

**Answer:** It's a composite score (1-10) combining:

- **Sentiment velocity** (0-4 points): How fast sentiment is changing
- **Phrase frequency** (0-3 points): Trending narrative detection
- **Influencer ratio** (0-3 points): What big accounts are saying
- **Divergence penalty** (-1 point): When price and sentiment disagree

Score ≥7 = Strong Buy Signal, Score ≤3 = Cooling Off

### 3. Why Pathway instead of Kafka/Spark?

**Answer:** Pathway provides:

- Native Python API (faster development)
- Built-in temporal windowing (`pw.temporal.sliding()`)
- Integrated vector store for RAG
- Single framework for streaming + ML
- Simpler deployment than Kafka+Spark combo

### 4. How do you handle rate limits from CoinGecko?

**Answer:** We implement:

- Token bucket rate limiter (50 calls/min)
- TTL-based caching (60 seconds default)
- Graceful fallback to cached/simulated data
- Batching requests where possible

### 5. What's the latency from message to alert?

**Answer:** Target is <5 seconds end-to-end. We track this via `transforms/performance.py` which logs warnings if latency exceeds threshold. In demo mode, alerts are near-instant.

### 6. How does the RAG system work without an LLM?

**Answer:** The RAG system has fallback responses that return raw metrics (pulse score, trending phrases, consensus) without LLM interpretation. When Ollama is available, it enriches responses with AI analysis.

### 7. How do you detect divergences?

**Answer:**

- **Bearish divergence**: Sentiment > 0.5 but price falling > 2% (potential top)
- **Bullish divergence**: Sentiment < -0.5 but price rising > 2% (potential bottom)

This is a classic technical analysis signal adapted for social sentiment.

### 8. What's the influencer threshold?

**Answer:** Influence score = (followers × 0.6) + (engagement × 0.4). Authors with score > 10,000 are classified as influencers. This weights both reach and engagement.

---

## Toughest Questions

### 1. How do you prevent manipulation/gaming of the Pulse Score?

**Answer:** Several safeguards:

- Influencer weighting means random bot accounts have minimal impact
- Phrase clustering detects coordinated messaging patterns
- Divergence detection flags when sentiment doesn't match price action
- Time-windowed aggregation smooths out spam bursts

**Honest limitation:** A sophisticated coordinated attack with high-follower accounts could still game it. Future work: anomaly detection, account age verification.

### 2. What's your false positive/negative rate for buy/sell signals?

**Answer:** We haven't done backtesting against historical data yet - this is a hackathon MVP. The system is designed for **awareness**, not automated trading. We explicitly disclaim it's not financial advice.

**Future work:** Backtest against historical price data to measure signal accuracy.

### 3. How does this scale to millions of messages per second?

**Answer:** Pathway is designed for high-throughput streaming. Current architecture:

- Horizontal scaling via Docker replicas
- Pathway's internal parallelization
- Windowed aggregation reduces state size

**Honest limitation:** We haven't load-tested beyond demo scale. Production would need:

- Kubernetes deployment
- Distributed vector store
- Message queue (Kafka) for ingestion buffering

### 4. Why not use FinBERT instead of VADER for sentiment?

**Answer:** Trade-off decision:

- VADER: Fast, no GPU needed, customizable lexicon
- FinBERT: More accurate but slower, requires GPU

We chose VADER + custom crypto lexicon (moon, rug, fomo, etc.) for speed. FinBERT could be a future enhancement for higher accuracy.

### 5. How do you handle multi-language content?

**Answer:** Currently English-only. VADER is English-specific.

**Future work:** Language detection + multilingual sentiment models, or translation layer.

### 6. What happens if Telegram/CoinGecko APIs go down?

**Answer:** Graceful degradation:

- CoinGecko: Falls back to cached prices, then simulated data
- Telegram: Alerts queue locally, retry with exponential backoff
- Pipeline continues processing; alerts resume when services recover

### 7. How is this different from existing tools like LunarCrush or Santiment?

**Answer:** Key differentiators:

- **Open source** - fully customizable
- **Real-time streaming** - not batch processing
- **RAG integration** - natural language queries
- **Self-hosted** - no API costs, data privacy

**Honest comparison:** Commercial tools have more data sources and historical data. We're a focused MVP demonstrating Pathway's capabilities.

### 8. What's the business model?

**Answer:** This is a hackathon demo, but potential models:

- SaaS for crypto traders (subscription)
- White-label for exchanges/funds
- Open-source with enterprise support
- Data/signal API licensing

### 9. How do you ensure the sentiment analysis is accurate for crypto slang?

**Answer:** Custom crypto lexicon added to VADER:

- Bullish: moon (+3.0), pump (+2.5), lfg (+2.5), wagmi (+2.0)
- Bearish: rug (-4.0), scam (-3.5), dump (-3.0), rekt (-3.0)

We tested with real crypto tweets to tune scores. Still imperfect - sarcasm and context are hard.

### 10. Why not use real Twitter/X data?

**Answer:** X API is expensive ($100+/month for basic access). For hackathon:

- Hype Simulator generates realistic data
- Demonstrates full pipeline capability
- Easy to swap in real data source later

The architecture supports any webhook-based data source.

---

## Demo Flow Suggestions

1. **Start with the problem** - Show chaotic Twitter feed, impossible to track
2. **Show the dashboard** - Clean pulse score, trending phrases
3. **Trigger an alert** - Run simulator, show Telegram notification
4. **Ask a question** - Demo RAG query via Telegram or API
5. **Show divergence** - Explain the warning system
6. **Technical deep-dive** - Pathway streaming, windowing, vector store

---

## Key Metrics to Mention

- **23 tests passing** - Comprehensive test coverage
- **<5 second latency** - Real-time alerts
- **4-phase hype cycle** - Realistic simulation
- **10+ crypto terms** - Custom sentiment lexicon
- **15 messages retrieved** - RAG context window
