# Crypto Narrative Pulse Tracker

## Real-Time Sentiment Intelligence for Crypto Traders

---

## The Problem

**Crypto markets move at the speed of social media.**

Every day, traders miss critical opportunities because:

- **Information Overload**: Thousands of tweets, Discord messages, and Telegram posts flood the crypto space every minute
- **Delayed Reactions**: By the time you manually spot a trend, it's often too late
- **Emotional Bias**: FOMO and FUD cloud judgment without objective data
- **Influencer Noise**: Hard to distinguish genuine signals from paid promotions
- **Price-Sentiment Disconnect**: Price action alone doesn't tell the full story

**The result?** Traders enter too late, exit too early, or miss narratives entirely.

---

## The Solution

**Crypto Narrative Pulse Tracker** transforms social chaos into actionable intelligence.

A real-time streaming analytics platform that:

1. **Ingests** social media signals from multiple sources
2. **Analyzes** sentiment using AI-enhanced NLP with crypto-specific vocabulary
3. **Calculates** a unified "Pulse Score" (1-10) combining multiple signals
4. **Detects** emerging narratives through phrase clustering
5. **Tracks** influencer consensus separately from retail sentiment
6. **Alerts** traders via Telegram when momentum shifts
7. **Answers** questions through RAG-powered AI chat

**One number. Real-time. Actionable.**

---

## Live Demo

### Dashboard Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  NARRATIVE PULSE                    Live Sentiment Stream Active    │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────────┐  ┌─────────┐  ┌─────────────────┐  │
│  │  PULSE   │  │  TRENDING    │  │  $MEME  │  │  PERFORMANCE    │  │
│  │   8.5    │  │  • to moon   │  │ $0.012  │  │  Latency: 50ms  │  │
│  │   🚀     │  │  • diamond   │  │  +2.3%  │  │  5.2 msg/sec    │  │
│  └──────────┘  └──────────────┘  └─────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│  [═══════════════════════════════════════════════════════] Score   │
│  Score History: Real-time pulse score over last 24 hours           │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐  │
│  │  INFLUENCER CONSENSUS       │  │  AI ASSISTANT               │  │
│  │  Strongly Bullish           │  │  "What's the sentiment?"    │  │
│  │  8 bullish / 2 bearish      │  │  > Based on current data... │  │
│  └─────────────────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Demo Simulation

The platform includes a **Hype Cycle Simulator** that demonstrates the full lifecycle:

| Phase    | Volume | Sentiment    | Typical Phrases                    |
|----------|--------|--------------|-----------------------------------|
| Seed     | 10%    | 0.1 - 0.4    | "early gem", "hidden alpha"       |
| Growth   | 40%    | 0.3 - 0.7    | "to the moon", "lfg", "bullish"   |
| Peak     | 30%    | 0.5 - 0.9    | "100x potential", "never selling" |
| Decline  | 20%    | -0.3 - 0.3   | "taking profits", "cooling off"   |

**Run the demo:**

```bash
python main.py --demo
```

Watch as the Pulse Score rises through seed → growth → peak, then cools off during decline, with real-time Telegram alerts at threshold crossings.

---

## Technical Deep Dive

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                  │
│  [Simulator] [Twitter] [Discord] [Telegram]                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PATHWAY STREAMING PIPELINE                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │  Webhook    │→ │  Sentiment  │→ │  Sliding    │→ │  Pulse     │ │
│  │  Ingestion  │  │  Analysis   │  │  Windows    │  │  Score     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
│         │                                                    │       │
│         ▼                                                    ▼       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │  Phrase     │  │  Influencer │  │  Divergence │  │  Live      │ │
│  │  Clustering │  │  Tracking   │  │  Detection  │  │  Metrics   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌───────────┐   ┌───────────┐   ┌───────────┐
       │  REST API │   │  Telegram │   │  RAG Chat │
       │  /metrics │   │  Alerts   │   │  /query   │
       └───────────┘   └───────────┘   └───────────┘
              │
              ▼
       ┌───────────────────────────────────────────┐
       │           NEXT.JS DASHBOARD               │
       │  Real-time charts, AI chat, alerts        │
       └───────────────────────────────────────────┘
```

### Core Components

#### 1. Sentiment Analysis (VADER + Crypto Lexicon)

Standard NLP fails on crypto slang. We enhanced VADER with 30+ crypto-specific terms:

```python
CRYPTO_LEXICON = {
    # Bullish signals
    "moon": 3.0, "mooning": 3.5, "lfg": 2.5, "wagmi": 2.0,
    "diamond": 2.0, "hodl": 1.5, "bullish": 2.5,

    # Bearish signals
    "rug": -4.0, "scam": -3.5, "rekt": -3.0, "ngmi": -2.0,
    "dump": -3.0, "fud": -2.0, "bearish": -2.5
}
```

**Result:** "$MEME to the moon! 🚀" → sentiment: +0.85 (vs +0.2 with vanilla VADER)

#### 2. Pulse Score Algorithm

The Pulse Score (1-10) combines four weighted signals:

| Component           | Points | Thresholds                          |
|---------------------|--------|-------------------------------------|
| Sentiment Velocity  | 0-4    | >0.7 = 4pts, >0.4 = 2pts           |
| Phrase Frequency    | 0-3    | >20 = 3pts, >10 = 1.5pts           |
| Influencer Ratio    | 0-3    | >0.7 = 3pts, >0.5 = 1.5pts         |
| Divergence Penalty  | -1-0   | Bearish divergence = -1pt          |

```python
def calculate_pulse_score(sentiment, phrases, influencers, divergence):
    score = 0
    score += 4 if sentiment > 0.7 else 2 if sentiment > 0.4 else 0
    score += 3 if phrases > 20 else 1.5 if phrases > 10 else 0
    score += 3 if influencers > 0.7 else 1.5 if influencers > 0.5 else 0
    score -= 1 if divergence == "bearish_divergence" else 0
    return max(1, min(10, score))
```

#### 3. Phrase Clustering (Bigram/Trigram Analysis)

Detects emerging narratives by tracking phrase frequency:

```python
# Input: 100 messages mentioning "$MEME"
# Output:
[
    {"phrase": "to the moon", "frequency": 23, "trending": True},
    {"phrase": "diamond hands", "frequency": 18, "trending": True},
    {"phrase": "early gem", "frequency": 7, "trending": True},
]
```

Phrases appearing ≥5 times in a 10-minute window are flagged as "trending."

#### 4. Influencer Tracking

Separates whale signals from retail noise:

```python
influence_score = (followers × 0.6) + (engagement × 0.4)
is_influencer = influence_score > 10,000
```

Tracks bullish/bearish ratio among influencers separately:

- **Strongly Bullish**: >70% bullish influencers
- **Strongly Bearish**: <30% bullish influencers

#### 5. Divergence Detection

Alerts when sentiment and price move opposite directions:

| Condition                              | Signal              |
|----------------------------------------|---------------------|
| Sentiment > 0.5 AND Price Δ < -2%      | Bearish Divergence  |
| Sentiment < -0.5 AND Price Δ > +2%     | Bullish Divergence  |

**Why it matters:** Divergences often precede trend reversals.

#### 6. Real-Time Streaming (Pathway)

Built on Pathway for true streaming analytics:

- **Sliding Windows**: 5-minute windows with 1-minute hops for sentiment velocity
- **Tumbling Windows**: 10-minute windows for influencer consensus
- **Live Subscriptions**: `pw.io.subscribe()` pushes updates to metrics store

```python
# Sentiment velocity calculation
sentiment_velocity = messages.windowby(
    pw.this.timestamp,
    window=pw.temporal.sliding(
        hop=pw.Duration.minutes(1),
        duration=pw.Duration.minutes(5),
    ),
).reduce(velocity=pw.reducers.avg(pw.this.sentiment))
```

#### 7. RAG-Powered AI Chat

Query market context with natural language:

```
User: "What's the sentiment on $MEME?"

AI: Based on current data:
📊 Pulse Score: 8.2/10 - Strong bullish momentum
🔥 Trending: "to the moon", "diamond hands", "lfg"
👥 Influencers: Strongly bullish (80% positive)
📈 Status: Aligned (sentiment matches price action)

The community is showing significant excitement with elevated
engagement. This typically indicates growing interest, but
remember - high momentum can reverse quickly.
```

### Performance Monitoring

Real-time tracking of system health:

| Metric              | Target    | Current |
|---------------------|-----------|---------|
| End-to-end Latency  | <5s       | ~50ms   |
| Throughput          | >1 msg/s  | 5+ msg/s|
| P95 Latency         | <3s       | ~150ms  |

Warnings logged when latency exceeds 5 seconds.

---

## Tech Stack

| Layer        | Technology                                    |
|--------------|-----------------------------------------------|
| Streaming    | Pathway (real-time data processing)           |
| Sentiment    | VADER + Custom Crypto Lexicon                 |
| Backend      | Flask + Flask-CORS + Flask-SocketIO           |
| Frontend     | Next.js 14 + Tailwind CSS + Recharts          |
| AI/RAG       | Ollama (local LLM) + Custom retriever         |
| Alerts       | Telegram Bot API                              |
| Price Data   | CoinGecko API (with caching + rate limiting)  |
| Testing      | pytest (153 tests)                            |
| Deployment   | Docker + Docker Compose                       |

---

## API Reference

| Endpoint              | Method | Description                        |
|-----------------------|--------|------------------------------------|
| `/api/metrics`        | GET    | Current pulse score and metrics    |
| `/api/metrics/history`| GET    | Historical scores for charting     |
| `/api/performance`    | GET    | Latency and throughput metrics     |
| `/api/query`          | POST   | RAG query for market insights      |
| `/api/config`         | POST   | Update tracked coin                |
| `/health`             | GET    | Health check                       |
| `/`                   | POST   | Webhook for message ingestion      |

---

## Quick Start

### Docker (Recommended)

```bash
# Start full stack
docker-compose up

# With demo simulator
docker-compose --profile demo up
```

### Local Development

```bash
# Backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py --demo

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

### Configuration

```bash
# .env
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHANNEL_ID=@your_channel
TRACKED_COIN=MEME
COINGECKO_API_KEY=CG-xxx  # Optional
```

---

## Test Coverage

**153 tests** covering:

- Core pipeline functionality
- Sentiment analysis accuracy
- Pulse score calculation
- Phrase clustering
- Influencer tracking
- Divergence detection
- Performance monitoring
- API endpoints

```bash
python -m pytest tests/ -v
```

---

## Roadmap

### Phase 1 ✅ Complete

- Real-time sentiment analysis
- Pulse score calculation
- Telegram alerts
- Basic dashboard

### Phase 2 ✅ Complete

- Phrase clustering
- Influencer tracking
- Divergence detection
- Performance monitoring

### Phase 3 🚧 In Progress

- Multi-coin support
- Historical backtesting
- Custom alert thresholds
- Mobile app

### Phase 4 📋 Planned

- Twitter/X integration
- Discord bot
- Portfolio tracking
- ML-based predictions

---

## Why This Matters

**For Traders:**

- Spot narratives before they peak
- Objective data over emotional decisions
- Real-time alerts, not delayed analysis

**For Projects:**

- Monitor community sentiment
- Track narrative adoption
- Identify influencer engagement

**For Researchers:**

- Quantified social signals
- Historical sentiment data
- Correlation analysis tools

---

## The Team

Built with ❤️ for the crypto community.

---

## Contact

- GitHub: [Repository Link]
- Telegram: [@CryptoPulseBot]
- Demo: [Live Dashboard URL]

---

*Disclaimer: This tool provides sentiment analysis only. It is NOT financial advice. Always DYOR.*
