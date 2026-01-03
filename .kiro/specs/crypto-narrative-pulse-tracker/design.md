# Design Document: Crypto Narrative Pulse Tracker

## Overview

The Crypto Narrative Pulse Tracker is a real-time streaming analytics system built on Pathway that processes social media data and price feeds to generate actionable momentum signals for crypto traders. The system demonstrates Pathway's streaming capabilities through multi-source ingestion, temporal windowing, and continuously-updating RAG.

### Key Design Decisions

1. **Build on Pathway RAG Template**: Extend the existing template (app.py, app.yaml) rather than building from scratch
2. **Pathway-First Architecture**: All streaming logic uses Pathway's native APIs for consistency and performance
3. **Unified Message Schema**: All data sources normalize to a common schema for pipeline simplicity
4. **Modular Components**: Each transformation stage is isolated for testability and maintainability
5. **Demo-Optimized**: Hype simulator is the primary data source; external APIs are optional enhancements
6. **Consistent Windowing**: All temporal operations use 5-minute sliding windows with 1-minute hops for simplicity
7. **Template Syntax Priority**: In case of API conflicts, use syntax from the cloned template

### Implementation Phases (MVP Strategy)

**Phase 1 - Core (Must Have):**

- Hype simulator with message generation
- Sentiment analysis with crypto lexicon
- Basic pulse score (sentiment-only initially)
- Console/log output for debugging

**Phase 2 - Demo-Ready:**

- Document store with Pathway VectorStoreServer
- Basic RAG queries via REST API
- Telegram alerts (score + signal)
- Simple Streamlit dashboard (score chart)

**Phase 3 - Impressive:**

- Price correlation and divergence detection
- Phrase clustering and trending detection
- Full RAG with context enrichment
- Influencer tracking

**Phase 4 - Stretch Goals:**

- Discord connector
- Advanced dashboard visualizations
- Comprehensive property-based testing

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION LAYER                               │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│  Hype Simulator │  HTTP Webhook   │ Discord Connector│   CoinGecko Price    │
│   (Primary)     │   (Optional)    │   (Stretch)      │      Fetcher         │
└────────┬────────┴────────┬────────┴────────┬─────────┴──────────┬───────────┘
         │                 │                 │                    │
         ▼                 ▼                 ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PATHWAY STREAMING PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Sentiment   │  │   Phrase     │  │  Influence   │  │    Price     │    │
│  │  Analyzer    │  │  Clusterer   │  │  Calculator  │  │  Correlator  │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │                 │            │
│         └─────────────────┴─────────────────┴─────────────────┘            │
│                                    │                                        │
│                           ┌────────▼────────┐                              │
│                           │  Pulse Score    │                              │
│                           │   Calculator    │                              │
│                           └────────┬────────┘                              │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Document Store │       │  Alert Engine   │       │    Dashboard    │
│   (RAG Index)   │       │ (Telegram Bot)  │       │   (Streamlit)   │
└────────┬────────┘       └────────┬────────┘       └─────────────────┘
         │                         │
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│   RAG System    │◄──────│  User Queries   │
│  (GPT-4 + RAG)  │       │  via Telegram   │
└─────────────────┘       └─────────────────┘
```

## Components and Interfaces

### 1. Data Ingestion Components

#### MessageSchema (Unified Data Model)

```python
class MessageSchema(pw.Schema):
    message_id: str          # Unique identifier
    text: str                # Message content
    author_id: str           # Author identifier
    author_followers: int    # Follower/member count
    timestamp: datetime      # Event timestamp
    tags: list[str]          # Hashtags, cashtags, channel tags
    engagement_count: int    # Likes, reactions, retweets
    source_platform: str     # "simulator", "twitter", "discord"
```

#### PriceSchema

```python
class PriceSchema(pw.Schema):
    coin_symbol: str         # e.g., "MEME", "SOL"
    price_usd: float         # Current price
    timestamp: datetime      # Price timestamp
    volume_24h: float        # 24h trading volume
```

#### HTTP Webhook Connector

```python
def create_webhook_connector(host: str, port: int) -> pw.Table:
    """Creates Pathway HTTP connector for webhook ingestion."""
    return pw.io.http.rest_connector(
        host=host,
        port=port,
        schema=MessageSchema,
        delete_completed_queries=False
    )
```

#### Discord Connector (Stretch Goal)

```python
def create_discord_connector(webhook_url: str) -> pw.Table:
    """Custom Pathway connector for Discord webhooks."""
    def discord_reader():
        # Implements pw.io.python.read() pattern
        # Transforms Discord messages to MessageSchema
        pass

    return pw.io.python.read(
        discord_reader,
        schema=MessageSchema
    )
```

#### Price Fetcher

```python
class PriceFetcher:
    """Fetches price data from CoinGecko with rate limiting."""

    def __init__(self, coin_id: str, cache_ttl: int = 60):
        self.coin_id = coin_id
        self.cache = {}
        self.rate_limiter = RateLimiter(max_calls=50, period=60)

    def get_price(self) -> PriceSchema:
        """Returns cached or fresh price data."""
        pass
```

### 2. Transformation Components

#### Sentiment Analyzer

```python
class SentimentAnalyzer:
    """Analyzes sentiment using VADER with crypto-specific lexicon."""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self._add_crypto_lexicon()

    def analyze(self, text: str) -> float:
        """Returns sentiment score from -1 (bearish) to 1 (bullish)."""
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']

    def _add_crypto_lexicon(self):
        """Adds crypto-specific terms to VADER lexicon."""
        crypto_terms = {
            'moon': 3.0, 'mooning': 3.5, 'pump': 2.5,
            'dump': -3.0, 'rug': -4.0, 'scam': -3.5,
            'bullish': 2.5, 'bearish': -2.5, 'hodl': 1.5,
            'fomo': 1.0, 'fud': -2.0, 'degen': 0.5
        }
        self.analyzer.lexicon.update(crypto_terms)
```

#### Sentiment Velocity Calculator

```python
# Standard window configuration for consistency across all components
STANDARD_WINDOW = pw.Duration.minutes(5)
STANDARD_HOP = pw.Duration.minutes(1)

def calculate_sentiment_velocity(messages: pw.Table) -> pw.Table:
    """Calculates sentiment velocity using 5-min sliding window."""

    # Add sentiment scores
    with_sentiment = messages.select(
        message_id=pw.this.message_id,
        sentiment=pw.apply(sentiment_analyzer.analyze, pw.this.text),
        timestamp=pw.this.timestamp
    )

    # Calculate velocity with sliding window
    velocity = with_sentiment.windowby(
        pw.this.timestamp,
        window=pw.temporal.sliding(
            hop=STANDARD_HOP,
            duration=STANDARD_WINDOW
        )
    ).reduce(
        velocity=pw.reducers.avg(pw.this.sentiment),
        message_count=pw.reducers.count(),
        window_end=pw.this._pw_window_end
    )

    return velocity
```

#### Phrase Clusterer

```python
class PhraseClusterer:
    """Extracts and clusters trending phrases from messages."""

    def extract_phrases(self, text: str) -> list[str]:
        """Extracts bigrams and trigrams from text."""
        tokens = self._tokenize(text)
        bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
        trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
        return bigrams + trigrams

    def _tokenize(self, text: str) -> list[str]:
        """Tokenizes and filters stopwords."""
        words = text.lower().split()
        return [w for w in words if w not in STOPWORDS and len(w) > 2]

def calculate_trending_phrases(messages: pw.Table) -> pw.Table:
    """Tracks phrase frequency in 10-min rolling window."""

    phrases = messages.select(
        phrases=pw.apply(phrase_clusterer.extract_phrases, pw.this.text),
        timestamp=pw.this.timestamp
    ).flatten(pw.this.phrases)

    trending = phrases.windowby(
        pw.this.timestamp,
        window=pw.temporal.sliding(
            hop=pw.Duration.minutes(1),
            duration=pw.Duration.minutes(10)
        )
    ).groupby(pw.this.phrases).reduce(
        phrase=pw.this.phrases,
        frequency=pw.reducers.count(),
        last_seen=pw.reducers.max(pw.this.timestamp)
    ).filter(pw.this.frequency >= 5)

    return trending
```

#### Influence Calculator

```python
def calculate_influence_score(followers: int, engagement: int) -> float:
    """Calculates influence score: (followers * 0.6) + (engagement * 0.4)."""
    return (followers * 0.6) + (engagement * 0.4)

def calculate_influencer_signals(messages: pw.Table) -> pw.Table:
    """Tracks influencer sentiment in 10-min tumbling windows."""

    with_influence = messages.select(
        message_id=pw.this.message_id,
        influence_score=pw.apply(
            calculate_influence_score,
            pw.this.author_followers,
            pw.this.engagement_count
        ),
        sentiment=pw.apply(sentiment_analyzer.analyze, pw.this.text),
        timestamp=pw.this.timestamp
    )

    # Filter to influencers only (score > 10000)
    influencers = with_influence.filter(pw.this.influence_score > 10000)

    # Aggregate in tumbling windows
    signals = influencers.windowby(
        pw.this.timestamp,
        window=pw.temporal.tumbling(duration=pw.Duration.minutes(10))
    ).reduce(
        bullish_count=pw.reducers.sum(
            pw.if_else(pw.this.sentiment > 0.3, 1, 0)
        ),
        bearish_count=pw.reducers.sum(
            pw.if_else(pw.this.sentiment < -0.3, 1, 0)
        ),
        total_count=pw.reducers.count()
    )

    return signals
```

#### Price Correlator

```python
def calculate_price_correlation(
    sentiment_velocity: pw.Table,
    price_stream: pw.Table
) -> pw.Table:
    """Joins sentiment with price data to detect divergences."""

    # Calculate price delta over 5-min window (using standard window config)
    price_delta = price_stream.windowby(
        pw.this.timestamp,
        window=pw.temporal.sliding(
            hop=STANDARD_HOP,
            duration=STANDARD_WINDOW
        )
    ).reduce(
        start_price=pw.reducers.earliest(pw.this.price_usd),
        end_price=pw.reducers.latest(pw.this.price_usd),
        window_end=pw.this._pw_window_end
    ).select(
        price_delta_pct=(pw.this.end_price - pw.this.start_price) / pw.this.start_price * 100,
        window_end=pw.this.window_end
    )

    # Join with sentiment velocity using interval_join for temporal alignment
    # Note: Using standard join on window_end for simplicity in MVP
    correlated = sentiment_velocity.join(
        price_delta,
        sentiment_velocity.window_end == price_delta.window_end
    ).select(
        velocity=sentiment_velocity.velocity,
        price_delta_pct=price_delta.price_delta_pct,
        divergence_type=pw.apply(detect_divergence, sentiment_velocity.velocity, price_delta.price_delta_pct),
        timestamp=sentiment_velocity.window_end
    )

    return correlated

def detect_divergence(sentiment: float, price_delta: float) -> str:
    """Detects sentiment-price divergences."""
    if sentiment > 0.5 and price_delta < -2.0:
        return "bearish_divergence"
    elif sentiment < -0.5 and price_delta > 2.0:
        return "bullish_divergence"
    return "aligned"
```

### 3. Pulse Score Calculator

```python
class PulseScoreCalculator:
    """Combines all signals into a 1-10 momentum score."""

    def calculate(
        self,
        sentiment_velocity: float,
        phrase_frequency: int,
        influencer_ratio: float,
        divergence_type: str
    ) -> float:
        """
        Scoring formula:
        - Sentiment velocity: 0-4 points
        - Phrase frequency spike: 0-3 points
        - Influencer bullish ratio: 0-3 points
        - Divergence modifier: -1 to 0 points
        """
        score = 0.0

        # Sentiment component (0-4 points)
        if sentiment_velocity > 0.7:
            score += 4.0
        elif sentiment_velocity > 0.4:
            score += 2.0
        elif sentiment_velocity > 0.0:
            score += 1.0

        # Phrase frequency component (0-3 points)
        if phrase_frequency > 20:
            score += 3.0
        elif phrase_frequency > 10:
            score += 1.5
        elif phrase_frequency > 5:
            score += 0.5

        # Influencer component (0-3 points)
        if influencer_ratio > 0.7:
            score += 3.0
        elif influencer_ratio > 0.5:
            score += 1.5
        elif influencer_ratio > 0.3:
            score += 0.5

        # Divergence modifier
        if divergence_type == "bearish_divergence":
            score -= 1.0

        # Clamp to 1-10 range
        return max(1.0, min(10.0, score))

def calculate_pulse_score(
    sentiment_velocity: pw.Table,
    trending_phrases: pw.Table,
    influencer_signals: pw.Table,
    price_correlation: pw.Table
) -> pw.Table:
    """Combines all signals into pulse score stream."""

    # Get max phrase frequency
    max_phrase_freq = trending_phrases.reduce(
        max_frequency=pw.reducers.max(pw.this.frequency)
    )

    # Calculate influencer ratio
    influencer_ratio = influencer_signals.select(
        ratio=pw.this.bullish_count / (pw.this.bullish_count + pw.this.bearish_count + 0.001)
    )

    # Join all signals and calculate score
    # Note: Both tables use STANDARD_WINDOW, so window_end values align
    pulse = sentiment_velocity.join(
        max_phrase_freq
    ).join(
        influencer_ratio
    ).join(
        price_correlation
    ).select(
        score=pw.apply(
            pulse_calculator.calculate,
            pw.this.velocity,
            pw.this.max_frequency,
            pw.this.ratio,
            pw.this.divergence_type
        ),
        # Use window_end from sentiment_velocity for timestamp
        timestamp=pw.this.window_end
    )

    return pulse
```

### 4. Document Store and RAG System

The RAG system uses Pathway's VectorStoreServer for continuous indexing with a simple custom query layer for context enrichment.

```python
from pathway.xpacks.llm import embedders
from pathway.xpacks.llm.vector_store import VectorStoreServer
from openai import OpenAI

# Shared state for live metrics (updated by pipeline subscriptions)
live_metrics = {
    "pulse_score": 5.0,
    "trending_phrases": [],
    "influencer_consensus": "neutral",
    "divergence_status": "aligned"
}

def create_document_store(messages: pw.Table) -> VectorStoreServer:
    """Creates Pathway vector store for continuous RAG indexing."""
    embedder = embedders.SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

    indexed = messages.select(
        text=pw.this.text,
        metadata={
            "message_id": pw.this.message_id,
            "timestamp": pw.this.timestamp,
            "sentiment": pw.apply(sentiment_analyzer.analyze, pw.this.text),
            "influence_score": pw.apply(
                calculate_influence_score,
                pw.this.author_followers,
                pw.this.engagement_count
            ),
            "source_platform": pw.this.source_platform
        }
    )

    server = VectorStoreServer(
        indexed,
        embedder=embedder,
        splitter=None  # Already chunked at message level
    )
    return server

def update_live_metrics(pulse_scores: pw.Table, phrases: pw.Table, influencers: pw.Table):
    """Subscribe to pipeline outputs to update shared metrics state."""
    pw.io.subscribe(
        pulse_scores,
        on_change=lambda row: live_metrics.update({"pulse_score": row['score']})
    )
    pw.io.subscribe(
        phrases.reduce(top_phrases=pw.reducers.tuple(pw.this.phrase)[:3]),
        on_change=lambda row: live_metrics.update({"trending_phrases": list(row['top_phrases'])})
    )

def answer_query(query: str, doc_store: VectorStoreServer) -> dict:
    """Answers user query with RAG and live context enrichment."""
    # Retrieve relevant messages from Pathway vector store
    retrieved = doc_store.query(query, k=15)

    # Build context-enriched prompt
    prompt = f"""You are a crypto momentum analyst. Answer based on LIVE social data.

Current Pulse Score: {live_metrics['pulse_score']}/10
Top Trending Phrases: {', '.join(live_metrics['trending_phrases'][:3])}
Influencer Consensus: {live_metrics['influencer_consensus']}
Price-Sentiment Status: {live_metrics['divergence_status']}

Recent messages:
{chr(10).join([f"- {r.text}" for r in retrieved[:10]])}

User question: {query}

Provide:
1. Direct answer with pulse score interpretation
2. Top 3 repeating phrases from recent threads
3. Risk assessment (are we early or late to this narrative?)"""

    # Call OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.choices[0].message.content,
        "pulse_score": live_metrics['pulse_score'],
        "sources": [r.metadata for r in retrieved],
        "relevance_scores": [r.score for r in retrieved]
    }
```

### 5. Alert and Output Components

#### Telegram Bot

```python
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, MessageHandler

class TelegramAlertBot:
    """Telegram bot for alerts and queries."""

    def __init__(self, token: str, channel_id: str):
        self.bot = Bot(token=token)
        self.channel_id = channel_id

    def send_alert(self, score: float, coin: str, divergence: str, phrases: list[str]):
        """Sends momentum alert to channel."""
        if score >= 7:
            emoji = "🚀"
            signal = "Strong Buy Signal"
        elif score <= 3:
            emoji = "❄️"
            signal = "Cooling Off"
        else:
            return  # No alert for mid-range scores

        divergence_warning = ""
        if divergence == "bearish_divergence":
            divergence_warning = "\n⚠️ Warning: Bearish divergence detected!"

        message = f"""{emoji} Momentum Alert: ${coin}
📊 Pulse Score: {score}/10
📈 Signal: {signal}
🔥 Trending: {', '.join(phrases[:3])}{divergence_warning}

Reply /query for detailed analysis"""

        self.bot.send_message(chat_id=self.channel_id, text=message)

    async def handle_query(self, update: Update, context):
        """Handles user queries via RAG."""
        query = update.message.text

        # Get current metrics (from shared state)
        metrics = get_current_metrics()

        response = self.rag.answer(
            query=query,
            pulse_score=metrics['pulse_score'],
            trending_phrases=metrics['phrases'],
            influencer_consensus=metrics['consensus'],
            divergence_status=metrics['divergence']
        )

        await update.message.reply_text(response['answer'])

def subscribe_to_alerts(pulse_scores: pw.Table, bot: TelegramAlertBot):
    """Subscribes to pulse score changes for alerts."""
    pw.io.subscribe(
        pulse_scores.filter(
            (pw.this.score >= 7) | (pw.this.score <= 3)
        ),
        on_change=lambda row: bot.send_alert(
            row['score'],
            TRACKED_COIN,
            row.get('divergence', 'aligned'),
            row.get('phrases', [])
        )
    )
```

#### Hype Simulator

```python
import random
import time
import requests
from datetime import datetime

class HypeSimulator:
    """Generates realistic hype cycle data for demos."""

    PHASES = {
        'seed': {'volume_pct': 0.10, 'sentiment_range': (0.1, 0.4), 'phrases': ['early gem', 'hidden alpha']},
        'growth': {'volume_pct': 0.40, 'sentiment_range': (0.3, 0.7), 'phrases': ['to the moon', 'bullish af', 'lfg']},
        'peak': {'volume_pct': 0.30, 'sentiment_range': (0.5, 0.9), 'phrases': ['100x potential', 'generational wealth', 'never selling']},
        'decline': {'volume_pct': 0.20, 'sentiment_range': (-0.3, 0.3), 'phrases': ['taking profits', 'be careful', 'top signal']}
    }

    INFLUENCERS = [
        {'id': 'crypto_whale_1', 'followers': 500000},
        {'id': 'degen_trader_2', 'followers': 250000},
        {'id': 'nft_guru_3', 'followers': 150000}
    ]

    def __init__(self, webhook_url: str, coin_symbol: str, total_messages: int = 200, duration_mins: float = 3.0):
        self.webhook_url = webhook_url
        self.coin = coin_symbol
        self.total_messages = total_messages
        self.duration_secs = duration_mins * 60
        self.base_price = 0.001  # Starting price for simulation

    def run(self):
        """Runs the full hype cycle simulation."""
        messages_sent = 0
        phase_order = ['seed', 'growth', 'peak', 'decline']

        for phase_name in phase_order:
            phase = self.PHASES[phase_name]
            phase_messages = int(self.total_messages * phase['volume_pct'])
            phase_duration = self.duration_secs * phase['volume_pct']
            delay = phase_duration / phase_messages

            for _ in range(phase_messages):
                message = self._generate_message(phase_name, phase)
                self._send_message(message)

                # Also send correlated price update
                price = self._generate_price(phase_name, messages_sent)
                self._send_price(price)

                messages_sent += 1
                time.sleep(delay)

    def _generate_message(self, phase_name: str, phase: dict) -> dict:
        """Generates a single message for the current phase."""
        is_influencer = random.random() < 0.1
        author = random.choice(self.INFLUENCERS) if is_influencer else {
            'id': f'user_{random.randint(1000, 9999)}',
            'followers': random.randint(100, 5000)
        }

        sentiment = random.uniform(*phase['sentiment_range'])
        phrase = random.choice(phase['phrases'])

        return {
            'message_id': f'sim_{int(time.time() * 1000)}_{random.randint(0, 999)}',
            'text': f'${self.coin} {phrase}! {"🚀" if sentiment > 0.5 else ""}',
            'author_id': author['id'],
            'author_followers': author['followers'],
            'timestamp': datetime.utcnow().isoformat(),
            'tags': [f'#{self.coin}', '#crypto'],
            'engagement_count': random.randint(10, 1000) if is_influencer else random.randint(1, 50),
            'source_platform': 'simulator'
        }

    def _generate_price(self, phase_name: str, progress: int) -> dict:
        """Generates correlated price data."""
        multipliers = {'seed': 1.0, 'growth': 1.5, 'peak': 2.5, 'decline': 1.8}
        noise = random.uniform(-0.05, 0.05)
        price = self.base_price * multipliers[phase_name] * (1 + noise)

        return {
            'coin_symbol': self.coin,
            'price_usd': price,
            'timestamp': datetime.utcnow().isoformat(),
            'volume_24h': random.uniform(100000, 1000000)
        }
```

## Data Models

### Core Data Structures

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum

class SourcePlatform(Enum):
    SIMULATOR = "simulator"
    TWITTER = "twitter"
    DISCORD = "discord"

class DivergenceType(Enum):
    ALIGNED = "aligned"
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"

@dataclass
class Message:
    message_id: str
    text: str
    author_id: str
    author_followers: int
    timestamp: datetime
    tags: list[str]
    engagement_count: int
    source_platform: SourcePlatform

@dataclass
class PricePoint:
    coin_symbol: str
    price_usd: float
    timestamp: datetime
    volume_24h: float

@dataclass
class SentimentResult:
    message_id: str
    sentiment_score: float  # -1 to 1
    timestamp: datetime

@dataclass
class PhraseFrequency:
    phrase: str
    frequency: int
    last_seen: datetime
    is_trending: bool  # frequency >= 5

@dataclass
class InfluencerSignal:
    window_start: datetime
    window_end: datetime
    bullish_count: int
    bearish_count: int
    total_count: int
    consensus_ratio: float  # bullish / total

@dataclass
class PulseScore:
    score: float  # 1-10
    sentiment_velocity: float
    top_phrase_frequency: int
    influencer_ratio: float
    divergence_type: DivergenceType
    timestamp: datetime

@dataclass
class RAGResponse:
    answer: str
    pulse_score: float
    trending_phrases: list[str]
    sources: list[dict]
    relevance_scores: list[float]
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Message Schema Validation Round-Trip

*For any* valid Message object, serializing it to JSON and deserializing it back SHALL produce an equivalent Message object with all fields preserved.

**Validates: Requirements 1.1, 1.2**

### Property 2: Invalid Message Rejection

*For any* message payload that is missing required fields (message_id, text, author_id, timestamp) or has invalid field types, the schema validator SHALL reject it and return an error.

**Validates: Requirements 1.3**

### Property 3: Tag Filtering Correctness

*For any* message and any tag pattern, the message SHALL pass the filter if and only if at least one of its tags matches the pattern.

**Validates: Requirements 1.4**

### Property 4: Discord Message Transformation

*For any* valid Discord webhook payload, the transformation to MessageSchema SHALL produce a valid Message with all required fields populated.

**Validates: Requirements 2.3**

### Property 5: Sentiment Score Range

*For any* text input, the Sentiment_Analyzer SHALL return a score in the range [-1, 1] inclusive.

**Validates: Requirements 3.1**

### Property 6: Sentiment Velocity Classification

*For any* sentiment velocity value:

- If velocity > 0.7, classification SHALL be "strong_bullish_momentum"
- If velocity < -0.7, classification SHALL be "strong_bearish_momentum"
- Otherwise, classification SHALL be "neutral" or "moderate"

**Validates: Requirements 3.3, 3.4**

### Property 7: Price Delta Calculation

*For any* two price values (start_price, end_price) where start_price > 0, the price delta percentage SHALL equal ((end_price - start_price) / start_price) * 100.

**Validates: Requirements 4.2**

### Property 8: Divergence Detection

*For any* (sentiment_velocity, price_delta_pct) pair:

- If sentiment > 0.5 AND price_delta < -2.0, divergence SHALL be "bearish_divergence"
- If sentiment < -0.5 AND price_delta > 2.0, divergence SHALL be "bullish_divergence"
- Otherwise, divergence SHALL be "aligned"

**Validates: Requirements 4.4, 4.5**

### Property 9: Phrase Extraction Validity

*For any* text input, all extracted phrases SHALL be valid bigrams or trigrams that appear consecutively in the tokenized input text.

**Validates: Requirements 5.1**

### Property 10: Trending Phrase Classification

*For any* phrase with frequency count, the phrase SHALL be marked as trending if and only if frequency >= 5.

**Validates: Requirements 5.3**

### Property 11: Phrase Ranking Order

*For any* list of trending phrases, the output list SHALL be sorted in descending order by frequency.

**Validates: Requirements 5.4**

### Property 12: Influence Score Calculation

*For any* (followers, engagement) pair where both are non-negative integers, the influence score SHALL equal (followers *0.6) + (engagement* 0.4).

**Validates: Requirements 6.1**

### Property 13: Influencer Classification

*For any* author with an influence score, the author SHALL be classified as an influencer if and only if influence_score > 10000.

**Validates: Requirements 6.2**

### Property 14: Influencer Consensus Ratio

*For any* (bullish_count, bearish_count) pair where total > 0, the consensus ratio SHALL equal bullish_count / (bullish_count + bearish_count).

**Validates: Requirements 6.4**

### Property 15: Pulse Score Calculation

*For any* valid combination of (sentiment_velocity, phrase_frequency, influencer_ratio, divergence_type):

- The Pulse_Score SHALL be in the range [1, 10] inclusive
- Sentiment velocity > 0.7 SHALL contribute 4 points
- Sentiment velocity in (0.4, 0.7] SHALL contribute 2 points
- Phrase frequency > 20 SHALL contribute 3 points
- Phrase frequency in (10, 20] SHALL contribute 1.5 points
- Influencer ratio > 0.7 SHALL contribute 3 points
- Influencer ratio in (0.5, 0.7] SHALL contribute 1.5 points
- Bearish divergence SHALL subtract 1 point
- Final score SHALL be clamped to [1, 10]

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9**

### Property 16: Document Store Metadata Completeness

*For any* message stored in the Document_Store, the stored metadata SHALL contain all required fields: message_id, timestamp, sentiment, influence_score, and source_platform.

**Validates: Requirements 8.2**

### Property 17: RAG Prompt Context Enrichment

*For any* RAG query response, the generated prompt SHALL contain: current Pulse_Score, top 3 trending phrases, influencer consensus, and price correlation status.

**Validates: Requirements 8.4**

### Property 18: Alert Type Classification

*For any* Pulse_Score value:

- If score >= 7, alert type SHALL be "Strong Buy Signal"
- If score <= 3, alert type SHALL be "Cooling Off"
- Otherwise, no alert SHALL be generated

**Validates: Requirements 9.1, 9.2**

### Property 19: Divergence Warning Inclusion

*For any* alert where divergence_type is not "aligned", the alert message SHALL include a warning indicator.

**Validates: Requirements 9.3**

### Property 20: Alert Emoji Formatting

*For any* alert message, the message SHALL contain appropriate emoji indicators (🚀 for buy signals, ❄️ for cooling off, ⚠️ for warnings).

**Validates: Requirements 9.5**

### Property 21: Hype Simulator Phase Distribution

*For any* simulation run with N total messages, the phase distribution SHALL be approximately:

- Seed phase: 10% of messages (±2%)
- Growth phase: 40% of messages (±2%)
- Peak phase: 30% of messages (±2%)
- Decline phase: 20% of messages (±2%)

**Validates: Requirements 10.1**

### Property 22: Hype Simulator Phase Sentiment

*For any* message generated during a simulation phase, the sentiment score SHALL fall within the expected range for that phase:

- Seed: [0.1, 0.4]
- Growth: [0.3, 0.7]
- Peak: [0.5, 0.9]
- Decline: [-0.3, 0.3]

**Validates: Requirements 10.3**

### Property 23: Hype Simulator Influencer Inclusion

*For any* simulation run, at least 5% of generated messages SHALL come from influencer accounts (followers > 100,000).

**Validates: Requirements 10.4**

### Property 24: Hype Simulator Price Correlation

*For any* simulation phase, the generated price data SHALL follow the expected pattern:

- Seed: price ≈ base_price
- Growth: price ≈ base_price * 1.5
- Peak: price ≈ base_price * 2.5
- Decline: price ≈ base_price * 1.8

**Validates: Requirements 10.5**

## Error Handling

### Input Validation Errors

- **Malformed JSON**: Return 400 Bad Request with error details
- **Missing Required Fields**: Return 400 with list of missing fields
- **Invalid Field Types**: Return 400 with type mismatch details
- **Rate Limit Exceeded**: Return 429 with retry-after header

### Processing Errors

- **Sentiment Analysis Failure**: Log error, use neutral score (0.0) as fallback
- **Embedding Failure**: Retry 3 times with exponential backoff, then log and skip
- **Price API Failure**: Use cached price, log warning
- **LLM API Failure**: Return cached response or error message to user

### System Errors

- **Pipeline Crash**: Pathway persistence recovers state from checkpoint
- **Container Restart**: Docker restart policy ensures automatic recovery
- **Network Partition**: Graceful degradation with cached data

## Deployment Architecture

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  pipeline:
    build: ./pipeline
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - COINGECKO_API_KEY=${COINGECKO_API_KEY}
      - TRACKED_COIN=${TRACKED_COIN:-MEME}
    ports:
      - "8080:8080"  # Webhook endpoint
      - "8765:8765"  # Vector store API
    volumes:
      - ./cache:/app/cache  # Pathway persistence
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  telegram-bot:
    build: ./bot
    environment:
      - TELEGRAM_TOKEN=${TELEGRAM_TOKEN}
      - TELEGRAM_CHANNEL_ID=${TELEGRAM_CHANNEL_ID}
      - PIPELINE_URL=http://pipeline:8080
      - VECTOR_STORE_URL=http://pipeline:8765
    depends_on:
      pipeline:
        condition: service_healthy
    restart: unless-stopped

  dashboard:
    build: ./dashboard
    ports:
      - "8501:8501"
    environment:
      - PIPELINE_URL=http://pipeline:8080
    depends_on:
      - pipeline
    restart: unless-stopped

  simulator:
    build: ./simulator
    environment:
      - WEBHOOK_URL=http://pipeline:8080/webhook
      - TRACKED_COIN=${TRACKED_COIN:-MEME}
      - DEMO_DURATION_MINS=${DEMO_DURATION_MINS:-3}
    depends_on:
      pipeline:
        condition: service_healthy
    profiles:
      - demo  # Only runs with: docker-compose --profile demo up
```

### Environment Variables

```bash
# .env.example
OPENAI_API_KEY=sk-...
COINGECKO_API_KEY=CG-...  # Optional, free tier available
TELEGRAM_TOKEN=123456:ABC...
TELEGRAM_CHANNEL_ID=-100...
TRACKED_COIN=MEME
DEMO_DURATION_MINS=3
```

## Testing Strategy

### Dual Testing Approach

This project uses both unit tests and property-based tests for comprehensive coverage:

- **Unit Tests**: Verify specific examples, edge cases, and error conditions
- **Property Tests**: Verify universal properties across all valid inputs using Hypothesis

### Property-Based Testing Configuration

- **Framework**: Hypothesis (Python PBT library)
- **Minimum Iterations**: 100 per property test
- **Tag Format**: `# Feature: crypto-narrative-pulse-tracker, Property N: {property_text}`

### MVP Property Tests (Priority)

For the hackathon, focus on these 5 critical properties first:

1. **Property 1**: Message Schema Round-Trip (validates data integrity)
2. **Property 5**: Sentiment Score Range [-1, 1] (validates core analysis)
3. **Property 8**: Divergence Detection Logic (validates price correlation)
4. **Property 15**: Pulse Score Calculation [1, 10] (validates core output)
5. **Property 22**: Simulator Phase Sentiment Ranges (validates demo data)

### Test Organization

```
tests/
├── unit/
│   ├── test_sentiment.py          # Sentiment analyzer unit tests
│   ├── test_pulse_score.py        # Pulse score unit tests
│   ├── test_alerts.py             # Alert formatting unit tests
│   └── test_simulator.py          # Hype simulator unit tests
├── property/
│   ├── test_mvp_props.py          # MVP Properties 1, 5, 8, 15, 22
│   └── test_extended_props.py     # Remaining properties (post-MVP)
└── integration/
    └── test_pipeline.py           # End-to-end pipeline tests
```

### Example Property Test

```python
from hypothesis import given, strategies as st, settings

# Feature: crypto-narrative-pulse-tracker, Property 15: Pulse Score Calculation
@settings(max_examples=100)
@given(
    sentiment_velocity=st.floats(min_value=-1.0, max_value=1.0),
    phrase_frequency=st.integers(min_value=0, max_value=100),
    influencer_ratio=st.floats(min_value=0.0, max_value=1.0),
    divergence_type=st.sampled_from(['aligned', 'bearish_divergence', 'bullish_divergence'])
)
def test_pulse_score_range_and_calculation(
    sentiment_velocity, phrase_frequency, influencer_ratio, divergence_type
):
    """Property 15: Pulse score is always in [1, 10] and follows scoring rules."""
    calculator = PulseScoreCalculator()
    score = calculator.calculate(
        sentiment_velocity, phrase_frequency, influencer_ratio, divergence_type
    )

    # Score must be in valid range
    assert 1.0 <= score <= 10.0

    # Verify individual component contributions
    expected_score = 0.0
    if sentiment_velocity > 0.7:
        expected_score += 4.0
    elif sentiment_velocity > 0.4:
        expected_score += 2.0

    if phrase_frequency > 20:
        expected_score += 3.0
    elif phrase_frequency > 10:
        expected_score += 1.5

    if influencer_ratio > 0.7:
        expected_score += 3.0
    elif influencer_ratio > 0.5:
        expected_score += 1.5

    if divergence_type == 'bearish_divergence':
        expected_score -= 1.0

    expected_score = max(1.0, min(10.0, expected_score))
    assert score == expected_score
```
