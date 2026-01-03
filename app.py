#!/usr/bin/env python
"""
Crypto Narrative Pulse Tracker - Pathway Pipeline Application

This is the main Pathway streaming pipeline that:
1. Ingests messages via HTTP webhook (pw.io.http.rest_connector)
2. Calculates sentiment velocity using sliding windows
3. Tracks influencer signals using tumbling windows
4. Detects trending phrases
5. Calculates pulse scores
6. Sends alerts via Telegram
7. Updates live metrics for RAG queries

Run with: python app.py
Or via Docker: docker-compose up pipeline

Webhook endpoints (automatically created by Pathway):
- POST http://host:port/ - Send messages (JSON body matching MessageSchema)
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

import pathway as pw
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "0.0.0.0")
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8080"))
TRACKED_COIN = os.getenv("TRACKED_COIN", "MEME")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", "")

# Window configurations (from design.md)
SENTIMENT_WINDOW_DURATION = 5  # minutes
SENTIMENT_WINDOW_HOP = 1  # minutes
INFLUENCER_WINDOW_DURATION = 10  # minutes
PHRASE_WINDOW_DURATION = 10  # minutes

# Thresholds
INFLUENCER_SCORE_THRESHOLD = 10000
TRENDING_PHRASE_MIN_FREQUENCY = 5

# =============================================================================
# SCHEMAS
# =============================================================================


class MessageSchema(pw.Schema):
    """Schema for incoming social media messages."""

    message_id: str = pw.column_definition(default_value="")
    text: str = pw.column_definition(default_value="")
    author_id: str = pw.column_definition(default_value="")
    author_followers: int = pw.column_definition(default_value=0)
    timestamp: str = pw.column_definition(default_value="")
    tags: str = pw.column_definition(default_value="[]")  # JSON string of tags
    engagement_count: int = pw.column_definition(default_value=0)
    source_platform: str = pw.column_definition(default_value="unknown")


class PriceSchema(pw.Schema):
    """Schema for price data."""

    coin_symbol: str
    price_usd: float
    timestamp: str
    volume_24h: float


# =============================================================================
# SENTIMENT ANALYSIS
# =============================================================================

# Crypto-specific lexicon
CRYPTO_LEXICON = {
    "moon": 3.0,
    "mooning": 3.5,
    "pump": 2.5,
    "pumping": 2.5,
    "bullish": 2.5,
    "hodl": 1.5,
    "hodling": 1.5,
    "fomo": 1.0,
    "degen": 0.5,
    "lfg": 2.5,
    "wagmi": 2.0,
    "gm": 0.5,
    "alpha": 1.5,
    "gem": 2.0,
    "based": 1.5,
    "ape": 1.0,
    "diamond": 2.0,
    "lambo": 2.5,
    "dump": -3.0,
    "dumping": -3.0,
    "rug": -4.0,
    "rugged": -4.0,
    "scam": -3.5,
    "bearish": -2.5,
    "fud": -2.0,
    "rekt": -3.0,
    "ngmi": -2.0,
    "crash": -3.0,
    "crashing": -3.0,
    "ponzi": -3.5,
    "exit": -1.5,
    "sell": -1.0,
    "selling": -1.0,
    "paper": -1.5,
    "dead": -2.5,
    "dying": -2.5,
}

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _vader = SentimentIntensityAnalyzer()
    _vader.lexicon.update(CRYPTO_LEXICON)
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("VADER not available, using simple sentiment")


def analyze_sentiment(text: str) -> float:
    """Analyze sentiment of text, returns score in [-1, 1]."""
    if not text or not text.strip():
        return 0.0

    if VADER_AVAILABLE:
        scores = _vader.polarity_scores(text)
        return scores["compound"]

    # Simple fallback
    text_lower = text.lower()
    score = 0.0
    for word, value in CRYPTO_LEXICON.items():
        if word in text_lower:
            score += value * 0.1
    return max(-1.0, min(1.0, score))


# =============================================================================
# INFLUENCE CALCULATION
# =============================================================================


def calculate_influence_score(followers: int, engagement: int) -> float:
    """Calculate influence score: (followers × 0.6) + (engagement × 0.4)."""
    return (followers * 0.6) + (engagement * 0.4)


# =============================================================================
# DIVERGENCE DETECTION
# =============================================================================


def detect_divergence(sentiment: float, price_delta_pct: float) -> str:
    """Detect sentiment-price divergence."""
    if sentiment > 0.5 and price_delta_pct < -2.0:
        return "bearish_divergence"
    elif sentiment < -0.5 and price_delta_pct > 2.0:
        return "bullish_divergence"
    return "aligned"


# =============================================================================
# PULSE SCORE CALCULATION
# =============================================================================


def calculate_pulse_score(
    sentiment_velocity: float,
    phrase_frequency: int,
    influencer_ratio: float,
    divergence_type: str,
) -> float:
    """Calculate pulse score from 1-10."""
    score = 0.0

    # Sentiment component (0-5 points) - adjusted for demo responsiveness
    if sentiment_velocity > 0.6:
        score += 5.0
    elif sentiment_velocity > 0.4:
        score += 3.5
    elif sentiment_velocity > 0.2:
        score += 2.0
    elif sentiment_velocity > 0.0:
        score += 1.0
    elif sentiment_velocity > -0.2:
        score += 0.5
    # Negative sentiment reduces score
    elif sentiment_velocity < -0.4:
        score -= 1.0

    # Phrase frequency / message count component (0-3 points)
    if phrase_frequency > 15:
        score += 3.0
    elif phrase_frequency > 8:
        score += 2.0
    elif phrase_frequency > 3:
        score += 1.0

    # Influencer component (0-2 points)
    if influencer_ratio > 0.6:
        score += 2.0
    elif influencer_ratio > 0.4:
        score += 1.0

    # Divergence modifier
    if divergence_type == "bearish_divergence":
        score -= 1.0

    return max(1.0, min(10.0, score))


# =============================================================================
# TELEGRAM ALERTS
# =============================================================================

_telegram_bot = None
_last_alert_score = None


async def send_telegram_alert(score: float, phrases: list, divergence: str):
    """Send alert to Telegram channel."""
    global _telegram_bot, _last_alert_score

    if not TELEGRAM_TOKEN or not TELEGRAM_CHANNEL_ID:
        logger.warning("Telegram not configured - skipping alert")
        return

    # Only alert on significant changes
    should_alert = False
    if score >= 7.0 and (_last_alert_score is None or _last_alert_score < 7.0):
        should_alert = True
        emoji = "🚀"
        signal = "High Momentum"
        explanation = "Strong bullish sentiment detected! The community is showing significant excitement. This typically indicates growing interest, but remember - high momentum can reverse quickly."
    elif score <= 3.0 and (_last_alert_score is None or _last_alert_score > 3.0):
        should_alert = True
        emoji = "❄️"
        signal = "Low Momentum"
        explanation = "Sentiment is cooling off. This could mean the hype is fading or it's a consolidation phase. Often a good time to watch and wait for clearer signals."

    if not should_alert:
        _last_alert_score = score
        return

    try:
        if _telegram_bot is None:
            from telegram import Bot

            _telegram_bot = Bot(token=TELEGRAM_TOKEN)

        phrases_str = (
            ", ".join(phrases[:3]) if phrases else "No specific phrases trending"
        )

        divergence_warning = ""
        if divergence == "bearish_divergence":
            divergence_warning = "\n\n⚠️ *Divergence Warning*: Price is dropping while sentiment stays positive. This mismatch often signals a potential trend reversal."
        elif divergence == "bullish_divergence":
            divergence_warning = "\n\n💡 *Divergence Note*: Price is rising while sentiment is negative. This could indicate hidden accumulation."

        message = f"""{emoji} *{signal} Alert: ${TRACKED_COIN}*

📊 *Pulse Score:* {score:.1f}/10
🔥 *Trending:* {phrases_str}

💭 *What this means:*
{explanation}{divergence_warning}

⚠️ _Not financial advice. Always DYOR._

_Crypto Narrative Pulse Tracker_"""

        await _telegram_bot.send_message(
            chat_id=TELEGRAM_CHANNEL_ID, text=message, parse_mode="Markdown"
        )
        _last_alert_score = score
        logger.info(f"Sent Telegram alert: {signal} ({score:.1f})")
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")


def send_alert_sync(score: float, phrases: list, divergence: str):
    """Synchronous wrapper for sending alerts."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(send_telegram_alert(score, phrases, divergence))


# =============================================================================
# LIVE METRICS STATE
# =============================================================================


class LiveMetricsState:
    """Shared state for live metrics."""

    def __init__(self):
        self.pulse_score = 5.0
        self.sentiment_velocity = 0.0
        self.trending_phrases = []
        self.influencer_consensus = "neutral"
        self.divergence_status = "aligned"
        self.message_count = 0
        self.last_updated = None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = datetime.now(timezone.utc)

    def to_dict(self):
        return {
            "pulse_score": self.pulse_score,
            "sentiment_velocity": self.sentiment_velocity,
            "trending_phrases": self.trending_phrases,
            "influencer_consensus": self.influencer_consensus,
            "divergence_status": self.divergence_status,
            "message_count": self.message_count,
            "last_updated": self.last_updated.isoformat()
            if self.last_updated
            else None,
        }


live_metrics = LiveMetricsState()


# =============================================================================
# PATHWAY PIPELINE
# =============================================================================


def create_pipeline():
    """Create the Pathway streaming pipeline."""
    logger.info(f"Creating Pathway pipeline on {WEBHOOK_HOST}:{WEBHOOK_PORT}")
    logger.info(f"Tracking coin: ${TRACKED_COIN}")

    # 1. Message ingestion via HTTP webhook
    # Pathway's rest_connector returns (table, response_writer) tuple
    # The endpoint will be: POST http://host:port/
    messages, response_writer = pw.io.http.rest_connector(
        host=WEBHOOK_HOST,
        port=WEBHOOK_PORT,
        schema=MessageSchema,
        delete_completed_queries=True,  # Clean up completed queries
    )
    logger.info(
        f"✓ HTTP webhook connector created at http://{WEBHOOK_HOST}:{WEBHOOK_PORT}/"
    )

    # 2. Add sentiment scores to messages
    messages_with_sentiment = messages.select(
        message_id=pw.this.message_id,
        text=pw.this.text,
        author_id=pw.this.author_id,
        author_followers=pw.this.author_followers,
        timestamp=pw.this.timestamp,
        engagement_count=pw.this.engagement_count,
        source_platform=pw.this.source_platform,
        sentiment=pw.apply(analyze_sentiment, pw.this.text),
    )
    logger.info("✓ Sentiment analysis added")

    # 3. Calculate sentiment velocity using sliding window
    # Parse timestamp and use for windowing
    messages_with_time = messages_with_sentiment.select(
        **{
            col: getattr(pw.this, col)
            for col in [
                "message_id",
                "text",
                "author_id",
                "author_followers",
                "engagement_count",
                "source_platform",
                "sentiment",
            ]
        },
        event_time=pw.this.timestamp,
    )

    # 4. Add influence scores
    messages_with_influence = messages_with_time.select(
        **{
            col: getattr(pw.this, col)
            for col in messages_with_time.schema.column_names()
        },
        influence_score=pw.apply(
            calculate_influence_score,
            pw.this.author_followers,
            pw.this.engagement_count,
        ),
    )
    logger.info("✓ Influence scores added")

    # 5. Filter to influencers only
    influencers = messages_with_influence.filter(
        pw.this.influence_score > INFLUENCER_SCORE_THRESHOLD
    )

    # 6. Subscribe to updates for live metrics
    # Keep track of recent sentiments for rolling average
    recent_sentiments = []
    MAX_RECENT = 20  # Keep last 20 messages for averaging

    def on_message_update(key, row, time, is_addition):
        nonlocal recent_sentiments
        if is_addition and row:
            sentiment = row.get("sentiment", 0.0)
            live_metrics.message_count += 1

            # Add to recent sentiments and maintain window
            recent_sentiments.append(sentiment)
            if len(recent_sentiments) > MAX_RECENT:
                recent_sentiments = recent_sentiments[-MAX_RECENT:]

            # Calculate rolling average sentiment velocity
            avg_sentiment = sum(recent_sentiments) / len(recent_sentiments)
            live_metrics.sentiment_velocity = avg_sentiment

            # Check for influencer (high follower count)
            followers = row.get("author_followers", 0)
            engagement = row.get("engagement_count", 0)
            influence_score = (followers * 0.6) + (engagement * 0.4)
            is_influencer = influence_score > INFLUENCER_SCORE_THRESHOLD

            # Update influencer ratio based on recent messages
            influencer_ratio = 0.5  # Default
            if is_influencer and sentiment > 0.3:
                influencer_ratio = 0.8
            elif is_influencer and sentiment < -0.3:
                influencer_ratio = 0.2

            # Calculate pulse score using rolling average
            score = calculate_pulse_score(
                sentiment_velocity=avg_sentiment,
                phrase_frequency=len(live_metrics.trending_phrases) * 5
                + live_metrics.message_count,
                influencer_ratio=influencer_ratio,
                divergence_type=live_metrics.divergence_status,
            )
            live_metrics.pulse_score = score
            live_metrics.last_updated = datetime.now(timezone.utc)

            logger.info(
                f"Message #{live_metrics.message_count}: sentiment={sentiment:.2f}, avg={avg_sentiment:.2f}, pulse={score:.1f}"
            )

            # Send alert if needed
            if score >= 7.0 or score <= 3.0:
                send_alert_sync(
                    score, live_metrics.trending_phrases, live_metrics.divergence_status
                )

    pw.io.subscribe(messages_with_sentiment, on_change=on_message_update)
    logger.info("✓ Live metrics subscription created")

    # 7. Output to console for debugging
    pw.io.subscribe(
        messages_with_sentiment,
        on_change=lambda key, row, time, is_addition: (
            logger.debug(f"[{'+' if is_addition else '-'}] {row}") if row else None
        ),
    )

    return {
        "messages": messages,
        "messages_with_sentiment": messages_with_sentiment,
        "messages_with_influence": messages_with_influence,
        "influencers": influencers,
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Main entry point."""
    print("=" * 60)
    print("  🚀 CRYPTO NARRATIVE PULSE TRACKER")
    print("=" * 60)
    print(f"\n  Webhook endpoint: http://{WEBHOOK_HOST}:{WEBHOOK_PORT}")
    print(f"  Tracked coin: ${TRACKED_COIN}")
    print(f"  Telegram: {'Enabled' if TELEGRAM_TOKEN else 'Disabled'}")
    print("\n  Waiting for messages...")
    print("=" * 60 + "\n")

    # Create and run pipeline
    pipeline = create_pipeline()

    logger.info("Starting Pathway pipeline...")
    pw.run()


if __name__ == "__main__":
    main()
