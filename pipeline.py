#!/usr/bin/env python
"""
Pathway Streaming Pipeline for Crypto Narrative Pulse Tracker.

This module implements the real Pathway streaming pipeline with:
- pw.io.http.rest_connector() for message ingestion
- pw.temporal.sliding() for sentiment velocity calculation
- pw.temporal.tumbling() for influencer tracking
- pw.join() for price-sentiment correlation
- pw.io.subscribe() for live metrics updates

Requirements: 1.5, 3.2, 4.3, 5.4, 6.3, 8.4

Usage:
    # Run the pipeline
    python pipeline.py

    # Or import and use programmatically
    from pipeline import create_pipeline, run_pipeline
"""

import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check if we can import real Pathway
PATHWAY_AVAILABLE = False
try:
    import pathway as pw
    from pathway.stdlib.temporal import sliding, tumbling

    PATHWAY_AVAILABLE = True
    logger.info("Pathway imported successfully")
except ImportError as e:
    logger.warning(f"Pathway not available: {e}")
    logger.warning("Pipeline will run in mock mode")

# Import local modules
sys.path.insert(0, ".")
from dotenv import load_dotenv

load_dotenv()

from rag.live_metrics import (
    get_live_metrics,
    update_influencer_consensus,
    update_pulse_score,
    update_sentiment_velocity,
    update_trending_phrases,
)
from transforms.pulse_score import calculate_pulse_score
from transforms.sentiment import analyze_sentiment

# =============================================================================
# CONFIGURATION
# =============================================================================

# Standard window configuration (from design.md)
WINDOW_DURATION_MINUTES = 5
WINDOW_HOP_MINUTES = 1
INFLUENCER_WINDOW_MINUTES = 10

# Thresholds
INFLUENCER_SCORE_THRESHOLD = 10000
TRENDING_PHRASE_MIN_FREQUENCY = 5

# Server configuration
WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "0.0.0.0")
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8080"))
TRACKED_COIN = os.getenv("TRACKED_COIN", "MEME")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def calculate_influence_score(followers: int, engagement: int) -> float:
    """
    Calculate influence score: (followers × 0.6) + (engagement × 0.4)

    Requirements: 6.1
    """
    return (followers * 0.6) + (engagement * 0.4)


def extract_phrases(text: str) -> list:
    """
    Extract bigrams and trigrams from text.

    Requirements: 5.1
    """
    # Simple stopwords
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "it",
        "its",
        "they",
        "them",
        "their",
    }

    # Tokenize and filter
    words = text.lower().split()
    tokens = [w for w in words if w not in stopwords and len(w) > 2]

    phrases = []
    # Bigrams
    for i in range(len(tokens) - 1):
        phrases.append(f"{tokens[i]} {tokens[i + 1]}")
    # Trigrams
    for i in range(len(tokens) - 2):
        phrases.append(f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}")

    return phrases


# =============================================================================
# PATHWAY PIPELINE (Real Implementation)
# =============================================================================

if PATHWAY_AVAILABLE:
    from schemas import MessageSchema

    def create_message_connector(
        host: str = WEBHOOK_HOST, port: int = WEBHOOK_PORT
    ) -> "pw.Table":
        """
        Create Pathway HTTP connector for message ingestion.

        Requirements: 1.5
        """
        logger.info(f"Creating webhook connector on {host}:{port}")
        return pw.io.http.rest_connector(
            host=host,
            port=port,
            schema=MessageSchema,
            delete_completed_queries=False,
        )

    def add_sentiment_scores(messages: "pw.Table") -> "pw.Table":
        """
        Add sentiment scores to messages using pw.apply().

        Requirements: 3.1
        """
        return messages.select(
            message_id=pw.this.message_id,
            text=pw.this.text,
            author_id=pw.this.author_id,
            author_followers=pw.this.author_followers,
            timestamp=pw.this.timestamp,
            tags=pw.this.tags,
            engagement_count=pw.this.engagement_count,
            source_platform=pw.this.source_platform,
            sentiment=pw.apply(analyze_sentiment, pw.this.text),
        )

    def calculate_sentiment_velocity(messages_with_sentiment: "pw.Table") -> "pw.Table":
        """
        Calculate sentiment velocity using pw.temporal.sliding().

        Uses 5-minute sliding window with 1-minute hops.

        Requirements: 3.2, 3.5
        """
        return messages_with_sentiment.windowby(
            pw.this.timestamp,
            window=pw.temporal.sliding(
                hop=pw.Duration.minutes(WINDOW_HOP_MINUTES),
                duration=pw.Duration.minutes(WINDOW_DURATION_MINUTES),
            ),
        ).reduce(
            velocity=pw.reducers.avg(pw.this.sentiment),
            message_count=pw.reducers.count(),
            window_end=pw.this._pw_window_end,
        )

    def add_influence_scores(messages: "pw.Table") -> "pw.Table":
        """
        Add influence scores to messages.

        Requirements: 6.1
        """
        return messages.select(
            **{col: getattr(pw.this, col) for col in messages.schema.column_names()},
            influence_score=pw.apply(
                calculate_influence_score,
                pw.this.author_followers,
                pw.this.engagement_count,
            ),
        )

    def calculate_influencer_signals(messages_with_influence: "pw.Table") -> "pw.Table":
        """
        Track influencer sentiment in tumbling windows.

        Requirements: 6.2, 6.3, 6.4
        """
        # Filter to influencers only
        influencers = messages_with_influence.filter(
            pw.this.influence_score > INFLUENCER_SCORE_THRESHOLD
        )

        # Aggregate in tumbling windows
        return influencers.windowby(
            pw.this.timestamp,
            window=pw.temporal.tumbling(
                duration=pw.Duration.minutes(INFLUENCER_WINDOW_MINUTES)
            ),
        ).reduce(
            bullish_count=pw.reducers.sum(pw.if_else(pw.this.sentiment > 0.3, 1, 0)),
            bearish_count=pw.reducers.sum(pw.if_else(pw.this.sentiment < -0.3, 1, 0)),
            total_count=pw.reducers.count(),
            window_end=pw.this._pw_window_end,
        )

    def calculate_trending_phrases(messages: "pw.Table") -> "pw.Table":
        """
        Track phrase frequency in sliding windows.

        Requirements: 5.2, 5.3, 5.4
        """
        # Extract phrases from messages
        with_phrases = messages.select(
            phrases=pw.apply(extract_phrases, pw.this.text),
            timestamp=pw.this.timestamp,
        ).flatten(pw.this.phrases)

        # Count in sliding window
        phrase_counts = (
            with_phrases.windowby(
                pw.this.timestamp,
                window=pw.temporal.sliding(
                    hop=pw.Duration.minutes(1),
                    duration=pw.Duration.minutes(10),
                ),
            )
            .groupby(pw.this.phrases)
            .reduce(
                phrase=pw.this.phrases,
                frequency=pw.reducers.count(),
                window_end=pw.this._pw_window_end,
            )
        )

        # Filter to trending (frequency >= 5)
        return phrase_counts.filter(pw.this.frequency >= TRENDING_PHRASE_MIN_FREQUENCY)

    def create_pipeline_subscriptions(
        sentiment_velocity: "pw.Table",
        influencer_signals: "pw.Table",
        trending_phrases: "pw.Table",
    ) -> None:
        """
        Subscribe to pipeline outputs to update live metrics.

        Requirements: 8.4
        """
        # Get performance monitor for throughput tracking
        perf_monitor = get_performance_monitor()

        # Subscribe to sentiment velocity
        def on_velocity_change(key, row, time, is_addition):
            if is_addition and row:
                velocity = row.get("velocity", 0.0)
                count = row.get("message_count", 0)
                update_sentiment_velocity(velocity, count)

                # Track throughput - record messages processed in this window
                for _ in range(count):
                    record_message_processed()

                # Calculate and update pulse score
                metrics = get_live_metrics()
                score = calculate_pulse_score(
                    sentiment_velocity=velocity,
                    phrase_frequency=len(metrics.trending_phrases) * 5,
                    influencer_ratio=0.5,  # Default
                    divergence_type=metrics.divergence_status,
                )
                update_pulse_score(score)

        pw.io.subscribe(sentiment_velocity, on_change=on_velocity_change)

        # Subscribe to influencer signals
        def on_influencer_change(key, row, time, is_addition):
            if is_addition and row:
                bullish = row.get("bullish_count", 0)
                bearish = row.get("bearish_count", 0)
                update_influencer_consensus(bullish, bearish)

        pw.io.subscribe(influencer_signals, on_change=on_influencer_change)

        # Subscribe to trending phrases
        def on_phrases_change(key, row, time, is_addition):
            if is_addition and row:
                phrase = row.get("phrase", "")
                if phrase:
                    metrics = get_live_metrics()
                    current = list(metrics.trending_phrases)
                    if phrase not in current:
                        current.append(phrase)
                    update_trending_phrases(current[:10])

        pw.io.subscribe(trending_phrases, on_change=on_phrases_change)

        logger.info("Pipeline subscriptions created")

    def create_full_pipeline(
        host: str = WEBHOOK_HOST,
        port: int = WEBHOOK_PORT,
    ) -> dict:
        """
        Create the full Pathway streaming pipeline.

        Returns dict with all pipeline tables for further processing.
        """
        logger.info("Creating full Pathway pipeline...")

        # 1. Message ingestion
        messages = create_message_connector(host, port)
        logger.info("✓ Message connector created")

        # 2. Add sentiment scores
        messages_with_sentiment = add_sentiment_scores(messages)
        logger.info("✓ Sentiment analysis added")

        # 3. Calculate sentiment velocity
        sentiment_velocity = calculate_sentiment_velocity(messages_with_sentiment)
        logger.info("✓ Sentiment velocity calculation added")

        # 4. Add influence scores
        messages_with_influence = add_influence_scores(messages_with_sentiment)
        logger.info("✓ Influence scores added")

        # 5. Calculate influencer signals
        influencer_signals = calculate_influencer_signals(messages_with_influence)
        logger.info("✓ Influencer signal tracking added")

        # 6. Calculate trending phrases
        trending_phrases = calculate_trending_phrases(messages)
        logger.info("✓ Phrase clustering added")

        # 7. Create subscriptions for live metrics
        create_pipeline_subscriptions(
            sentiment_velocity,
            influencer_signals,
            trending_phrases,
        )
        logger.info("✓ Live metrics subscriptions created")

        return {
            "messages": messages,
            "messages_with_sentiment": messages_with_sentiment,
            "sentiment_velocity": sentiment_velocity,
            "messages_with_influence": messages_with_influence,
            "influencer_signals": influencer_signals,
            "trending_phrases": trending_phrases,
        }

    def run_pipeline(host: str = WEBHOOK_HOST, port: int = WEBHOOK_PORT):
        """
        Run the Pathway pipeline.

        This starts the streaming pipeline and blocks until interrupted.
        """
        logger.info(f"Starting Pathway pipeline on {host}:{port}")
        logger.info(f"Tracking coin: ${TRACKED_COIN}")

        # Create pipeline
        pipeline = create_full_pipeline(host, port)

        # Run the pipeline
        logger.info("Pipeline running. Press Ctrl+C to stop.")
        pw.run()

else:
    # Mock implementations for when Pathway is not available
    def create_full_pipeline(*args, **kwargs):
        logger.warning("Pathway not available - using mock pipeline")
        return {}

    def run_pipeline(*args, **kwargs):
        logger.error("Cannot run pipeline - Pathway not installed")
        logger.error("Install with: pip install pathway")
        sys.exit(1)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Main entry point for the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Crypto Pulse Pathway Pipeline")
    parser.add_argument("--host", default=WEBHOOK_HOST, help="Webhook host")
    parser.add_argument("--port", type=int, default=WEBHOOK_PORT, help="Webhook port")
    parser.add_argument(
        "--check", action="store_true", help="Check if Pathway is available"
    )

    args = parser.parse_args()

    if args.check:
        if PATHWAY_AVAILABLE:
            print("✓ Pathway is available")
            sys.exit(0)
        else:
            print("✗ Pathway is NOT available")
            print("  Install with: pip install pathway")
            sys.exit(1)

    if not PATHWAY_AVAILABLE:
        logger.error("Pathway is not installed. Install with: pip install pathway")
        logger.error("Note: Pathway requires Linux or WSL on Windows")
        sys.exit(1)

    run_pipeline(args.host, args.port)


if __name__ == "__main__":
    main()
