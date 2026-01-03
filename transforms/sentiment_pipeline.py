"""
Sentiment Pipeline Integration for the Crypto Narrative Pulse Tracker.

Integrates sentiment analysis into the Pathway streaming pipeline using:
- pw.apply() for adding sentiment scores to message streams
- pw.temporal.sliding() for sentiment velocity calculation
- pw.reducers.avg() for velocity aggregation

Requirements: 3.2, 3.3, 3.4, 3.5
"""

import pathway as pw

from transforms.sentiment import SentimentAnalyzer, analyze_sentiment

# =============================================================================
# WINDOW CONFIGURATION
# =============================================================================
# Standard window configuration for consistency across all components
# 5-minute sliding window with 1-minute hop (as per design doc)

STANDARD_WINDOW_DURATION = pw.Duration(minutes=5)
STANDARD_WINDOW_HOP = pw.Duration(minutes=1)


# =============================================================================
# SENTIMENT SCHEMAS
# =============================================================================


class SentimentResultSchema(pw.Schema):
    """Schema for messages with sentiment scores."""

    message_id: str
    text: str
    author_id: str
    author_followers: int
    timestamp: pw.DateTimeUtc
    tags: list
    engagement_count: int
    source_platform: str
    sentiment_score: float  # Added sentiment score [-1, 1]


class SentimentVelocitySchema(pw.Schema):
    """Schema for sentiment velocity aggregations."""

    velocity: float  # Average sentiment over window
    message_count: int  # Number of messages in window
    window_end: pw.DateTimeUtc  # End timestamp of the window
    momentum_class: str  # Classification: strong_bullish, strong_bearish, etc.


# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================


def add_sentiment_scores(messages: pw.Table) -> pw.Table:
    """
    Add sentiment scores to a message stream using pw.apply().

    Takes a Pathway table of messages and adds a sentiment_score column
    by applying the SentimentAnalyzer to each message's text.

    Args:
        messages: Pathway table with MessageSchema columns

    Returns:
        Pathway table with additional sentiment_score column

    Requirements: 3.1

    Example:
        >>> messages = pw.io.http.rest_connector(...)
        >>> with_sentiment = add_sentiment_scores(messages)
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
        sentiment_score=pw.apply(analyze_sentiment, pw.this.text),
    )


def calculate_sentiment_velocity(
    messages_with_sentiment: pw.Table,
    window_duration: pw.Duration = STANDARD_WINDOW_DURATION,
    window_hop: pw.Duration = STANDARD_WINDOW_HOP,
) -> pw.Table:
    """
    Calculate sentiment velocity using sliding windows.

    Sentiment velocity is the average sentiment score over a time window,
    which indicates the momentum of sentiment change. Uses:
    - pw.temporal.sliding() with 5-min duration, 1-min hop
    - pw.reducers.avg() for velocity aggregation

    Args:
        messages_with_sentiment: Table with sentiment_score column
        window_duration: Duration of sliding window (default: 5 minutes)
        window_hop: Hop interval between windows (default: 1 minute)

    Returns:
        Pathway table with velocity, message_count, window_end, momentum_class

    Requirements: 3.2, 3.3, 3.4, 3.5

    Example:
        >>> with_sentiment = add_sentiment_scores(messages)
        >>> velocity = calculate_sentiment_velocity(with_sentiment)
    """
    # Create the sentiment analyzer for classification
    analyzer = SentimentAnalyzer()

    # Apply sliding window aggregation
    velocity_table = messages_with_sentiment.windowby(
        pw.this.timestamp,
        window=pw.temporal.sliding(
            hop=window_hop,
            duration=window_duration,
        ),
    ).reduce(
        velocity=pw.reducers.avg(pw.this.sentiment_score),
        message_count=pw.reducers.count(),
        window_end=pw.this._pw_window_end,
    )

    # Add momentum classification
    return velocity_table.select(
        velocity=pw.this.velocity,
        message_count=pw.this.message_count,
        window_end=pw.this.window_end,
        momentum_class=pw.apply(
            analyzer.classify_momentum,
            pw.this.velocity,
        ),
    )


def classify_momentum(velocity: float) -> str:
    """
    Classify sentiment velocity into momentum categories.

    Standalone function for use with pw.apply() when analyzer
    instance is not available.

    Args:
        velocity: Average sentiment over a time window

    Returns:
        Momentum classification string

    Requirements: 3.3, 3.4
    """
    if velocity > 0.7:
        return "strong_bullish_momentum"
    elif velocity < -0.7:
        return "strong_bearish_momentum"
    elif velocity > 0.3:
        return "moderate_bullish"
    elif velocity < -0.3:
        return "moderate_bearish"
    else:
        return "neutral"


def create_sentiment_pipeline(messages: pw.Table) -> tuple[pw.Table, pw.Table]:
    """
    Create the complete sentiment analysis pipeline.

    Combines sentiment scoring and velocity calculation into a single
    pipeline that can be integrated with the main Pathway application.

    Args:
        messages: Raw message stream from connectors

    Returns:
        Tuple of (messages_with_sentiment, sentiment_velocity) tables

    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5

    Example:
        >>> messages = pw.io.http.rest_connector(...)
        >>> with_sentiment, velocity = create_sentiment_pipeline(messages)
        >>> # Use with_sentiment for document store indexing
        >>> # Use velocity for pulse score calculation
    """
    # Step 1: Add sentiment scores to messages
    messages_with_sentiment = add_sentiment_scores(messages)

    # Step 2: Calculate sentiment velocity over sliding windows
    sentiment_velocity = calculate_sentiment_velocity(messages_with_sentiment)

    return messages_with_sentiment, sentiment_velocity


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def filter_by_momentum(
    velocity_table: pw.Table,
    momentum_types: list[str],
) -> pw.Table:
    """
    Filter velocity table by momentum classification.

    Useful for triggering alerts only on strong momentum signals.

    Args:
        velocity_table: Table with momentum_class column
        momentum_types: List of momentum types to include

    Returns:
        Filtered table containing only specified momentum types

    Example:
        >>> strong_signals = filter_by_momentum(
        ...     velocity,
        ...     ["strong_bullish_momentum", "strong_bearish_momentum"]
        ... )
    """
    return velocity_table.filter(
        pw.apply(lambda m: m in momentum_types, pw.this.momentum_class)
    )


def get_latest_velocity(velocity_table: pw.Table) -> pw.Table:
    """
    Get the most recent velocity reading.

    Useful for dashboard displays and alert triggers.

    Args:
        velocity_table: Table with window_end column

    Returns:
        Table with single row containing latest velocity
    """
    return velocity_table.reduce(
        velocity=pw.reducers.latest(pw.this.velocity),
        message_count=pw.reducers.latest(pw.this.message_count),
        window_end=pw.reducers.max(pw.this.window_end),
        momentum_class=pw.reducers.latest(pw.this.momentum_class),
    )
