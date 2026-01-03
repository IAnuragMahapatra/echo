"""
HTTP Webhook Connector for the Crypto Narrative Pulse Tracker.

Provides Pathway HTTP connector for ingesting messages from external sources
(Hype Simulator, external webhooks) and tag filtering functionality.

Requirements: 1.4, 1.5
"""

import logging
from typing import Callable

import pathway as pw

from schemas import MessageSchema

# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# HTTP WEBHOOK CONNECTOR
# =============================================================================


def create_webhook_connector(
    host: str = "0.0.0.0",
    port: int = 8080,
    delete_completed_queries: bool = False,
) -> pw.Table:
    """
    Create a Pathway HTTP REST connector for message ingestion.

    Uses pw.io.http.rest_connector() to create an HTTP endpoint that accepts
    JSON payloads conforming to MessageSchema. Messages from the Hype Simulator
    and external webhooks are ingested through this connector.

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number for the HTTP endpoint (default: 8080)
        delete_completed_queries: Whether to delete completed queries (default: False)

    Returns:
        Pathway Table with MessageSchema columns

    Requirements: 1.5

    Example:
        >>> messages = create_webhook_connector(host="0.0.0.0", port=8080)
        >>> # Messages are now available as a Pathway table
        >>> filtered = filter_by_tags(messages, ["#Solana", "$MEME"])
    """
    logger.info(f"Creating webhook connector on {host}:{port}")

    return pw.io.http.rest_connector(
        host=host,
        port=port,
        schema=MessageSchema,
        delete_completed_queries=delete_completed_queries,
    )


# =============================================================================
# TAG FILTERING
# =============================================================================


def matches_tag_pattern(tags: list, patterns: list[str]) -> bool:
    """
    Check if any tag matches any of the given patterns.

    Supports exact matches and simple wildcard patterns:
    - Exact match: "#Solana" matches "#Solana"
    - Case-insensitive: "#solana" matches "#Solana"
    - Prefix match with *: "#Sol*" matches "#Solana", "#SolanaEcosystem"

    Args:
        tags: List of tags from the message
        patterns: List of tag patterns to match against

    Returns:
        True if at least one tag matches at least one pattern

    Requirements: 1.4
    """
    if not tags or not patterns:
        return False

    for tag in tags:
        if not isinstance(tag, str):
            continue
        tag_lower = tag.lower()

        for pattern in patterns:
            pattern_lower = pattern.lower()

            # Handle wildcard patterns
            if pattern_lower.endswith("*"):
                prefix = pattern_lower[:-1]
                if tag_lower.startswith(prefix):
                    return True
            # Exact match (case-insensitive)
            elif tag_lower == pattern_lower:
                return True

    return False


def filter_by_tags(
    messages: pw.Table,
    tag_patterns: list[str],
) -> pw.Table:
    """
    Filter messages by configurable tag patterns.

    Uses pw.Table.filter() with tag matching logic to filter messages
    that contain at least one tag matching the specified patterns.

    Supports:
    - Hashtags: #Solana, #crypto
    - Cashtags: $MEME, $SOL
    - Wildcards: #Sol* matches #Solana, #SolanaEcosystem

    Args:
        messages: Pathway table with MessageSchema columns
        tag_patterns: List of tag patterns to filter by (e.g., ["#Solana", "$MEME"])

    Returns:
        Filtered Pathway table containing only messages with matching tags

    Requirements: 1.4

    Example:
        >>> messages = create_webhook_connector()
        >>> solana_messages = filter_by_tags(messages, ["#Solana", "$SOL"])
        >>> meme_messages = filter_by_tags(messages, ["$MEME", "#memecoin"])
    """
    if not tag_patterns:
        logger.warning("No tag patterns provided, returning all messages")
        return messages

    logger.info(f"Filtering messages by tag patterns: {tag_patterns}")

    return messages.filter(
        pw.apply(
            lambda tags: matches_tag_pattern(tags, tag_patterns),
            pw.this.tags,
        )
    )


def filter_by_tags_any(
    messages: pw.Table,
    tag_patterns: list[str],
) -> pw.Table:
    """
    Alias for filter_by_tags - filters messages matching ANY of the patterns.

    Args:
        messages: Pathway table with MessageSchema columns
        tag_patterns: List of tag patterns (OR logic - matches any)

    Returns:
        Filtered Pathway table

    Requirements: 1.4
    """
    return filter_by_tags(messages, tag_patterns)


def create_tag_filter(tag_patterns: list[str]) -> Callable[[list], bool]:
    """
    Create a reusable tag filter function for the given patterns.

    Useful when you need to apply the same filter multiple times
    or pass the filter to other functions.

    Args:
        tag_patterns: List of tag patterns to match

    Returns:
        Callable that takes a list of tags and returns True if any match

    Requirements: 1.4

    Example:
        >>> solana_filter = create_tag_filter(["#Solana", "$SOL"])
        >>> is_solana = solana_filter(["#crypto", "#Solana"])  # True
        >>> is_solana = solana_filter(["#Bitcoin"])  # False
    """

    def filter_func(tags: list) -> bool:
        return matches_tag_pattern(tags, tag_patterns)

    return filter_func


# =============================================================================
# COMBINED CONNECTOR WITH FILTERING
# =============================================================================


def create_filtered_webhook_connector(
    host: str = "0.0.0.0",
    port: int = 8080,
    tag_patterns: list[str] | None = None,
    delete_completed_queries: bool = False,
) -> pw.Table:
    """
    Create a webhook connector with optional tag filtering.

    Convenience function that combines connector creation and filtering
    into a single call.

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number for the HTTP endpoint (default: 8080)
        tag_patterns: Optional list of tag patterns to filter by
        delete_completed_queries: Whether to delete completed queries

    Returns:
        Pathway Table with MessageSchema columns (filtered if patterns provided)

    Requirements: 1.4, 1.5

    Example:
        >>> # Get all messages
        >>> all_messages = create_filtered_webhook_connector()
        >>>
        >>> # Get only Solana-related messages
        >>> solana_messages = create_filtered_webhook_connector(
        ...     tag_patterns=["#Solana", "$SOL", "#Sol*"]
        ... )
    """
    messages = create_webhook_connector(
        host=host,
        port=port,
        delete_completed_queries=delete_completed_queries,
    )

    if tag_patterns:
        return filter_by_tags(messages, tag_patterns)

    return messages
