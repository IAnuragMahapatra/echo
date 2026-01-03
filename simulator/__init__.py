"""
Hype Simulator module for Crypto Narrative Pulse Tracker.

Provides realistic hype cycle data generation for demos.
"""

from .hype_simulator import (
    INFLUENCER_PROBABILITY,
    INFLUENCERS,
    PHASE_ORDER,
    PHASES,
    PRICE_MULTIPLIERS,
    HypeSimulator,
    generate_single_message,
    generate_single_price,
    get_phase_sentiment_range,
    get_phase_volume_percentage,
)

__all__ = [
    "HypeSimulator",
    "PHASES",
    "PHASE_ORDER",
    "PRICE_MULTIPLIERS",
    "INFLUENCERS",
    "INFLUENCER_PROBABILITY",
    "generate_single_message",
    "generate_single_price",
    "get_phase_sentiment_range",
    "get_phase_volume_percentage",
]
