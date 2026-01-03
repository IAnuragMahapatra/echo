"""
Hype Simulator for Crypto Narrative Pulse Tracker.

Generates realistic hype cycle data for demos, including:
- Phase-based message generation (seed, growth, peak, decline)
- Influencer accounts with high follower counts
- Correlated price data following hype cycle phases
- HTTP webhook sender for pipeline integration

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

import logging
import os
import random
import time
from datetime import datetime
from typing import Any

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# PHASE CONFIGURATION
# =============================================================================

# Phase distribution: seed 10%, growth 40%, peak 30%, decline 20%
# Requirements: 10.1
PHASES = {
    "seed": {
        "volume_pct": 0.10,
        "sentiment_range": (0.1, 0.4),
        "phrases": [
            "early gem",
            "hidden alpha",
            "under the radar",
            "accumulating quietly",
            "smart money loading",
            "low cap gem",
            "sleeping giant",
        ],
    },
    "growth": {
        "volume_pct": 0.40,
        "sentiment_range": (0.3, 0.7),
        "phrases": [
            "to the moon",
            "bullish af",
            "lfg",
            "breaking out",
            "momentum building",
            "volume increasing",
            "chart looking good",
            "higher lows",
            "accumulation phase over",
        ],
    },
    "peak": {
        "volume_pct": 0.30,
        "sentiment_range": (0.5, 0.9),
        "phrases": [
            "100x potential",
            "generational wealth",
            "never selling",
            "life changing money",
            "this is it",
            "parabolic",
            "price discovery",
            "new ath incoming",
            "unstoppable",
        ],
    },
    "decline": {
        "volume_pct": 0.20,
        "sentiment_range": (-0.3, 0.3),
        "phrases": [
            "taking profits",
            "be careful",
            "top signal",
            "cooling off",
            "healthy pullback",
            "buying the dip",
            "shaking out weak hands",
            "consolidation",
        ],
    },
}

# Phase order for simulation
PHASE_ORDER = ["seed", "growth", "peak", "decline"]

# Price multipliers for each phase (relative to base price)
# Requirements: 10.5
PRICE_MULTIPLIERS = {
    "seed": 1.0,
    "growth": 1.5,
    "peak": 2.5,
    "decline": 1.8,
}

# =============================================================================
# INFLUENCER CONFIGURATION
# =============================================================================

# Simulated influencer accounts with high follower counts
# Requirements: 10.4
INFLUENCERS = [
    {"id": "crypto_whale_1", "followers": 500000, "name": "CryptoWhale"},
    {"id": "degen_trader_2", "followers": 250000, "name": "DegenTrader"},
    {"id": "nft_guru_3", "followers": 150000, "name": "NFTGuru"},
    {"id": "alpha_hunter_4", "followers": 350000, "name": "AlphaHunter"},
    {"id": "moon_boy_5", "followers": 200000, "name": "MoonBoy"},
]

# Probability of a message being from an influencer
INFLUENCER_PROBABILITY = 0.10


# =============================================================================
# HYPE SIMULATOR CLASS
# =============================================================================


class HypeSimulator:
    """
    Generates realistic hype cycle data for demos.

    The simulator creates messages in four phases:
    - Seed (10%): Early discovery, low sentiment
    - Growth (40%): Building momentum, moderate sentiment
    - Peak (30%): Maximum hype, high sentiment
    - Decline (20%): Cooling off, mixed sentiment

    Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
    """

    def __init__(
        self,
        webhook_url: str,
        coin_symbol: str = "MEME",
        total_messages: int = 200,
        duration_mins: float = 3.0,
        base_price: float = 0.001,
    ):
        """
        Initialize the Hype Simulator.

        Args:
            webhook_url: URL to send messages to (pipeline webhook endpoint)
            coin_symbol: Cryptocurrency symbol to simulate (default: MEME)
            total_messages: Total number of messages to generate (default: 200)
            duration_mins: Duration of simulation in minutes (default: 3.0)
            base_price: Starting price for simulation (default: 0.001)
        """
        self.webhook_url = webhook_url
        self.coin_symbol = coin_symbol
        self.total_messages = total_messages
        self.duration_secs = duration_mins * 60
        self.base_price = base_price
        self.messages_sent = 0
        self.prices_sent = 0

    def run(self) -> dict[str, Any]:
        """
        Run the full hype cycle simulation.

        Generates messages and price data across all phases,
        sending them to the configured webhook endpoint.

        Returns:
            Dictionary with simulation statistics

        Requirements: 10.1, 10.2
        """
        logger.info(
            f"Starting hype simulation for ${self.coin_symbol}: "
            f"{self.total_messages} messages over {self.duration_secs}s"
        )

        start_time = time.time()
        phase_stats = {}

        for phase_name in PHASE_ORDER:
            phase_config = PHASES[phase_name]
            phase_messages = int(self.total_messages * phase_config["volume_pct"])
            phase_duration = self.duration_secs * phase_config["volume_pct"]

            if phase_messages == 0:
                continue

            delay = phase_duration / phase_messages

            logger.info(
                f"Phase '{phase_name}': {phase_messages} messages, "
                f"{phase_duration:.1f}s duration, {delay:.2f}s delay"
            )

            phase_start = time.time()
            phase_sent = 0
            phase_influencer_count = 0

            for i in range(phase_messages):
                # Generate and send message
                message = self._generate_message(phase_name, phase_config)
                success = self._send_message(message)

                if success:
                    phase_sent += 1
                    self.messages_sent += 1
                    if message.get("_is_influencer"):
                        phase_influencer_count += 1

                # Generate and send correlated price data
                price = self._generate_price(phase_name)
                self._send_price(price)
                self.prices_sent += 1

                # Wait before next message (except for last message)
                if i < phase_messages - 1:
                    time.sleep(delay)

            phase_stats[phase_name] = {
                "messages_sent": phase_sent,
                "influencer_messages": phase_influencer_count,
                "duration": time.time() - phase_start,
            }

        total_duration = time.time() - start_time

        stats = {
            "total_messages_sent": self.messages_sent,
            "total_prices_sent": self.prices_sent,
            "total_duration_secs": total_duration,
            "phase_stats": phase_stats,
            "coin_symbol": self.coin_symbol,
        }

        logger.info(f"Simulation complete: {stats}")
        return stats

    def _generate_message(self, phase_name: str, phase_config: dict) -> dict[str, Any]:
        """
        Generate a single message for the current phase.

        Args:
            phase_name: Current phase name
            phase_config: Phase configuration dictionary

        Returns:
            Message dictionary conforming to MessageSchema

        Requirements: 10.3, 10.4
        """
        # Determine if this is an influencer message
        is_influencer = random.random() < INFLUENCER_PROBABILITY

        if is_influencer:
            author = random.choice(INFLUENCERS)
            author_id = author["id"]
            author_followers = author["followers"]
            engagement_base = random.randint(500, 2000)
        else:
            author_id = f"user_{random.randint(1000, 99999)}"
            author_followers = random.randint(50, 5000)
            engagement_base = random.randint(1, 100)

        # Generate sentiment-appropriate text
        sentiment_min, sentiment_max = phase_config["sentiment_range"]
        sentiment = random.uniform(sentiment_min, sentiment_max)
        phrase = random.choice(phase_config["phrases"])

        # Build message text with emoji based on sentiment
        emoji = self._get_sentiment_emoji(sentiment)
        text = f"${self.coin_symbol} {phrase}! {emoji}"

        # Add extra hype for high sentiment
        if sentiment > 0.7:
            text += " " + random.choice(["🔥", "💎", "🚀", "💰"])

        # Generate unique message ID
        message_id = f"sim_{int(time.time() * 1000)}_{random.randint(0, 9999)}"

        # Calculate engagement (higher for influencers and high sentiment)
        engagement_multiplier = 1.0 + sentiment
        engagement_count = int(engagement_base * engagement_multiplier)

        message = {
            "message_id": message_id,
            "text": text,
            "author_id": author_id,
            "author_followers": author_followers,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "tags": [f"#{self.coin_symbol}", "#crypto", f"${self.coin_symbol}"],
            "engagement_count": engagement_count,
            "source_platform": "simulator",
            "_is_influencer": is_influencer,  # Internal tracking, not sent
            "_phase": phase_name,  # Internal tracking
            "_sentiment": sentiment,  # Internal tracking
        }

        return message

    def _generate_price(self, phase_name: str) -> dict[str, Any]:
        """
        Generate correlated price data for the current phase.

        Price follows the hype cycle with some noise:
        - Seed: ~base_price
        - Growth: ~1.5x base_price
        - Peak: ~2.5x base_price
        - Decline: ~1.8x base_price

        Args:
            phase_name: Current phase name

        Returns:
            Price dictionary conforming to PriceSchema

        Requirements: 10.5
        """
        multiplier = PRICE_MULTIPLIERS[phase_name]

        # Add some noise (±10%)
        noise = random.uniform(-0.10, 0.10)
        price = self.base_price * multiplier * (1 + noise)

        # Generate volume (higher during peak)
        volume_base = 500000
        volume_multiplier = {
            "seed": 0.5,
            "growth": 1.0,
            "peak": 2.0,
            "decline": 1.2,
        }
        volume = volume_base * volume_multiplier[phase_name] * random.uniform(0.8, 1.2)

        return {
            "coin_symbol": self.coin_symbol,
            "price_usd": round(price, 8),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "volume_24h": round(volume, 2),
        }

    def _get_sentiment_emoji(self, sentiment: float) -> str:
        """Get appropriate emoji based on sentiment score."""
        if sentiment > 0.7:
            return random.choice(["🚀", "🔥", "💎", "🌙"])
        elif sentiment > 0.4:
            return random.choice(["📈", "💪", "✨", "👀"])
        elif sentiment > 0.0:
            return random.choice(["🤔", "👍", "📊"])
        else:
            return random.choice(["⚠️", "🤷", "📉"])

    def _send_message(self, message: dict[str, Any]) -> bool:
        """
        Send a message to the pipeline webhook endpoint.

        Args:
            message: Message dictionary to send

        Returns:
            True if successful, False otherwise

        Requirements: 10.2
        """
        # Remove internal tracking fields before sending
        payload = {k: v for k, v in message.items() if not k.startswith("_")}

        # Convert tags list to JSON string for Pathway schema
        if isinstance(payload.get("tags"), list):
            import json

            payload["tags"] = json.dumps(payload["tags"])

        try:
            # Pathway rest_connector processes async and may not respond immediately
            # Use a longer timeout and treat ReadTimeout as success
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )
            if response.status_code in (200, 201, 202):
                logger.debug(f"Message sent: {message['message_id']}")
                return True
            else:
                logger.warning(
                    f"Failed to send message: {response.status_code} - {response.text}"
                )
                return False
        except requests.exceptions.ReadTimeout:
            # Pathway processes async - timeout doesn't mean failure
            logger.debug(f"Message sent (async): {message['message_id']}")
            return True
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error sending message: {e}")
            return False

    def _send_price(self, price: dict[str, Any]) -> bool:
        """
        Send price data to the pipeline webhook endpoint.

        Note: Price data is logged but not sent to the main message pipeline.
        The pipeline focuses on social messages; price data is handled separately.

        Args:
            price: Price dictionary to send

        Returns:
            True (always succeeds as price is just logged)

        Requirements: 10.2
        """
        # For now, just log price data - the main pipeline handles messages
        # Price correlation can be added via a separate price stream if needed
        logger.debug(
            f"Price update: {price['coin_symbol']} @ ${price['price_usd']:.8f}"
        )
        return True


# =============================================================================
# STANDALONE FUNCTIONS FOR TESTING
# =============================================================================


def generate_single_message(
    coin_symbol: str = "MEME",
    phase: str = "growth",
) -> dict[str, Any]:
    """
    Generate a single message for testing purposes.

    Args:
        coin_symbol: Cryptocurrency symbol
        phase: Phase name (seed, growth, peak, decline)

    Returns:
        Message dictionary conforming to MessageSchema
    """
    if phase not in PHASES:
        raise ValueError(
            f"Invalid phase: {phase}. Must be one of {list(PHASES.keys())}"
        )

    simulator = HypeSimulator(
        webhook_url="http://localhost:8080",
        coin_symbol=coin_symbol,
    )
    return simulator._generate_message(phase, PHASES[phase])


def generate_single_price(
    coin_symbol: str = "MEME",
    phase: str = "growth",
    base_price: float = 0.001,
) -> dict[str, Any]:
    """
    Generate a single price point for testing purposes.

    Args:
        coin_symbol: Cryptocurrency symbol
        phase: Phase name (seed, growth, peak, decline)
        base_price: Base price for calculation

    Returns:
        Price dictionary conforming to PriceSchema
    """
    if phase not in PHASES:
        raise ValueError(
            f"Invalid phase: {phase}. Must be one of {list(PHASES.keys())}"
        )

    simulator = HypeSimulator(
        webhook_url="http://localhost:8080",
        coin_symbol=coin_symbol,
        base_price=base_price,
    )
    return simulator._generate_price(phase)


def get_phase_sentiment_range(phase: str) -> tuple[float, float]:
    """
    Get the sentiment range for a given phase.

    Args:
        phase: Phase name (seed, growth, peak, decline)

    Returns:
        Tuple of (min_sentiment, max_sentiment)
    """
    if phase not in PHASES:
        raise ValueError(
            f"Invalid phase: {phase}. Must be one of {list(PHASES.keys())}"
        )
    return PHASES[phase]["sentiment_range"]


def get_phase_volume_percentage(phase: str) -> float:
    """
    Get the volume percentage for a given phase.

    Args:
        phase: Phase name (seed, growth, peak, decline)

    Returns:
        Volume percentage (0.0 to 1.0)
    """
    if phase not in PHASES:
        raise ValueError(
            f"Invalid phase: {phase}. Must be one of {list(PHASES.keys())}"
        )
    return PHASES[phase]["volume_pct"]


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """
    Main entry point for running the simulator from command line.

    Environment variables:
        WEBHOOK_URL: Pipeline webhook URL (default: http://localhost:8080)
        TRACKED_COIN: Cryptocurrency symbol (default: MEME)
        SIMULATOR_DURATION_MINS: Duration in minutes (default: 3.0)
        SIMULATOR_MESSAGE_COUNT: Total messages (default: 200)
    """
    # Load configuration from environment
    webhook_url = os.environ.get("WEBHOOK_URL", "http://localhost:8080")
    coin_symbol = os.environ.get("TRACKED_COIN", "MEME")
    duration_mins = float(os.environ.get("SIMULATOR_DURATION_MINS", "3.0"))
    message_count = int(os.environ.get("SIMULATOR_MESSAGE_COUNT", "200"))

    logger.info("Configuration:")
    logger.info(f"  Webhook URL: {webhook_url}")
    logger.info(f"  Coin Symbol: {coin_symbol}")
    logger.info(f"  Duration: {duration_mins} minutes")
    logger.info(f"  Message Count: {message_count}")

    # Create and run simulator
    simulator = HypeSimulator(
        webhook_url=webhook_url,
        coin_symbol=coin_symbol,
        total_messages=message_count,
        duration_mins=duration_mins,
    )

    try:
        stats = simulator.run()
        logger.info("Simulation completed successfully!")
        logger.info(f"Total messages sent: {stats['total_messages_sent']}")
        logger.info(f"Total duration: {stats['total_duration_secs']:.1f}s")
        return 0
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
