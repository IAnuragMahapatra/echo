#!/usr/bin/env python
"""
Simple test script to send messages to the Crypto Pulse Pipeline.

Usage:
    python send_test_message.py                    # Send a single test message
    python send_test_message.py --count 10         # Send 10 messages
    python send_test_message.py --bullish          # Send bullish message
    python send_test_message.py --bearish          # Send bearish message
    python send_test_message.py --url http://localhost:8080  # Custom URL
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone

import requests

# Default webhook URL (matches docker-compose port mapping)
DEFAULT_URL = "http://localhost:8080"

# Sample messages for different sentiments
BULLISH_MESSAGES = [
    "$MEME to the moon! 🚀",
    "$MEME looking bullish af! LFG! 🔥",
    "$MEME 100x potential, never selling! 💎",
    "$MEME breaking out, momentum building! 📈",
    "$MEME generational wealth incoming! 🌙",
]

BEARISH_MESSAGES = [
    "$MEME looking weak, be careful ⚠️",
    "$MEME might be a rug, do your research 🤔",
    "$MEME top signal, taking profits 📉",
    "$MEME fud spreading, stay cautious",
    "$MEME cooling off, healthy pullback maybe",
]

NEUTRAL_MESSAGES = [
    "$MEME consolidating, watching closely 👀",
    "$MEME interesting project, need more research",
    "$MEME volume picking up, something brewing",
    "$MEME chart looking interesting 📊",
]


def generate_message(sentiment: str = "random", coin: str = "MEME") -> dict:
    """Generate a test message."""
    if sentiment == "bullish":
        text = random.choice(BULLISH_MESSAGES)
    elif sentiment == "bearish":
        text = random.choice(BEARISH_MESSAGES)
    elif sentiment == "neutral":
        text = random.choice(NEUTRAL_MESSAGES)
    else:
        text = random.choice(BULLISH_MESSAGES + BEARISH_MESSAGES + NEUTRAL_MESSAGES)

    # Replace $MEME with actual coin if different
    text = text.replace("$MEME", f"${coin}")

    # Random author (10% chance of being an influencer)
    is_influencer = random.random() < 0.1
    if is_influencer:
        author_id = f"influencer_{random.randint(1, 5)}"
        followers = random.randint(100000, 500000)
        engagement = random.randint(500, 2000)
    else:
        author_id = f"user_{random.randint(1000, 99999)}"
        followers = random.randint(50, 5000)
        engagement = random.randint(1, 100)

    return {
        "message_id": f"test_{int(time.time() * 1000)}_{random.randint(0, 9999)}",
        "text": text,
        "author_id": author_id,
        "author_followers": followers,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tags": json.dumps([f"#{coin}", "#crypto"]),
        "engagement_count": engagement,
        "source_platform": "test_script",
    }


def send_message(url: str, message: dict) -> bool:
    """Send a message to the pipeline."""
    try:
        # Pathway rest_connector processes async, may not return immediately
        response = requests.post(url, json=message, timeout=10)
        if response.status_code in (200, 201, 202, 405):
            # 405 can happen on GET, but POST should work
            print(f"✓ Sent: {message['text'][:50]}...")
            return True
        else:
            print(f"✗ Failed ({response.status_code}): {response.text[:100]}")
            return False
    except requests.exceptions.ReadTimeout:
        # Pathway may not respond immediately but still processes the message
        print(f"✓ Sent (async): {message['text'][:50]}...")
        return True
    except requests.exceptions.ConnectionError:
        print(f"✗ Connection failed - is the pipeline running at {url}?")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Send test messages to Crypto Pulse Pipeline"
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL, help=f"Webhook URL (default: {DEFAULT_URL})"
    )
    parser.add_argument(
        "--count", type=int, default=1, help="Number of messages to send"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Delay between messages (seconds)"
    )
    parser.add_argument(
        "--bullish", action="store_true", help="Send only bullish messages"
    )
    parser.add_argument(
        "--bearish", action="store_true", help="Send only bearish messages"
    )
    parser.add_argument(
        "--neutral", action="store_true", help="Send only neutral messages"
    )
    parser.add_argument("--coin", default="MEME", help="Coin symbol (default: MEME)")

    args = parser.parse_args()

    # Determine sentiment
    if args.bullish:
        sentiment = "bullish"
    elif args.bearish:
        sentiment = "bearish"
    elif args.neutral:
        sentiment = "neutral"
    else:
        sentiment = "random"

    print(f"Sending {args.count} {sentiment} message(s) to {args.url}")
    print("-" * 50)

    success_count = 0
    for i in range(args.count):
        message = generate_message(sentiment, args.coin)
        if send_message(args.url, message):
            success_count += 1

        if i < args.count - 1:
            time.sleep(args.delay)

    print("-" * 50)
    print(f"Sent {success_count}/{args.count} messages successfully")


if __name__ == "__main__":
    main()
