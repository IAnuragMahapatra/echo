![](https://github.com/PSonakshi/echo/blob/main/assets/architecture.jpeg)

# Crypto Narrative Pulse Tracker

Real-time streaming analytics system built on Pathway that processes social media data to generate momentum signals for crypto traders.

## Quick Start with Docker

### Prerequisites

- Docker Desktop running
- `.env` file configured (see `.env.example`)

### Run the Pipeline

```bash
# Build and start the pipeline
docker-compose up pipeline

# In another terminal, send test messages
python send_test_message.py --count 10 --bullish
```

### Run with Simulator (Demo Mode)

```bash
# Start pipeline + simulator together
docker-compose --profile demo up
```

### Test Endpoints

The pipeline exposes a webhook at `http://localhost:8080/` that accepts POST requests:

```bash
# Send a test message using curl
curl -X POST http://localhost:8080/ \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "test_001",
    "text": "$MEME to the moon! 🚀",
    "author_id": "user_123",
    "author_followers": 5000,
    "timestamp": "2026-01-04T00:00:00Z",
    "tags": "[\"#MEME\", \"#crypto\"]",
    "engagement_count": 100,
    "source_platform": "test"
  }'
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TELEGRAM_TOKEN` | Telegram bot token | - |
| `TELEGRAM_CHANNEL_ID` | Telegram channel (e.g., `@echo_pathway`) | - |
| `TRACKED_COIN` | Cryptocurrency to track | `MEME` |
| `WEBHOOK_PORT` | Port for webhook endpoint | `8080` |

## Local Development (Windows)

Since Pathway requires Linux, use the demo modes for local development:

```bash
# Activate virtual environment
.venv_demo\Scripts\activate

# Run demo mode (simulated Pathway)
python main.py

# Or run live showcase with LLM
python showcase_live.py
```

## Architecture

- This is temporary README with a nano banana generated architecture diagram for reference of developers
