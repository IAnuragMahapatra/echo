# Project Notes & How to Run

## Important Constraints to Remember

1. **Pathway only works inside Docker** - `main.py` and `run_demo.py` fake/mock Pathway for Windows compatibility. The real Pathway streaming pipeline only runs via `docker-compose`.

2. **Demo CoinGecko API only** - We only have the demo API, meaning NO WebSocket support. Only polling is available via `connectors/price_fetcher.py` with caching.

3. **X (Twitter) API is paid** - Creating fake social media messages is expensive. We use the Hype Simulator instead with predefined phrase lists.

4. **No Ollama/LLM for now** - To save costs, we're not using any LLM. The RAG system has fallback responses that work without LLM.

5. **Saved texts repeat in stream** - The simulator uses predefined phrase lists per phase (seed, growth, peak, decline) that randomly repeat.

6. **docker-compose is the only way to run real Pathway pipeline** - For demos on Windows, use `run_demo.py` which simulates the pipeline behavior.

---

## How to Run

### Option 1: Demo Mode (Windows - No Docker Required)

```bash
# Run demo simulation with Telegram alerts
python run_demo.py

# Run demo without Telegram
python run_demo.py --no-telegram

# Start API server only (for frontend development)
python main.py

# Start API server on custom port
python main.py --port 5000

# Run demo mode via main.py
python main.py --demo
python main.py --demo --no-telegram
```

### Option 2: Docker - Pipeline + API Only

```bash
# Start the main pipeline and API server
docker-compose up

# Or in detached mode
docker-compose up -d

# View logs
docker-compose logs -f pipeline
```

### Option 3: Docker - Full Stack (Pipeline + Frontend + Telegram)

```bash
# Start everything including frontend dashboard
docker-compose --profile full up

# Or in detached mode
docker-compose --profile full up -d
```

### Option 4: Docker - With Hype Simulator

```bash
# Run simulator to generate demo data
docker-compose --profile demo up simulator

# Run full stack with simulator
docker-compose --profile full --profile demo up
```

### Option 5: Docker - With Ollama LLM (for RAG queries)

```bash
# Start with Ollama for AI-powered responses
docker-compose --profile llm up

# Full stack with LLM
docker-compose --profile full --profile llm up
```

### Frontend Development (Standalone)

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:3000
```

---

## API Endpoints (when running main.py or docker)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/metrics` | GET | Current pulse score and metrics |
| `/api/metrics/history` | GET | Historical data for charts |
| `/api/config` | GET/POST | Get or update tracked coin |
| `/api/query` | POST | RAG query endpoint |
| `/api/performance` | GET | Latency/throughput metrics |
| `/api/influencers` | GET | Influencer leaderboard |
| `/api/rag/stats` | GET | RAG relevance statistics |
| `/ws` | WebSocket | Real-time metrics updates |

---

## Environment Variables (.env)

```bash
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHANNEL_ID=your_channel_id
TRACKED_COIN=MEME
COINGECKO_API_KEY=CG-xxx  # Optional, demo key
CORS_ORIGINS=http://localhost:3000
```

---

## Quick Test Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run checkpoint tests only
python -m pytest tests/test_checkpoint_impressive.py -v

# Check if modules work
python -c "from transforms.sentiment import SentimentAnalyzer; print('OK')"
python -c "from transforms.pulse_score import PulseScoreCalculator; print('OK')"
```
