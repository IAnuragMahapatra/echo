# Requirements Document

## Introduction

The Crypto Narrative Pulse Tracker is a real-time streaming analytics system that monitors social media and price feeds for cryptocurrency-related discussions, analyzes sentiment velocity, detects emerging narratives through phrase clustering, correlates with live price data, and provides traders with actionable momentum signals via a Telegram bot. The system uses Pathway for live data processing and RAG-based LLM queries, demonstrating Pathway's extensibility through custom connectors.

## Glossary

- **Pipeline**: The Pathway streaming data processing system that ingests, transforms, and stores data using pw.temporal, pw.join, pw.groupby, and pw.reducers APIs
- **Social_Stream**: The continuous flow of incoming social data from webhooks, Discord connector, or the simulator
- **Price_Stream**: Live cryptocurrency price data from CoinGecko API
- **Sentiment_Analyzer**: Component that scores sentiment using VADER or FinBERT on a -1 to 1 scale
- **Phrase_Clusterer**: Component that extracts and groups repeating n-gram phrases
- **Influence_Calculator**: Component that weights posts based on author metrics and engagement
- **Pulse_Score**: A 1-10 momentum score combining sentiment velocity, phrase frequency, influencer signals, and price correlation
- **Document_Store**: Pathway's vector store for continuous indexing and RAG retrieval
- **RAG_System**: Retrieval-Augmented Generation system using OpenAI GPT-4 with sentence-transformers embeddings
- **Telegram_Bot**: The output interface that sends alerts and handles user queries
- **Hype_Simulator**: Python script that generates realistic fake social data for demos
- **Discord_Connector**: Custom Pathway connector demonstrating framework extensibility
- **Dashboard**: Streamlit-based real-time visualization interface

## Requirements

### Requirement 1: Multi-Source Data Ingestion

**User Story:** As a system operator, I want to ingest social data from multiple sources, so that the pipeline can process real-time crypto discussions from diverse channels.

#### Acceptance Criteria

1. WHEN a social data webhook payload is received, THE Pipeline SHALL parse and validate it against the MessageSchema (message_id, text, author_id, author_followers, timestamp, tags, engagement_count, source_platform)
2. WHEN the Hype_Simulator generates a message, THE Pipeline SHALL accept it via the HTTP endpoint as the primary demo data source
3. IF a message payload is malformed or missing required fields, THEN THE Pipeline SHALL reject it and log the error with structured logging
4. THE Pipeline SHALL support filtering messages by configurable tag patterns (e.g., #Solana, $MEME)
5. THE Pipeline SHALL use pw.io.http.rest_connector() for webhook-based ingestion

### Requirement 2: Custom Discord Connector

**User Story:** As a developer, I want to create a custom Pathway connector for crypto Discord channels, so that I can demonstrate Pathway's extensibility.

**Note:** Discord connector is a stretch goal to demonstrate extensibility. Primary data source is Hype_Simulator with optional webhook fallback.

#### Acceptance Criteria

1. THE Discord_Connector SHALL implement Pathway's custom connector API using pw.io.python.read()
2. THE Discord_Connector SHALL subscribe to Discord webhook events for specified channels
3. THE Discord_Connector SHALL transform Discord messages to the unified MessageSchema format
4. IF the Discord webhook fails, THEN THE Discord_Connector SHALL retry 3 times with exponential backoff
5. THE Discord_Connector SHALL log connection status and message throughput

### Requirement 3: Sentiment Analysis and Velocity Tracking

**User Story:** As a trader, I want to see how sentiment is changing over time, so that I can detect momentum shifts before price moves.

#### Acceptance Criteria

1. WHEN a message is ingested, THE Sentiment_Analyzer SHALL compute a sentiment score between -1 (bearish) and 1 (bullish) using VADER or FinBERT
2. THE Pipeline SHALL calculate sentiment velocity using pw.temporal.sliding() with a 5-minute duration and 1-minute hops
3. WHEN sentiment velocity exceeds 0.7, THE Pipeline SHALL classify it as "strong bullish momentum"
4. WHEN sentiment velocity falls below -0.7, THE Pipeline SHALL classify it as "strong bearish momentum"
5. THE Pipeline SHALL use pw.reducers.avg() for velocity aggregation

### Requirement 4: Live Price Data Correlation

**User Story:** As a trader, I want to see how sentiment correlates with actual price movements, so that I can validate the pulse score and detect divergences.

#### Acceptance Criteria

1. THE Pipeline SHALL ingest live price data from CoinGecko API at 1-minute intervals
2. THE Pipeline SHALL calculate price delta percentage over 5-minute windows
3. THE Pipeline SHALL use pw.join() with pw.temporal.windowby() to correlate price deltas with sentiment windows
4. WHEN sentiment is high (>0.5) but price is falling (>-2%), THE Pipeline SHALL flag a "bearish divergence" warning
5. WHEN sentiment is low (<-0.5) but price is rising (>2%), THE Pipeline SHALL flag a "bullish divergence" warning
6. THE Pulse_Score SHALL incorporate price-sentiment correlation as a modifier
7. THE Price_Stream SHALL handle CoinGecko rate limits (50 calls/min) by caching and batching requests

### Requirement 5: Phrase Clustering and Narrative Detection

**User Story:** As a trader, I want to see what phrases are trending, so that I can understand the narrative driving momentum.

#### Acceptance Criteria

1. WHEN a message is processed, THE Phrase_Clusterer SHALL extract key bigram and trigram phrases
2. THE Pipeline SHALL track phrase frequency within a 10-minute rolling window using pw.temporal.sliding()
3. WHEN a phrase appears 5 or more times within the window, THE Pipeline SHALL mark it as "trending"
4. THE Pipeline SHALL maintain a ranked list of top trending phrases sorted by frequency using pw.groupby() and pw.reducers.count()

### Requirement 6: Influencer Signal Tracking

**User Story:** As a trader, I want to know what influential accounts are saying, so that I can weight their opinions more heavily.

#### Acceptance Criteria

1. WHEN a message is processed, THE Influence_Calculator SHALL compute an influence score as (followers × 0.6) + (engagement × 0.4)
2. THE Pipeline SHALL classify authors with influence score above 10,000 as "influencers"
3. THE Pipeline SHALL track bullish vs bearish influencer counts in 10-minute tumbling windows using pw.temporal.tumbling()
4. THE Pipeline SHALL calculate an influencer consensus ratio (bullish_count / total_influencer_count)

### Requirement 7: Pulse Score Calculation

**User Story:** As a trader, I want a single momentum score, so that I can quickly assess whether to enter or exit a position.

#### Acceptance Criteria

1. THE Pipeline SHALL calculate a Pulse_Score from 1-10 by combining sentiment velocity (0-4 points), phrase frequency spike (0-3 points), influencer ratio (0-3 points), and price correlation modifier
2. WHEN sentiment velocity exceeds 0.7, THE Pipeline SHALL add 4 points to the Pulse_Score
3. WHEN sentiment velocity is between 0.4 and 0.7, THE Pipeline SHALL add 2 points to the Pulse_Score
4. WHEN trending phrase frequency exceeds 20, THE Pipeline SHALL add 3 points to the Pulse_Score
5. WHEN trending phrase frequency is between 10 and 20, THE Pipeline SHALL add 1.5 points to the Pulse_Score
6. WHEN influencer bullish ratio exceeds 0.7, THE Pipeline SHALL add 3 points to the Pulse_Score
7. WHEN influencer bullish ratio is between 0.5 and 0.7, THE Pipeline SHALL add 1.5 points to the Pulse_Score
8. WHEN a bearish divergence is detected, THE Pipeline SHALL subtract 1 point from the Pulse_Score
9. THE Pipeline SHALL cap the Pulse_Score between 1 and 10

### Requirement 8: Document Store and RAG Integration

**User Story:** As a trader, I want to query the system with natural language, so that I can get contextual answers about current momentum.

#### Acceptance Criteria

1. WHEN a message is processed, THE Document_Store SHALL embed it using sentence-transformers and index it within 1 second
2. THE Document_Store SHALL store message metadata including message_id, timestamp, sentiment, influence_score, and source_platform
3. WHEN a RAG query is received, THE RAG_System SHALL retrieve the top 15 most relevant recent messages
4. THE RAG_System SHALL enrich the LLM prompt with current Pulse_Score, top 3 trending phrases, influencer consensus, and price correlation status
5. IF embedding fails, THEN THE Document_Store SHALL retry 3 times before logging the error

### Requirement 9: Telegram Bot Alerts and Queries

**User Story:** As a trader, I want to receive alerts and ask questions via Telegram, so that I can stay informed on mobile.

#### Acceptance Criteria

1. WHEN the Pulse_Score reaches 7 or above, THE Telegram_Bot SHALL send a "Strong Buy Signal" alert with emoji indicators
2. WHEN the Pulse_Score drops to 3 or below, THE Telegram_Bot SHALL send a "Cooling Off" alert with emoji indicators
3. WHEN a divergence is detected, THE Telegram_Bot SHALL send a warning alert
4. WHEN a user sends a query to the Telegram_Bot, THE RAG_System SHALL respond with pulse score, top phrases, price correlation, and risk assessment
5. THE Telegram_Bot SHALL format responses with emoji indicators for quick scanning

### Requirement 10: Demo Hype Simulator

**User Story:** As a demo presenter, I want to simulate realistic hype cycles, so that I can demonstrate the system's capabilities in 3 minutes.

#### Acceptance Criteria

1. THE Hype_Simulator SHALL generate messages in four phases: seed (10% volume), growth (40% volume), peak (30% volume), decline (20% volume)
2. WHEN running in demo mode, THE Hype_Simulator SHALL inject 200 messages over 3-5 minutes (configurable duration)
3. THE Hype_Simulator SHALL vary sentiment and phrases according to the current phase
4. THE Hype_Simulator SHALL include simulated influencer accounts with high follower counts
5. THE Hype_Simulator SHALL also generate correlated price data that follows the hype cycle

### Requirement 11: Live Dashboard

**User Story:** As a demo presenter and trader, I want a visual dashboard, so that I can see real-time analytics at a glance.

#### Acceptance Criteria

1. THE Dashboard SHALL display a real-time line chart of Pulse_Score over time
2. THE Dashboard SHALL display a word cloud of trending phrases updated every 30 seconds
3. THE Dashboard SHALL display an influencer leaderboard showing top contributors and their sentiment
4. THE Dashboard SHALL display current price and price-sentiment correlation status
5. THE Dashboard SHALL be built using Streamlit for rapid development

### Requirement 12: Performance Monitoring

**User Story:** As a system operator, I want to monitor system performance, so that I can ensure the pipeline meets latency requirements.

#### Acceptance Criteria

1. THE Pipeline SHALL measure and log end-to-end latency from ingestion to alert delivery
2. THE Pipeline SHALL track and display processing throughput in messages per second
3. THE Dashboard SHALL display current latency and throughput metrics
4. WHEN end-to-end latency exceeds 5 seconds, THE Pipeline SHALL log a warning
5. THE System SHALL log RAG retrieval relevance scores for monitoring query quality

### Requirement 13: Docker Deployment and Fault Tolerance

**User Story:** As a system operator, I want to deploy the entire system with one command and have it recover from failures, so that setup is simple and the system is reliable.

#### Acceptance Criteria

1. THE System SHALL provide a docker-compose.yml that starts all services (pipeline, document store, telegram bot, dashboard, price fetcher)
2. WHEN docker-compose up is executed, THE System SHALL be fully operational within 60 seconds
3. THE System SHALL use Pathway's persistence API (pw.persistence.Backend.filesystem) for state recovery
4. THE System SHALL demonstrate fault tolerance by surviving container restarts without losing stream position
5. THE System SHALL expose configurable environment variables for API keys, tokens, and tracked coin symbols
