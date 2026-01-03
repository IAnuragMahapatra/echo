# Frontend Integration Guide - Crypto Narrative Pulse Tracker

## Overview

This guide covers everything you need to integrate a Next.js frontend with the existing Python backend for the Crypto Narrative Pulse Tracker.

---

## Project Structure

```
frontend/
├── app/
│   ├── page.tsx                 # Landing page
│   ├── login/page.tsx           # Login page
│   ├── signup/page.tsx          # Sign up page
│   ├── dashboard/page.tsx       # Main dashboard
│   ├── team/page.tsx            # Meet the team
│   └── layout.tsx               # Root layout
├── components/
│   ├── ui/                      # Reusable UI components
│   ├── dashboard/               # Dashboard-specific components
│   │   ├── PulseScoreCard.tsx
│   │   ├── PulseChart.tsx
│   │   ├── TrendingPhrases.tsx
│   │   ├── DivergenceStatus.tsx
│   │   └── CoinSelector.tsx
│   └── landing/                 # Landing page components
├── lib/
│   ├── api.ts                   # API client functions
│   ├── auth.ts                  # Auth utilities
│   └── types.ts                 # TypeScript types
├── hooks/
│   ├── usePulseScore.ts         # Real-time pulse score hook
│   ├── useMetrics.ts            # Metrics polling hook
│   └── useAuth.ts               # Auth hook
└── public/
    └── team/                    # Team member photos
```

---

## Backend API Endpoints to Create

You'll need these endpoints in the Python backend. Add to `app.py` or create `api/routes.py`:

### 1. Metrics Endpoint (Required)

```python
# GET /api/metrics
# Returns current pulse tracker metrics

@app.route('/api/metrics')
def get_metrics():
    from rag.live_metrics import get_live_metrics
    metrics = get_live_metrics()
    return {
        "pulse_score": metrics.pulse_score,
        "trending_phrases": metrics.trending_phrases,
        "influencer_consensus": metrics.influencer_consensus,
        "divergence_status": metrics.divergence_status,
        "timestamp": metrics.last_updated.isoformat()
    }
```

### 2. Metrics History Endpoint (For Charts)

```python
# GET /api/metrics/history?hours=24
# Returns historical pulse scores for charting

@app.route('/api/metrics/history')
def get_metrics_history():
    hours = request.args.get('hours', 24, type=int)
    # Return list of {timestamp, pulse_score, sentiment} objects
    return {"history": [...]}
```

### 3. Coin Configuration Endpoint

```python
# GET /api/config
# POST /api/config  (body: {"coin": "SOL"})

@app.route('/api/config', methods=['GET', 'POST'])
def config():
    if request.method == 'POST':
        coin = request.json.get('coin')
        # Update tracked coin
        return {"success": True, "coin": coin}
    return {"coin": os.getenv("TRACKED_COIN", "MEME")}
```

### 4. RAG Query Endpoint

```python
# POST /api/query
# body: {"question": "What's the sentiment on SOL?"}

@app.route('/api/query', methods=['POST'])
def query():
    from rag.crypto_rag import CryptoRAG
    question = request.json.get('question')
    rag = CryptoRAG()
    response = rag.answer(question)
    return response.to_dict()
```

---

## TypeScript Types

Create `lib/types.ts`:

```typescript
export interface PulseMetrics {
  pulse_score: number;          // 1-10
  trending_phrases: string[];   // Top 5 phrases
  influencer_consensus: string; // "strongly bullish" | "moderately bullish" | "neutral" | etc.
  divergence_status: string;    // "aligned" | "bearish_divergence" | "bullish_divergence"
  timestamp: string;            // ISO timestamp
}

export interface MetricsHistory {
  timestamp: string;
  pulse_score: number;
  sentiment: number;
}

export interface User {
  id: string;
  email: string;
  tracked_coin: string;
  created_at: string;
}

export interface RAGResponse {
  answer: string;
  pulse_score: number;
  trending_phrases: string[];
  sources: object[];
  relevance_scores: number[];
}
```

---

## Real-Time Data Fetching

### Option 1: Polling (Simpler)

```typescript
// hooks/usePulseScore.ts
import { useState, useEffect } from 'react';
import { PulseMetrics } from '@/lib/types';

export function usePulseScore(refreshInterval = 5000) {
  const [metrics, setMetrics] = useState<PulseMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMetrics = async () => {
      const res = await fetch('/api/metrics');
      const data = await res.json();
      setMetrics(data);
      setLoading(false);
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshInterval]);

  return { metrics, loading };
}
```

### Option 2: WebSocket (Real-Time)

Add WebSocket support to backend:

```python
# Using flask-socketio
from flask_socketio import SocketIO, emit

socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('subscribe_metrics')
def handle_subscribe():
    # Send metrics updates when they change
    pass
```

Frontend:

```typescript
// hooks/useRealtimeMetrics.ts
import { useEffect, useState } from 'react';
import { io } from 'socket.io-client';

export function useRealtimeMetrics() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    const socket = io(process.env.NEXT_PUBLIC_API_URL);
    socket.emit('subscribe_metrics');
    socket.on('metrics_update', setMetrics);
    return () => socket.disconnect();
  }, []);

  return metrics;
}
```

---

## Page-by-Page Implementation Tips

### Landing Page (`/`)

**Key Sections:**

- Hero with animated pulse score demo
- Feature highlights (real-time alerts, sentiment analysis, divergence detection)
- How it works (3-step flow)
- CTA to sign up

**Tips:**

- Use Framer Motion for animations
- Show a live demo pulse score (can be simulated)
- Include testimonials section if you have beta users

**Suggested Components:**

```
<Hero />
<FeatureGrid />
<HowItWorks />
<LiveDemo />        // Embed a mini dashboard preview
<Testimonials />
<CTASection />
<Footer />
```

### Login/Signup (`/login`, `/signup`)

**Auth Options:**

1. **NextAuth.js** - Easy OAuth (Google, GitHub, Discord)
2. **Supabase Auth** - Full auth + database
3. **Custom JWT** - Roll your own with backend

**Recommended: Supabase**

- Free tier is generous
- Built-in user management
- Easy to integrate with Next.js

```bash
npm install @supabase/supabase-js @supabase/auth-helpers-nextjs
```

**User Settings to Store:**

- `tracked_coin` (default: "MEME")
- `alert_threshold_high` (default: 7)
- `alert_threshold_low` (default: 3)
- `telegram_chat_id` (optional)

### Dashboard (`/dashboard`)

**Layout:**

```
┌─────────────────────────────────────────────────────────┐
│  Header: Logo | Coin Selector | User Menu              │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────────────────────────┐ │
│  │ PULSE SCORE  │  │  Pulse Score Chart (24h)         │ │
│  │     8.5      │  │  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~    │ │
│  │   🚀 BUY     │  │                                  │ │
│  └──────────────┘  └──────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────────────────┐ │
│  │ Trending Phrases │  │ Divergence Status            │ │
│  │ • to the moon    │  │ ✅ Aligned                   │ │
│  │ • bullish af     │  │ Sentiment: +0.65             │ │
│  │ • lfg            │  │ Price Δ: +12.5%              │ │
│  └──────────────────┘  └──────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐│
│  │ Ask AI: [What's the sentiment on $MEME?] [Send]     ││
│  │ Response: Based on recent data...                   ││
│  └──────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

**Key Components:**

1. **PulseScoreCard** - Big number with color coding
   - Green (≥7): "🚀 Strong Buy"
   - Yellow (4-6): "➡️ Hold"
   - Red (≤3): "❄️ Cooling Off"

2. **PulseChart** - Line chart with Recharts or Chart.js

   ```bash
   npm install recharts
   ```

3. **CoinSelector** - Dropdown to change tracked coin
   - Popular coins: MEME, SOL, BTC, ETH, DOGE
   - Save to user settings

4. **TrendingPhrases** - Word cloud or simple list

   ```bash
   npm install react-wordcloud  # Optional
   ```

5. **DivergenceStatus** - Color-coded status card
   - Green: Aligned
   - Yellow: Bullish divergence
   - Red: Bearish divergence

6. **AIQueryBox** - Chat-like interface for RAG queries

### Meet the Team (`/team`)

**Structure:**

```typescript
const team = [
  {
    name: "Your Name",
    role: "Founder & Lead Developer",
    image: "/team/you.jpg",
    bio: "Building the future of crypto analytics...",
    socials: {
      twitter: "https://twitter.com/...",
      linkedin: "https://linkedin.com/in/...",
      github: "https://github.com/..."
    }
  },
  // ... more team members
];
```

**Tips:**

- Use consistent photo dimensions (400x400 recommended)
- Add hover effects on cards
- Include social links

---

## Environment Variables

### Frontend (`.env.local`)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Supabase (if using)
NEXT_PUBLIC_SUPABASE_URL=your-project-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key

# Analytics (optional)
NEXT_PUBLIC_GA_ID=G-XXXXXXXXXX
```

### Backend Updates (`.env`)

Add CORS origins:

```env
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

---

## CORS Configuration

Update backend to allow frontend requests:

```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=os.getenv("CORS_ORIGINS", "*").split(","))
```

---

## Recommended Tech Stack

| Category | Recommendation | Why |
|----------|---------------|-----|
| Framework | Next.js 14 (App Router) | Best DX, built-in API routes |
| Styling | Tailwind CSS + shadcn/ui | Fast, beautiful components |
| Charts | Recharts | Easy React integration |
| Auth | Supabase or NextAuth | Free, easy setup |
| State | Zustand or React Query | Lightweight, powerful |
| Animations | Framer Motion | Smooth, declarative |
| Icons | Lucide React | Clean, consistent |

```bash
# Quick setup
npx create-next-app@latest frontend --typescript --tailwind --app
cd frontend
npx shadcn-ui@latest init
npm install recharts framer-motion lucide-react @tanstack/react-query
```

---

## Feature Suggestions

### Must Have

- [ ] Real-time pulse score display
- [ ] Historical chart (24h/7d/30d)
- [ ] Coin selector with persistence
- [ ] Divergence alerts
- [ ] Mobile responsive design

### Nice to Have

- [ ] Dark/light mode toggle
- [ ] Push notifications (Web Push API)
- [ ] Price alerts configuration
- [ ] Export data to CSV
- [ ] Shareable dashboard links

### Impressive Additions

- [ ] Animated pulse score gauge
- [ ] Sound alerts for threshold crossings
- [ ] Telegram bot connection from dashboard
- [ ] Multi-coin comparison view
- [ ] Sentiment heatmap calendar
- [ ] AI chat history

---

## API Client Example

```typescript
// lib/api.ts
const API_URL = process.env.NEXT_PUBLIC_API_URL;

export async function getMetrics() {
  const res = await fetch(`${API_URL}/api/metrics`);
  if (!res.ok) throw new Error('Failed to fetch metrics');
  return res.json();
}

export async function getMetricsHistory(hours = 24) {
  const res = await fetch(`${API_URL}/api/metrics/history?hours=${hours}`);
  if (!res.ok) throw new Error('Failed to fetch history');
  return res.json();
}

export async function updateCoin(coin: string) {
  const res = await fetch(`${API_URL}/api/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ coin }),
  });
  if (!res.ok) throw new Error('Failed to update coin');
  return res.json();
}

export async function queryRAG(question: string) {
  const res = await fetch(`${API_URL}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error('Failed to query');
  return res.json();
}
```

---

## Deployment Checklist

### Frontend (Vercel)

- [ ] Set environment variables in Vercel dashboard
- [ ] Configure custom domain
- [ ] Enable analytics

### Backend

- [ ] Add CORS for production domain
- [ ] Set up SSL/HTTPS
- [ ] Configure rate limiting
- [ ] Add API authentication (API keys or JWT)

### Integration

- [ ] Test all API endpoints
- [ ] Verify WebSocket connection (if using)
- [ ] Test auth flow end-to-end
- [ ] Load test with multiple users

---

## Quick Start Commands

```bash
# Create frontend
npx create-next-app@latest frontend --typescript --tailwind --app
cd frontend

# Install dependencies
npm install recharts framer-motion lucide-react @tanstack/react-query zustand
npx shadcn-ui@latest init
npx shadcn-ui@latest add button card input

# Run development
npm run dev
```

---

## Questions to Decide

1. **Auth provider?** Supabase / NextAuth / Custom
2. **Database for user settings?** Supabase / PostgreSQL / MongoDB
3. **Hosting?** Vercel (frontend) + Railway/Render (backend)
4. **Real-time method?** Polling (simpler) vs WebSocket (faster)
5. **Multi-coin support?** One coin per user vs multiple watchlist

---

Good luck with the frontend! The backend is ready to serve data via the endpoints above. 🚀
