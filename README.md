# movies-dashboard

One repo, two apps — a Streamlit dashboard and a Next.js/Vercel dashboard with AI film recommendations, both reading from the same `movies.csv`.

---

## Repo Structure

```
movies-dashboard/
├── public/
│   └── movies.csv              ← THE SINGLE SOURCE OF TRUTH for both apps
├── streamlit/
│   └── app.py                  ← Streamlit dashboard
├── app/                        ← Next.js app (Vercel)
│   ├── api/
│   │   ├── chat/route.ts       ← AI recommendation API (Claude Haiku + prompt caching)
│   │   ├── movies/route.ts     ← Movie data API
│   │   ├── track/route.ts      ← Analytics event tracking (page views, chat events)
│   │   └── analytics/route.ts  ← Analytics dashboard API (password-protected)
│   ├── analytics/
│   │   └── page.tsx            ← Analytics dashboard UI (/analytics)
│   ├── stats/
│   │   └── page.tsx            ← Collection stats page (/stats)
│   ├── page.tsx                ← Main dashboard page
│   ├── layout.tsx
│   └── globals.css
├── components/
│   └── ChatPanel.tsx           ← AI chat panel with disambiguation chips
├── lib/
│   ├── movies.ts               ← Data loading logic
│   ├── heuristic.ts            ← Fallback recommendation logic
│   ├── supabase.ts             ← Supabase client
│   └── track.ts                ← Client-side analytics helper
├── update_movies.py            ← Script to add new movies from Excel
├── requirements.txt            ← Streamlit dependencies
├── package.json                ← Next.js dependencies
└── .env.local.example          ← API key template
```

---

## Features

### Next.js Dashboard (Vercel)
- **Main page** — Browse and filter 800+ films by language, genre, director, rating
- **Stats page** (`/stats`) — Collection analytics: top directors, languages, genres, ratings distribution
- **Chat recommender** — AI-powered film recommendations via Claude Haiku
  - Fuzzy title matching (Levenshtein distance) for typos like "vazha" → "Vaazha"
  - Disambiguation chips when multiple films match a query
  - Prompt caching to reduce API cost (~4× cheaper per conversation)
  - Falls back to heuristic recommender on network errors
- **Analytics dashboard** (`/analytics`) — Password-protected view of site usage

### Analytics (Supabase)
Tracks the following in a Supabase Postgres database:
- Page views (path + hashed IP)
- Chat opens and queries
- Unique visitors (by hashed IP — no raw IPs stored)
- Returning visitors (visited more than once)
- Daily page views (last 30 days chart)
- Top pages by view count

---

## How Updates Work

Every time you watch a new film:

1. Add it to `Movies.xlsx` on your laptop
2. Run `Update_Movies.command` (double-click)
3. The script fetches TMDb data and pushes to `public/movies.csv` on GitHub
4. **Streamlit** auto-reloads within seconds
5. **Vercel** rebuilds automatically within ~30 seconds

One push → both apps updated. ✓

---

## Setup Instructions

### A. GitHub

1. Go to github.com → **New repository**
2. Name it `movies-dashboard`
3. Make it **Public**
4. Do NOT initialise with README (we'll push our own)

### B. Your Laptop

**One-time setup:**

```bash
cd ~/Documents
unzip movies-dashboard.zip
cd movies-dashboard

git init
git add .
git commit -m "Unified repo — Streamlit + Vercel"
git remote add origin https://github.com/JSam-M/movies-dashboard.git
git push -u origin main
```

**Update `update_movies.py` on your laptop:**
```python
GITHUB_REPO = "https://github.com/JSam-M/movies-dashboard.git"
GITHUB_PAT  = "your_github_pat_here"
EXCEL_FILE  = "/Users/jsam/.../Movies.xlsx"
```

---

### C. Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Repository: `JSam-M/movies-dashboard`, Branch: `main`, Main file: `streamlit/app.py`

---

### D. Vercel

1. Go to [vercel.com](https://vercel.com) → **New Project**
2. Import `JSam-M/movies-dashboard`
3. Framework preset: **Next.js** (auto-detected)
4. Root directory: `.`
5. Add all environment variables (see table below)
6. Click **Deploy**

---

### E. Supabase

1. Create a project at [supabase.com](https://supabase.com)
2. Run the following SQL in the Supabase SQL editor:

```sql
create table page_views (
  id bigserial primary key,
  path text not null,
  ip_hash text not null,
  created_at timestamptz default now()
);

create table chat_events (
  id bigserial primary key,
  event_type text not null,
  ip_hash text not null,
  created_at timestamptz default now()
);
```

3. Copy your **Project URL** and **anon public key** from Project Settings → API
4. Add them as Vercel environment variables (see table below)

---

### F. UptimeRobot (keep Streamlit awake)

1. Go to [uptimerobot.com](https://uptimerobot.com) → free account
2. Add monitor: HTTP(s), URL: `https://jsam-movies-dashboard.streamlit.app/`, interval: 5 min

---

## Environment Variables

| Variable | Where | Value |
|---|---|---|
| `ANTHROPIC_API_KEY` | Vercel | From [console.anthropic.com](https://console.anthropic.com) |
| `SUPABASE_URL` | Vercel | Your Supabase project URL |
| `SUPABASE_ANON_KEY` | Vercel | Your Supabase anon public key |
| `ANALYTICS_PASSWORD` | Vercel | Password to access `/analytics` |

---

## Sharing

- **Streamlit URL**: `https://jsam-movies-dashboard.streamlit.app`
- **Vercel URL**: `https://movies-dashboard-jsam.vercel.app`
- **Analytics**: `https://movies-dashboard-jsam.vercel.app/analytics` (password-protected)
