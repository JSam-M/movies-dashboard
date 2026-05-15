# movies-dashboard

A personal film collection dashboard with AI-powered recommendations, collection stats, and analytics. Built with Next.js (Vercel) and Streamlit (Streamlit Cloud), both reading from the same `movies.csv`.

**Live URLs:**
- Next.js dashboard: `https://movies-dashboard-jsam.vercel.app`
- Streamlit dashboard: `https://jsam-movies-dashboard.streamlit.app`
- Analytics: `https://movies-dashboard-jsam.vercel.app/analytics` (password-protected)

---

## Table of Contents

1. [What This Is](#what-this-is)
2. [Repo Structure](#repo-structure)
3. [Data Model](#data-model)
4. [Pages & Features](#pages--features)
5. [Architecture](#architecture)
6. [AI Chat Recommender](#ai-chat-recommender)
7. [Analytics System](#analytics-system)
8. [Design System](#design-system)
9. [How to Add New Films](#how-to-add-new-films)
10. [Setup from Scratch](#setup-from-scratch)
11. [Environment Variables](#environment-variables)
12. [Database Schema](#database-schema)
13. [Deployment](#deployment)

---

## What This Is

A personal film archive built by JSam. Tracks every film watched since 2019 — currently 800+ films. The collection is stored in a single Excel file (`Movies.xlsx`) and synced to `public/movies.csv` via a Python update script. Both the Next.js and Streamlit apps read from that CSV.

The Next.js app is the primary interface — Apple-inspired glass UI, dark/light mode, AI chat recommender, and analytics. The Streamlit app is a secondary data exploration tool.

---

## Repo Structure

```
movies-dashboard/
│
├── public/
│   └── movies.csv              ← Single source of truth. All apps read from here.
│
├── app/                        ← Next.js App Router
│   ├── layout.tsx              ← Root layout (fonts, dark-mode inline script, metadata)
│   ├── globals.css             ← Design tokens (CSS vars), Tailwind, animations
│   ├── page.tsx                ← Main page: hero, AI search, daily picks, browse/filter
│   ├── stats/
│   │   └── page.tsx            ← Stats page: charts, filters, full catalogue table
│   ├── analytics/
│   │   └── page.tsx            ← Analytics dashboard (password-gated, sessionStorage auth)
│   └── api/
│       ├── movies/route.ts     ← GET /api/movies — returns parsed movies + stats
│       ├── chat/route.ts       ← POST /api/chat — Claude Haiku recommender
│       ├── track/route.ts      ← POST /api/track — records page_view/chat events to Supabase
│       └── analytics/route.ts ← GET /api/analytics — returns aggregated analytics (password-protected)
│
├── components/
│   ├── ChatPanel.tsx           ← Floating chat UI, message history, disambiguation chips
│   ├── StatsContent.tsx        ← All charts for the stats page (Recharts)
│   ├── FilterBar.tsx           ← Filter sidebar for stats page
│   ├── CatalogueTab.tsx        ← Full film table in stats page
│   ├── CompositionTab.tsx      ← Composition charts tab
│   ├── RankingsTab.tsx         ← Rankings tab
│   ├── TrendsTab.tsx           ← Trends over time tab
│   ├── KPIBar.tsx              ← KPI row (total films, hours, avg rating, rewatched)
│   ├── MultiSelect.tsx         ← Reusable searchable multi-select dropdown
│   ├── AboutModal.tsx          ← "About" modal accessible from nav
│   ├── FeedbackWidget.tsx      ← Feedback widget
│   ├── ScrollJump.tsx          ← Floating scroll-to-top button
│   └── ThemeToggle.tsx         ← Dark/light mode toggle (persists to localStorage)
│
├── lib/
│   ├── movies.ts               ← CSV parsing, Movie type, getMovies(), getUniqueMovies(), getStats()
│   ├── heuristic.ts            ← Client-side fallback recommender (no API needed)
│   ├── supabase.ts             ← Supabase client (server-side only)
│   └── track.ts                ← Client-side analytics fire-and-forget helper
│
├── streamlit/
│   └── app.py                  ← Streamlit dashboard (reads same movies.csv)
│
├── update_movies.py            ← Python script: Excel → TMDb enrichment → CSV → GitHub push
├── requirements.txt            ← Python deps for update script + Streamlit
├── package.json                ← Node deps: next, anthropic, @supabase/supabase-js, recharts, papaparse
├── tailwind.config.ts
├── tsconfig.json
└── .env.local                  ← Local secrets (never committed)
```

---

## Data Model

### movies.csv columns

| Column | Description |
|---|---|
| `No.` | Row number |
| `Date` | Watch date (`DD/MM/YY`) |
| `Name` | Film title |
| `Language` | Language (Tamil, Hindi, English, Malayalam, Korean, etc.) |
| `Year` | Year first watched (not release year) |
| `Good?` | Personal rating tag |
| `N'th time of watching` | How many times watched (1, 2, 3…) |
| `Location` | Where watched (Theatre, Home, etc.) |
| `Director` | Director name(s), comma-separated |
| `Runtime` | Runtime in minutes |
| `Genre` | TMDb genres, comma-separated |
| `TMDb_Rating` | TMDb/IMDb rating (0–10) |
| `Release_Year` | Year the film was released |
| `Overview` | TMDb plot summary |
| `API_Status` | Whether TMDb data was fetched |

### Movie TypeScript interface (`lib/movies.ts`)

```ts
interface Movie {
  no: number
  date: string          // "DD/MM/YY"
  name: string
  language: string
  year: number
  good: string
  timesWatched: number  // derived from "N'th time of watching"
  location: string
  director: string      // may be comma-separated
  runtime: string       // e.g. "142"
  genre: string         // e.g. "Drama,Thriller"
  tmdbRating: number
  releaseYear: number
  overview: string
  apiStatus: string
  runtimeMins: number   // parsed integer version of runtime
}
```

### Key data logic

- **`getMovies()`** — parses the full CSV (includes duplicate rows for rewatches)
- **`getUniqueMovies()`** — deduplicates by title; if a film appears multiple times, the highest `timesWatched` value is kept. This is the list shown in the UI and used for AI recommendations.
- **`getStats()`** — computes totals, avg rating, genre/language/director counts, watch-year histogram

---

## Pages & Features

### `/` — Main Dashboard (`app/page.tsx`)

- **Hero** with animated gradient heading and AI search pill
- **Quick prompt chips** — generated from the actual collection on every page load (a top director, a language, a highly-rated film, a genre, and two mood prompts). Refreshes on every visit.
- **Daily Picks** — 6 films selected deterministically each day using a seeded shuffle (seed = YYYYMMDD, hashed per full film name). Always pulls from films rated ≥7.0. Picks rotate every midnight. Clicks open a detail modal.
- **Movie detail modal** — shows genre tag, rating, year/language/runtime, overview, director, genre tags
- **Browse/filter** — search by title or director, filter by genre (multi-select) and language (multi-select), sort by rating/rewatched/date, filter to rewatched favourites only
- Shows first 60 results; "View full collection →" links to stats page
- Floating chat button (bottom-right) opens ChatPanel
- Tracks `page_view` event on mount

### `/stats` — Stats Page (`app/stats/page.tsx` + `components/StatsContent.tsx`)

- Full filter sidebar: by language, genre, director, minimum rating, watch year, rewatch status, individual film search
- **KPI bar**: total films, total watch time (hours/days), average rating, rewatched count
- **Charts** (all via Recharts):
  - Films by language (bar chart)
  - Films by genre (bar chart)
  - Top directors (bar chart)
  - Films watched per year (line chart)
  - Rating distribution (bar chart)
  - Language composition (pie chart)
- **Full catalogue table** (`CatalogueTab`) — sortable, paginated list of all films
- Rankings, Trends, Composition sub-tabs
- Tracks `page_view` event on mount

### `/analytics` — Analytics Dashboard (`app/analytics/page.tsx`)

- Password-gated with `sessionStorage` (password saved for the session, not persisted across browser sessions)
- Fetches from `/api/analytics` using `x-analytics-password` header
- **KPI cards**: Total Page Views, Unique Visitors, Returning Visitors, Chat Opens, Total Queries, Queries per Session
- **Daily page views chart** (line, last 30 days, Recharts)
- **Top pages table**
- Password: stored in `ANALYTICS_PASSWORD` env var on Vercel

---

## Architecture

### API Routes

#### `GET /api/movies`
Returns `{ movies: Movie[], allEntries: Movie[], stats: Stats }`.
- `movies` = deduplicated list via `getUniqueMovies()`
- `allEntries` = full list with duplicate rows (used for watch-year stats)
- `stats` = computed totals (total, totalHours, avgRating, etc.)

#### `POST /api/chat`
Body: `{ messages: { role: 'user'|'assistant', content: string }[] }`

Flow:
1. Extract last user message as `query`
2. **Fuzzy match**: if query is ≤4 words, compute Levenshtein edit distance between the first word of the query and the first word of each film title. If edit distance ≤1:
   - 2+ matches → return `{ disambiguate: [{ name, year, language }] }` (disambiguation chips)
   - 1 match → rewrite the user message to `I liked "Film Name" — recommend similar films from the catalogue.`
3. **Pill detection**: if message starts with `I liked "X"`, extract film name as `referencedFilm`
4. Build catalogue: top 400 films by TMDb rating. If the referenced film is outside top 400, append it anyway. If a director name is detected in the query, all their films outside top 400 are appended too.
5. Call **Claude Haiku** (`claude-haiku-4-5-20251001`) with:
   - `max_tokens: 500`
   - System prompt (cached with `cache_control: ephemeral`)
   - Last 4 messages from conversation history
6. Return `{ message: string }`

On overload error: return `{ error: 'overloaded' }` → client falls back to heuristic.

#### `POST /api/track`
Body: `{ event: 'page_view'|'chat_open'|'chat_query', path?: string }`

- Reads client IP from `x-forwarded-for` or `x-real-ip`
- SHA-256 hashes the IP (no raw IPs stored)
- Writes to Supabase: `page_views` (path, ip_hash) or `chat_events` (event_type, ip_hash)

#### `GET /api/analytics`
Header: `x-analytics-password: <password>`

- Verifies password against `ANALYTICS_PASSWORD` env var
- Queries both Supabase tables
- Returns: `{ totalViews, uniqueVisitors, returningVisitors, chatOpens, chatQueries, dailyViews, topPages }`

---

## AI Chat Recommender

### Model
**Claude Haiku** (`claude-haiku-4-5-20251001`) — chosen for speed and low cost.

### Prompt Caching
The system prompt (which contains the 400-film catalogue) is tagged with `cache_control: { type: 'ephemeral' }`. This caches the prompt for ~5 minutes, reducing input token cost by ~4× for repeat queries within that window.

### System Prompt Rules
1. Only recommend films that appear in the provided catalogue
2. If a referenced film isn't in the catalogue, infer taste from it and find similar films that ARE in the catalogue
3. Start with one short intro line ("Since you enjoyed X…" or "For a feel-good night…")
4. If user mentions a language, filter strictly to that language only
5. Format: `**Name** (Year, Language) — one thematic sentence about why it suits taste`
6. Give 3–5 recommendations; ★ marks personally rewatched films
7. End with one short closing line inviting refinement

### Fuzzy Matching (Levenshtein)
- Handles typos: "vazha" → "Vaazha", "parsite" → "Parasite"
- Compares first word of query vs first word of each film title
- Edit distance ≤1 triggers a match
- 2+ matches → disambiguation chips (tappable cards showing name, year, language)
- 1 match → auto-rewrites query before sending to Claude

### Disambiguation Chips
When multiple films match a fuzzy query, ChatPanel renders tappable pill cards. Clicking one sends: `I liked "Film Name" — recommend similar films from the catalogue.`

### Heuristic Fallback (`lib/heuristic.ts`)
Used when: API is overloaded, network error, or explicitly for fallback.
- Mood → genre mapping (feel-good → Comedy/Romance, etc.)
- Language detection (Tamil/kollywood, Korean/korea, etc.)
- Runtime filtering ("under 2 hours")
- Scoring: genre match + rating boost + rewatch boost
- Does NOT use Claude, runs instantly

### Token Cost Estimate
- ~800 input tokens (system prompt) + ~100 (messages) per query
- With caching: ~200 cached input tokens + ~700 prompt cache read tokens + ~150 output tokens
- Approximate cost per query: ~$0.0003–0.0005 (Claude Haiku pricing)

---

## Analytics System

### Storage: Supabase (Postgres)

Two tables (see [Database Schema](#database-schema)):
- `page_views` — every page visit with hashed IP and path
- `chat_events` — every `chat_open` or `chat_query` event

### Privacy
IPs are SHA-256 hashed before storage. No raw IPs are ever written to the database. Unique visitor counting is based on distinct `ip_hash` values.

### Client tracking (`lib/track.ts`)
```ts
track('page_view', '/') // called in page.tsx
track('page_view', '/stats') // called in stats page
track('chat_open')  // called when ChatPanel mounts
track('chat_query') // called before each send
```
All calls are fire-and-forget (`fetch(...).catch(() => {})`). Tracking failures never affect the user.

### Metrics computed in `/api/analytics`
| Metric | How |
|---|---|
| Total page views | `page_views` row count |
| Unique visitors | Distinct `ip_hash` values in `page_views` |
| Returning visitors | `ip_hash` values appearing more than once |
| Chat opens | `chat_events` rows where `event_type = 'chat_open'` |
| Total queries | `chat_events` rows where `event_type = 'chat_query'` |
| Queries per session | `chatQueries / chatOpens` |
| Daily views (last 30d) | Group `page_views` by `created_at` date |
| Top pages | Group `page_views` by `path`, sorted by count |

---

## Design System

### Fonts
- **Display font**: Cormorant Garamond (300, 400, 600 weights + italic) — used for headings, film titles, large numbers
- **Body font**: DM Sans (300, 400, 500, 600) — used for all UI text, labels, buttons

### CSS Design Tokens (CSS variables)
All colors and surfaces use CSS custom properties defined in `globals.css`. Light/dark mode flips the values.

| Token | Light | Dark |
|---|---|---|
| `--bg` | `#f5f5f7` | `#000000` |
| `--surface` | `#ffffff` | `#1c1c1e` |
| `--text` | `#1d1d1f` | `#f5f5f7` |
| `--sub` | `#6e6e73` | `#8e8e93` |
| `--muted` | `#86868b` | `#636366` |
| `--blue` | `#0071e3` | `#0a84ff` |
| `--glass` | `rgba(255,255,255,0.72)` | `rgba(28,28,30,0.82)` |
| `--modal-bg` | `rgba(255,255,255,0.97)` | `rgba(28,28,30,0.97)` |

### Dark Mode
Toggled by adding/removing `class="dark"` on `<html>`. Preference persisted to `localStorage`. An inline script in `layout.tsx` applies the class before paint to prevent flash.

### Glass UI
`.glass` class applies `backdrop-filter: blur(20px) saturate(1.8)` — used for cards, panels, modals.

### Rating Color Scale
Films are color-coded by TMDb rating in the browse list:
- ≥8.5 → green (`#34c759`)
- ≥7.5 → blue (`#0071e3`)
- ≥6.5 → orange (`#ff9500`)
- <6.5 → red (`#ff3b30`)

---

## How to Add New Films

1. Open `Movies.xlsx` on your Mac
2. Add new rows at the top (newest first)
3. Double-click `Update_Movies.command`
4. The script:
   - Reads the Excel file
   - Fetches missing TMDb metadata (rating, overview, genre, director, runtime, release year)
   - Writes updated `public/movies.csv`
   - Commits and pushes to GitHub
5. Vercel detects the push and redeploys in ~30 seconds
6. Streamlit auto-refreshes

---

## Setup from Scratch

### 1. Clone and install

```bash
git clone https://github.com/JSam-M/movies-dashboard.git
cd movies-dashboard
npm install
```

### 2. Create `.env.local`

```
ANTHROPIC_API_KEY=your_key_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here
ANALYTICS_PASSWORD=your_chosen_password
```

### 3. Create Supabase tables

Run this SQL in the Supabase SQL editor:

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

### 4. Run locally

```bash
npm run dev
```

App runs at `http://localhost:3000`.

### 5. Configure `update_movies.py`

Edit these three lines:
```python
GITHUB_REPO = "https://github.com/JSam-M/movies-dashboard.git"
GITHUB_PAT  = "your_github_pat_here"
EXCEL_FILE  = "/Users/jsam/path/to/Movies.xlsx"
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key from [console.anthropic.com](https://console.anthropic.com) |
| `SUPABASE_URL` | Yes | Supabase project URL (Project Settings → API) |
| `SUPABASE_ANON_KEY` | Yes | Supabase anon public key (Project Settings → API) |
| `ANALYTICS_PASSWORD` | Yes | Password to access `/analytics` dashboard |

All four must be set in Vercel (Settings → Environment Variables) for production.

---

## Database Schema

```sql
-- Tracks every page visit
create table page_views (
  id         bigserial primary key,
  path       text        not null,   -- e.g. "/", "/stats"
  ip_hash    text        not null,   -- SHA-256 of visitor IP
  created_at timestamptz default now()
);

-- Tracks chat interactions
create table chat_events (
  id         bigserial primary key,
  event_type text        not null,   -- "chat_open" or "chat_query"
  ip_hash    text        not null,   -- SHA-256 of visitor IP
  created_at timestamptz default now()
);
```

---

## Deployment

### Vercel (Next.js)
- Connects to GitHub repo `JSam-M/movies-dashboard`
- Auto-deploys on every push to `main`
- Framework: Next.js (auto-detected)
- Root directory: `.`
- All 4 environment variables must be set in Vercel dashboard

### Streamlit Cloud
- Repository: `JSam-M/movies-dashboard`
- Branch: `main`
- Main file: `streamlit/app.py`
- No environment variables needed

### UptimeRobot
- Pings `https://jsam-movies-dashboard.streamlit.app/` every 5 minutes to keep Streamlit from sleeping

### Key dependencies
| Package | Version | Purpose |
|---|---|---|
| `next` | 14.2.5 | React framework |
| `@anthropic-ai/sdk` | latest | Claude API client |
| `@supabase/supabase-js` | latest | Supabase database client |
| `recharts` | latest | Charts in stats and analytics pages |
| `papaparse` | latest | CSV parsing |
| `tailwindcss` | latest | Utility CSS |
