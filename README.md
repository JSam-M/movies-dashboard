# movies-dashboard

One repo, two apps — a Streamlit dashboard and a Next.js/Vercel dashboard with AI film recommendations, both reading from the same `movies.csv`.

---

## Repo Structure

```
movies-dashboard/
├── public/
│   └── movies.csv          ← THE SINGLE SOURCE OF TRUTH for both apps
├── streamlit/
│   └── app.py              ← Streamlit dashboard
├── app/                    ← Next.js app (Vercel)
│   ├── api/
│   │   ├── chat/route.ts   ← AI recommendation API
│   │   └── movies/route.ts ← Movie data API
│   ├── page.tsx            ← Main dashboard page
│   ├── layout.tsx
│   └── globals.css
├── components/             ← Next.js React components
├── lib/
│   └── movies.ts           ← Data loading logic
├── update_movies.py        ← Script to add new movies from Excel
├── requirements.txt        ← Streamlit dependencies
├── package.json            ← Next.js dependencies
└── .env.local.example      ← API key template
```

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
2. Name it `movies-dashboard` (replacing your old one, or create new)
3. Make it **Public**
4. Do NOT initialise with README (we'll push our own)

### B. Your Laptop

**One-time setup:**

```bash
# Unzip the project somewhere permanent, e.g. your Documents folder
cd ~/Documents
unzip movies-dashboard.zip

# Go into the folder
cd movies-dashboard

# Initialise git and push to GitHub
git init
git add .
git commit -m "Unified repo — Streamlit + Vercel"
git remote add origin https://github.com/JSam-M/movies-dashboard.git
git push -u origin main
```

**Update `update_movies.py` on your laptop** — open it and check these three lines:
```python
GITHUB_REPO = "https://github.com/JSam-M/movies-dashboard.git"   # ← your repo
GITHUB_PAT  = "your_github_pat_here"                              # ← your PAT (unchanged)
EXCEL_FILE  = "/Users/jsam/.../Movies.xlsx"                       # ← your Excel path (unchanged)
```
The CSV path is already fixed to `public/movies.csv` — no change needed.

**To run updates going forward:**
- Just double-click `Update_Movies.command` as before
- It will now push to `public/movies.csv` in the unified repo

---

### C. Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click your existing app → **Settings** → **Repository**
3. Change the **Main file path** from `app.py` to `streamlit/app.py`
4. Save — Streamlit will reboot and find the new path

> If you're creating a fresh Streamlit app:
> - Repository: `JSam-M/movies-dashboard`
> - Branch: `main`
> - Main file path: `streamlit/app.py`

---

### D. Vercel

1. Go to [vercel.com](https://vercel.com) → **New Project**
2. Import `JSam-M/movies-dashboard` from GitHub
3. Framework preset: **Next.js** (auto-detected)
4. Root directory: `.` (leave as default)
5. Under **Environment Variables**, add:
   - Name: `ANTHROPIC_API_KEY`
   - Value: your key from [console.anthropic.com](https://console.anthropic.com)
6. Click **Deploy**

That's it. Vercel will build and give you a URL like `movies-dashboard.vercel.app`.

---

### E. UptimeRobot (keep Streamlit awake)

1. Go to [uptimerobot.com](https://uptimerobot.com) → free account
2. **Add New Monitor**:
   - Type: HTTP(s)
   - Name: Films Dashboard
   - URL: `https://jsam-movies-dashboard.streamlit.app/`
   - Interval: 5 minutes
3. Save — Streamlit will never sleep

---

## Environment Variables

| Variable | Where | Value |
|---|---|---|
| `ANTHROPIC_API_KEY` | Vercel dashboard | From console.anthropic.com |

The Streamlit app does not need any API keys.

---

## Sharing

- **Streamlit URL**: `https://jsam-movies-dashboard.streamlit.app`
- **Vercel URL**: `https://movies-dashboard-jsam.vercel.app` (assigned after deploy)

Share either link — both always show your latest data.
