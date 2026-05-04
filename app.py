import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Films",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

NTH = "N'th time of watching"

# ─────────────────────────────────────────────────────────────
# LIQUID GLASS CSS  — Apple visionOS / iOS 26 aesthetic
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Geist:wght@200;300;400;500;600&display=swap');

/* ── Reset ── */
#MainMenu, footer, .stDeployButton, header[data-testid="stHeader"] { display:none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; overflow-x: hidden; }
* { box-sizing: border-box; }

/* ── Root canvas ── */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #08090a !important;
    color: #f0ede8 !important;
    font-family: 'Geist', system-ui, sans-serif !important;
}

/* ── Ambient orbs (pure CSS, no images needed) ── */
[data-testid="stMain"]::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 80vw 60vh at 15% 10%, rgba(99,102,241,.13) 0%, transparent 70%),
        radial-gradient(ellipse 60vw 50vh at 85% 80%, rgba(168,85,247,.10) 0%, transparent 70%),
        radial-gradient(ellipse 50vw 40vh at 60% 40%, rgba(20,184,166,.07) 0%, transparent 70%);
}

/* ── Glass mixin (reused via class) ── */
.glass {
    background: rgba(255,255,255,.055);
    backdrop-filter: blur(28px) saturate(1.6);
    -webkit-backdrop-filter: blur(28px) saturate(1.6);
    border: 1px solid rgba(255,255,255,.10);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.12),
        0 8px 32px rgba(0,0,0,.4),
        0 1px 2px rgba(0,0,0,.3);
}
.glass-subtle {
    background: rgba(255,255,255,.03);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,.07);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 4px 16px rgba(0,0,0,.25);
}
.glass-strong {
    background: rgba(255,255,255,.09);
    backdrop-filter: blur(40px) saturate(1.8);
    -webkit-backdrop-filter: blur(40px) saturate(1.8);
    border: 1px solid rgba(255,255,255,.15);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.18),
        0 16px 48px rgba(0,0,0,.5);
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: rgba(8,9,10,.85) !important;
    backdrop-filter: blur(40px) !important;
    border-right: 1px solid rgba(255,255,255,.08) !important;
}
[data-testid="stSidebar"] * { color: rgba(240,237,232,.8) !important; }
[data-testid="stSidebar"] label { font-size: 0.72rem !important; font-weight: 400 !important; letter-spacing: .04em !important; }
[data-testid="stSidebar"] .stTextInput input {
    background: rgba(255,255,255,.06) !important;
    border: 1px solid rgba(255,255,255,.10) !important;
    border-radius: 10px !important; color: #f0ede8 !important;
}
[data-testid="collapsedControl"] {
    background: rgba(255,255,255,.07) !important;
    border: 1px solid rgba(255,255,255,.10) !important;
    border-left: none !important;
    backdrop-filter: blur(20px) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,.08) !important;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Geist', sans-serif !important;
    font-size: 0.65rem !important; font-weight: 500 !important;
    letter-spacing: .12em !important; text-transform: uppercase !important;
    color: rgba(240,237,232,.35) !important;
    border: none !important;
    border-bottom: 1.5px solid transparent !important;
    padding: 14px 28px !important; margin-bottom: -1px !important;
    background: transparent !important;
    transition: color .2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: rgba(240,237,232,.7) !important; }
.stTabs [aria-selected="true"] {
    color: #f0ede8 !important;
    border-bottom-color: rgba(255,255,255,.6) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,.08) !important;
}

/* ── RADIO ── */
.stRadio label { font-size: 0.75rem !important; color: rgba(240,237,232,.5) !important; letter-spacing: .04em !important; }
.stRadio [aria-checked="true"] + div label { color: #f0ede8 !important; }

/* ── SLIDER ── */
[data-testid="stSlider"] [data-testid="stTickBar"] { background: rgba(255,255,255,.1) !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,.15); border-radius: 2px; }

/* ── Typography classes ── */
.t-display {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: clamp(3rem, 6vw, 5.5rem);
    font-weight: 300; line-height: .95; letter-spacing: -.03em;
    color: #f0ede8;
}
.t-display em {
    font-style: italic; font-weight: 300;
    background: linear-gradient(135deg, #e2d9f3 0%, #c4b5fd 40%, #a78bfa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.t-eyebrow {
    font-family: 'Geist', sans-serif;
    font-size: 0.6rem; font-weight: 500; letter-spacing: .2em;
    text-transform: uppercase; color: rgba(240,237,232,.35);
}
.t-body { font-family: 'Geist', sans-serif; font-size: .85rem; line-height: 1.65; color: rgba(240,237,232,.55); }
.t-section {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem; font-weight: 300; letter-spacing: -.02em; color: #f0ede8; line-height: 1.1;
}

/* ── KPI ── */
.kpi-glass {
    border-radius: 20px; padding: 28px 24px; position: relative; overflow: hidden;
}
.kpi-glass::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.25), transparent);
}
.kpi-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3rem; font-weight: 300; color: #f0ede8;
    line-height: 1; letter-spacing: -.03em;
}
.kpi-value sup { font-size: 1rem; font-weight: 400; color: rgba(240,237,232,.45); vertical-align: super; }
.kpi-label {
    font-family: 'Geist', sans-serif;
    font-size: 0.6rem; font-weight: 500; letter-spacing: .16em;
    text-transform: uppercase; color: rgba(240,237,232,.35);
    margin-top: 8px;
}

/* ── Chart label ── */
.chart-label {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.25rem; font-weight: 300; color: #f0ede8;
    letter-spacing: -.01em; margin-bottom: 4px;
}
.chart-sub {
    font-family: 'Geist', sans-serif;
    font-size: .75rem; color: rgba(240,237,232,.35); margin-bottom: 16px;
}

/* ── Divider ── */
.div-line { border: none; border-top: 1px solid rgba(255,255,255,.07); margin: 0; }

/* ── Page wrapper ── */
.page { max-width: 1380px; margin: 0 auto; padding: 48px 56px 80px 56px; position: relative; z-index: 1; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv('movies.csv')
    df['TMDb_Rating']  = pd.to_numeric(df['TMDb_Rating'],  errors='coerce')
    df['Release_Year'] = pd.to_numeric(df['Release_Year'], errors='coerce')
    df[NTH] = pd.to_numeric(df[NTH], errors='coerce').fillna(1)
    df['Date_Parsed'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    mask = df['Date_Parsed'].isna() & df['Date'].notna()
    df.loc[mask, 'Date_Parsed'] = pd.to_datetime(df.loc[mask, 'Date'], errors='coerce')
    df['Watch_Year'] = df['Date_Parsed'].dt.year
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df_unique = df.groupby('Name', as_index=False).agg({
        'Date': 'first', 'Date_Parsed': 'first', 'Language': 'first',
        'Year': 'first', 'Watch_Year': 'first', 'Good?': 'first',
        NTH: 'max', 'Location': 'first', 'Director': 'first', 'Runtime': 'first',
        'Genre': 'first', 'TMDb_Rating': 'first', 'Release_Year': 'first',
        'Overview': 'first', 'API_Status': 'first'
    })
    return df_unique, df

df, df_original = load_data()
original_count  = len(df)

def parse_runtime(val):
    if pd.isna(val): return 0
    try: return int(str(val).split()[0])
    except: return 0

GLASS_COLORS = ["#a78bfa","#67e8f9","#fbbf24","#f472b6","#34d399","#fb923c","#60a5fa","#e879f9"]

def glass_fig(fig, height=360):
    fig.update_layout(
        font=dict(family="sans-serif", size=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=8, r=28, t=24, b=8),
        hoverlabel=dict(
            bgcolor="rgba(20,20,28,.95)",
            font_size=12, font_family="sans-serif",
            font_color="#f0ede8",
            bordercolor="rgba(255,255,255,.12)"
        ),
        xaxis=dict(
            showgrid=False, zeroline=False, showline=False,
            tickfont=dict(size=10, color="rgba(240,237,232,.3)"),
            tickcolor="transparent"
        ),
        yaxis=dict(
            showgrid=True, zeroline=False, showline=False,
            gridcolor="rgba(255,255,255,.05)",
            tickfont=dict(size=10, color="rgba(240,237,232,.3)")
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────
df_full, _ = load_data()

st.sidebar.markdown('<p class="t-eyebrow" style="padding:12px 0 16px;">Refine</p>', unsafe_allow_html=True)
search = st.sidebar.text_input("", placeholder="Search films…", label_visibility="collapsed")
if search: df = df[df['Name'].str.contains(search, case=False, na=False)]

langs = ['All'] + sorted(df_full['Language'].dropna().unique().tolist())
sel_lang = st.sidebar.multiselect("Language", langs, default=['All'])
if 'All' not in sel_lang and sel_lang: df = df[df['Language'].isin(sel_lang)]

all_genres = set()
for g in df_full['Genre'].dropna(): all_genres.update([x.strip() for x in str(g).split(',')])
sel_genre = st.sidebar.multiselect("Genre", ['All']+sorted(all_genres), default=['All'])
if 'All' not in sel_genre and sel_genre:
    df = df[df['Genre'].apply(lambda x: any(g in str(x) for g in sel_genre) if pd.notna(x) else False)]

sel_dir = st.sidebar.multiselect("Director", ['All']+sorted(df_full['Director'].dropna().unique().tolist()), default=['All'])
if 'All' not in sel_dir and sel_dir: df = df[df['Director'].isin(sel_dir)]

min_r = st.sidebar.slider("Min Rating", 0.0, 10.0, 0.0, 0.5)
df = df[df['TMDb_Rating'] >= min_r]

if df_full['Release_Year'].notna().any():
    yl, yh = int(df_full['Release_Year'].min()), int(df_full['Release_Year'].max())
    if yl < yh:
        yr_range = st.sidebar.slider("Release Year", yl, yh, (yl, yh))
        df = df[(df['Release_Year'] >= yr_range[0]) & (df['Release_Year'] <= yr_range[1])]

wy_range = None
if df_full['Watch_Year'].notna().any():
    wl, wh = int(df_full['Watch_Year'].min()), int(df_full['Watch_Year'].max())
    if wl < wh:
        wy_range = st.sidebar.slider("Watch Year", wl, wh, (wl, wh))
        df = df[(df['Watch_Year'] >= wy_range[0]) & (df['Watch_Year'] <= wy_range[1])]

st.sidebar.markdown("---")
rw = st.sidebar.radio("", ["All", "Rewatched", "First watch"], label_visibility="collapsed")
if rw == "Rewatched":    df = df[df[NTH] >= 2]
elif rw == "First watch": df = df[df[NTH] <= 1]

st.sidebar.markdown(f'<p style="font-size:.7rem;color:rgba(240,237,232,.25);margin-top:12px;">{len(df)} of {original_count} films</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# COMPUTED STATS
# ─────────────────────────────────────────────────────────────
fe = df_original[df_original['Name'].isin(df['Name'].tolist())].copy()
if wy_range is not None:
    fe = fe[(fe['Watch_Year'] >= wy_range[0]) & (fe['Watch_Year'] <= wy_range[1])]
fe['Runtime_mins'] = fe['Runtime'].apply(parse_runtime)

total_hours = fe['Runtime_mins'].sum() / 60
avg_r       = df['TMDb_Rating'].mean() if df['TMDb_Rating'].notna().any() else 0
n_rec       = int((df[NTH] >= 2).sum())
n_langs     = df['Language'].nunique()


# ─────────────────────────────────────────────────────────────
# PAGE OPEN
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="page">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────
h_left, h_right = st.columns([3, 1], gap="large")
with h_left:
    st.markdown(f"""
    <div style="padding: 24px 0 48px;">
        <p class="t-eyebrow" style="margin-bottom:20px;">Personal Archive &nbsp;·&nbsp; Since 2019</p>
        <h1 class="t-display">A life in<br><em>cinema</em></h1>
        <p class="t-body" style="margin-top:20px; max-width:380px;">
            {original_count} films across {n_langs} languages,
            {total_hours:.0f} hours of storytelling —
            every frame logged since the first watch.
        </p>
    </div>
    """, unsafe_allow_html=True)

with h_right:
    # Top-rated film callout
    if df['TMDb_Rating'].notna().any():
        top_film = df.nlargest(1,'TMDb_Rating').iloc[0]
        genre_str = str(top_film.get('Genre','')).split(',')[0].strip() if pd.notna(top_film.get('Genre')) else ''
        st.markdown(f"""
        <div class="glass-strong" style="border-radius:24px; padding:28px; margin-top:28px;">
            <p class="t-eyebrow" style="margin-bottom:12px;">Highest rated</p>
            <p style="font-family:'Cormorant Garamond',serif; font-size:1.4rem; font-weight:300;
                      color:#f0ede8; letter-spacing:-.01em; line-height:1.2; margin:0 0 6px;">
                {top_film['Name']}
            </p>
            <p style="font-family:'Geist',sans-serif; font-size:.7rem; color:rgba(240,237,232,.35); margin:0 0 16px;">
                {int(top_film['Release_Year']) if pd.notna(top_film['Release_Year']) else ''}&ensp;·&ensp;{genre_str}
            </p>
            <p style="font-family:'Cormorant Garamond',serif; font-size:2.8rem; font-weight:300;
                      color:#c4b5fd; line-height:1; margin:0;">
                {top_film['TMDb_Rating']:.1f}
                <span style="font-size:1rem; color:rgba(196,181,253,.5);">/ 10</span>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────
st.markdown('<hr class="div-line" style="margin:8px 0 32px;">', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5, gap="medium")
for col, val, unit, lbl, accent in [
    (k1, str(len(df)),          "",  "Films",         "#a78bfa"),
    (k2, f"{avg_r:.1f}",        "",  "Avg Rating",    "#67e8f9"),
    (k3, f"{total_hours:.0f}",  "h", "Hours Watched", "#fbbf24"),
    (k4, str(n_rec),            "",  "Rewatched",     "#f472b6"),
    (k5, str(n_langs),          "",  "Languages",     "#34d399"),
]:
    with col:
        st.markdown(f"""
        <div class="glass kpi-glass">
            <div style="position:absolute;top:0;right:0;width:60px;height:60px;
                border-radius:0 20px 0 60px;
                background:linear-gradient(135deg,{accent}18,transparent);"></div>
            <div class="kpi-value">{val}<sup>{unit}</sup></div>
            <div class="kpi-label">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div style="height:40px;"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["CATALOGUE", "RANKINGS", "COMPOSITION", "TRENDS"])


# ── TAB 1: CATALOGUE ─────────────────────────────────────────
with tab1:
    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <p class="t-eyebrow" style="margin-bottom:6px;">Browse</p>
    <p class="t-section" style="margin-bottom:4px;">Complete Collection</p>
    <p class="chart-sub">Sorted by TMDb rating &nbsp;·&nbsp; ★ marks a personal rewatch</p>
    """, unsafe_allow_html=True)

    disp = df[['Name','Release_Year','TMDb_Rating',NTH,'Genre','Director','Runtime','Language']].copy()
    disp.columns = ['Film','Year','Rating','Watches','Genre','Director','Runtime','Language']
    disp['★'] = disp['Watches'].apply(lambda x: '★' if x >= 2 else '')
    disp = disp[['Film','Year','Rating','★','Genre','Director','Runtime','Language']].sort_values('Rating', ascending=False)

    st.dataframe(disp, hide_index=True, use_container_width=True, height=620,
                 column_config={
                     "Rating": st.column_config.NumberColumn(format="%.1f"),
                     "Year":   st.column_config.NumberColumn(format="%d"),
                 })


# ── TAB 2: RANKINGS ──────────────────────────────────────────
with tab2:
    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <p class="t-eyebrow" style="margin-bottom:6px;">By Rating</p>
        <p class="chart-label">Highest Rated</p>
        """, unsafe_allow_html=True)
        top = df.nlargest(10,'TMDb_Rating')[['Name','Release_Year','TMDb_Rating']].reset_index(drop=True)
        top['Label'] = top.apply(
            lambda r: r['Name'] + f"  ({int(r['Release_Year'])})" if pd.notna(r['Release_Year']) else r['Name'], axis=1)
        fig = go.Figure(go.Bar(
            x=top['TMDb_Rating'], y=top['Label'], orientation='h',
            marker=dict(
                color=top['TMDb_Rating'],
                colorscale=[[0,'#4c1d95'],[0.5,'#7c3aed'],[1,'#a78bfa']],
                showscale=False,
                line=dict(width=0),
            ),
            text=top['TMDb_Rating'].apply(lambda v: f"{v:.1f}"),
            textposition='outside',
            textfont=dict(size=11, color='rgba(240,237,232,.5)'),
            hovertemplate='<b>%{y}</b><br>%{x:.1f}<extra></extra>',
        ))
        fig.update_layout(
            yaxis=dict(categoryorder='total ascending', title=''),
            xaxis=dict(title='', range=[0, top['TMDb_Rating'].max()*1.13]))
        glass_fig(fig, 420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("""
        <p class="t-eyebrow" style="margin-bottom:6px;">Personal Picks</p>
        <p class="chart-label">Most Rewatched</p>
        """, unsafe_allow_html=True)
        rwd = df[df[NTH]>1].nlargest(10,NTH)[['Name','Release_Year',NTH]].reset_index(drop=True)
        if len(rwd):
            rwd['Label'] = rwd.apply(
                lambda r: r['Name'] + f"  ({int(r['Release_Year'])})" if pd.notna(r['Release_Year']) else r['Name'], axis=1)
            fig = go.Figure(go.Bar(
                x=rwd[NTH], y=rwd['Label'], orientation='h',
                marker=dict(
                    color=rwd[NTH],
                    colorscale=[[0,'#92400e'],[0.5,'#d97706'],[1,'#fbbf24']],
                    showscale=False, line=dict(width=0),
                ),
                text=rwd[NTH].astype(int).astype(str)+'×',
                textposition='outside',
                textfont=dict(size=11, color='rgba(240,237,232,.5)'),
                hovertemplate='<b>%{y}</b><br>%{x}×<extra></extra>',
            ))
            fig.update_layout(
                yaxis=dict(categoryorder='total ascending', title=''),
                xaxis=dict(title='', range=[0, rwd[NTH].max()*1.25]))
            glass_fig(fig, 420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rewatched films in selection.")


# ── TAB 3: COMPOSITION ───────────────────────────────────────
with tab3:
    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <p class="t-eyebrow" style="margin-bottom:6px;">Language</p>
        <p class="chart-label">By Language</p>
        """, unsafe_allow_html=True)
        ld = df['Language'].value_counts().reset_index()
        ld.columns = ['Language','Count']
        fig = go.Figure(go.Pie(
            labels=ld['Language'], values=ld['Count'], hole=0.68,
            marker=dict(colors=GLASS_COLORS, line=dict(color='rgba(8,9,10,1)', width=2)),
            textinfo='label+percent',
            textfont=dict(size=10, color='rgba(240,237,232,.6)'),
            hovertemplate='<b>%{label}</b><br>%{value} films<extra></extra>',
        ))
        fig.update_layout(
            font=dict(family="sans-serif", size=10),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=380, showlegend=False,
            margin=dict(l=0,r=0,t=24,b=0),
            hoverlabel=dict(bgcolor='rgba(20,20,28,.95)',font_size=12,font_color='#f0ede8')
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("""
        <p class="t-eyebrow" style="margin-bottom:6px;">Genre</p>
        <p class="chart-label">Top Genres</p>
        """, unsafe_allow_html=True)
        gc = {}
        for gs in df['Genre'].dropna():
            for g in str(gs).split(','): g=g.strip(); gc[g]=gc.get(g,0)+1
        gdf = pd.DataFrame(list(gc.items()),columns=['Genre','Count']).sort_values('Count',ascending=False).head(10)
        fig = go.Figure(go.Bar(
            x=gdf['Count'], y=gdf['Genre'], orientation='h',
            marker=dict(
                color=list(range(len(gdf))),
                colorscale=[[0,'#1e1b4b'],[0.5,'#4c1d95'],[1,'#a78bfa']],
                showscale=False, line=dict(width=0),
            ),
            text=gdf['Count'], textposition='outside',
            textfont=dict(size=10, color='rgba(240,237,232,.4)'),
            hovertemplate='<b>%{y}</b><br>%{x} films<extra></extra>',
        ))
        fig.update_layout(
            yaxis=dict(categoryorder='total ascending',title=''),
            xaxis=dict(title='',range=[0,gdf['Count'].max()*1.18]))
        glass_fig(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="div-line" style="margin:8px 0 28px;">', unsafe_allow_html=True)

    st.markdown("""
    <p class="t-eyebrow" style="margin-bottom:6px;">Filmmakers</p>
    <p class="t-section" style="margin-bottom:20px;">Top Directors</p>
    """, unsafe_allow_html=True)
    dc = df['Director'].value_counts().head(15).reset_index()
    dc.columns = ['Director','Films']
    cc, ct = st.columns([2.5,1], gap="large")
    with cc:
        fig = go.Figure(go.Bar(
            x=dc['Films'], y=dc['Director'], orientation='h',
            marker=dict(
                color=list(range(len(dc))),
                colorscale=[[0,'#134e4a'],[0.5,'#0f766e'],[1,'#67e8f9']],
                showscale=False, line=dict(width=0),
            ),
            text=dc['Films'], textposition='outside',
            textfont=dict(size=10, color='rgba(240,237,232,.4)'),
            hovertemplate='<b>%{y}</b><br>%{x} films<extra></extra>',
        ))
        fig.update_layout(
            yaxis=dict(categoryorder='total ascending',title=''),
            xaxis=dict(title='',range=[0,dc['Films'].max()*1.2]))
        glass_fig(fig, 500)
        st.plotly_chart(fig, use_container_width=True)
    with ct:
        st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)
        st.dataframe(dc, hide_index=True, use_container_width=True, height=500)


# ── TAB 4: TRENDS ────────────────────────────────────────────
with tab4:
    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <p class="t-eyebrow" style="margin-bottom:6px;">Analytics</p>
    <p class="t-section" style="margin-bottom:4px;">Viewing Over Time</p>
    <p class="chart-sub">Films watched and hours invested across your collection history.</p>
    """, unsafe_allow_html=True)

    tdf = fe.copy()
    tdf['Watch_Year'] = tdf['Date_Parsed'].dt.year
    tdf['Month_Name'] = tdf['Date_Parsed'].dt.month_name()
    tdf['Year-Month'] = tdf['Date_Parsed'].dt.to_period('M').astype(str)
    tdf = tdf.dropna(subset=['Watch_Year'])
    tdf['Watch_Year'] = tdf['Watch_Year'].astype(int)

    view_by = st.radio("", ["By Year","By Month","All Time"], horizontal=True, label_visibility="collapsed")

    if view_by == "By Year":
        gdf = tdf.groupby('Watch_Year').agg(Movies=('Name','count'),Minutes=('Runtime_mins','sum')).reset_index()
        gdf.columns=['Period','Movies','Minutes']; gdf['Period']=gdf['Period'].astype(str)
        gdf['Hours']=gdf['Minutes']/60; x_ang,show_lbl=0,True
    elif view_by == "By Month":
        mo=['January','February','March','April','May','June','July','August','September','October','November','December']
        gdf=tdf.dropna(subset=['Month_Name']).groupby('Month_Name').agg(Movies=('Name','count'),Minutes=('Runtime_mins','sum')).reset_index()
        gdf.columns=['Period','Movies','Minutes']; gdf['Hours']=gdf['Minutes']/60
        gdf['Period']=pd.Categorical(gdf['Period'],categories=mo,ordered=True); gdf=gdf.sort_values('Period')
        x_ang,show_lbl=45,True
    else:
        gdf=tdf.groupby('Year-Month').agg(Movies=('Name','count'),Minutes=('Runtime_mins','sum')).reset_index()
        gdf.columns=['Period','Movies','Minutes']; gdf['Hours']=gdf['Minutes']/60
        x_ang,show_lbl=45,False

    tot_m=len(tdf); tot_h=tdf['Runtime_mins'].sum()/60
    days=tot_h/24; avg_rt=tdf['Runtime_mins'].mean() if tot_m else 0

    # Stat strip
    st.markdown('<hr class="div-line" style="margin:16px 0 28px;">', unsafe_allow_html=True)
    s1,s2,s3,s4 = st.columns(4, gap="medium")
    for col,val,unit,lbl,clr in [
        (s1,str(tot_m),       "",  "Total Watches",  "#a78bfa"),
        (s2,f"{tot_h:.0f}",   "h", "Hours Spent",    "#67e8f9"),
        (s3,f"{days:.1f}",    "d", "Days in Cinema", "#fbbf24"),
        (s4,f"{avg_rt:.0f}",  "m", "Avg Runtime",    "#f472b6"),
    ]:
        with col:
            st.markdown(f"""
            <div class="glass kpi-glass">
                <div style="position:absolute;top:0;right:0;width:50px;height:50px;
                    border-radius:0 20px 0 50px;background:linear-gradient(135deg,{clr}20,transparent);"></div>
                <div class="kpi-value" style="font-size:2.4rem;">{val}<sup style="font-size:.85rem;">{unit}</sup></div>
                <div class="kpi-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="div-line" style="margin:28px 0;">', unsafe_allow_html=True)
    cl,cr = st.columns(2, gap="large")

    with cl:
        st.markdown('<p class="t-eyebrow" style="margin-bottom:8px;">Volume</p>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=gdf['Period'], y=gdf['Movies'],
            marker=dict(
                color=gdf['Movies'],
                colorscale=[[0,'#1e1b4b'],[0.5,'#4c1d95'],[1,'#a78bfa']],
                showscale=False, line=dict(width=0),
            ),
            text=gdf['Movies'] if show_lbl else None,
            textposition='outside', textfont=dict(size=10,color='rgba(240,237,232,.4)'),
            hovertemplate='<b>%{x}</b><br>%{y} films<extra></extra>',
        ))
        fig.update_layout(xaxis=dict(tickangle=x_ang,type='category'),yaxis=dict(title=''))
        if show_lbl: fig.update_layout(yaxis=dict(range=[0,gdf['Movies'].max()*1.2]))
        glass_fig(fig, 420)
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown('<p class="t-eyebrow" style="margin-bottom:8px;">Duration</p>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=gdf['Period'], y=gdf['Hours'],
            marker=dict(
                color=gdf['Hours'],
                colorscale=[[0,'#134e4a'],[0.5,'#0f766e'],[1,'#67e8f9']],
                showscale=False, line=dict(width=0),
            ),
            text=gdf['Hours'].apply(lambda v:f"{v:.0f}h") if show_lbl else None,
            textposition='outside', textfont=dict(size=10,color='rgba(240,237,232,.4)'),
            hovertemplate='<b>%{x}</b><br>%{y:.0f}h<extra></extra>',
        ))
        fig.update_layout(xaxis=dict(tickangle=x_ang,type='category'),yaxis=dict(title=''))
        if show_lbl: fig.update_layout(yaxis=dict(range=[0,gdf['Hours'].max()*1.2]))
        glass_fig(fig, 420)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="height:40px;"></div>
<hr class="div-line">
<p style="font-family:'Geist',sans-serif; font-size:.65rem; color:rgba(240,237,232,.2);
          letter-spacing:.12em; text-transform:uppercase; text-align:center; padding:28px 0 8px;">
    {original_count} films &nbsp;·&nbsp; Dashboard v3.0 &nbsp;·&nbsp; {datetime.now().strftime("%B %Y")}
</p>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close .page
