import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Films",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded"
)

NTH = "N'th time of watching"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Inter:wght@300;400;500;600&display=swap');

/* ── Reset ── */
#MainMenu, footer, .stDeployButton, header[data-testid="stHeader"] { display:none !important; }
.block-container { padding: 4rem 5rem 5rem 5rem !important; max-width: 1380px !important; }
* { box-sizing: border-box; }

/* ── Root canvas — Apple white ── */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #f5f5f7 !important;
    color: #1d1d1f !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Subtle gradient mesh behind everything ── */
[data-testid="stMain"]::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 70vw 50vh at 10% 0%,   rgba(0,125,250,.06)  0%, transparent 60%),
        radial-gradient(ellipse 60vw 40vh at 90% 100%, rgba(100,210,255,.07) 0%, transparent 60%),
        radial-gradient(ellipse 40vw 40vh at 55% 50%,  rgba(255,149,0,.04)  0%, transparent 60%);
}

/* ── Glass cards ── */
.glass {
    background: rgba(255,255,255,.72);
    backdrop-filter: blur(20px) saturate(1.8);
    -webkit-backdrop-filter: blur(20px) saturate(1.8);
    border: 1px solid rgba(255,255,255,.9);
    box-shadow:
        0 2px 12px rgba(0,0,0,.06),
        0 1px 3px rgba(0,0,0,.04),
        inset 0 1px 0 rgba(255,255,255,1);
    border-radius: 18px;
}
.glass-card {
    background: rgba(255,255,255,.85);
    backdrop-filter: blur(24px) saturate(1.6);
    -webkit-backdrop-filter: blur(24px) saturate(1.6);
    border: 1px solid rgba(255,255,255,.95);
    box-shadow:
        0 4px 24px rgba(0,0,0,.07),
        0 1px 4px rgba(0,0,0,.05),
        inset 0 1px 0 rgba(255,255,255,1);
    border-radius: 20px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,.8) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(0,0,0,.08) !important;
}
[data-testid="stSidebar"] * { color: #1d1d1f !important; }
[data-testid="stSidebar"] label {
    font-size: 0.72rem !important; font-weight: 500 !important;
    color: #6e6e73 !important; letter-spacing: .02em !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background: rgba(0,0,0,.04) !important;
    border: 1px solid rgba(0,0,0,.1) !important;
    border-radius: 10px !important; color: #1d1d1f !important;
}
[data-testid="collapsedControl"] {
    background: #0071e3 !important;
    border: none !important;
    border-radius: 0 20px 20px 0 !important;
    box-shadow: 0 4px 12px rgba(0,113,227,.35) !important;
    width: 32px !important;
    color: white !important;
    top: 50% !important;
}
[data-testid="collapsedControl"]:hover {
    background: #0077ed !important;
    box-shadow: 0 6px 16px rgba(0,113,227,.45) !important;
}
[data-testid="collapsedControl"] svg { fill: white !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(0,0,0,.08) !important;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.65rem !important; font-weight: 600 !important;
    letter-spacing: .1em !important; text-transform: uppercase !important;
    color: rgba(0,0,0,.3) !important;
    border: none !important;
    border-bottom: 1.5px solid transparent !important;
    padding: 14px 28px !important; margin-bottom: -1px !important;
    background: transparent !important;
    transition: color .2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: rgba(0,0,0,.6) !important; }
.stTabs [aria-selected="true"] {
    color: #1d1d1f !important;
    border-bottom-color: #1d1d1f !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid rgba(0,0,0,.06) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,.05) !important;
}

/* ── Radio ── */
.stRadio label {
    font-size: 0.75rem !important; color: #6e6e73 !important;
    letter-spacing: .02em !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,0,0,.12); border-radius: 2px; }

/* ── Typography ── */
.t-display {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: clamp(3rem, 5vw, 5rem);
    font-weight: 300; line-height: .95; letter-spacing: -.03em;
    color: #1d1d1f;
}
.t-display em {
    font-style: italic;
    background: linear-gradient(135deg, #0071e3 0%, #34aadc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.t-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.6rem; font-weight: 600; letter-spacing: .16em;
    text-transform: uppercase; color: #6e6e73;
}
.t-body {
    font-family: 'Inter', sans-serif;
    font-size: .88rem; line-height: 1.7; color: #6e6e73;
}
.t-section {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.7rem; font-weight: 300; letter-spacing: -.02em;
    color: #1d1d1f; line-height: 1.1;
}
.chart-label {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.3rem; font-weight: 300; color: #1d1d1f;
    letter-spacing: -.01em; margin-bottom: 4px;
}
.chart-sub {
    font-family: 'Inter', sans-serif;
    font-size: .75rem; color: #6e6e73; margin-bottom: 16px;
}

/* ── KPI cards ── */
.kpi-glass {
    padding: 24px 22px; position: relative; overflow: hidden;
    height: 120px; display: flex; flex-direction: column; justify-content: flex-end;
}
.kpi-glass::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,1), transparent);
}
.kpi-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.8rem; font-weight: 300; color: #1d1d1f;
    line-height: 1; letter-spacing: -.03em;
}
.kpi-value sup {
    font-size: .9rem; font-weight: 400; color: #6e6e73;
    vertical-align: super;
}
.kpi-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.6rem; font-weight: 600; letter-spacing: .14em;
    text-transform: uppercase; color: #6e6e73; margin-top: 8px;
}

/* ── Dividers ── */
.div-line { border: none; border-top: 1px solid rgba(0,0,0,.07); margin: 0; }

/* ── Highest rated callout ── */
.top-card {
    background: linear-gradient(135deg, rgba(255,255,255,.95), rgba(255,255,255,.75));
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255,255,255,1);
    box-shadow: 0 8px 32px rgba(0,125,250,.1), 0 2px 8px rgba(0,0,0,.06);
    border-radius: 22px; padding: 28px; margin-top: 24px;
}

/* ── Footer ── */
.footer-txt {
    font-family: 'Inter', sans-serif; font-size: .65rem;
    color: rgba(0,0,0,.2); letter-spacing: .1em;
    text-transform: uppercase; text-align: center; padding: 28px 0 8px;
}
</style>
""", unsafe_allow_html=True)


# ── DATA ──────────────────────────────────────────────────────
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

CHART_COLORS = ["#0071e3","#34aadc","#30b0c7","#32ade6","#5ac8fa","#007aff","#64d2ff","#0a84ff"]
QUAL_COLORS  = ["#0071e3","#ff9500","#34c759","#ff3b30","#5856d6","#ff2d55","#af52de","#00c7be"]

def apple_fig(fig, height=360, dark_bg=False):
    bg   = "rgba(0,0,0,0)"
    grid = "rgba(0,0,0,.05)"
    tick = "#86868b"
    fig.update_layout(
        font=dict(family="sans-serif", size=10),
        paper_bgcolor=bg, plot_bgcolor=bg,
        height=height,
        margin=dict(l=8, r=28, t=24, b=8),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,.95)",
            font_size=12, font_family="sans-serif",
            font_color="#1d1d1f",
            bordercolor="rgba(0,0,0,.1)"
        ),
        xaxis=dict(
            showgrid=False, zeroline=False, showline=False,
            tickfont=dict(size=10, color=tick), ticks="",
        ),
        yaxis=dict(
            showgrid=True, zeroline=False, showline=False,
            gridcolor=grid,
            tickfont=dict(size=10, color=tick), ticks="",
        ),
    )
    return fig


# ── FILTERS (sidebar) ─────────────────────────────────────────
df_full, _ = load_data()

st.sidebar.markdown('<p style="font-family:Inter,sans-serif;font-size:0.6rem;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:#6e6e73;padding:8px 0 16px;">Refine</p>', unsafe_allow_html=True)

search = st.sidebar.text_input("Search", placeholder="Film title…")
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

min_r = st.sidebar.slider("Min TMDb Rating", 0.0, 10.0, 0.0, 0.5)
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
rw = st.sidebar.radio("View", ["All", "Rewatched", "First watch"], label_visibility="collapsed")
if rw == "Rewatched":     df = df[df[NTH] >= 2]
elif rw == "First watch": df = df[df[NTH] <= 1]

st.sidebar.markdown(f'<p style="font-family:Inter,sans-serif;font-size:.7rem;color:#86868b;margin-top:8px;">{len(df)} of {original_count} films</p>', unsafe_allow_html=True)


# ── STATS ─────────────────────────────────────────────────────
fe = df_original[df_original['Name'].isin(df['Name'].tolist())].copy()
if wy_range is not None:
    fe = fe[(fe['Watch_Year'] >= wy_range[0]) & (fe['Watch_Year'] <= wy_range[1])]
fe['Runtime_mins'] = fe['Runtime'].apply(parse_runtime)

total_hours = fe['Runtime_mins'].sum() / 60
avg_r       = df['TMDb_Rating'].mean() if df['TMDb_Rating'].notna().any() else 0
n_rec       = int((df[NTH] >= 2).sum())
n_langs     = df['Language'].nunique()


# ── HERO ──────────────────────────────────────────────────────
h_left, h_right = st.columns([3, 1], gap="large")



with h_left:
    st.markdown(f"""
    <div style="padding: 16px 0 44px;">
        <p class="t-eyebrow" style="margin-bottom:18px;">Personal Archive &nbsp;·&nbsp; Since 2019</p>
        <h1 class="t-display">A life in<br><em>cinema</em></h1>
        <p class="t-body" style="margin-top:18px; max-width:360px;">
            {original_count} films across {n_langs} languages —
            {total_hours:.0f} hours of storytelling logged since the first watch.
        </p>
    </div>
    """, unsafe_allow_html=True)

with h_right:
    if df['TMDb_Rating'].notna().any():
        top_film  = df.nlargest(1, 'TMDb_Rating').iloc[0]
        genre_str = str(top_film.get('Genre','')).split(',')[0].strip() if pd.notna(top_film.get('Genre')) else ''
        yr_str    = str(int(top_film['Release_Year'])) if pd.notna(top_film.get('Release_Year')) else ''
        st.markdown(f"""
        <div class="top-card">
            <p class="t-eyebrow" style="margin-bottom:12px; color:#6e6e73;">Highest Rated</p>
            <p style="font-family:'Cormorant Garamond',serif; font-size:1.45rem; font-weight:300;
                      color:#1d1d1f; letter-spacing:-.01em; line-height:1.2; margin:0 0 6px;">
                {top_film['Name']}
            </p>
            <p style="font-family:'Inter',sans-serif; font-size:.7rem; color:#6e6e73; margin:0 0 18px;">
                {yr_str}&ensp;·&ensp;{genre_str}
            </p>
            <p style="font-family:'Cormorant Garamond',serif; font-size:3rem; font-weight:300;
                      color:#0071e3; line-height:1; margin:0;">
                {top_film['TMDb_Rating']:.1f}
                <span style="font-size:1rem; color:#86868b;">/ 10</span>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ── KPI ROW ───────────────────────────────────────────────────
st.markdown('<hr class="div-line" style="margin:8px 0 28px;">', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5, gap="medium")
accent_dots = ["#0071e3","#ff9500","#34c759","#ff3b30","#5856d6"]
for col, val, unit, lbl, dot in zip(
    [k1,k2,k3,k4,k5],
    [str(len(df)), f"{avg_r:.1f}", f"{total_hours:.0f}", str(n_rec), str(n_langs)],
    ["","","h","",""],
    ["Films","Avg Rating","Hours Watched","Rewatched","Languages"],
    accent_dots
):
    with col:
        st.markdown(f"""
        <div class="glass kpi-glass">
            <div style="position:absolute;top:16px;right:16px;width:8px;height:8px;
                border-radius:50%;background:{dot};opacity:.7;"></div>
            <div class="kpi-value">{val}<sup>{unit}</sup></div>
            <div class="kpi-label">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div style="height:36px;"></div>', unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["CATALOGUE", "RANKINGS", "COMPOSITION", "TRENDS"])


# ── TAB 1 ─────────────────────────────────────────────────────
with tab1:
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
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


# ── TAB 2 ─────────────────────────────────────────────────────
with tab2:
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
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
                colorscale=[[0,'#a2c4f5'],[1,'#0071e3']],
                showscale=False, line=dict(width=0),
            ),
            text=top['TMDb_Rating'].apply(lambda v: f"{v:.1f}"),
            textposition='outside',
            textfont=dict(size=11, color='#86868b'),
            hovertemplate='<b>%{y}</b><br>%{x:.1f}<extra></extra>',
        ))
        fig.update_layout(
            yaxis=dict(categoryorder='total ascending', title=''),
            xaxis=dict(title='', range=[0, top['TMDb_Rating'].max()*1.13]))
        apple_fig(fig, 420)
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
                    colorscale=[[0,'#ffd599'],[1,'#ff9500']],
                    showscale=False, line=dict(width=0),
                ),
                text=rwd[NTH].astype(int).astype(str)+'×',
                textposition='outside',
                textfont=dict(size=11, color='#86868b'),
                hovertemplate='<b>%{y}</b><br>%{x}×<extra></extra>',
            ))
            fig.update_layout(
                yaxis=dict(categoryorder='total ascending', title=''),
                xaxis=dict(title='', range=[0, rwd[NTH].max()*1.25]))
            apple_fig(fig, 420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rewatched films in selection.")


# ── TAB 3 ─────────────────────────────────────────────────────
with tab3:
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
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
            marker=dict(colors=QUAL_COLORS, line=dict(color='#f5f5f7', width=2)),
            textinfo='label+percent',
            textfont=dict(size=10, color='#1d1d1f'),
            hovertemplate='<b>%{label}</b><br>%{value} films<extra></extra>',
        ))
        fig.update_layout(
            font=dict(family="sans-serif", size=10),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=380, showlegend=False,
            margin=dict(l=0,r=0,t=24,b=0),
            hoverlabel=dict(bgcolor='rgba(255,255,255,.95)',font_size=12,font_color='#1d1d1f')
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
                colorscale=[[0,'#a2c4f5'],[1,'#0071e3']],
                showscale=False, line=dict(width=0),
            ),
            text=gdf['Count'], textposition='outside',
            textfont=dict(size=10, color='#86868b'),
            hovertemplate='<b>%{y}</b><br>%{x} films<extra></extra>',
        ))
        fig.update_layout(
            yaxis=dict(categoryorder='total ascending',title=''),
            xaxis=dict(title='',range=[0,gdf['Count'].max()*1.18]))
        apple_fig(fig, 380)
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
                colorscale=[[0,'#b3e8e5'],[1,'#00c7be']],
                showscale=False, line=dict(width=0),
            ),
            text=dc['Films'], textposition='outside',
            textfont=dict(size=10, color='#86868b'),
            hovertemplate='<b>%{y}</b><br>%{x} films<extra></extra>',
        ))
        fig.update_layout(
            yaxis=dict(categoryorder='total ascending',title=''),
            xaxis=dict(title='',range=[0,dc['Films'].max()*1.2]))
        apple_fig(fig, 500)
        st.plotly_chart(fig, use_container_width=True)
    with ct:
        st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)
        st.dataframe(dc, hide_index=True, use_container_width=True, height=500)


# ── TAB 4 ─────────────────────────────────────────────────────
with tab4:
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
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

    # Favourite Genre (from filtered unique films)
    gc = {}
    for gs in df['Genre'].dropna():
        for g in str(gs).split(','): g=g.strip(); gc[g]=gc.get(g,0)+1
    fav_genre = max(gc, key=gc.get).strip() if gc else "—"

    # Binge Streak — longest consecutive days with at least one watch
    binge = 0
    binge_period = ""
    if not tdf.empty:
        watch_dates = sorted(set(tdf['Date_Parsed'].dt.date.dropna()))
        if watch_dates:
            streak = best = 1
            cur_start = best_start = watch_dates[0]
            for i in range(1, len(watch_dates)):
                if (watch_dates[i] - watch_dates[i-1]).days == 1:
                    streak += 1
                    if streak > best:
                        best = streak
                        best_start = cur_start
                else:
                    streak = 1
                    cur_start = watch_dates[i]
            binge = best
            best_end = best_start + __import__('datetime').timedelta(days=best - 1)
            if best_start.month == best_end.month:
                binge_period = best_start.strftime("%-d") + "–" + best_end.strftime("%-d %b %Y")
            else:
                binge_period = best_start.strftime("%-d %b") + "–" + best_end.strftime("%-d %b %Y")

    st.markdown('<hr class="div-line" style="margin:16px 0 28px;">', unsafe_allow_html=True)
    s1,s2,s3,s4,s5 = st.columns(5, gap="medium")
    for col,val,unit,lbl,dot in [
        (s1, str(tot_m),       "",   "Total Watches",   "#0071e3"),
        (s2, fav_genre,        "",   "Favourite Genre", "#ff9500"),
        (s3, str(binge),       "d",  f"Binge Streak<br><span style='font-size:.5rem;letter-spacing:.1em;font-weight:400;opacity:.7;'>{binge_period}</span>",    "#34c759"),
        (s4, f"{days:.1f}",    "d",  "Days in Cinema",  "#5856d6"),
        (s5, f"{avg_rt:.0f}", "m",  "Avg Runtime",     "#ff3b30"),
    ]:
        with col:
            is_text = lbl in ("Favourite Genre",)
            font_size   = "1.2rem" if is_text else "2.4rem"
            font_family = "'Inter', sans-serif" if is_text else "'Cormorant Garamond', serif"
            st.markdown(f"""
            <div class="glass kpi-glass">
                <div style="position:absolute;top:16px;right:16px;width:8px;height:8px;
                    border-radius:50%;background:{dot};opacity:.7;"></div>
                <div class="kpi-value" style="font-size:{font_size};font-family:{font_family};line-height:1.25;">{val}<sup style="font-size:.75rem;">{unit}</sup></div>
                <div class="kpi-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="div-line" style="margin:28px 0;">', unsafe_allow_html=True)
    cl, cr = st.columns(2, gap="large")

    with cl:
        st.markdown('<p class="t-eyebrow" style="margin-bottom:8px;">Volume</p>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=gdf['Period'], y=gdf['Movies'],
            marker=dict(
                color=gdf['Movies'],
                colorscale=[[0,'#a2c4f5'],[1,'#0071e3']],
                showscale=False, line=dict(width=0),
            ),
            text=gdf['Movies'] if show_lbl else None,
            textposition='outside', textfont=dict(size=10, color='#86868b'),
            hovertemplate='<b>%{x}</b><br>%{y} films<extra></extra>',
        ))
        fig.update_layout(xaxis=dict(tickangle=x_ang,type='category'),yaxis=dict(title=''))
        if show_lbl: fig.update_layout(yaxis=dict(range=[0,gdf['Movies'].max()*1.2]))
        apple_fig(fig, 420)
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown('<p class="t-eyebrow" style="margin-bottom:8px;">Duration</p>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=gdf['Period'], y=gdf['Hours'],
            marker=dict(
                color=gdf['Hours'],
                colorscale=[[0,'#ffd599'],[1,'#ff9500']],
                showscale=False, line=dict(width=0),
            ),
            text=gdf['Hours'].apply(lambda v:f"{v:.0f}h") if show_lbl else None,
            textposition='outside', textfont=dict(size=10, color='#86868b'),
            hovertemplate='<b>%{x}</b><br>%{y:.0f}h<extra></extra>',
        ))
        fig.update_layout(xaxis=dict(tickangle=x_ang,type='category'),yaxis=dict(title=''))
        if show_lbl: fig.update_layout(yaxis=dict(range=[0,gdf['Hours'].max()*1.2]))
        apple_fig(fig, 420)
        st.plotly_chart(fig, use_container_width=True)


# ── FOOTER ────────────────────────────────────────────────────
st.markdown('<hr class="div-line" style="margin-top:40px;">', unsafe_allow_html=True)
st.markdown(
    f'<p class="footer-txt">{original_count} films &nbsp;·&nbsp; '
    f'v4.0 &nbsp;·&nbsp; {datetime.now().strftime("%B %Y")}</p>',
    unsafe_allow_html=True)
