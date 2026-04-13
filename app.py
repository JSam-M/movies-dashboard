import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Film Collection",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── THEME STATE ───────────────────────────────────────────────
if "dark" not in st.session_state:
    st.session_state.dark = True

dark = st.session_state.dark

# ── DESIGN TOKENS (switch on theme) ───────────────────────────
if dark:
    BG      = "#0D1117"
    SURFACE = "#161B22"
    BORDER  = "#21262D"
    TEXT    = "#E6EDF3"
    SUB     = "#8B949E"
    MUTED   = "#484F58"
    BLUE    = "#58A6FF"
    AMBER   = "#E3B341"
    TOGGLE  = "☀"
else:
    BG      = "#FFFFFF"
    SURFACE = "#F8FAFC"
    BORDER  = "#E2E8F0"
    TEXT    = "#1A2332"
    SUB     = "#64748B"
    MUTED   = "#94A3B8"
    BLUE    = "#1B5EC7"
    AMBER   = "#E0951A"
    TOGGLE  = "🌙"

NTH  = "N'th time of watching"
QUAL = [BLUE, AMBER, "#3FB950" if dark else "#2CA89A",
        "#A371F7", "#F78166", "#56D364", "#79C0FF", "#FFA657"]

PLOTLY = dict(
    font=dict(family="'DM Sans', 'Helvetica Neue', sans-serif", color=SUB, size=11),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=24, t=32, b=0),
    hoverlabel=dict(
        bgcolor=SURFACE,
        font_size=12,
        font_family="'DM Sans', sans-serif",
        font_color=TEXT
    ),
)


# ── CSS ──────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

/* ── Reset ── */
#MainMenu, footer, .stDeployButton {{ visibility: hidden; display: none; }}
.block-container {{ padding: 2.5rem 3rem 2rem 3rem !important; max-width: 1300px; }}

/* ── App background ── */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {{ background-color: {BG} !important; }}
header[data-testid="stHeader"] {{
    background-color: {BG} !important;
    border-bottom: 1px solid {BORDER} !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {SURFACE} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
[data-testid="stSidebar"] label {{ color: {SUB} !important; font-size: 0.78rem !important; }}

/* ── Sidebar toggle: hide the ugly icon text, show a clean arrow ── */
[data-testid="collapsedControl"] {{
    background-color: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-left: none !important;
    color: {SUB} !important;
    width: 24px !important;
}}
[data-testid="collapsedControl"] span[data-testid="tooltipHoverTarget"] {{
    display: none !important;
}}
[data-testid="collapsedControl"]:hover {{ background-color: {BORDER} !important; }}

/* ── Typography ── */
p, li, span, div, label {{ font-family: 'DM Sans', sans-serif !important; }}
h1 {{
    font-family: 'Libre Baskerville', Georgia, serif !important;
    font-size: 2rem !important; font-weight: 700 !important;
    color: {TEXT} !important; letter-spacing: -0.03em !important;
    line-height: 1.15 !important; margin: 0 !important;
}}
h2 {{
    font-family: 'Libre Baskerville', serif !important;
    font-size: 1.15rem !important; font-weight: 400 !important;
    color: {TEXT} !important; margin: 0 !important;
}}
h3 {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.62rem !important; font-weight: 600 !important;
    color: {MUTED} !important; letter-spacing: 0.14em !important;
    text-transform: uppercase !important; margin: 0 !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0; border-bottom: 1px solid {BORDER} !important; background: transparent !important;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.68rem !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    color: {MUTED} !important; border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 14px 28px !important; margin-bottom: -1px !important;
    background: transparent !important;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: {TEXT} !important; }}
.stTabs [aria-selected="true"] {{
    color: {TEXT} !important; border-bottom-color: {BLUE} !important;
}}

/* ── Radio ── */
.stRadio label {{ font-size: 0.8rem !important; color: {SUB} !important; }}

/* ── Theme toggle button ── */
.theme-btn button {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
    border-radius: 6px !important;
    font-size: 1rem !important;
    padding: 4px 10px !important;
    min-height: unset !important;
    height: 36px !important;
}}
.theme-btn button:hover {{ background: {BORDER} !important; }}

/* ── Divider ── */
hr {{ border: none !important; border-top: 1px solid {BORDER} !important; margin: 2rem 0 !important; }}

/* ── Custom classes ── */
.eyebrow {{
    display: block; font-family: 'DM Sans', sans-serif;
    font-size: 0.62rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.14em; color: {MUTED}; margin-bottom: 6px;
}}
.chart-title {{
    display: block; font-family: 'Libre Baskerville', Georgia, serif;
    font-size: 1.1rem; font-weight: 400; color: {TEXT};
    letter-spacing: -0.02em; line-height: 1.3; margin: 0 0 18px 0;
}}
.chart-sub {{
    display: block; font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem; color: {SUB}; line-height: 1.55; margin: -10px 0 18px 0;
}}
.kpi-wrap {{ border-top: 2px solid {BLUE}; padding: 18px 0 12px 0; }}
.kpi-num {{
    font-family: 'Libre Baskerville', Georgia, serif;
    font-size: 2.5rem; font-weight: 700;
    color: {TEXT}; line-height: 1; letter-spacing: -0.03em;
}}
.kpi-unit {{ font-family: 'DM Sans', sans-serif; font-size: 0.95rem; color: {MUTED}; }}
.kpi-lbl {{
    font-family: 'DM Sans', sans-serif; font-size: 0.62rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em; color: {MUTED}; margin-top: 6px;
}}
.divider {{ border-top: 1px solid {BORDER}; margin: 28px 0; }}
.footer-line {{
    font-family: 'DM Sans', sans-serif; font-size: 0.7rem; color: {MUTED};
    text-align: center; padding: 28px 0 8px; letter-spacing: 0.06em;
}}
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

def style_fig(fig, height=380, y_grid=False):
    fig.update_layout(
        **PLOTLY, height=height,
        xaxis=dict(showgrid=False, zeroline=False, showline=False,
                   tickfont=dict(size=10, color=SUB)),
        yaxis=dict(showgrid=y_grid, zeroline=False, showline=False,
                   gridcolor=BORDER, tickfont=dict(size=10, color=SUB)),
    )
    return fig


# ── HEADER ────────────────────────────────────────────────────
hcol, tcol = st.columns([11, 1])
with hcol:
    st.markdown(f"""
    <div style="padding:0 0 32px 0; border-bottom:1px solid {BORDER}; margin-bottom:36px;">
        <span class="eyebrow">Personal Archive · {datetime.now().year}</span>
        <h1>Film Collection</h1>
        <p style="font-family:'DM Sans',sans-serif; font-size:0.88rem; color:{SUB};
                  margin:10px 0 0 0; max-width:480px; line-height:1.6;">
            A curated catalogue of every film watched, rated, and logged since 2019.
        </p>
    </div>
    """, unsafe_allow_html=True)
with tcol:
    st.markdown('<div class="theme-btn" style="padding-top:8px;">', unsafe_allow_html=True)
    if st.button(TOGGLE, help="Toggle light / dark mode"):
        st.session_state.dark = not st.session_state.dark
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────
st.sidebar.markdown(
    f'<p style="font-family:DM Sans,sans-serif; font-size:0.62rem; font-weight:600; '
    f'letter-spacing:0.14em; text-transform:uppercase; color:{MUTED}; '
    f'padding:8px 0 12px;">Filters</p>', unsafe_allow_html=True)
df_full, _ = load_data()

search = st.sidebar.text_input("Search", placeholder="Film title…")
if search:
    df = df[df['Name'].str.contains(search, case=False, na=False)]

languages = ['All'] + sorted(df_full['Language'].dropna().unique().tolist())
sel_lang  = st.sidebar.multiselect("Language", languages, default=['All'])
if 'All' not in sel_lang and sel_lang:
    df = df[df['Language'].isin(sel_lang)]

all_genres = set()
for g in df_full['Genre'].dropna():
    all_genres.update([x.strip() for x in str(g).split(',')])
all_genres = ['All'] + sorted(all_genres)
sel_genres = st.sidebar.multiselect("Genre", all_genres, default=['All'])
if 'All' not in sel_genres and sel_genres:
    df = df[df['Genre'].apply(
        lambda x: any(g in str(x) for g in sel_genres) if pd.notna(x) else False)]

directors = ['All'] + sorted(df_full['Director'].dropna().unique().tolist())
sel_dir   = st.sidebar.multiselect("Director", directors, default=['All'])
if 'All' not in sel_dir and sel_dir:
    df = df[df['Director'].isin(sel_dir)]

min_rating = st.sidebar.slider("Min TMDb Rating", 0.0, 10.0, 0.0, 0.5)
df = df[df['TMDb_Rating'] >= min_rating]

if df_full['Release_Year'].notna().any():
    yr_lo, yr_hi = int(df_full['Release_Year'].min()), int(df_full['Release_Year'].max())
    if yr_lo < yr_hi:
        yr_range = st.sidebar.slider("Release Year", yr_lo, yr_hi, (yr_lo, yr_hi))
        df = df[(df['Release_Year'] >= yr_range[0]) & (df['Release_Year'] <= yr_range[1])]

wy_range = None
if df_full['Watch_Year'].notna().any():
    wy_lo, wy_hi = int(df_full['Watch_Year'].min()), int(df_full['Watch_Year'].max())
    if wy_lo < wy_hi:
        wy_range = st.sidebar.slider("Watch Year", wy_lo, wy_hi, (wy_lo, wy_hi))
        df = df[(df['Watch_Year'] >= wy_range[0]) & (df['Watch_Year'] <= wy_range[1])]

st.sidebar.markdown("---")
rw_opt = st.sidebar.radio(
    "View", ["All Films", "Rewatched Only", "First Watch Only"],
    label_visibility="collapsed")
if rw_opt == "Rewatched Only":   df = df[df[NTH] >= 2]
elif rw_opt == "First Watch Only": df = df[df[NTH] <= 1]

st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<p style="font-family:DM Sans,sans-serif; font-size:0.76rem; color:{MUTED};">'
    f'Showing <strong style="color:{TEXT};">{len(df)}</strong> of {original_count} films</p>',
    unsafe_allow_html=True)


# ── KPI BAR ───────────────────────────────────────────────────
filtered_names   = df['Name'].tolist()
filtered_entries = df_original[df_original['Name'].isin(filtered_names)].copy()
if wy_range is not None:
    filtered_entries = filtered_entries[
        (filtered_entries['Watch_Year'] >= wy_range[0]) &
        (filtered_entries['Watch_Year'] <= wy_range[1])
    ]
filtered_entries['Runtime_mins'] = filtered_entries['Runtime'].apply(parse_runtime)
total_hours = filtered_entries['Runtime_mins'].sum() / 60
avg_r = df['TMDb_Rating'].mean() if df['TMDb_Rating'].notna().any() else 0
n_rec = (df[NTH] >= 2).sum()

k1, k2, k3, k4 = st.columns(4, gap="large")
for col, num, unit, lbl in [
    (k1, str(len(df)),          "",  "Films"),
    (k2, f"{avg_r:.1f}",        "",  "Avg TMDb Rating"),
    (k3, f"{total_hours:.0f}",  "h", "Hours Watched"),
    (k4, str(n_rec),            "",  "Rewatched"),
]:
    with col:
        st.markdown(
            f'<div class="kpi-wrap">'
            f'<div class="kpi-num">{num}<span class="kpi-unit">{unit}</span></div>'
            f'<div class="kpi-lbl">{lbl}</div></div>',
            unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["CATALOGUE", "RANKINGS", "COMPOSITION", "TRENDS"])


# ── TAB 1: CATALOGUE ─────────────────────────────────────────
with tab1:
    st.markdown("""
    <span class="eyebrow">Browse</span>
    <span class="chart-title">Complete Film Catalogue</span>
    <span class="chart-sub">All films sorted by TMDb rating. ★ denotes a personal rewatch.</span>
    """, unsafe_allow_html=True)

    disp = df[['Name','Release_Year','TMDb_Rating',NTH,
               'Genre','Director','Runtime','Language']].copy()
    disp.columns = ['Film','Year','Rating','Watches','Genre','Director','Runtime','Language']
    disp['★'] = disp['Watches'].apply(lambda x: '★' if x >= 2 else '')
    disp = disp[['Film','Year','Rating','Watches','★','Genre','Director','Runtime','Language']]
    disp = disp.sort_values('Rating', ascending=False)

    st.dataframe(disp, hide_index=True, use_container_width=True, height=600,
                 column_config={
                     "Rating":  st.column_config.NumberColumn(format="%.1f"),
                     "Watches": st.column_config.NumberColumn(format="%d"),
                     "Year":    st.column_config.NumberColumn(format="%d"),
                 })


# ── TAB 2: RANKINGS ──────────────────────────────────────────
with tab2:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <span class="eyebrow">By Rating</span>
        <span class="chart-title">Highest-Rated Films</span>
        """, unsafe_allow_html=True)
        top = df.nlargest(10, 'TMDb_Rating')[
            ['Name','Release_Year','TMDb_Rating']].reset_index(drop=True)
        top['Label'] = top['Name'] + '  (' + top['Release_Year'].astype(int).astype(str) + ')'
        fig = go.Figure(go.Bar(
            x=top['TMDb_Rating'], y=top['Label'], orientation='h',
            marker=dict(color=BLUE),
            text=top['TMDb_Rating'].apply(lambda v: f"{v:.1f}"),
            textposition='outside', textfont=dict(size=11, color=SUB),
            hovertemplate='<b>%{y}</b><br>%{x:.1f}<extra></extra>',
        ))
        fig.update_layout(
            yaxis=dict(categoryorder='total ascending', title=''),
            xaxis=dict(title='', range=[0, top['TMDb_Rating'].max() * 1.12]))
        style_fig(fig, 420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("""
        <span class="eyebrow">Personal Picks</span>
        <span class="chart-title">Most Rewatched Films</span>
        """, unsafe_allow_html=True)
        rw = df[df[NTH] > 1].nlargest(10, NTH)[
            ['Name','Release_Year',NTH]].reset_index(drop=True)
        if len(rw) > 0:
            rw['Label'] = rw['Name'] + '  (' + rw['Release_Year'].astype(int).astype(str) + ')'
            fig = go.Figure(go.Bar(
                x=rw[NTH], y=rw['Label'], orientation='h',
                marker=dict(color=AMBER),
                text=rw[NTH].astype(int).astype(str) + '×',
                textposition='outside', textfont=dict(size=11, color=SUB),
                hovertemplate='<b>%{y}</b><br>%{x}×<extra></extra>',
            ))
            fig.update_layout(
                yaxis=dict(categoryorder='total ascending', title=''),
                xaxis=dict(title='', range=[0, rw[NTH].max() * 1.22]))
            style_fig(fig, 420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rewatched films in current selection.")


# ── TAB 3: COMPOSITION ───────────────────────────────────────
with tab3:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <span class="eyebrow">Language</span>
        <span class="chart-title">Distribution by Language</span>
        """, unsafe_allow_html=True)
        ld = df['Language'].value_counts().reset_index()
        ld.columns = ['Language','Count']
        fig = go.Figure(go.Pie(
            labels=ld['Language'], values=ld['Count'], hole=0.62,
            marker=dict(colors=QUAL, line=dict(color=BG, width=2)),
            textinfo='label+percent',
            textfont=dict(size=11, family="DM Sans", color=TEXT),
            hovertemplate='<b>%{label}</b><br>%{value} films · %{percent}<extra></extra>',
        ))
        fig.update_layout(**PLOTLY, height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("""
        <span class="eyebrow">Genre</span>
        <span class="chart-title">Top Genres</span>
        """, unsafe_allow_html=True)
        gc = {}
        for gs in df['Genre'].dropna():
            for g in str(gs).split(','):
                g = g.strip(); gc[g] = gc.get(g, 0) + 1
        gdf = pd.DataFrame(list(gc.items()), columns=['Genre','Count']
                           ).sort_values('Count', ascending=False).head(10)
        fig = go.Figure(go.Bar(
            x=gdf['Count'], y=gdf['Genre'], orientation='h',
            marker=dict(color=BLUE),
            text=gdf['Count'], textposition='outside',
            textfont=dict(size=10, color=SUB),
            hovertemplate='<b>%{y}</b><br>%{x} films<extra></extra>',
        ))
        fig.update_layout(
            yaxis=dict(categoryorder='total ascending', title=''),
            xaxis=dict(title='', range=[0, gdf['Count'].max() * 1.18]))
        style_fig(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <span class="eyebrow">Filmmakers</span>
    <span class="chart-title">Top Directors in Collection</span>
    """, unsafe_allow_html=True)
    dc = df['Director'].value_counts().head(15).reset_index()
    dc.columns = ['Director','Films']
    cc, ct = st.columns([2.2, 1], gap="large")
    with cc:
        fig = go.Figure(go.Bar(
            x=dc['Films'], y=dc['Director'], orientation='h',
            marker=dict(color=BLUE),
            text=dc['Films'], textposition='outside',
            textfont=dict(size=10, color=SUB),
            hovertemplate='<b>%{y}</b><br>%{x} films<extra></extra>',
        ))
        fig.update_layout(
            yaxis=dict(categoryorder='total ascending', title=''),
            xaxis=dict(title='', range=[0, dc['Films'].max() * 1.2]))
        style_fig(fig, 480)
        st.plotly_chart(fig, use_container_width=True)
    with ct:
        st.dataframe(dc, hide_index=True, use_container_width=True, height=480)


# ── TAB 4: TRENDS ────────────────────────────────────────────
with tab4:
    st.markdown("""
    <span class="eyebrow">Analytics</span>
    <span class="chart-title">Viewing Patterns Over Time</span>
    <span class="chart-sub">Films watched and hours logged across the full collection history.</span>
    """, unsafe_allow_html=True)

    tdf = filtered_entries.copy()
    tdf['Watch_Year'] = tdf['Date_Parsed'].dt.year
    tdf['Month_Name'] = tdf['Date_Parsed'].dt.month_name()
    tdf['Year-Month'] = tdf['Date_Parsed'].dt.to_period('M').astype(str)
    tdf = tdf.dropna(subset=['Watch_Year'])
    tdf['Watch_Year'] = tdf['Watch_Year'].astype(int)

    view_by = st.radio("Aggregate by", ["Year", "Month", "All Time"], horizontal=True)

    if view_by == "Year":
        gdf = tdf.groupby('Watch_Year').agg(
            Movies=('Name','count'), Minutes=('Runtime_mins','sum')).reset_index()
        gdf.columns = ['Period','Movies','Minutes']
        gdf['Period'] = gdf['Period'].astype(str)
        gdf['Hours'] = gdf['Minutes'] / 60
        x_ang, show_lbl = 0, True
    elif view_by == "Month":
        mo = ['January','February','March','April','May','June',
              'July','August','September','October','November','December']
        gdf = tdf.dropna(subset=['Month_Name']).groupby('Month_Name').agg(
            Movies=('Name','count'), Minutes=('Runtime_mins','sum')).reset_index()
        gdf.columns = ['Period','Movies','Minutes']
        gdf['Hours'] = gdf['Minutes'] / 60
        gdf['Period'] = pd.Categorical(gdf['Period'], categories=mo, ordered=True)
        gdf = gdf.sort_values('Period')
        x_ang, show_lbl = 45, True
    else:
        gdf = tdf.groupby('Year-Month').agg(
            Movies=('Name','count'), Minutes=('Runtime_mins','sum')).reset_index()
        gdf.columns = ['Period','Movies','Minutes']
        gdf['Hours'] = gdf['Minutes'] / 60
        x_ang, show_lbl = 45, False

    tot_m  = len(tdf)
    tot_h  = tdf['Runtime_mins'].sum() / 60
    days   = tot_h / 24
    avg_rt = tdf['Runtime_mins'].mean() if tot_m else 0

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4, gap="large")
    for col, num, unit, lbl in [
        (m1, str(tot_m),        "",  "Total Watches"),
        (m2, f"{tot_h:.0f}",    "h", "Hours Spent"),
        (m3, f"{days:.1f}",     "d", "Days Spent"),
        (m4, f"{avg_rt:.0f}",   "m", "Avg Runtime"),
    ]:
        with col:
            st.markdown(
                f'<div class="kpi-wrap">'
                f'<div class="kpi-num">{num}<span class="kpi-unit">{unit}</span></div>'
                f'<div class="kpi-lbl">{lbl}</div></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    cl, cr = st.columns(2, gap="large")
    with cl:
        st.markdown('<span class="eyebrow">Volume</span>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=gdf['Period'], y=gdf['Movies'], marker=dict(color=BLUE),
            text=gdf['Movies'] if show_lbl else None,
            textposition='outside', textfont=dict(size=10, color=SUB),
            hovertemplate='<b>%{x}</b><br>%{y} films<extra></extra>',
        ))
        fig.update_layout(xaxis=dict(tickangle=x_ang, type='category'), yaxis=dict(title=''))
        if show_lbl: fig.update_layout(yaxis=dict(range=[0, gdf['Movies'].max() * 1.2]))
        style_fig(fig, 400, y_grid=True)
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown('<span class="eyebrow">Duration</span>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=gdf['Period'], y=gdf['Hours'],
            marker=dict(color=AMBER),
            text=gdf['Hours'].apply(lambda v: f"{v:.0f}h") if show_lbl else None,
            textposition='outside', textfont=dict(size=10, color=SUB),
            hovertemplate='<b>%{x}</b><br>%{y:.1f}h<extra></extra>',
        ))
        fig.update_layout(xaxis=dict(tickangle=x_ang, type='category'), yaxis=dict(title=''))
        if show_lbl: fig.update_layout(yaxis=dict(range=[0, gdf['Hours'].max() * 1.2]))
        style_fig(fig, 400, y_grid=True)
        st.plotly_chart(fig, use_container_width=True)


# ── FOOTER ────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    f'<p class="footer-line">{original_count} films catalogued'
    f'&ensp;·&ensp;Dashboard v2.2'
    f'&ensp;·&ensp;Updated {datetime.now().strftime("%B %Y")}</p>',
    unsafe_allow_html=True)
