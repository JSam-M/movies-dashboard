import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Film Collection",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# THEME STATE
# ──────────────────────────────────────────────
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

dark = st.session_state.dark_mode

# ──────────────────────────────────────────────
# DESIGN TOKENS
# ──────────────────────────────────────────────
if dark:
    T = dict(
        bg="#0e1117", bg_card="#1a1d26", bg_sidebar="#12151c",
        border="#2a2d38", border_light="#22252e",
        text_primary="#e8eaed", text_heading="#f0f2f5",
        text_secondary="#9ca3af", text_muted="#6b7280",
        accent="#60a5fa", accent_deep="#3b82f6",
        navy="#60a5fa", teal="#2dd4bf", teal_light="#5eead4",
        grid="rgba(255,255,255,0.06)", hover_row="#1f2330",
        tab_active="#60a5fa", tab_border="#2a2d38",
        stat_color="#93c5fd", rec_color="#2dd4bf",
        chart_font="#9ca3af", chart_text="#d1d5db",
    )
    CHART_QUAL = ["#60a5fa","#2dd4bf","#fbbf24","#a78bfa","#fb7185","#34d399","#fb923c","#818cf8"]
else:
    T = dict(
        bg="#fafbfc", bg_card="#ffffff", bg_sidebar="#ffffff",
        border="#e5e7eb", border_light="#f0f0f0",
        text_primary="#2d2d3f", text_heading="#1a1a2e",
        text_secondary="#6b7280", text_muted="#9ca3af",
        accent="#2563eb", accent_deep="#1e3a5f",
        navy="#1e3a5f", teal="#0d9488", teal_light="#14b8a6",
        grid="rgba(229,231,235,0.6)", hover_row="#f8fafc",
        tab_active="#1e3a5f", tab_border="#e5e7eb",
        stat_color="#1e3a5f", rec_color="#0d9488",
        chart_font="#6b7280", chart_text="#2d2d3f",
    )
    CHART_QUAL = ["#1e3a5f","#0d9488","#d97706","#7c3aed","#e11d48","#059669","#ea580c","#4f46e5"]

PLOTLY_LAYOUT = dict(
    font=dict(family="'Source Sans Pro', 'Segoe UI', sans-serif", color=T["chart_font"]),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=20, t=30, b=20),
    hoverlabel=dict(
        bgcolor="#1a1a2e" if not dark else "#2a2d38",
        font_size=13,
        font_family="'Source Sans Pro', sans-serif",
        font_color="#ffffff",
    ),
)

# ──────────────────────────────────────────────
# CSS — driven by theme tokens
# ──────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Source+Sans+Pro:wght@300;400;600;700&display=swap');

    .stApp {{ background-color: {T['bg']}; }}
    #MainMenu, footer {{ visibility: hidden; }}
    .stDeployButton {{ display: none; }}
    .block-container {{ padding-top: 3.5rem; padding-bottom: 2rem; max-width: 1200px; }}

    h1 {{
        font-family: 'Playfair Display', Georgia, serif !important;
        font-weight: 600 !important; color: {T['text_heading']} !important;
        font-size: 2.1rem !important; letter-spacing: -0.02em !important;
        margin-bottom: 0 !important;
    }}
    h2, .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        font-family: 'Playfair Display', Georgia, serif !important;
        font-weight: 500 !important; color: {T['text_heading']} !important;
        font-size: 1.3rem !important; letter-spacing: -0.01em !important;
    }}
    h3 {{
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 600 !important; color: {T['text_primary']} !important;
        font-size: 1.05rem !important; text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }}
    p, li, span, div {{ font-family: 'Source Sans Pro', sans-serif; }}

    [data-testid="stMetric"] {{
        background: {T['bg_card']}; border: 1px solid {T['border']};
        border-radius: 8px; padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,{'0.15' if dark else '0.04'});
    }}
    [data-testid="stMetricLabel"] {{
        font-family: 'Source Sans Pro', sans-serif !important;
        font-size: 0.75rem !important; font-weight: 600 !important;
        text-transform: uppercase !important; letter-spacing: 0.1em !important;
        color: {T['text_secondary']} !important;
    }}
    [data-testid="stMetricValue"] {{
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 700 !important; color: {T['text_heading']} !important;
        font-size: 1.7rem !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 0; border-bottom: 2px solid {T['tab_border']}; background: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 600; font-size: 0.85rem;
        text-transform: uppercase; letter-spacing: 0.06em;
        color: {T['text_secondary']}; border: none;
        border-bottom: 2px solid transparent;
        padding: 12px 24px; margin-bottom: -2px; background: transparent;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {T['tab_active']}; border-bottom-color: {T['accent']};
    }}
    .stTabs [aria-selected="true"] {{
        color: {T['tab_active']} !important; border-bottom-color: {T['tab_active']} !important;
        background: transparent !important;
    }}

    [data-testid="stSidebar"] {{ background: {T['bg_sidebar']}; border-right: 1px solid {T['border']}; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {{
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 700 !important; font-size: 0.85rem !important;
        text-transform: uppercase !important; letter-spacing: 0.1em !important;
        color: {T['text_heading']} !important;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {T['border']}; border-radius: 8px; overflow: hidden;
    }}
    .stRadio > div {{ gap: 0.3rem; }}
    .stRadio [data-testid="stMarkdownContainer"] p {{ font-size: 0.85rem; }}
    hr {{ border: none; border-top: 1px solid {T['border']}; margin: 1.5rem 0; }}

    .card {{
        background: {T['bg_card']}; border: 1px solid {T['border']};
        border-radius: 8px; padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,{'0.15' if dark else '0.04'}); margin-bottom: 16px;
    }}
    .section-label {{
        font-family: 'Source Sans Pro', sans-serif; font-size: 0.7rem;
        font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.12em; color: {T['text_secondary']}; margin-bottom: 4px;
    }}
    .section-title {{
        font-family: 'Playfair Display', Georgia, serif; font-size: 1.4rem;
        font-weight: 600; color: {T['text_heading']}; margin-bottom: 4px;
        letter-spacing: -0.02em;
    }}
    .section-subtitle {{
        font-family: 'Source Sans Pro', sans-serif; font-size: 0.88rem;
        color: {T['text_secondary']}; margin-bottom: 20px; line-height: 1.5;
    }}
    .stat-highlight {{
        font-family: 'Source Sans Pro', sans-serif; font-size: 2.2rem;
        font-weight: 700; color: {T['stat_color']}; line-height: 1;
    }}
    .stat-label {{
        font-family: 'Source Sans Pro', sans-serif; font-size: 0.72rem;
        font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.1em; color: {T['text_muted']}; margin-top: 2px;
    }}
    .footer-text {{
        font-family: 'Source Sans Pro', sans-serif; font-size: 0.75rem;
        color: {T['text_muted']}; text-align: center;
        padding: 24px 0 8px 0; letter-spacing: 0.04em;
    }}
    .kpi-row {{ display: flex; gap: 32px; flex-wrap: wrap; }}
    .kpi-item {{ flex: 1; min-width: 100px; }}

    /* Toggle styling */
    .theme-label {{
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.1em; color: {T['text_secondary']};
    }}

    /* Streamlit widget overrides for dark mode */
    {f"""
    .stApp p, .stApp span, .stApp label, .stApp div {{
        color: {T['text_primary']};
    }}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {{
        color: {T['text_primary']} !important;
    }}
    .stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] {{
        color: {T['text_secondary']} !important;
    }}
    .stTextInput input {{
        background: {T['bg_card']} !important;
        color: {T['text_primary']} !important;
        border-color: {T['border']} !important;
    }}
    .stMultiSelect [data-baseweb="tag"] {{
        background-color: {T['accent_deep']} !important;
    }}
    .stRadio label span {{
        color: {T['text_primary']} !important;
    }}
    [data-testid="stMarkdownContainer"] p {{
        color: {T['text_primary']};
    }}
    [data-testid="stMarkdownContainer"] strong {{
        color: {T['text_heading']};
    }}
    """ if dark else ""}

</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv('movies.csv')
    df['TMDb_Rating'] = pd.to_numeric(df['TMDb_Rating'], errors='coerce')
    df['Release_Year'] = pd.to_numeric(df['Release_Year'], errors='coerce')
    df['N\'th time of watching'] = pd.to_numeric(df['N\'th time of watching'], errors='coerce').fillna(1)

    df['Date_Parsed'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    mask = df['Date_Parsed'].isna() & df['Date'].notna()
    df.loc[mask, 'Date_Parsed'] = pd.to_datetime(df.loc[mask, 'Date'], errors='coerce')

    df['Watch_Year'] = df['Date_Parsed'].dt.year
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    df_unique = df.groupby('Name', as_index=False).agg({
        'Date': 'last', 'Date_Parsed': 'last', 'Language': 'first',
        'Year': 'first', 'Watch_Year': 'last', 'Good?': 'first',
        'N\'th time of watching': 'max', 'Location': 'last',
        'Director': 'first', 'Runtime': 'first', 'Genre': 'first',
        'TMDb_Rating': 'first', 'Release_Year': 'first',
        'Overview': 'first', 'API_Status': 'first'
    })
    return df_unique, df

df, df_original = load_data()
original_count = len(df)


def style_fig(fig, height=420, show_grid_y=True):
    fig.update_layout(
        **PLOTLY_LAYOUT, height=height,
        xaxis=dict(showgrid=False, gridcolor=T["grid"],
                   zeroline=False, tickfont=dict(size=11, color=T["chart_font"])),
        yaxis=dict(showgrid=show_grid_y, gridcolor=T["grid"],
                   zeroline=False, tickfont=dict(size=11, color=T["chart_font"])),
    )
    return fig

def parse_runtime(val):
    if pd.isna(val): return 0
    try: return int(str(val).split()[0])
    except: return 0


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:8px;">
    <p class="section-label">PERSONAL COLLECTION</p>
    <p class="section-title" style="font-size:2.1rem;">Film Collection &amp; Recommendations</p>
    <p class="section-subtitle" style="margin-bottom:0;">A curated catalogue of films — rated, reviewed, and personally recommended.</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
# Theme toggle at top
st.sidebar.markdown('<p class="theme-label" style="margin-top:0;">APPEARANCE</p>', unsafe_allow_html=True)
theme_toggle = st.sidebar.toggle("Dark Mode", value=st.session_state.dark_mode, key="theme_toggle")
if theme_toggle != st.session_state.dark_mode:
    st.session_state.dark_mode = theme_toggle
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f'<p class="section-label" style="margin-top:0;">FILTERS</p>', unsafe_allow_html=True)
df_full, _ = load_data()

search = st.sidebar.text_input("Search", placeholder="Film title…")
if search:
    df = df[df['Name'].str.contains(search, case=False, na=False)]

languages = ['All'] + sorted(df_full['Language'].dropna().unique().tolist())
sel_lang = st.sidebar.multiselect("Language", languages, default=['All'])
if 'All' not in sel_lang and sel_lang:
    df = df[df['Language'].isin(sel_lang)]

all_genres = set()
for g in df_full['Genre'].dropna():
    all_genres.update([x.strip() for x in str(g).split(',')])
all_genres = ['All'] + sorted(all_genres)
sel_genres = st.sidebar.multiselect("Genre", all_genres, default=['All'])
if 'All' not in sel_genres and sel_genres:
    df = df[df['Genre'].apply(lambda x: any(g in str(x) for g in sel_genres) if pd.notna(x) else False)]

directors = ['All'] + sorted(df_full['Director'].dropna().unique().tolist())
sel_dir = st.sidebar.multiselect("Director", directors, default=['All'])
if 'All' not in sel_dir and sel_dir:
    df = df[df['Director'].isin(sel_dir)]

min_rating = st.sidebar.slider("Minimum TMDb Rating", 0.0, 10.0, 0.0, 0.5)
df = df[df['TMDb_Rating'] >= min_rating]

if df_full['Release_Year'].notna().any():
    yr_lo, yr_hi = int(df_full['Release_Year'].min()), int(df_full['Release_Year'].max())
    if yr_lo < yr_hi:
        yr_range = st.sidebar.slider("Release Year", yr_lo, yr_hi, (yr_lo, yr_hi))
        df = df[(df['Release_Year'] >= yr_range[0]) & (df['Release_Year'] <= yr_range[1])]

if df_full['Watch_Year'].notna().any():
    wy_lo, wy_hi = int(df_full['Watch_Year'].min()), int(df_full['Watch_Year'].max())
    if wy_lo < wy_hi:
        wy_range = st.sidebar.slider("Watch Year", wy_lo, wy_hi, (wy_lo, wy_hi))
        df = df[(df['Watch_Year'] >= wy_range[0]) & (df['Watch_Year'] <= wy_range[1])]

st.sidebar.markdown("---")
st.sidebar.markdown(f'<p class="section-label">RECOMMENDATION</p>', unsafe_allow_html=True)
rw_opt = st.sidebar.radio("Filter by", ["All Films", "Recommended (Rewatched)", "First Watch Only"],
                           label_visibility="collapsed")
if rw_opt == "Recommended (Rewatched)":
    df = df[df['N\'th time of watching'] >= 2]
elif rw_opt == "First Watch Only":
    df = df[df['N\'th time of watching'] <= 1]

st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<p style="font-family:Source Sans Pro;font-size:0.82rem;color:{T["text_secondary"]};">'
    f'Showing <strong style="color:{T["text_heading"]};">{len(df)}</strong> of {original_count} films</p>',
    unsafe_allow_html=True)


# ──────────────────────────────────────────────
# KPI BAR
# ──────────────────────────────────────────────
filtered_names = df['Name'].tolist()
filtered_entries = df_original[df_original['Name'].isin(filtered_names)]

avg_r = df['TMDb_Rating'].mean() if df['TMDb_Rating'].notna().any() else 0
max_r = df['TMDb_Rating'].max() if df['TMDb_Rating'].notna().any() else 0
n_genres = len(set(g.strip() for gs in df['Genre'].dropna() for g in str(gs).split(',')))
n_rec = (df['N\'th time of watching'] >= 2).sum()

st.markdown(f"""
<div class="card" style="padding:20px 28px;">
  <div class="kpi-row">
    <div class="kpi-item"><div class="stat-highlight">{len(df)}</div><div class="stat-label">Films</div></div>
    <div class="kpi-item"><div class="stat-highlight">{df['Language'].nunique()}</div><div class="stat-label">Languages</div></div>
    <div class="kpi-item"><div class="stat-highlight">{avg_r:.1f}</div><div class="stat-label">Avg Rating</div></div>
    <div class="kpi-item"><div class="stat-highlight">{max_r:.1f}</div><div class="stat-label">Top Rating</div></div>
    <div class="kpi-item"><div class="stat-highlight">{n_genres}</div><div class="stat-label">Genres</div></div>
    <div class="kpi-item"><div class="stat-highlight" style="color:{T['rec_color']};">{n_rec}</div><div class="stat-label">Recommended</div></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["BROWSE", "TOP PICKS", "BY CATEGORY", "VIEWING STATS"])

# ── TAB 1: BROWSE ─────────────────────────────
with tab1:
    st.markdown("""
    <p class="section-label">CATALOGUE</p>
    <p class="section-title">Browse All Films</p>
    <p class="section-subtitle">Complete collection sorted by rating. Films marked ★ are personally recommended rewatches.</p>
    """, unsafe_allow_html=True)

    disp = df[['Name','Release_Year','TMDb_Rating','N\'th time of watching','Genre','Director','Runtime','Language']].copy()
    disp.columns = ['Film','Year','Rating','Watches','Genre','Director','Runtime','Language']
    disp['Pick'] = disp['Watches'].apply(lambda x: '★' if x >= 2 else '')
    disp = disp[['Film','Year','Rating','Watches','Pick','Genre','Director','Runtime','Language']]
    disp = disp.sort_values('Rating', ascending=False)

    st.dataframe(disp, hide_index=True, use_container_width=True, height=600,
                 column_config={
                     "Rating": st.column_config.NumberColumn(format="%.1f"),
                     "Watches": st.column_config.NumberColumn(format="%d"),
                     "Year": st.column_config.NumberColumn(format="%d"),
                 })


# ── TAB 2: TOP PICKS ──────────────────────────
with tab2:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <p class="section-label">BY RATING</p>
        <p class="section-title">Highest Rated</p>
        <p class="section-subtitle">Top 10 films by TMDb community rating.</p>
        """, unsafe_allow_html=True)

        top = df.nlargest(10, 'TMDb_Rating')[['Name','Release_Year','TMDb_Rating','N\'th time of watching']].reset_index(drop=True)
        top['Label'] = top['Name'] + '  (' + top['Release_Year'].astype(int).astype(str) + ')'

        fig = go.Figure(go.Bar(
            x=top['TMDb_Rating'], y=top['Label'], orientation='h',
            marker=dict(color=top['TMDb_Rating'],
                        colorscale=[[0,T["accent"]],[1,T["accent_deep"]]], cornerradius=3),
            text=top['TMDb_Rating'].apply(lambda v: f"{v:.1f}"),
            textposition='outside',
            textfont=dict(size=12, color=T["chart_text"], family="Source Sans Pro"),
            hovertemplate='<b>%{y}</b><br>Rating: %{x:.1f}<extra></extra>',
        ))
        fig.update_layout(yaxis=dict(categoryorder='total ascending', title=''),
                          xaxis=dict(title='TMDb Rating', range=[0, top['TMDb_Rating'].max()*1.12]))
        style_fig(fig, 440, show_grid_y=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("""
        <p class="section-label">PERSONAL PICKS</p>
        <p class="section-title">Most Rewatched</p>
        <p class="section-subtitle">Films watched more than once — strongest personal endorsements.</p>
        """, unsafe_allow_html=True)

        rw = df[df['N\'th time of watching']>1].nlargest(10,'N\'th time of watching')[
            ['Name','Release_Year','N\'th time of watching','TMDb_Rating']].reset_index(drop=True)
        if len(rw) > 0:
            rw['Label'] = rw['Name'] + '  (' + rw['Release_Year'].astype(int).astype(str) + ')'
            fig = go.Figure(go.Bar(
                x=rw['N\'th time of watching'], y=rw['Label'], orientation='h',
                marker=dict(color=rw['N\'th time of watching'],
                            colorscale=[[0,T["teal_light"]],[1,T["teal"]]], cornerradius=3),
                text=rw['N\'th time of watching'].astype(int).astype(str)+'x',
                textposition='outside',
                textfont=dict(size=12, color=T["chart_text"], family="Source Sans Pro"),
                hovertemplate='<b>%{y}</b><br>Watches: %{x}<extra></extra>',
            ))
            fig.update_layout(yaxis=dict(categoryorder='total ascending', title=''),
                              xaxis=dict(title='Times Watched', range=[0, rw['N\'th time of watching'].max()*1.2]))
            style_fig(fig, 440, show_grid_y=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rewatched films in current selection.")


# ── TAB 3: BY CATEGORY ────────────────────────
with tab3:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <p class="section-label">LANGUAGE</p>
        <p class="section-title">Distribution by Language</p>
        """, unsafe_allow_html=True)
        ld = df['Language'].value_counts().reset_index()
        ld.columns = ['Language','Count']
        fig = go.Figure(go.Pie(
            labels=ld['Language'], values=ld['Count'], hole=0.52,
            marker=dict(colors=CHART_QUAL),
            textinfo='label+percent',
            textfont=dict(size=12, family="Source Sans Pro", color=T["chart_text"]),
            hovertemplate='<b>%{label}</b><br>Films: %{value}<br>Share: %{percent}<extra></extra>',
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("""
        <p class="section-label">GENRE</p>
        <p class="section-title">Top Genres</p>
        """, unsafe_allow_html=True)
        gc = {}
        for gs in df['Genre'].dropna():
            for g in str(gs).split(','):
                g = g.strip(); gc[g] = gc.get(g,0)+1
        gdf = pd.DataFrame(list(gc.items()), columns=['Genre','Count']).sort_values('Count', ascending=False).head(10)
        fig = go.Figure(go.Bar(
            x=gdf['Count'], y=gdf['Genre'], orientation='h',
            marker=dict(color=T["navy"], cornerradius=3),
            text=gdf['Count'], textposition='outside',
            textfont=dict(size=11, color=T["chart_text"], family="Source Sans Pro"),
            hovertemplate='<b>%{y}</b><br>Films: %{x}<extra></extra>',
        ))
        fig.update_layout(yaxis=dict(categoryorder='total ascending', title=''),
                          xaxis=dict(title='Films', range=[0, gdf['Count'].max()*1.18]))
        style_fig(fig, 400, show_grid_y=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <p class="section-label">FILMMAKERS</p>
    <p class="section-title">Top Directors in Collection</p>
    <p class="section-subtitle">Directors with the most films in the catalogue.</p>
    """, unsafe_allow_html=True)

    dc = df['Director'].value_counts().head(15).reset_index()
    dc.columns = ['Director','Films']
    cc, ct = st.columns([2.2, 1], gap="large")
    with cc:
        fig = go.Figure(go.Bar(
            x=dc['Films'], y=dc['Director'], orientation='h',
            marker=dict(color=T["accent"], cornerradius=3),
            text=dc['Films'], textposition='outside',
            textfont=dict(size=11, color=T["chart_text"], family="Source Sans Pro"),
            hovertemplate='<b>%{y}</b><br>Films: %{x}<extra></extra>',
        ))
        fig.update_layout(yaxis=dict(categoryorder='total ascending', title=''),
                          xaxis=dict(title='Films', range=[0, dc['Films'].max()*1.2]))
        style_fig(fig, 480, show_grid_y=False)
        st.plotly_chart(fig, use_container_width=True)
    with ct:
        st.dataframe(dc, hide_index=True, use_container_width=True, height=480)


# ── TAB 4: VIEWING STATS ──────────────────────
with tab4:
    st.markdown("""
    <p class="section-label">ANALYTICS</p>
    <p class="section-title">Viewing Patterns</p>
    <p class="section-subtitle">Personal viewing history derived from watch dates — how many films and hours across time.</p>
    """, unsafe_allow_html=True)

    tdf = filtered_entries.copy()
    tdf['Runtime_mins'] = tdf['Runtime'].apply(parse_runtime)

    tdf['Watch_Year'] = tdf['Date_Parsed'].dt.year
    tdf['Month_Name'] = tdf['Date_Parsed'].dt.month_name()
    tdf['Year-Month'] = tdf['Date_Parsed'].dt.to_period('M').astype(str)
    tdf = tdf.dropna(subset=['Watch_Year'])
    tdf['Watch_Year'] = tdf['Watch_Year'].astype(int)

    view_by = st.radio("Aggregate by", ["Year", "Month", "All Time"], horizontal=True)

    if view_by == "Year":
        gdf = tdf.groupby('Watch_Year').agg(Movies=('Name','count'), Minutes=('Runtime_mins','sum')).reset_index()
        gdf.columns = ['Period','Movies','Minutes']
        gdf['Period'] = gdf['Period'].astype(str)
        gdf['Hours'] = gdf['Minutes']/60
        x_lbl, x_ang, show_lbl = 'Year', 0, True

    elif view_by == "Month":
        mo = ['January','February','March','April','May','June',
              'July','August','September','October','November','December']
        tmp = tdf.dropna(subset=['Month_Name'])
        gdf = tmp.groupby('Month_Name').agg(Movies=('Name','count'), Minutes=('Runtime_mins','sum')).reset_index()
        gdf.columns = ['Period','Movies','Minutes']
        gdf['Hours'] = gdf['Minutes']/60
        gdf['Period'] = pd.Categorical(gdf['Period'], categories=mo, ordered=True)
        gdf = gdf.sort_values('Period')
        x_lbl, x_ang, show_lbl = 'Month', 45, True

    else:
        gdf = tdf.groupby('Year-Month').agg(Movies=('Name','count'), Minutes=('Runtime_mins','sum')).reset_index()
        gdf.columns = ['Period','Movies','Minutes']
        gdf['Hours'] = gdf['Minutes']/60
        x_lbl, x_ang, show_lbl = 'Month', 45, False

    tot_m = len(tdf)
    tot_min = tdf['Runtime_mins'].sum()
    tot_h = tot_min/60
    tot_d = tot_h/24
    avg_rt = tot_min/tot_m if tot_m else 0

    st.markdown("---")
    st.markdown(f"""
    <div class="card" style="padding:18px 28px;">
      <div class="kpi-row">
        <div class="kpi-item"><div class="stat-highlight">{tot_m}</div><div class="stat-label">Total Watches</div></div>
        <div class="kpi-item"><div class="stat-highlight">{tot_h:.0f}<span style="font-size:1rem;color:{T['text_secondary']};">h</span></div><div class="stat-label">Hours Spent</div></div>
        <div class="kpi-item"><div class="stat-highlight">{tot_d:.1f}<span style="font-size:1rem;color:{T['text_secondary']};">d</span></div><div class="stat-label">Days Spent</div></div>
        <div class="kpi-item"><div class="stat-highlight">{avg_rt:.0f}<span style="font-size:1rem;color:{T['text_secondary']};">m</span></div><div class="stat-label">Avg Runtime</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    cl, cr = st.columns(2, gap="large")

    with cl:
        st.markdown(f'<p class="section-label" style="margin-top:8px;">VOLUME</p>', unsafe_allow_html=True)
        st.markdown("**Viewing Activity**")
        fig = go.Figure(go.Bar(
            x=gdf['Period'], y=gdf['Movies'],
            marker=dict(color=T["navy"], cornerradius=3),
            text=gdf['Movies'] if show_lbl else None,
            textposition='outside' if show_lbl else None,
            textfont=dict(size=11, color=T["chart_text"], family="Source Sans Pro"),
            hovertemplate='<b>%{x}</b><br>Films: %{y}<extra></extra>',
        ))
        fig.update_layout(xaxis=dict(tickangle=x_ang, title=x_lbl, type='category'),
                          yaxis=dict(title='Films Watched'))
        if show_lbl:
            fig.update_layout(yaxis=dict(range=[0, gdf['Movies'].max()*1.18]))
        style_fig(fig, 460)
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown(f'<p class="section-label" style="margin-top:8px;">DURATION</p>', unsafe_allow_html=True)
        st.markdown("**Time Invested**")
        fig = go.Figure(go.Bar(
            x=gdf['Period'], y=gdf['Hours'],
            marker=dict(color=T["teal"], cornerradius=3),
            text=gdf['Hours'].apply(lambda v: f"{v:.0f}h") if show_lbl else None,
            textposition='outside' if show_lbl else None,
            textfont=dict(size=11, color=T["chart_text"], family="Source Sans Pro"),
            hovertemplate='<b>%{x}</b><br>Hours: %{y:.1f}<extra></extra>',
        ))
        fig.update_layout(xaxis=dict(tickangle=x_ang, title=x_lbl, type='category'),
                          yaxis=dict(title='Hours'))
        if show_lbl:
            fig.update_layout(yaxis=dict(range=[0, gdf['Hours'].max()*1.18]))
        style_fig(fig, 460)
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(f'<p class="footer-text">{original_count} films catalogued&ensp;·&ensp;Dashboard v4.1</p>',
            unsafe_allow_html=True)
