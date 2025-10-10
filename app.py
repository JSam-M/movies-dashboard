import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="My Movie Dashboard",
    page_icon="🎬",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('movies.csv')
    # Clean data
    df['TMDb_Rating'] = pd.to_numeric(df['TMDb_Rating'], errors='coerce')
    df['Release_Year'] = pd.to_numeric(df['Release_Year'], errors='coerce')
    df['N\'th time of watching'] = pd.to_numeric(df['N\'th time of watching'], errors='coerce').fillna(1)
    
    # Get unique movies with max rewatch count
    df_unique = df.groupby('Name', as_index=False).agg({
        'Date': 'last',  # Most recent watch date
        'Language': 'first',
        'Year': 'first',
        'Good?': 'first',
        'N\'th time of watching': 'max',  # Take the highest rewatch number
        'Location': 'last',  # Most recent location
        'Director': 'first',
        'Runtime': 'first',
        'Genre': 'first',
        'TMDb_Rating': 'first',
        'Release_Year': 'first',
        'Overview': 'first',
        'API_Status': 'first'
    })
    
    return df_unique, df  # Return both unique and original

df, df_original = load_data()
original_count = len(df)

# Title
st.title("🎬 My Movie Collection")
st.markdown("---")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Get full list of unique values from original unique dataset
df_full, _ = load_data()

# Search - back to text input
search = st.sidebar.text_input("Search movie name")
if search:
    df = df[df['Name'].str.contains(search, case=False, na=False)]

# Language filter
languages = ['All'] + sorted(df_full['Language'].dropna().unique().tolist())
selected_language = st.sidebar.multiselect("Language", languages, default=['All'])
if 'All' not in selected_language and selected_language:
    df = df[df['Language'].isin(selected_language)]

# Year range
if df['Release_Year'].notna().any():
    min_year = int(df['Release_Year'].min())
    max_year = int(df['Release_Year'].max())
    
    # Only show slider if there's a range
    if min_year < max_year:
        year_range = st.sidebar.slider(
            "Release Year",
            min_year, max_year,
            (min_year, max_year)
        )
        df = df[(df['Release_Year'] >= year_range[0]) & (df['Release_Year'] <= year_range[1])]
    else:
        # Just show the year as text if only one year
        st.sidebar.text(f"Release Year: {min_year}")

# Genre filter
all_genres = set()
for genres in df_full['Genre'].dropna():
    all_genres.update([g.strip() for g in str(genres).split(',')])
all_genres = ['All'] + sorted(list(all_genres))
selected_genres = st.sidebar.multiselect("Genre", all_genres, default=['All'])
if 'All' not in selected_genres and selected_genres:
    df = df[df['Genre'].apply(lambda x: any(genre in str(x) for genre in selected_genres) if pd.notna(x) else False)]

# Director filter
directors = ['All'] + sorted(df_full['Director'].dropna().unique().tolist())
selected_directors = st.sidebar.multiselect("Director", directors, default=['All'])
if 'All' not in selected_directors and selected_directors:
    df = df[df['Director'].isin(selected_directors)]

# Rewatch filter
rewatch_options = st.sidebar.radio(
    "Rewatches",
    ["All", "First time only", "Rewatched (2+)"]
)
if rewatch_options == "First time only":
    df = df[df['N\'th time of watching'] <= 1]
elif rewatch_options == "Rewatched (2+)":
    df = df[df['N\'th time of watching'] >= 2]

# Location filter - check if column exists
df_check, _ = load_data()
if 'Location' in df_check.columns:
    location_options = st.sidebar.radio(
        "Viewing Location",
        ["All", "Theatre", "Home"]
    )
    if location_options == "Theatre":
        df = df[df['Location'].str.lower().str.contains('theatre', na=False)]
    elif location_options == "Home":
        df = df[df['Location'].str.lower().str.contains('home', na=False)]
else:
    # Debug: show available columns
    st.sidebar.error(f"⚠️ Location column not found. Available columns: {', '.join(df_check.columns[:5])}...")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Showing {len(df)} of {original_count} unique movies**")

# Calculate total time spent
def parse_runtime(runtime_str):
    """Extract minutes from runtime string like '120 min'"""
    if pd.isna(runtime_str):
        return 0
    try:
        return int(str(runtime_str).split()[0])
    except:
        return 0

# Calculate total time from ALL viewings (including rewatches)
total_minutes = df_original['Runtime'].apply(parse_runtime).sum()
total_hours = total_minutes / 60
total_days = total_hours / 24

# Top stats row
st.subheader("📊 Collection Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Entries", len(df_original), help="Including rewatches")

with col2:
    st.metric("Unique Movies", original_count)

with col3:
    if total_hours >= 24:
        st.metric("Total Time Spent", f"{total_days:.1f} days", help=f"{total_hours:.0f} hours = {total_minutes:,} minutes")
    else:
        st.metric("Total Time Spent", f"{total_hours:.1f} hours", help=f"{total_minutes:,} minutes")

st.markdown("---")

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Unique Movies", len(df))

with col2:
    top_language = df['Language'].value_counts().index[0] if len(df) > 0 else "N/A"
    st.metric("Top Language", top_language)

with col3:
    avg_rating = df['TMDb_Rating'].mean() if df['TMDb_Rating'].notna().any() else 0
    st.metric("Avg TMDb Rating", f"{avg_rating:.1f}")

with col4:
    # Count movies with rewatch value >= 2
    total_rewatches = df[df['N\'th time of watching'] >= 2].shape[0]
    st.metric("Movies Rewatched", total_rewatches)

st.markdown("---")

# Visualizations
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎭 Genres & Languages", "⭐ Ratings", "📋 Full List"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Movies over time
        st.subheader("Movies Watched Over Time")
        df_time = df.copy()
        df_time['Date'] = pd.to_datetime(df_time['Date'])
        df_time['Year-Month'] = df_time['Date'].dt.to_period('M').astype(str)
        time_data = df_time.groupby('Year-Month').size().reset_index(name='Count')
        
        fig = px.line(time_data, x='Year-Month', y='Count', 
                     title='Movies per Month',
                     markers=True)
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top directors
        st.subheader("Top Directors")
        director_counts = df['Director'].value_counts().head(10).reset_index()
        director_counts.columns = ['Director', 'Count']
        fig = px.bar(
            director_counts,
            x='Count',
            y='Director',
            orientation='h',
            labels={'Count': 'Movies', 'Director': 'Director'}
        )
        fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Language breakdown
        st.subheader("Movies by Language")
        lang_data = df['Language'].value_counts()
        fig = px.pie(values=lang_data.values, names=lang_data.index,
                    title='Language Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Genre breakdown
        st.subheader("Top Genres")
        genre_counts = {}
        for genres in df['Genre'].dropna():
            for genre in str(genres).split(','):
                genre = genre.strip()
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count'])
        genre_df = genre_df.sort_values('Count', ascending=False).head(10)
        
        fig = px.bar(genre_df, x='Count', y='Genre', orientation='h')
        fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Rating distribution
    st.subheader("TMDb Rating Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='TMDb_Rating', nbins=20,
                          labels={'TMDb_Rating': 'Rating'},
                          title='Rating Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top rated movies
        st.subheader("Highest Rated Movies")
        top_rated = df.nlargest(10, 'TMDb_Rating')[['Name', 'Release_Year', 'TMDb_Rating', 'Director']]
        st.dataframe(top_rated, hide_index=True, use_container_width=True)
    
    # Note about your rating system
    st.info("💡 Your 2x2 Rating System will appear here once you fill the 'Good?' column in your Excel!")
    st.markdown("""
    **Your Rating System:**
    - 🟢 **Good movie, I liked it** - Quality + Enjoyment ✓
    - 🟡 **Good movie, I didn't like it** - Quality ✓, Personal taste ✗
    - 🟠 **Bad movie, I liked it** - Guilty pleasure!
    - 🔴 **Bad movie, I didn't like it** - Skip it
    """)

with tab4:
    # Full table
    st.subheader("Complete Movie List")
    
    # Select columns to display
    display_cols = ['Name', 'Release_Year', 'Language', 'Director', 'Genre', 'TMDb_Rating', 'N\'th time of watching']
    display_df = df[display_cols].copy()
    display_df.columns = ['Movie', 'Year', 'Language', 'Director', 'Genre', 'Rating', 'Rewatches']
    
    st.dataframe(
        display_df.sort_values('Rating', ascending=False),
        hide_index=True,
        use_container_width=True,
        height=600
    )

# Footer
st.markdown("---")
st.markdown("🎬 **Unique movies tracked** | TMDb enriched data | Made with Streamlit")
st.markdown("<p style='text-align: center; color: #666; font-size: 10px; margin-top: 20px;'>Dashboard v1.5 | Last updated: 2025-01-08</p>", unsafe_allow_html=True)
