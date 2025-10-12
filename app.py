import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Movie Recommendations",
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
        'Date': 'last',
        'Language': 'first',
        'Year': 'first',
        'Good?': 'first',
        'N\'th time of watching': 'max',
        'Location': 'last',
        'Director': 'first',
        'Runtime': 'first',
        'Genre': 'first',
        'TMDb_Rating': 'first',
        'Release_Year': 'first',
        'Overview': 'first',
        'API_Status': 'first'
    })
    
    return df_unique, df

df, df_original = load_data()
original_count = len(df)

# Title
st.title("🎬 Movie Recommendations")
st.caption("Curated collection with personal recommendations")
st.markdown("---")

# Sidebar filters
st.sidebar.header("🔍 Find Movies")

# Get full list of unique values from original unique dataset
df_full, _ = load_data()

# Search
search = st.sidebar.text_input("🔤 Search movie name")
if search:
    df = df[df['Name'].str.contains(search, case=False, na=False)]

# Language filter
languages = ['All'] + sorted(df_full['Language'].dropna().unique().tolist())
selected_language = st.sidebar.multiselect("🗣️ Language", languages, default=['All'])
if 'All' not in selected_language and selected_language:
    df = df[df['Language'].isin(selected_language)]

# Genre filter
all_genres = set()
for genres in df_full['Genre'].dropna():
    all_genres.update([g.strip() for g in str(genres).split(',')])
all_genres = ['All'] + sorted(list(all_genres))
selected_genres = st.sidebar.multiselect("🎭 Genre", all_genres, default=['All'])
if 'All' not in selected_genres and selected_genres:
    df = df[df['Genre'].apply(lambda x: any(genre in str(x) for genre in selected_genres) if pd.notna(x) else False)]

# Director filter
directors = ['All'] + sorted(df_full['Director'].dropna().unique().tolist())
selected_directors = st.sidebar.multiselect("🎬 Director", directors, default=['All'])
if 'All' not in selected_directors and selected_directors:
    df = df[df['Director'].isin(selected_directors)]

# Rating filter
min_rating = st.sidebar.slider(
    "⭐ Minimum Rating",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.5,
    help="Filter by TMDb rating"
)
df = df[df['TMDb_Rating'] >= min_rating]

# Year range
if df['Release_Year'].notna().any():
    min_year = int(df['Release_Year'].min())
    max_year = int(df['Release_Year'].max())
    
    if min_year < max_year:
        year_range = st.sidebar.slider(
            "📅 Release Year",
            min_year, max_year,
            (min_year, max_year)
        )
        df = df[(df['Release_Year'] >= year_range[0]) & (df['Release_Year'] <= year_range[1])]

# Personal recommendation filter (rewatches)
st.sidebar.markdown("---")
st.sidebar.subheader("💡 Personal Picks")
rewatch_options = st.sidebar.radio(
    "Recommendation Level",
    ["All Movies", "Personally Recommended (Rewatched)", "First Watch Only"],
    help="Rewatched movies = Strong personal recommendations!"
)
if rewatch_options == "Personally Recommended (Rewatched)":
    df = df[df['N\'th time of watching'] >= 2]
elif rewatch_options == "First Watch Only":
    df = df[df['N\'th time of watching'] <= 1]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Showing {len(df)} of {original_count} movies**")

# Calculate stats
filtered_movie_names = df['Name'].tolist()
filtered_entries = df_original[df_original['Name'].isin(filtered_movie_names)]

# Compact metrics row
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Movies", len(df))

with col2:
    langs_count = df['Language'].nunique()
    st.metric("Languages", langs_count)

with col3:
    avg_rating = df['TMDb_Rating'].mean() if df['TMDb_Rating'].notna().any() else 0
    st.metric("Avg Rating", f"{avg_rating:.1f}")

with col4:
    max_rating = df['TMDb_Rating'].max() if df['TMDb_Rating'].notna().any() else 0
    st.metric("Highest", f"{max_rating:.1f}")

with col5:
    genres_count = len(set(g.strip() for genres in df['Genre'].dropna() for g in str(genres).split(',')))
    st.metric("Genres", genres_count)

with col6:
    recommended = df[df['N\'th time of watching'] >= 2].shape[0]
    st.metric("⭐ Recommended", recommended, help="Personally rewatched movies")

st.markdown("---")

# Tabs - reordered for recommendation focus
tab1, tab2, tab3, tab4 = st.tabs(["📋 Browse Movies", "⭐ Top Picks", "🎭 By Category", "📊 My Stats"])

with tab1:
    # Primary view - Full browsable list
    st.subheader("Browse All Movies")
    
    # Prepare display dataframe
    display_df = df[['Name', 'Release_Year', 'TMDb_Rating', 'N\'th time of watching', 'Genre', 'Director', 'Runtime', 'Language']].copy()
    display_df.columns = ['Movie', 'Year', 'Rating', 'Rewatches', 'Genre', 'Director', 'Runtime', 'Language']
    
    # Add recommendation indicator
    display_df['💡'] = display_df['Rewatches'].apply(lambda x: '⭐ Recommended' if x >= 2 else '')
    
    # Reorder columns
    display_df = display_df[['Movie', 'Year', 'Rating', 'Rewatches', '💡', 'Genre', 'Director', 'Runtime', 'Language']]
    
    # Sort by rating by default
    display_df = display_df.sort_values('Rating', ascending=False)
    
    # Show count
    st.caption(f"Showing {len(display_df)} movies • Sorted by rating • ⭐ = Personally recommended")
    
    # Display table
    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        height=600
    )

with tab2:
    # Top picks tab
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Highest Rated")
        top_rated = df.nlargest(10, 'TMDb_Rating')[['Name', 'Release_Year', 'TMDb_Rating', 'N\'th time of watching']].reset_index(drop=True)
        top_rated['Movie'] = top_rated['Name'] + ' (' + top_rated['Release_Year'].astype(str) + ')'
        top_rated['Recommended'] = top_rated['N\'th time of watching'].apply(lambda x: '⭐' if x >= 2 else '')
        
        fig = px.bar(
            top_rated,
            x='TMDb_Rating',
            y='Movie',
            orientation='h',
            labels={'TMDb_Rating': 'Rating', 'Movie': ''},
            color='TMDb_Rating',
            color_continuous_scale='Viridis',
            hover_data={'Recommended': True}
        )
        fig.update_layout(
            showlegend=False, 
            yaxis={'categoryorder': 'total ascending'},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⭐ Personally Recommended")
        st.caption("Movies I've rewatched - my strongest recommendations!")
        most_rewatched = df[df['N\'th time of watching'] > 1].nlargest(10, 'N\'th time of watching')[['Name', 'Release_Year', 'N\'th time of watching', 'TMDb_Rating']].reset_index(drop=True)
        
        if len(most_rewatched) > 0:
            most_rewatched['Movie'] = most_rewatched['Name'] + ' (' + most_rewatched['Release_Year'].astype(str) + ')'
            
            fig = px.bar(
                most_rewatched,
                x='N\'th time of watching',
                y='Movie',
                orientation='h',
                labels={'N\'th time of watching': 'Times Watched', 'Movie': ''},
                color='N\'th time of watching',
                color_continuous_scale='Blues',
                hover_data={'TMDb_Rating': True}
            )
            fig.update_layout(
                showlegend=False, 
                yaxis={'categoryorder': 'total ascending'},
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rewatched movies in current filters")

with tab3:
    # By category
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Movies by Language")
        lang_data = df['Language'].value_counts()
        fig = px.pie(values=lang_data.values, names=lang_data.index,
                    title='Language Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Movies by Genre")
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
    
    # Directors
    st.subheader("Top Directors in Collection")
    director_counts = df['Director'].value_counts().head(15).reset_index()
    director_counts.columns = ['Director', 'Movies']
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.bar(
            director_counts,
            x='Movies',
            y='Director',
            orientation='h',
            labels={'Movies': 'Number of Movies', 'Director': ''}
        )
        fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'}, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(director_counts, hide_index=True, use_container_width=True, height=500)

with tab4:
    # Personal stats tab
    st.subheader("📊 My Viewing Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Movies over time
        st.subheader("Movies Watched Over Time")
        df_time = df.copy()
        df_time['Date'] = pd.to_datetime(df_time['Date'])
        df_time['Year-Month'] = df_time['Date'].dt.to_period('M').astype(str)
        time_data = df_time.groupby('Year-Month').size().reset_index(name='Count')
        
        fig = px.line(time_data, x='Year-Month', y='Count', 
                     title='Viewing Activity',
                     markers=True)
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Viewing location breakdown
        st.subheader("Viewing Preferences")
        if 'Location' in df.columns:
            location_data = df['Location'].value_counts()
            fig = px.pie(values=location_data.values, names=location_data.index,
                        title='Theatre vs Home')
            st.plotly_chart(fig, use_container_width=True)
    
    # Time spent over months/years
    st.subheader("Time Spent Watching Movies")
    
    def parse_runtime(runtime_str):
        if pd.isna(runtime_str):
            return 0
        try:
            return int(str(runtime_str).split()[0])
        except:
            return 0
    
    # Get all entries with dates and runtime
    time_df = filtered_entries.copy()
    time_df['Date'] = pd.to_datetime(time_df['Date'])
    time_df['Runtime_mins'] = time_df['Runtime'].apply(parse_runtime)
    time_df['Year-Month'] = time_df['Date'].dt.to_period('M').astype(str)
    
    # Calculate time per month
    time_by_month = time_df.groupby('Year-Month')['Runtime_mins'].sum().reset_index()
    time_by_month['Hours'] = time_by_month['Runtime_mins'] / 60
    time_by_month['Cumulative_Hours'] = time_by_month['Hours'].cumsum()
    time_by_month['Cumulative_Days'] = time_by_month['Cumulative_Hours'] / 24
    
    # Create dual-axis chart
    fig = go.Figure()
    
    # Monthly hours (bars)
    fig.add_trace(go.Bar(
        x=time_by_month['Year-Month'],
        y=time_by_month['Hours'],
        name='Hours per Month',
        marker_color='lightblue',
        yaxis='y'
    ))
    
    # Cumulative days (line)
    fig.add_trace(go.Scatter(
        x=time_by_month['Year-Month'],
        y=time_by_month['Cumulative_Days'],
        name='Cumulative Total (Days)',
        mode='lines+markers',
        line=dict(color='darkblue', width=3),
        yaxis='y2'
    ))
    
    fig.update_layout(
        xaxis=dict(tickangle=45, title=''),
        yaxis=dict(title='Hours per Month', side='left'),
        yaxis2=dict(title='Cumulative Days', overlaying='y', side='right'),
        hovermode='x unified',
        height=400,
        legend=dict(x=0.01, y=0.99)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary stats
    total_minutes = time_by_month['Runtime_mins'].sum()
    total_hours = total_minutes / 60
    total_days = total_hours / 24
    st.caption(f"📊 Total: {total_days:.1f} days ({total_hours:.0f} hours) spent watching movies")
    
    # Rewatch stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rewatch Distribution")
        rewatch_dist = df['N\'th time of watching'].value_counts().sort_index()
        rewatch_dist.index = rewatch_dist.index.map(lambda x: f"{int(x)}x" if x > 1 else "Once")
        
        fig = px.bar(x=rewatch_dist.values, y=rewatch_dist.index, 
                    orientation='h',
                    labels={'x': 'Number of Movies', 'y': 'Times Watched'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Average Movie Length")
        avg_runtime = time_df['Runtime_mins'].mean()
        st.metric("Average Runtime", f"{avg_runtime:.0f} minutes", help=f"{avg_runtime/60:.1f} hours")
        
        # Runtime distribution
        fig = px.histogram(time_df, x='Runtime_mins', nbins=20,
                          labels={'Runtime_mins': 'Runtime (minutes)'},
                          title='Runtime Distribution')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("🎬 **Movie recommendation database** | Powered by TMDb")
st.markdown("<p style='text-align: center; color: #666; font-size: 10px; margin-top: 20px;'>Dashboard v2.2 | Optimized for recommendations</p>", unsafe_allow_html=True)
