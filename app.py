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
    help="Filter by rating"
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
                hover_data={'TMDb_Rating': ':.1f'}
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
    # Personal stats tab - FIXED VERSION
    st.subheader("📊 My Viewing Patterns")
    
    # Helper function
    def parse_runtime(runtime_str):
        if pd.isna(runtime_str):
            return 0
        try:
            return int(str(runtime_str).split()[0])
        except:
            return 0
    
    # Prepare time data
    time_df = filtered_entries.copy()
    time_df['Date'] = pd.to_datetime(time_df['Date'], dayfirst=True)
    time_df['Runtime_mins'] = time_df['Runtime'].apply(parse_runtime)
    time_df['Year'] = time_df['Date'].dt.year
    time_df['Month'] = time_df['Date'].dt.month
    time_df['Month_Name'] = time_df['Date'].dt.month_name()
    time_df['Year-Month'] = time_df['Date'].dt.to_period('M').astype(str)
    
    # Time period selector
    view_by = st.radio("View by", ["All Time", "By Year", "By Month"], horizontal=True)
    
    # Calculate stats based on view
    if view_by == "All Time":
        # Monthly breakdown
        group_df = time_df.groupby('Year-Month').agg({
            'Name': 'count',
            'Runtime_mins': 'sum'
        }).reset_index()
        group_df.columns = ['Period', 'Movies', 'Minutes']
        group_df['Hours'] = group_df['Minutes'] / 60
        x_label = 'Month'
        x_angle = 45
        show_labels = False
        
    elif view_by == "By Year":
        # Aggregate by YEAR (compare all years)
        group_df = time_df.groupby('Year').agg({
            'Name': 'count',
            'Runtime_mins': 'sum'
        }).reset_index()
        group_df.columns = ['Period', 'Movies', 'Minutes']
        group_df['Period'] = group_df['Period'].astype(str)
        group_df['Hours'] = group_df['Minutes'] / 60
        x_label = 'Year'
        x_angle = 0
        show_labels = True
        
    else:  # By Month
        # Aggregate by MONTH NAME (compare Jan vs Feb vs Mar, etc.)
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        group_df = time_df.groupby('Month_Name').agg({
            'Name': 'count',
            'Runtime_mins': 'sum'
        }).reset_index()
        group_df.columns = ['Period', 'Movies', 'Minutes']
        group_df['Hours'] = group_df['Minutes'] / 60
        
        # Sort by month order
        group_df['Period'] = pd.Categorical(group_df['Period'], categories=month_order, ordered=True)
        group_df = group_df.sort_values('Period')
        x_label = 'Month'
        x_angle = 45
        show_labels = True
    
    # Calculate totals
    total_movies = time_df['Name'].count()
    total_minutes = time_df['Runtime_mins'].sum()
    total_hours = total_minutes / 60
    total_days = total_hours / 24
    avg_per_movie = total_minutes / total_movies if total_movies > 0 else 0
    
    # Display stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Movies Watched", total_movies)
    
    with col2:
        st.metric("Total Hours", f"{total_hours:.1f}h")
    
    with col3:
        st.metric("Total Days", f"{total_days:.2f}d")
    
    with col4:
        st.metric("Avg Runtime", f"{avg_per_movie:.0f}m")
    
    st.markdown("---")
    
    # Two main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Viewing Activity")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=group_df['Period'],
            y=group_df['Movies'],
            name='Movies',
            marker_color='lightblue',
            text=group_df['Movies'] if show_labels else None,
            textposition='outside' if show_labels else None
        ))
        fig.update_layout(
            xaxis=dict(tickangle=x_angle, title=x_label),
            yaxis_title='Movies Watched',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Time Spent")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=group_df['Period'],
            y=group_df['Hours'],
            name='Hours',
            marker_color='#4ECDC4',
            text=group_df['Hours'].apply(lambda x: f"{x:.1f}h") if show_labels else None,
            textposition='outside' if show_labels else None
        ))
        fig.update_layout(
            xaxis=dict(tickangle=x_angle, title=x_label),
            yaxis_title='Hours Spent',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("🎬 **Movie recommendation database** | Curated collection")
st.markdown("<p style='text-align: center; color: #666; font-size: 10px; margin-top: 20px;'>Dashboard v3.2 | Optimized for recommendations</p>", unsafe_allow_html=True)
