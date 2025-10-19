import add_genres_to_playlist

def analyze_genre_patterns(df):
    """Analyze what genres appear in your curated collection"""
    from collections import Counter
    
    # Extract all genres
    all_genres = []
    for genres_str in df['All_Genres']:
        if genres_str:
            genres = [g.strip() for g in genres_str.split(',')]
            all_genres.extend(genres)
    
    # Count genre frequency
    genre_counter = Counter(all_genres)
    
    print("TOP GENRES IN YOUR COLLECTION:")
    print("=" * 40)
    for genre, count in genre_counter.most_common(30):
        print(f"{genre:25} : {count:3} occurrences")
    
    # Analyze by artist
    print("\nTOP ARTISTS AND THEIR GENRES:")
    print("=" * 40)
    artist_genres = df.groupby('Artist')['All_Genres'].first()
    for artist, genres in artist_genres.head(15).items():
        print(f"{artist:25} : {genres}")
    
    return genre_counter

def build_genre_profile(df, top_n=20):
    """Build a personalized genre profile from your collection"""
    from collections import Counter
    
    all_genres = []
    for genres_str in df['All_Genres']:
        if genres_str:
            genres = [g.strip() for g in genres_str.split(',')]
            all_genres.extend(genres)
    
    genre_counter = Counter(all_genres)
    
    # Create genre weights based on frequency
    total_tracks = len(df)
    genre_weights = {}
    
    for genre, count in genre_counter.most_common(top_n):
        weight = count / total_tracks
        genre_weights[genre] = weight
    
    print("YOUR PERSONAL GENRE PROFILE:")
    print("=" * 35)
    for genre, weight in list(genre_weights.items())[:15]:
        print(f"{genre:25} : {weight:.3f} ({genre_counter[genre]} songs)")
    
    return genre_weights

# Analyze your music taste
genre_patterns = analyze_genre_patterns(add_genres_to_playlist.enhanced_df)

# Build your genre profile
my_genre_profile = build_genre_profile(add_genres_to_playlist.enhanced_df)
