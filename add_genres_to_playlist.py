import pandas as pd
import requests
import os
from dotenv import load_dotenv
import time

load_dotenv()

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
LASTFM_BASE_URL = "http://ws.audioscrobbler.com/2.0/"

INPUT_CSV = '2020s_RnB_Hip_Hop.csv'
OUTPUT_CSV = INPUT_CSV.replace('.csv', '_with_genres.csv')

def clean_genres(genres, artist_name, track_name, album_name):
    """
    Remove artist names, track names, and album names from genre list
    """
    # Convert to lowercase for comparison
    artist_lower = artist_name.lower()
    track_lower = track_name.lower() 
    album_lower = album_name.lower()
    
    # Split into words for partial matching
    artist_words = set(artist_lower.split())
    track_words = set(track_lower.split())
    album_words = set(album_lower.split())
    
    cleaned_genres = []
    
    for genre in genres:
        genre_lower = genre.lower()
        
        # Skip if genre contains artist name, track name, or album name
        if (artist_lower in genre_lower or 
            track_lower in genre_lower or 
            album_lower in genre_lower or
            any(word in genre_lower for word in artist_words if len(word) > 2) or
            any(word in genre_lower for word in track_words if len(word) > 2) or
            any(word in genre_lower for word in album_words if len(word) > 2)):
            continue
            
        cleaned_genres.append(genre)
    
    return cleaned_genres

def get_genres_for_artist_track(artist_name, track_name, album_name=""):
    """Get genres for artist/track with error handling"""
    params = {
        'api_key': LASTFM_API_KEY,
        'format': 'json'
    }
    
    # Try track-specific genres first
    if track_name:
        params.update({
            'method': 'track.getTopTags',
            'artist': artist_name,
            'track': track_name
        })
        try:
            response = requests.get(LASTFM_BASE_URL, params=params, timeout=10)
            data = response.json()
            
            if 'toptags' in data and 'tag' in data['toptags']:
                genres = [tag['name'].lower() for tag in data['toptags']['tag']]
                if genres:
                    # CLEAN THE GENRES
                    cleaned_genres = clean_genres(genres, artist_name, track_name, album_name)
                    return cleaned_genres
        except:
            pass
    
    # Fall back to artist genres
    params.update({
        'method': 'artist.getTopTags',
        'artist': artist_name
    })
    
    try:
        response = requests.get(LASTFM_BASE_URL, params=params, timeout=10)
        data = response.json()
        
        if 'toptags' in data and 'tag' in data['toptags']:
            genres = [tag['name'] for tag in data['toptags']['tag']]
            # CLEAN THE GENRES
            cleaned_genres = clean_genres(genres, artist_name, track_name, album_name)
            return cleaned_genres
    except:
        pass
    
    return []

def process_curated_csv(input_csv, output_csv):
    """Add genres to your curated CSV"""
    # Read your existing CSV
    df = pd.read_csv(input_csv)
    
    print(f"Processing {len(df)} tracks...")
    
    # Add new columns for genres
    df['All_Genres'] = ""
    df['Primary_Genres'] = ""
    df['Genre_Count'] = 0
    
    # Process each row
    for idx, row in df.iterrows():
        artist = row['Artist']
        track = row['Name']
        
        print(f"Processing {idx+1}/{len(df)}: {artist} - {track}")
        
        # Get genres from Last.fm
        genres = get_genres_for_artist_track(artist, track, row.get('Album', ''))
        
        # Update the dataframe
        df.at[idx, 'All_Genres'] = ", ".join(genres)
        df.at[idx, 'Primary_Genres'] = ", ".join(genres[:5])  # Top 5 genres
        df.at[idx, 'Genre_Count'] = len(genres)
        
        # Rate limiting to be nice to Last.fm API
        time.sleep(0.2)
    
    # Save the enhanced CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved enhanced data to {output_csv}")
    
    return df

# Run the processing
enhanced_df = process_curated_csv(INPUT_CSV, OUTPUT_CSV)