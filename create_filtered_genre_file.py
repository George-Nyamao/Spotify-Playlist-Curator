import pandas as pd
import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
LASTFM_BASE_URL = "http://ws.audioscrobbler.com/2.0/"

output_folder = "data"

def get_genres_for_artist_track(artist_name, track_name):
    """Get genres for artist/track from Last.fm"""
    if not artist_name or not track_name or pd.isna(artist_name) or pd.isna(track_name):
        return []
    
    artist_name = str(artist_name).strip()
    track_name = str(track_name).strip()
    
    if not artist_name or not track_name:
        return []
    
    params = {
        'api_key': LASTFM_API_KEY,
        'format': 'json'
    }
    
    # Try track-specific genres first
    params.update({
        'method': 'track.getTopTags',
        'artist': artist_name,
        'track': track_name
    })
    
    try:
        response = requests.get(LASTFM_BASE_URL, params=params, timeout=10)
        data = response.json()
        
        if 'toptags' in data and 'tag' in data['toptags']:
            return [tag['name'].lower() for tag in data['toptags']['tag']]
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
            return [tag['name'].lower() for tag in data['toptags']['tag']]
    except:
        pass
    
    return []

def contains_target_genre(genres_list):
    """Check if genres contain any of our target genres"""
    target_genres = [
        'hip-hop', 'hip hop', 'rap', 'dirty south', 'southern rap', 
        'gangsta rap', 'east coast rap', 'pop rap', 'trap', 'west coast rap',
        'rnb', 'r&b', 'soul', 'neo-soul', 'urban', 'neo soul', 
        'alternative rnb', 'r and b', 'funk', 'afrobeat', 'nigerian', 
        'afropop', 'african', 'dancehall', 'reggae', 'funk rock'
    ]
    
    for genre in genres_list:
        if any(target in genre for target in target_genres):
            return True
    return False

def filter_csv_by_genres(input_csv, output_suffix='_curated'):
    """
    Filter CSV for target genres and save with renumbered rows
    """
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    print(f"Original CSV shape: {df.shape}")
    print(f"Processing {len(df)} tracks...")
    
    # Map column names
    title_col = 'Title'
    artist_col = 'Artist(s)'
    
    # Filter tracks
    filtered_indices = []
    
    for idx, row in df.iterrows():
        artist = row.get(artist_col, '')
        track = row.get(title_col, '')
        
        if pd.isna(artist) or pd.isna(track) or artist == '' or track == '':
            continue
        
        print(f"Checking {idx+1}/{len(df)}: {artist} - {track}")
        
        # Get genres from Last.fm
        genres = get_genres_for_artist_track(artist, track)
        
        # Check if contains target genres
        if contains_target_genre(genres):
            filtered_indices.append(idx)
            print(f"  ✓ INCLUDED - Genres: {genres[:3]}...")
        else:
            print(f"  ✗ EXCLUDED - Genres: {genres[:3]}...")
        
        # Rate limiting
        time.sleep(0.3)
    
    # Create filtered dataframe
    filtered_df = df.iloc[filtered_indices].copy()
    
    # Renumber the "No." column
    if 'No.' in filtered_df.columns:
        filtered_df['No.'] = range(1, len(filtered_df) + 1)
    
    # Generate output filename
    if input_csv.endswith('.csv'):
        output_csv = input_csv.replace('.csv', f'{output_suffix}.csv')
    else:
        output_csv = f"{input_csv}{output_suffix}.csv"
    
    # Save filtered results
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    full_path = os.path.join(output_folder, output_csv)
    filtered_df.to_csv(full_path, index=False)
    
    print(f"\nRESULTS:")
    print(f"Original tracks: {len(df)}")
    print(f"Filtered tracks: {len(filtered_df)}")
    print(f"Match rate: {len(filtered_df)/len(df)*100:.1f}%")
    print(f"Saved to: {output_csv}")
    
    return filtered_df

# Usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "2024_Year_End_Hot_100.csv"
    
    try:
        result = filter_csv_by_genres(input_file)
    except FileNotFoundError:
        print(f"File '{input_file}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")