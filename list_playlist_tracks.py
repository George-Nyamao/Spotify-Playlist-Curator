import os
import spotipy
import pandas as pd
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials
from typing import List, Dict, Any, Optional

PLAYLIST_ID = '3NwAKlyGXkYqjWouJFtJuN' # Example playlist ID
OUTPUT_CSV = '2020s_RnB_Hip_Hop.csv'

load_dotenv()

def get_spotify_client() -> spotipy.Spotify:
    """Initialize and return a Spotify client using client credentials."""
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        raise ValueError("SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET must be set")
    
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    ))

def get_playlist_tracks(sp: spotipy.Spotify, playlist_id: str) -> List[Dict[str, Any]]:
    """
    Get all tracks from a Spotify playlist, handling pagination.
    
    Args:
        sp: Spotify client instance
        playlist_id: The Spotify playlist ID
        
    Returns:
        List of track items from the playlist
    """
    playlist_tracks = []
    results = sp.playlist_tracks(playlist_id)
    playlist_tracks.extend(results['items'])
    
    # Handle pagination if the playlist has more than 100 tracks
    while results['next']:
        results = sp.next(results)
        playlist_tracks.extend(results['items'])
    
    return playlist_tracks

def extract_track_data(tracks_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract track data from Spotify track information.
    
    Args:
        tracks_info: List of track information from Spotify API
        
    Returns:
        List of dictionaries containing track data
    """
    table_data = []
    for track in tracks_info:
        if track is None:
            continue
            
        # Extract the main artist's name (handling multiple artists)
        main_artist = track['artists'][0]['name'] if track['artists'] else 'Unknown Artist'
        
        # Create a row of data for the track
        row = {
            'Name': track['name'],
            'Artist': main_artist,
            'Album': track['album']['name'],
            'Release Date': track['album']['release_date'],
            'Popularity': track['popularity'],
            'Explicit': track['explicit'],
            'Duration (ms)': track['duration_ms'],
            'Track ID': track['id'],
            'Spotify URL': track['external_urls']['spotify']
        }
        table_data.append(row)
    
    return table_data

def get_playlist_dataframe(sp: spotipy.Spotify, playlist_id: str) -> pd.DataFrame:
    """
    Get playlist tracks and return as a pandas DataFrame.
    
    Args:
        sp: Spotify client instance
        playlist_id: The Spotify playlist ID
        
    Returns:
        pandas DataFrame containing track information
    """
    # Get all tracks from the playlist
    playlist_tracks = get_playlist_tracks(sp, playlist_id)
    
    # Extract track IDs and get detailed audio features for each
    track_ids = [item['track']['id'] for item in playlist_tracks if item['track'] is not None]
    
    if not track_ids:
        return pd.DataFrame()
    
    # Process tracks in batches of 50 (Spotify API limit)
    batch_size = 50
    all_tracks_info = []
    
    for i in range(0, len(track_ids), batch_size):
        batch_ids = track_ids[i:i + batch_size]
        batch_tracks = sp.tracks(batch_ids)['tracks']
        all_tracks_info.extend(batch_tracks)
    
    # Extract track data
    table_data = extract_track_data(all_tracks_info)
    
    # Create DataFrame
    return pd.DataFrame(table_data)

def save_playlist_to_csv(df: pd.DataFrame, filename: str = OUTPUT_CSV) -> None:
    """
    Save playlist DataFrame to CSV file.
    
    Args:
        df: pandas DataFrame containing track information
        filename: Output CSV filename
    """
    df.to_csv(filename, index=False)

def main():
    """Main function to demonstrate the playlist tracks functionality."""
    try:
        sp = get_spotify_client()
        
        # Your playlist ID
        playlist_id = PLAYLIST_ID
        
        # Get playlist data
        df_playlist = get_playlist_dataframe(sp, playlist_id)
        
        if df_playlist.empty:
            print("No tracks found in the playlist.")
            return
        
        print(f"Found {len(df_playlist)} tracks in the playlist:")
        print(df_playlist.head())  # Display the first few rows
        
        # Save to CSV
        save_playlist_to_csv(df_playlist)
        print(f"Playlist data saved to {OUTPUT_CSV}")
        
    except Exception as e:
        print(f"Error processing playlist: {e}")

if __name__ == "__main__":
    main()

