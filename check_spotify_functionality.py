# test_audio_rewritten.py
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# Load credentials from environment
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

# A list of track IDs to test
TEST_TRACK_IDS = ['6iEvECKDbtcbfbTLNoQGe1', '1G32fy7VMCDLl92iGXvBEm']

auth_manager = SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET
)
sp = spotipy.Spotify(auth_manager=auth_manager)

def extract_available_track_features(track_data):
    """Extract all available features from track data"""
    features = {
        # Basic track info
        'id': track_data['id'],
        'name': track_data['name'],
        'uri': track_data['uri'],
        'duration_ms': track_data['duration_ms'],
        'explicit': track_data['explicit'],
        'popularity': track_data['popularity'],
        'preview_url': track_data['preview_url'],
        'is_local': track_data['is_local'],
        
        # Artist information
        'artists': [artist['name'] for artist in track_data['artists']],
        'artist_ids': [artist['id'] for artist in track_data['artists']],
        
        # Album information
        'album_id': track_data['album']['id'],
        'album_name': track_data['album']['name'],
        'album_type': track_data['album']['album_type'],
        'album_release_date': track_data['album']['release_date'],
        'album_release_date_precision': track_data['album']['release_date_precision'],
        'album_total_tracks': track_data['album']['total_tracks'],
        
        # Availability
        'available_markets': track_data['available_markets'],
        'disc_number': track_data['disc_number'],
        'track_number': track_data['track_number']
    }
    
    # Extract release year for filtering
    release_date = track_data['album']['release_date']
    if release_date:
        features['release_year'] = int(release_date.split('-')[0])
    
    return features

# Extract features for all test tracks
all_track_features = []
for track_id in TEST_TRACK_IDS:
    try:
        track = sp.track(track_id)
        features = extract_available_track_features(track)
        all_track_features.append(features)
        
        # Print basic info
        print(f"Track: {features['name']}")
        print(f"Artist: {', '.join(features['artists'])}")
        print(f"Album: {features['album_name']}")
        print(f"Release Date: {features['album_release_date']}")
        print(f"Popularity: {features['popularity']}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error processing track {track_id}: {str(e)}")

# Create DataFrame for analysis
df = pd.DataFrame(all_track_features)
print("\nAvailable features DataFrame:")
print(df.columns.tolist())
print(f"\nShape: {df.shape}")

# Save to CSV for further analysis
df.to_csv('extracted_track_features.csv', index=False)
print("\nFeatures saved to 'extracted_track_features.csv'")