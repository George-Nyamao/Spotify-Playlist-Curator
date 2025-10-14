import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Get Spotify credentials from environment variables
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

# Check if credentials are set
if not SPOTIPY_CLIENT_ID or not SPOTIPY_CLIENT_SECRET:
    raise Exception("Spotify API credentials not found. Please set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in a .env file.")

# Initialize Spotipy with client credentials and a longer timeout
auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager, requests_timeout=30)

def get_song_versions(song_title):
    """
    Searches for a song on Spotify and returns its explicit and non-explicit versions.
    """
    results = sp.search(q=song_title, type='track', limit=50)
    tracks = results['tracks']['items']

    versions = []
    seen_tracks = set()

    for track in tracks:
        # Normalize title and artists to avoid duplicates with minor variations
        normalized_title = track['name'].lower()
        artist_names = sorted([artist['name'].lower() for artist in track['artists']])

        # Create a unique key for the track based on title, artists, and explicit status
        track_key = (normalized_title, tuple(artist_names), track['explicit'])

        if track_key not in seen_tracks:
            # Get genre from the first artist
            artist_id = track['artists'][0]['id']
            artist_info = sp.artist(artist_id)
            genres = artist_info.get('genres', ['N/A'])

            versions.append({
                'Title': track['name'],
                'Artist(s)': ", ".join([artist['name'] for artist in track['artists']]),
                'Genre': ", ".join(genres) if genres else 'N/A',
                'Explicit': track['explicit'],
                'Track ID': track['id']
            })
            seen_tracks.add(track_key)

    return versions

def main():
    """
    Main function to run the script.
    """
    song_list = ["bad guy", "Blinding Lights", "Dance Monkey"]  # Example song list
    all_songs_data = []

    for song in song_list:
        print(f"Searching for '{song}'...")
        versions = get_song_versions(song)
        if versions:
            all_songs_data.extend(versions)
        else:
            print(f"Could not find any versions for '{song}'")

    if not all_songs_data:
        print("No song data found.")
        return

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(all_songs_data)
    df.index = df.index + 1  # Start numbering from 1
    df.index.name = "No."
    df.to_csv("explicit_songs.csv")

    print("\nSuccessfully created 'explicit_songs.csv' with the following data:")
    print(df)


if __name__ == "__main__":
    main()