import os
import sys
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

def get_song_details(title, artist):
    """
    Searches for a song on Spotify using title and artist and returns its details.
    """
    query = f"track:{title} artist:{artist}"
    results = sp.search(q=query, type='track', limit=1)
    tracks = results['tracks']['items']

    if not tracks:
        return {'Genre': 'N/A', 'Explicit': 'N/A', 'Track ID': 'N/A'}

    track = tracks[0]

    # Get genre from the artist
    artist_id = track['artists'][0]['id']
    artist_info = sp.artist(artist_id)
    genres = artist_info.get('genres', ['N/A'])

    return {
        'Genre': ", ".join(genres) if genres else 'N/A',
        'Explicit': track['explicit'],
        'Track ID': track['id']
    }

def main():
    """
    Main function to run the script.
    """
    if len(sys.argv) < 2:
        print("Usage: python Explicit.py <input_csv_file>")
        sys.exit(1)

    input_filename = sys.argv[1]

    try:
        df = pd.read_csv(input_filename)
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
        sys.exit(1)

    song_details = []
    for index, row in df.iterrows():
        title = row['Title']
        artist = row['Artist(s)']
        print(f"Processing '{title}' by '{artist}'...")
        details = get_song_details(title, artist)
        song_details.append(details)

    details_df = pd.DataFrame(song_details)

    # Combine original df with new details
    result_df = pd.concat([df, details_df], axis=1)

    # Generate output filename
    base, ext = os.path.splitext(input_filename)
    output_filename = f"{base}_test{ext}"

    result_df.to_csv(output_filename, index=False)

    print(f"\nSuccessfully created '{output_filename}'")
    print(result_df)

if __name__ == "__main__":
    main()