import spotipy
import os
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# Load dotenv
load_dotenv()

# Get market from env, default to US
MARKET = os.getenv("MARKET", "US")

# Setup authentication
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")
))

def find_relevant_song_versions(track_name, artist_name, market=None):
    """
    Finds different versions of a song, filtering for relevant results
    and sorting by popularity.
    """
    # Create a search query
    query = f'track:"{track_name}" artist:"{artist_name}"'
    # Use market to improve relevance
    results = sp.search(q=query, type='track', limit=50, market=market)
    tracks = results['tracks']['items']
    
    relevant_versions = []
    
    for track in tracks:
        # Get the main artist (first artist listed)
        main_artist = track['artists'][0]['name'] if track['artists'] else ''
        
        # FILTER 1: Check if the main artist matches the query
        if main_artist.lower() != artist_name.lower():
            continue
            
        # FILTER 2: Check if the track name starts with or closely matches the query
        track_name_lower = track['name'].lower()
        search_name_lower = track_name.lower()
        
        # Accept tracks that start with the search term or contain it as a distinct word
        if not (track_name_lower.startswith(search_name_lower) or 
                f" {search_name_lower} " in f" {track_name_lower} "):
            continue
        
        # Analyze the track name to categorize the version
        name_lower = track['name'].lower()
        is_remix = 'remix' in name_lower
        is_live = 'live' in name_lower or 'tiny desk' in name_lower
        is_acoustic = 'acoustic' in name_lower
        is_sped_up = 'sped up' in name_lower or 'chopped' in name_lower
        is_edit = 'edit' in name_lower
        is_original = not (is_remix or is_live or is_acoustic or is_edit or is_sped_up)
        
        # Create version descriptor
        version_type = "Original"
        if is_remix:
            version_type = "Remix"
        elif is_live:
            version_type = "Live"
        elif is_acoustic:
            version_type = "Acoustic"
        elif is_sped_up:
            version_type = "Alt Version"
        elif is_edit:
            version_type = "Edit"
        
        # Store relevant track info, including popularity
        relevant_versions.append({
            'id': track['id'],
            'name': track['name'],
            'artist': main_artist,
            'album': track['album']['name'],
            'explicit': track['explicit'],
            'popularity': track.get('popularity', 0), # Include popularity
            'version_type': version_type,
            'url': track['external_urls']['spotify'],
            'release_date': track['album']['release_date']
        })
    
    # Sort by popularity (descending)
    relevant_versions.sort(key=lambda x: x['popularity'], reverse=True)

    return relevant_versions

# Example usage
versions = find_relevant_song_versions("MUTT", "Leon Thomas", market=MARKET)

print(f"Found {len(versions)} relevant versions of MUTT by Leon Thomas (sorted by popularity):")
print("=" * 80)

for v in versions:
    if v['popularity'] < 50:  # Filter out very low popularity tracks
        continue
    print(f"Name: {v['name']}")
    print(f"Album: {v['album']}")
    print(f"Type: {v['version_type']}")
    print(f"Popularity: {v['popularity']}") # Print popularity
    print(f"Explicit: {v['explicit']}")
    print(f"URL: {v['url']}")
    print(f"Release Date: {v['release_date']}")
    print("-" * 60)