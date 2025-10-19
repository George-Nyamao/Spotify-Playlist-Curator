import requests
import os
from dotenv import load_dotenv

load_dotenv()

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
LASTFM_BASE_URL = "http://ws.audioscrobbler.com/2.0/"

def get_simple_genres(artist_name, track_name=None):
    """
    Simple function to get genres for artist or track
    Returns a list of genre strings
    """
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
        response = requests.get(LASTFM_BASE_URL, params=params)
        data = response.json()
        
        if 'toptags' in data and 'tag' in data['toptags']:
            genres = [tag['name'].lower() for tag in data['toptags']['tag'][:3]]
            if genres:
                return genres
    
    # Fall back to artist genres
    params.update({
        'method': 'artist.getTopTags',
        'artist': artist_name
    })
    
    response = requests.get(LASTFM_BASE_URL, params=params)
    data = response.json()
    
    if 'toptags' in data and 'tag' in data['toptags']:
        return [tag['name'].lower() for tag in data['toptags']['tag'][:3]]
    
    return []

def is_rnb_or_hiphop(genres_list):
    """
    Check if genres contain R&B or Hip-Hop
    """
    rnb_indicators = ['r&b', 'rnb', 'rhythm and blues', 'soul', 'urban']
    hiphop_indicators = ['hip hop', 'hip-hop', 'rap', 'trap']
    
    all_indicators = rnb_indicators + hiphop_indicators
    
    for genre in genres_list:
        if any(indicator in genre for indicator in all_indicators):
            return True
    return False

# Test it immediately
def quick_test():
    test_cases = [
        ("Beyoncé", "Crazy in Love"),
        ("Leon Thomas", "MUTT"),
        ("Kendrick Lamar", "HUMBLE"), 
        ("SZA", "Kill Bill"),
        ("Frank Ocean", "Thinkin Bout You"),
        ("Summer Walker", "Girls Need Love"),
        ("Daniel Caesar", "Get You"),
        ("Tyler, The Creator", "EARFQUAKE"),
        ("Jhene Aiko", "Triggered"),
        ("Brent Faiyaz", "ALL MINE")
    ]
    
    for artist, track in test_cases:
        genres = get_simple_genres(artist, track)
        is_target_genre = is_rnb_or_hiphop(genres)
        print(f"{artist} - {track}:")
        print(f"  Genres: {genres}")
        print(f"  R&B/Hip-Hop: {is_target_genre}")
        print()

# Run the test
quick_test()