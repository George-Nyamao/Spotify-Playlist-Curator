# test_audio.py
import os
import pytest
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

# Load credentials from environment
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

# A list of track IDs to test, taken from the original script
TEST_TRACK_IDS = ['6iEvECKDbtcbfbTLNoQGe1', '6y0igZArWVi6Iz0rj35c1Y']

@pytest.mark.skipif(not SPOTIPY_CLIENT_ID or not SPOTIPY_CLIENT_SECRET, reason="Spotify credentials (SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET) not set in environment")
@pytest.mark.parametrize("track_id", TEST_TRACK_IDS)
def test_fetch_spotify_track_data(track_id):
    """
    Tests that the Spotify API can be reached and that track data can be fetched
    for a given track ID using the Client Credentials authentication flow.
    """
    auth_manager = SpotifyClientCredentials(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET
    )
    sp = spotipy.Spotify(auth_manager=auth_manager)

    track = sp.track(track_id)

    # Assert that we got a valid track object back
    assert track is not None, "Track data should not be None"
    assert track['id'] == track_id
    assert 'name' in track and track['name']
    assert 'artists' in track and isinstance(track['artists'], list) and track['artists']
    assert 'album' in track and isinstance(track['album'], dict)
    assert 'release_date' in track['album']
