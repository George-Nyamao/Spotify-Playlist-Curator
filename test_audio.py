# test_audio.py
import os
from dotenv import load_dotenv

load_dotenv()

import Spotipy
from spotipy.oauth2 import SpotifyOAuth

# Set credentials
# Spotify OAuth
SPOTIPY_CLIENT_ID     = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id='SPOTIPY_CLIENT_ID',
    client_secret='SPOTIPY_CLIENT_SECRET'
))

print("current_user id:", sp.current_user().get("id"))

track_id = '6iEvECKDbtcbfbTLNoQGe1'
track = sp.track(track_id)

print(f"Track: {track['name']}")
print(f"Artist: {track['artists'][0]['name']}")
print(f"Album: {track['album']['name']}")
print(f"Release Date: {track['album']['release_date']}")

track_id = '6y0igZArWVi6Iz0rj35c1Y'
track = sp.track(track_id)

print(f"Track: {track['name']}")
print(f"Artist: {track['artists'][0]['name']}")
print(f"Album: {track['album']['name']}")
print(f"Release Date: {track['album']['release_date']}")