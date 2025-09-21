# track_test.py
import os
import requests
import pytest
from dotenv import load_dotenv

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "track-analysis.p.rapidapi.com")

# You can override this in your env if you want a specific track
TEST_SPOTIFY_ID = os.getenv("TEST_SPOTIFY_ID", "7s25THrKz86DM225dOYwnr")

@pytest.mark.skipif(not RAPIDAPI_KEY, reason="RAPIDAPI_KEY not set")
def test_rapidapi_track_analysis_by_spotify_id_smoke():
    """
    Calls the RapidAPI endpoint shown in your screenshot:
    GET https://track-analysis.p.rapidapi.com/pktx/spotify/{spotify_id}
    Asserts a 200 and that tempo-ish info is present.
    """
    url = f"https://{RAPIDAPI_HOST}/pktx/spotify/{TEST_SPOTIFY_ID}"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
        "accept": "application/json",
    }
    r = requests.get(url, headers=headers, timeout=20)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
    data = r.json()
    # Provider field names vary; accept any of these for tempo:
    tempo_like_keys = {"bpm", "tempo", "beatsperminute", "beatsPerMinute"}
    has_tempo = any(k in data for k in tempo_like_keys)
    assert has_tempo, f"Expected one of {tempo_like_keys} in response keys: {list(data.keys())[:15]}"

@pytest.mark.skipif(not RAPIDAPI_KEY, reason="RAPIDAPI_KEY not set")
def test_response_has_basic_analysis_fields():
    """
    Looser check: at least one of (key, mode, energy, danceability, valence)
    so our feature-mapper has something to work with.
    """
    url = f"https://{RAPIDAPI_HOST}/pktx/spotify/{TEST_SPOTIFY_ID}"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
        "accept": "application/json",
    }
    r = requests.get(url, headers=headers, timeout=20)
    data = r.json()
    expected_any = {"key", "mode", "energy", "danceability", "valence", "happiness", "intensity"}
    assert any(k in data for k in expected_any), f"No expected analysis fields found. Keys: {list(data.keys())[:20]}"
