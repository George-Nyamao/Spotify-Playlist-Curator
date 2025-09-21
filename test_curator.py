import pytest
from curator import dedupe_by_name_artist

def test_dedupe_by_name_artist_keeps_popular():
    """
    Tests that dedupe_by_name_artist correctly keeps the most popular
    track when a duplicate is found.
    """
    # trackA_less_popular and trackA_more_popular are duplicates by name and artist.
    trackA_less_popular = {
        "name": "Song A",
        "artists": [{"name": "Artist X"}],
        "popularity": 50,
        "id": "a1"
    }
    trackA_more_popular = {
        "name": "Song A",
        "artists": [{"name": "Artist X"}],
        "popularity": 80,
        "id": "a2"
    }
    trackB = {
        "name": "Song B",
        "artists": [{"name": "Artist Y"}],
        "popularity": 60,
        "id": "b1"
    }

    # The more popular track appears after the less popular one.
    tracks_in = [trackA_less_popular, trackB, trackA_more_popular]

    # The buggy function will return [trackA_less_popular, trackB].
    # The fixed function should return [trackA_more_popular, trackB] (or a different order).
    tracks_out = dedupe_by_name_artist(tracks_in)

    # Convert to a set of IDs for easy comparison, ignoring order.
    out_ids = {t["id"] for t in tracks_out}

    assert out_ids == {"a2", "b1"}
    assert len(tracks_out) == 2
