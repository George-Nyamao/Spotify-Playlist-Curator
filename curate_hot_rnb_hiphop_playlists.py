import os
import time
import csv
import json
import argparse
from typing import List, Dict, Optional, Tuple

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

# Load dotenv
load_dotenv()

# Get market from env, default to US
MARKET = os.getenv("MARKET", "US")

# Constants
DECADE_LABEL = "2020s"
GENERAL_FMT = "{year} Hot R&B/Hip-Hop"
CLEAN_FMT = "{year} Hot R&B/Hip-Hop (Clean)"
DECADE_GENERAL = f"{DECADE_LABEL} Hot R&B/Hip-Hop"
DECADE_CLEAN = f"{DECADE_LABEL} Hot R&B/Hip-Hop (Clean)"

GENERAL_DESC = "Hot R&B and Hip-Hop for {year}."
CLEAN_DESC = "Hot R&B and Hip-Hop for {year} (mostly clean versions)."

# Spotify Web API limits
ADD_ITEMS_LIMIT = 100  # per request limit for adding items to playlists

# Setup authentication (Authorization Code Flow: user-based)
# This will open a browser for authorization the first time it's run.
# The token will be cached in .spotify_token_cache
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    scope="playlist-modify-public playlist-modify-private",
    redirect_uri="http://127.0.0.1:60000", # Updated to an available port
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    cache_path=".spotify_token_cache" # Cache file for the user's token
))

# -----------------------------
# Utilities
# -----------------------------
def backoff_sleep(attempt: int):
    time.sleep(min(2 ** attempt, 30))


def chunked(iterable: List[str], size: int) -> List[List[str]]:
    return [iterable[i:i + size] for i in range(0, len(iterable), size)]


def curated_csv_path(year: int, input_dir: str) -> str:
    """
    Build path for the curated CSV based on your new naming convention.
    Example: data/2024_Year_End_Hot_100_curated.csv
    """
    fname = f"{year}_Year_End_Hot_100_curated.csv"
    return os.path.join(input_dir, fname)


def read_year_csv(year: int, input_dir: str) -> List[Dict[str, str]]:
    """
    Reads the curated Year-End Hot 100 CSV for a given year.
    Assumes CSV contains at least: title, artist (or track_name, artist_name).
    If 'spotify_track_id' or 'spotify_uri' are present, they are respected.
    """
    path = curated_csv_path(year, input_dir)
    rows: List[Dict[str, str]] = []
    if not os.path.exists(path):
        print(f"Warning: Missing CSV for {year}: {path}")
        return rows

    print(f"Attempting to read CSV: {path}") # Debugging line
    try:
        with open(path, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            print(f"CSV Headers for {year}: {reader.fieldnames}") # Debugging line
            for r in reader:
                rows.append(r)
        print(f"Successfully read {len(rows)} rows from {path}") # Debugging line
    except Exception as e:
        print(f"Error reading CSV {path}: {e}") # Debugging line
        return []
    return rows


# -----------------------------
# Caching layer (title+artist -> URI)
# -----------------------------
def norm_key(title: str, artist: str) -> str:
    return f"{title.strip().lower()}||{artist.strip().lower()}"


def load_cache(cache_path: str) -> Dict[str, Dict[str, str]]:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache_path: str, cache: Dict[str, Dict[str, str]]):
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, cache_path)


# -----------------------------
# Core search/selection logic
# -----------------------------
def search_track(title: str, artist: str, market: Optional[str] = None) -> List[Dict]:
    """
    Search by track+artist. Returns list of track objects (items).
    """
    q = f'track:"{title}" artist:"{artist}"'
    for attempt in range(5):
        try:
            res = sp.search(q=q, type="track", limit=20, market=market)
            return res.get("tracks", {}).get("items", [])
        except spotipy.SpotifyException as e:
            if getattr(e, "http_status", None) == 429:
                retry_after = int(e.headers.get("Retry-After", "2"))
                time.sleep(retry_after)
                continue
            if attempt < 4:
                backoff_sleep(attempt + 1)
                continue
            raise
        except Exception:
            if attempt < 4:
                backoff_sleep(attempt + 1)
                continue
            raise


def is_clean(track_obj: Dict) -> bool:
    return not track_obj.get("explicit", False)


def popularity(track_obj: Dict) -> int:
    return int(track_obj.get("popularity", 0))


def resolve_uris_with_cache(
    title: str,
    artist: str,
    market: Optional[str],
    cache: Dict[str, Dict[str, str]],
    cache_dirty: List[bool]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (best_any_uri, best_clean_or_fallback_uri).
    Caches both under the normalized key.
    """
    key = norm_key(title, artist)
    cached = cache.get(key, {})

    best_any_uri = cached.get("best_any_uri")
    best_clean_or_fallback_uri = cached.get("best_clean_or_fallback_uri")

    if best_any_uri and best_clean_or_fallback_uri:
        return best_any_uri, best_clean_or_fallback_uri

    items = search_track(title=title, artist=artist, market=market)
    if not items:
        cache[key] = {
            "best_any_uri": None,
            "best_clean_or_fallback_uri": None
        }
        cache_dirty[0] = True
        return None, None

    # Most popular overall
    best_any = max(items, key=popularity)
    best_any_uri = best_any.get("uri")

    # Most popular clean if exists, else fallback to best_any
    clean_candidates = [t for t in items if is_clean(t)]
    if clean_candidates:
        best_clean = max(clean_candidates, key=popularity)
        best_clean_or_fallback_uri = best_clean.get("uri")
    else:
        best_clean_or_fallback_uri = best_any_uri

    cache[key] = {
        "best_any_uri": best_any_uri,
        "best_clean_or_fallback_uri": best_clean_or_fallback_uri
    }
    cache_dirty[0] = True
    return best_any_uri, best_clean_or_fallback_uri


# -----------------------------
# Playlist helpers
# -----------------------------
def get_or_create_playlist(user_id: str, name: str, description: str, public: bool = True) -> str:
    """
    Return playlist_id for given name. Create if not found.
    """
    # Search current user's playlists for an exact name match
    for attempt in range(5):
        try:
            results = sp.current_user_playlists(limit=50)
            while results:
                for pl in results.get("items", []):
                    if pl.get("name") == name:
                        return pl.get("id")
                results = sp.next(results) if results.get("next") else None
            break
        except spotipy.SpotifyException as e:
            if getattr(e, "http_status", None) == 429:
                retry_after = int(e.headers.get("Retry-After", "2"))
                time.sleep(retry_after)
                continue
            if attempt < 4:
                backoff_sleep(attempt + 1)
                continue
            raise
        except Exception:
            if attempt < 4:
                backoff_sleep(attempt + 1)
                continue
            raise

    # Create
    for attempt in range(5):
        try:
            created = sp.user_playlist_create(user=user_id, name=name, public=public, description=description)
            return created.get("id")
        except spotipy.SpotifyException as e:
            if getattr(e, "http_status", None) == 429:
                retry_after = int(e.headers.get("Retry-After", "2"))
                time.sleep(retry_after)
                continue
            if attempt < 4:
                backoff_sleep(attempt + 1)
                continue
            raise
        except Exception:
            if attempt < 4:
                backoff_sleep(attempt + 1)
                continue
            raise


def add_tracks_to_playlist(playlist_id: str, uris: List[str]):
    """
    Add tracks in chunks using Spotify API: https://developer.spotify.com/documentation/web-api/reference/add-tracks-to-playlist
    """
    if not uris:
        return
    for batch in chunked(uris, ADD_ITEMS_LIMIT):
        for attempt in range(5):
            try:
                sp.playlist_add_items(playlist_id, batch)
                break
            except spotipy.SpotifyException as e:
                if getattr(e, "http_status", None) == 429:
                    retry_after = int(e.headers.get("Retry-After", "2"))
                    time.sleep(retry_after)
                    continue
                if attempt < 4:
                    backoff_sleep(attempt + 1)
                    continue
                raise
            except Exception:
                if attempt < 4:
                    backoff_sleep(attempt + 1)
                    continue
                raise


# -----------------------------
# Yearly build
# -----------------------------
def build_year_playlists(
    user_id: str,
    year: int,
    rows: List[Dict[str, str]],
    decade_general_pl_id: str,
    decade_clean_pl_id: str,
    market: Optional[str],
    limit: Optional[int],
    dry_run: bool,
    cache: Dict[str, Dict[str, str]],
    cache_dirty: List[bool],
):
    general_name = GENERAL_FMT.format(year=year)
    clean_name = CLEAN_FMT.format(year=year)

    general_desc = GENERAL_DESC.format(year=year)
    clean_desc = CLEAN_DESC.format(year=year)

    general_pl_id = get_or_create_playlist(user_id, general_name, general_desc, public=True)
    clean_pl_id = get_or_create_playlist(user_id, clean_name, clean_desc, public=True)

    general_uris: List[str] = []
    clean_uris: List[str] = []

    count = 0
    for r in rows:
        # Corrected to use exact CSV header names
        title = r.get("Title")
        artist = r.get("Artist(s)")

        if not title or not artist:
            print(f"Skipping row due to missing title or artist: {r}") # Debugging line
            continue

        if limit and count >= limit:
            break

        # If CSV already has a Spotify ID/URI, prefer it to avoid search
        csv_uri = None
        if r.get("spotify_uri"):
            csv_uri = r["spotify_uri"]
        elif r.get("spotify_track_id"):
            csv_uri = f"spotify:track:{r['spotify_track_id']}"

        if csv_uri:
            general_uri = csv_uri
            # For clean playlist, still resolve clean-or-fallback via search+cache unless you prefer to reuse csv_uri
            _, clean_uri = resolve_uris_with_cache(title, artist, market, cache, cache_dirty)
            clean_uri = clean_uri or csv_uri
        else:
            general_uri, clean_uri = resolve_uris_with_cache(title, artist, market, cache, cache_dirty)

        if general_uri:
            general_uris.append(general_uri)
        if clean_uri:
            clean_uris.append(clean_uri)

        count += 1

    if dry_run:
        print(f"[DRY-RUN] {year}: +{len(general_uris)} -> {general_name}")
        print(f"[DRY-RUN] {year}: +{len(clean_uris)} -> {clean_name}")
        print(f"[DRY-RUN] {year}: +{len(general_uris)} -> {DECADE_GENERAL}")
        print(f"[DRY-RUN] {year}: +{len(clean_uris)} -> {DECADE_CLEAN}")
        return

    if general_uris:
        add_tracks_to_playlist(general_pl_id, general_uris)
        add_tracks_to_playlist(decade_general_pl_id, general_uris)

    if clean_uris:
        add_tracks_to_playlist(clean_pl_id, clean_uris)
        add_tracks_to_playlist(decade_clean_pl_id, clean_uris)

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Curate yearly and decade Hot R&B/Hip-Hop playlists.")
    parser.add_argument("--years", nargs="+", type=int, required=True, help="Years to process, e.g. 2020 2021 2022 2023 2024")
    parser.add_argument("--input-dir", type=str, default="data", help="Directory with per-year CSVs")
    parser.add_argument("--owner-user-id", type=str, required=True, help="Spotify user ID that owns the playlists")
    parser.add_argument("--market", type=str, default=MARKET, help="Market code, e.g., US")
    parser.add_argument("--limit-per-year", type=int, default=None, help="Limit tracks processed per year")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to Spotify")
    parser.add_argument("--cache-path", type=str, default=".playlist_uri_cache.json", help="Path to URI cache file")

    args = parser.parse_args()

    # Load cache
    cache = load_cache(args.cache_path)
    cache_dirty = [False]  # mutable flag to mark if cache changed

    # Ensure decade playlists exist
    decade_general_pl_id = get_or_create_playlist(args.owner_user_id, DECADE_GENERAL, f"Hot R&B and Hip-Hop for the {DECADE_LABEL}.", public=True)
    decade_clean_pl_id = get_or_create_playlist(args.owner_user_id, DECADE_CLEAN, f"Hot R&B and Hip-Hop (mostly clean) for the {DECADE_LABEL}.", public=True)

    for year in args.years:
        rows = read_year_csv(year, args.input_dir)
        if not rows:
            continue

        build_year_playlists(
            user_id=args.owner_user_id,
            year=year,
            rows=rows,
            decade_general_pl_id=decade_general_pl_id,
            decade_clean_pl_id=decade_clean_pl_id,
            market=args.market,
            limit=args.limit_per_year,
            dry_run=args.dry_run,
            cache=cache,
            cache_dirty=cache_dirty,
        )

    # Persist cache if modified
    if cache_dirty[0]:
        save_cache(args.cache_path, cache)


if __name__ == "__main__":
    main()