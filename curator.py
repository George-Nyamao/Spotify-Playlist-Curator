import os, json, time, requests, threading
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ===================== CONFIG =====================
load_dotenv()

# Spotify OAuth
SPOTIPY_CLIENT_ID     = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIPY_REDIRECT_URI  = os.getenv("SPOTIPY_REDIRECT_URI")

# RapidAPI
RAPIDAPI_KEY  = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "track-analysis.p.rapidapi.com")
RAPIDAPI_MAX_RPS = float(os.getenv("RAPIDAPI_MAX_RPS", "1"))
# default burst = 1 token to avoid initial burst
RAPIDAPI_BURST = float(os.getenv("RAPIDAPI_BURST", "1"))

# Token-bucket state for RapidAPI throttling
_rb_lock = threading.Lock()
_rb_tokens = 0.0  # start empty to avoid an initial burst
_rb_last = None    # timestamp of last refill

def rapidapi_acquire():
    """Acquire one token from the token-bucket, waiting if necessary.
    Uses RAPIDAPI_MAX_RPS (tokens/second) and RAPIDAPI_BURST (max tokens).
    If RAPIDAPI_MAX_RPS <= 0, doesn't throttle.
    """
    global _rb_tokens, _rb_last
    if RAPIDAPI_MAX_RPS <= 0:
        return

    # fast path: try to get token under lock
    with _rb_lock:
        now = time.time()
        # initialize last timestamp if first use
        if _rb_last is None:
            _rb_last = now

        # refill tokens based on elapsed time
        elapsed = now - _rb_last
        if elapsed > 0:
            refill = elapsed * RAPIDAPI_MAX_RPS
            _rb_tokens = min(RAPIDAPI_BURST, (_rb_tokens or 0.0) + refill)
            _rb_last = now

        if (_rb_tokens or 0.0) >= 1.0:
            _rb_tokens -= 1.0
            return

        # calculate time to wait for next token
        need = (1.0 - (_rb_tokens or 0.0)) / RAPIDAPI_MAX_RPS

    # sleep outside the lock
    if need > 0:
        time.sleep(need)

    # after sleeping, consume one token (update under lock)
    with _rb_lock:
        now = time.time()
        elapsed = now - _rb_last
        if elapsed > 0:
            refill = elapsed * RAPIDAPI_MAX_RPS
            _rb_tokens = min(RAPIDAPI_BURST, (_rb_tokens or 0.0) + refill)
            _rb_last = now

        if _rb_tokens >= 1.0:
            _rb_tokens -= 1.0
            return

        # fallback: consume whatever is left
        _rb_tokens = max(0.0, (_rb_tokens or 0.0) - 1.0)
        return

# Markets / playlist source
MARKET = os.getenv("MARKET", "US")
SOURCE_PLAYLIST_ID = os.getenv("SOURCE_PLAYLIST_ID", "0nqnvBL1fG8EKOXqv1FCIf")  # your 2000–2019 Spotify playlist

# Target playlist (Spotify)
TARGET_PLAYLIST_NAME = os.getenv("TARGET_PLAYLIST_NAME", "R&B 2020s")
TARGET_DESC = os.getenv("TARGET_DESC", "Auto-curated R&B from 2020+ based on my 2000-2019 taste — clean-first, remix-aware")

# Year / size / caps
YEAR_MIN = int(os.getenv("YEAR_MIN", "2020"))
TARGET_SIZE = int(os.getenv("TARGET_SIZE", "120"))
ARTIST_CAP = int(os.getenv("ARTIST_CAP", "2"))

# RapidAPI-driven feature extraction knobs
SOURCE_SAMPLE_N = int(os.getenv("SOURCE_SAMPLE_N", "150"))          # sample size from source to build taste vector
MAX_RAPID_SOURCE_CALLS = int(os.getenv("MAX_RAPID_SOURCE_CALLS", "200"))
MAX_RAPID_CANDIDATE_CALLS = int(os.getenv("MAX_RAPID_CANDIDATE_CALLS", "250"))

# Misc heuristics
REMIX_TERMS = ["remix","radio edit","edit","mix","rework","version","club mix","extended"]
CLEAN_TERMS = ["clean","radio edit","clean edit","clean version"]

# Feature order we'll produce from RapidAPI
AUDIO_KEYS = ["danceability","energy","loudness","speechiness","acousticness",
              "instrumentalness","liveness","valence","tempo"]

# Cache file for RapidAPI payloads
CACHE_PATH = Path(os.getenv("RAPID_CACHE_PATH", "rapid_cache.json"))

# DRY_RUN safety: default true (do not create playlists unless explicitly disabled)
DRY_RUN = os.getenv("DRY_RUN", "1") == "1"

# ===================== AUTH =====================
def check_spotify_credentials():
    """Validate Spotify credentials are configured"""
    missing = []
    if not SPOTIPY_CLIENT_ID:
        missing.append("SPOTIPY_CLIENT_ID")
    if not SPOTIPY_CLIENT_SECRET:
        missing.append("SPOTIPY_CLIENT_SECRET")
    
    if missing:
        print("❌ Missing Spotify credentials in .env file:")
        for cred in missing:
            print(f"   - {cred}")
        print("\n📋 Setup Instructions:")
        print("1. Go to https://developer.spotify.com/dashboard")
        print("2. Create a new app or use existing")
        print("3. Add redirect URI in app settings: http://127.0.0.1:8888/callback")
        print("4. Copy Client ID and Client Secret to your .env file")
        return False
    return True

# Initialize Spotify client
sp = None
if check_spotify_credentials():
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=SPOTIPY_CLIENT_ID,
            client_secret=SPOTIPY_CLIENT_SECRET,
            redirect_uri=SPOTIPY_REDIRECT_URI,
            scope="playlist-read-private playlist-modify-private playlist-modify-public user-library-read",
            cache_path=".cache-spotify",
            show_dialog=True  # Always show login dialog for clarity
        ))
    except Exception as e:
        print(f"❌ Spotify authentication failed: {e}")
        print("Check your credentials and redirect URI configuration.")
        sp = None

# Shared HTTP session
http = requests.Session()
http.headers.update({"Accept": "application/json", "User-Agent": "rb-curator/1.0"})

# ===================== UTILS =====================
def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def year_of(track) -> int:
    d = (track.get("album", {}).get("release_date") or "0000")[:4]
    try:
        return int(d)
    except Exception:
        return 0

def is_remix_title(name: str) -> bool:
    n = name.lower()
    return any(term in n for term in REMIX_TERMS)

def is_clean_title(name: str) -> bool:
    n = name.lower()
    return any(term in n for term in CLEAN_TERMS)

def explicit_flag_spotify(track) -> bool:
    return bool(track.get("explicit"))

def duration_bonus(ms: int) -> float:
    # favor ~2–4m
    return 1.0 if 150_000 <= ms <= 255_000 else 0.6 if 120_000 <= ms <= 330_000 else 0.0

def remix_bonus(name: str) -> float:
    n = name.lower()
    if "radio edit" in n:
        return 0.25
    if any(t in n for t in REMIX_TERMS):
        return 0.15
    return 0.0

def spotify_me():
    me = sp.current_user()
    print("Spotify user:", me["id"])
    return me

# ===================== CACHE =====================
def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_cache(cache: dict):
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

# ===================== SPOTIFY LOAD =====================
def get_playlist_tracks(pid):
    items = []
    results = sp.playlist_items(pid, market=MARKET, additional_types=["track"])
    items += results.get("items", [])
    while results.get("next"):
        results = sp.next(results)
        items += results.get("items", [])
    tracks = []
    for it in items:
        t = it.get("track")
        if t and t.get("id"):
            tracks.append(t)
    return tracks

# ===================== RAPIDAPI ANALYSIS =====================
rapid_calls_made_source = 0
rapid_calls_made_candidates = 0

def feature_row_from_rapid(payload):
    """
    Map RapidAPI response to our AUDIO_KEYS order:
    ["danceability","energy","loudness","speechiness","acousticness",
     "instrumentalness","liveness","valence","tempo"]

    RapidAPI typical fields: energy, danceability, happiness(≈valence),
    acousticness, instrumentalness, liveness, speechiness (0–100 ints),
    tempo (bpm), loudness like "-4 dB"
    """
    if not payload or not isinstance(payload, dict):
        return None

    def pct(name):
        v = payload.get(name)
        if v is None:
            return None
        try:
            return float(v) / 100.0
        except Exception:
            return None

    def parse_loudness_db(x):
        if x is None:
            return None
        try:
            if isinstance(x, (int, float)):
                return float(x)
            s = str(x).lower().replace("db", "").strip()
            return float(s)
        except Exception:
            return None

    row = [
        pct("danceability"),
        pct("energy"),
        parse_loudness_db(payload.get("loudness")),
        pct("speechiness"),
        pct("acousticness"),
        pct("instrumentalness"),
        pct("liveness"),
        pct("happiness"),  # treat "happiness" as valence
        float(payload["tempo"]) if payload.get("tempo") is not None else None,
    ]
    if all(v is None for v in row):
        return None
    return row

def analyze_track_via_rapidapi(spotify_id: str, cache: dict, is_candidate: bool):
    """
    GET https://track-analysis.p.rapidapi.com/pktx/spotify/{spotify_id}
    Uses on-disk cache; separate budgets for source vs candidates.
    """
    global rapid_calls_made_source, rapid_calls_made_candidates

    cache_key = f"id:{spotify_id}"
    if cache_key in cache:
        return cache[cache_key]

    # enforce budgets
    if is_candidate:
        if rapid_calls_made_candidates >= MAX_RAPID_CANDIDATE_CALLS:
            return None
    else:
        if rapid_calls_made_source >= MAX_RAPID_SOURCE_CALLS:
            return None

    if not RAPIDAPI_KEY:
        return None

    url = f"https://{RAPIDAPI_HOST}/pktx/spotify/{spotify_id}"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
        "accept": "application/json",
    }

    # throttle via token-bucket before sending request
    rapidapi_acquire()
    # perform request with retries on 429/5xx
    max_attempts = 4
    attempt = 0
    r = None
    while attempt < max_attempts:
        attempt += 1
        rapidapi_acquire()
        try:
            r = http.get(url, headers=headers, timeout=(5, 20))
        except Exception as e:
            print(f"RapidAPI request failed (attempt {attempt}):", e)
            # backoff and retry
            time.sleep(min(2 ** attempt, 10))
            continue

        if r.status_code == 200:
            break
        if r.status_code == 429:
            # respect Retry-After header if present
            ra = r.headers.get("Retry-After")
            try:
                wait = float(ra) if ra is not None else min(2 ** attempt, 10)
            except Exception:
                wait = min(2 ** attempt, 10)
            print(f"RapidAPI 429: backing off for {wait}s (attempt {attempt})")
            time.sleep(wait)
            continue
        if 500 <= r.status_code < 600:
            wait = min(2 ** attempt, 10)
            print(f"RapidAPI server error {r.status_code}; retry {attempt} after {wait}s")
            time.sleep(wait)
            continue
        # other status codes: stop retrying
        break

    if r.status_code == 200:
        payload = r.json()
        cache[cache_key] = payload
        # update and persist counters in cache metadata
        if is_candidate:
            rapid_calls_made_candidates += 1
        else:
            rapid_calls_made_source += 1
        cache.setdefault("__meta__", {})["rapid_calls_made_source"] = rapid_calls_made_source
        cache.setdefault("__meta__", {})["rapid_calls_made_candidates"] = rapid_calls_made_candidates
        save_cache(cache)
        return payload
    elif r.status_code == 404:
        # Not found; cache tombstone to avoid re-tries
        cache[cache_key] = {"__not_found__": True}
        cache.setdefault("__meta__", {})["rapid_calls_made_source"] = rapid_calls_made_source
        cache.setdefault("__meta__", {})["rapid_calls_made_candidates"] = rapid_calls_made_candidates
        save_cache(cache)
    else:
        print(f"RapidAPI error {r.status_code}: {r.text[:200]}")
    return None

# ===================== TASTE VECTOR (RapidAPI-only) =====================
def taste_vector_from_source_tracks(src_tracks, rapid_cache):
    """
    Build scaler+centroid from a sample of the source playlist using RapidAPI.
    No Spotify audio-features calls are made.
    """
    sample = src_tracks[:SOURCE_SAMPLE_N] if len(src_tracks) > SOURCE_SAMPLE_N else src_tracks
    print(f"Building taste vector from {len(sample)} source tracks (RapidAPI)…")

    rows = []
    for t in sample:
        tid = t["id"]
        payload = analyze_track_via_rapidapi(tid, cache=rapid_cache, is_candidate=False)
        row = feature_row_from_rapid(payload)
        if row:
            rows.append(row)

    if not rows:
        print("No RapidAPI features found for source sample; similarity will be disabled.")
        return None, None, {}

    X = np.array(rows, dtype=float)
    # Replace NaNs per column with column means
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    centroid = Xs.mean(axis=0)
    return scaler, centroid, {}

# ===================== SCORING =====================
def similarity_score(row_vec, scaler, centroid):
    arr = np.array([v if v is not None else np.nan for v in row_vec], dtype=float)
    if np.isnan(arr).any():
        m = np.nanmean(arr)
        arr = np.where(np.isnan(arr), m, arr)
    vs = scaler.transform([arr])[0]
    denom = (np.linalg.norm(vs) * np.linalg.norm(centroid) + 1e-9)
    return float(np.dot(vs, centroid) / denom)

def score_track(track, feat_row, scaler, centroid):
    pop = (track.get("popularity") or 0) / 100.0
    rec = max(0, year_of(track) - YEAR_MIN) / 6.0
    dur = duration_bonus(track.get("duration_ms") or 0)
    rmx = remix_bonus(track["name"])
    sim = 0.0 if (scaler is None or centroid is None or feat_row is None) else similarity_score(feat_row, scaler, centroid)
    return 0.50 * sim + 0.20 * pop + 0.20 * rec + 0.05 * dur + 0.05 * rmx

# ===================== CANDIDATE GENERATION (Spotify) =====================
def top_artists_from_playlist(tracks, k=10):
    counts = Counter()
    for t in tracks:
        for a in t.get("artists", []):
            if a.get("id"):
                counts[a["id"]] += 1
    return [aid for aid, _ in counts.most_common(k)]

def recommend_from_spotify(seed_artists, seed_tracks, rounds=4, limit=80):
    """
    Replacement for Spotify recommendations endpoint.

    Strategy:
    - For each seed artist: gather their top tracks and related artists' top tracks.
    - For each seed track: fetch the track and add top tracks for its primary artist.
    - De-duplicate by track id and filter by YEAR_MIN.

    This avoids the deprecated /v1/recommendations endpoint.
    """
    out = []
    seen = set()

    # helper to add tracks safely
    def add_tracks(tracks):
        for tr in tracks:
            tid = tr.get("id")
            if not tid or tid in seen:
                continue
            if year_of(tr) >= YEAR_MIN:
                out.append(tr)
                seen.add(tid)

    # 1) seed artists: top tracks and related artists' top tracks
    if seed_artists:
        for aid in seed_artists:
            try:
                top = sp.artist_top_tracks(aid, country=MARKET).get("tracks", [])
                add_tracks(top)
            except Exception:
                pass
            try:
                rel = sp.artist_related_artists(aid).get("artists", [])
                for ra in rel[:5]:
                    try:
                        top2 = sp.artist_top_tracks(ra.get("id"), country=MARKET).get("tracks", [])
                        add_tracks(top2)
                    except Exception:
                        pass
            except Exception:
                pass

    # 2) seed tracks: add top tracks from their primary artists
    if seed_tracks:
        for tid in seed_tracks:
            try:
                tr = sp.track(tid)
                artists = tr.get("artists", [])
                if artists:
                    aid = artists[0].get("id")
                    if aid:
                        try:
                            top = sp.artist_top_tracks(aid, country=MARKET).get("tracks", [])
                            add_tracks(top)
                        except Exception:
                            pass
            except Exception:
                pass

    # 3) if still empty, try searching for popular tracks by seed artist names
    if not out and seed_artists:
        for aid in seed_artists:
            try:
                artist = sp.artist(aid)
                name = artist.get("name")
                if not name:
                    continue
                q = f'artist:"{name}" year:{YEAR_MIN}-2030'
                res = sp.search(q, type="track", limit=10, market=MARKET)
                items = res.get("tracks", {}).get("items", [])
                add_tracks(items)
            except Exception:
                pass

    return out

def search_recent_remixes(artist_name, limit=10):
    """
    Search recent tracks for the artist with remix-friendly keywords.
    """
    q = f'artist:"{artist_name}" year:{YEAR_MIN}-2030'
    try:
        res = sp.search(q, type="track", limit=limit, market=MARKET)
        items = res.get("tracks", {}).get("items", [])
        return [t for t in items if is_remix_title(t["name"]) and year_of(t) >= YEAR_MIN]
    except Exception as e:
        print("search_recent_remixes error:", e)
        return []

# ===================== CLEAN-FIRST + DEDUPE =====================
def find_clean_variant_spotify(base_track):
    if not explicit_flag_spotify(base_track):
        return base_track
    name = base_track["name"]
    art = base_track["artists"][0]["name"]
    q = f'track:"{name}" artist:"{art}" year:{YEAR_MIN}-2030'
    try:
        res = sp.search(q, type="track", market=MARKET, limit=12)
        cands = res.get("tracks", {}).get("items", [])
    except Exception:
        cands = []
    for c in cands:
        if (not c.get("explicit")) and (is_clean_title(c["name"]) or "radio edit" in c["name"].lower()):
            return c
    dur = base_track.get("duration_ms") or 0
    for c in cands:
        if (not c.get("explicit")) and abs((c.get("duration_ms") or 0) - dur) < 8000:
            return c
    return base_track

def dedupe_by_name_artist(tracks):
    seen = {}
    ordered = []
    for t in tracks:
        key = (t["name"].split(" - ")[0].strip().lower(), t["artists"][0]["name"].lower())
        if key in seen:
            if (t.get("popularity") or 0) > (seen[key].get("popularity") or 0):
                seen[key] = t
        else:
            seen[key] = t
            ordered.append(t)
    return list(seen.values())

# ===================== SPOTIFY PLAYLIST CREATION =====================
def create_spotify_playlist_and_fill(user_id, title, description, track_uris):
    """
    Create a new Spotify playlist and add tracks to it.
    """
    # Create the playlist
    playlist = sp.user_playlist_create(
        user=user_id,
        name=title,
        public=False,  # private playlist
        description=description
    )
    playlist_id = playlist["id"]
    print(f"Created Spotify playlist: {title} (ID: {playlist_id})")
    
    # Add tracks in batches of 100 (Spotify's limit)
    for chunk in batched(track_uris, 100):
        sp.playlist_add_items(playlist_id, chunk)
    
    return playlist_id

# ===================== MAIN =====================
def main():
    # Check if Spotify client is initialized
    if sp is None:
        print("❌ Cannot proceed without Spotify authentication")
        return
        
    user = spotify_me()
    user_id = user["id"]

    rapid_cache = load_cache()
    # restore counters from cache meta if present
    meta = rapid_cache.get("__meta__", {}) if isinstance(rapid_cache, dict) else {}
    global rapid_calls_made_source, rapid_calls_made_candidates
    rapid_calls_made_source = int(meta.get("rapid_calls_made_source", rapid_calls_made_source))
    rapid_calls_made_candidates = int(meta.get("rapid_calls_made_candidates", rapid_calls_made_candidates))

    print(f"Loaded RapidAPI cache with {len(rapid_cache)} entries.")
    print(f"RapidAPI calls made - Source: {rapid_calls_made_source}, Candidates: {rapid_calls_made_candidates}")

    # 1) SOURCE PLAYLIST & TASTE
    print("Loading source playlist…")
    src_tracks = get_playlist_tracks(SOURCE_PLAYLIST_ID)
    print(f"Source tracks: {len(src_tracks)}")

    scaler, centroid, _ = taste_vector_from_source_tracks(src_tracks, rapid_cache)

    # 2) CANDIDATE GENERATION (Spotify 2020+)
    print("Generating candidates from Spotify…")
    seed_artists = top_artists_from_playlist(src_tracks, k=12) or [a["id"] for a in src_tracks[0].get("artists", [])]
    seed_track_ids = [t["id"] for t in src_tracks[:10] if t.get("id")]
    candidates = recommend_from_spotify(seed_artists, seed_track_ids, rounds=5, limit=100)

    # also grab some recent remixes for top artists
    for aid in seed_artists[:6]:
        try:
            a = sp.artist(aid)
            candidates += search_recent_remixes(a["name"], limit=10)
        except Exception:
            pass

    # 3) CLEAN-FIRST SWAP + DEDUPE, and filter to 2020+
    print("Selecting clean variants & deduping…")
    clean = []
    for t in candidates:
        if year_of(t) < YEAR_MIN:
            continue
        c = find_clean_variant_spotify(t)
        if not c.get("explicit"):
            clean.append(c)
    clean = dedupe_by_name_artist(clean)
    if not clean:
        print("No clean candidates found; stopping.")
        return

    print(f"Found {len(clean)} clean candidate tracks")

    # 4) FEATURES FOR CANDIDATES (RapidAPI only)
    print("Fetching candidate features via RapidAPI (cached, budgeted)…")
    scored = []
    for i, t in enumerate(clean):
        if i % 20 == 0:
            print(f"  Processing candidate {i+1}/{len(clean)}")
        tid = t["id"]
        payload = analyze_track_via_rapidapi(tid, cache=rapid_cache, is_candidate=True)
        feat_row = feature_row_from_rapid(payload)
        s = score_track(t, feat_row, scaler, centroid)
        scored.append((s, t))
    scored.sort(key=lambda x: x[0], reverse=True)

    # 5) Diversity cap per artist
    out = []
    per_artist = defaultdict(int)
    for s, t in scored:
        art = t["artists"][0]["name"]
        if per_artist[art] >= ARTIST_CAP:
            continue
        per_artist[art] += 1
        out.append(t)
        if len(out) >= TARGET_SIZE:
            break

    print(f"Selected {len(out)} tracks for final playlist")
    
    # Print top 10 tracks for preview
    print("\nTop 10 tracks:")
    for i, t in enumerate(out[:10]):
        artist = t["artists"][0]["name"]
        title = t["name"]
        year = year_of(t)
        print(f"  {i+1:2d}. {artist} - {title} ({year})")

    if not out:
        print("No tracks selected; stopping.")
        return

    # 6) CREATE SPOTIFY PLAYLIST (respect DRY_RUN)
    track_uris = [f"spotify:track:{t['id']}" for t in out]
    
    if DRY_RUN:
        print(f"\n[DRY RUN] Would create Spotify playlist: {TARGET_PLAYLIST_NAME}")
        print(f"[DRY RUN] Would add {len(track_uris)} tracks")
        print(f"[DRY RUN] Set DRY_RUN=0 in your .env file to actually create the playlist")
        return

    playlist_id = create_spotify_playlist_and_fill(
        user_id=user_id,
        title=TARGET_PLAYLIST_NAME,
        description=TARGET_DESC,
        track_uris=track_uris
    )
    
    print(f"\n✅ Created Spotify playlist '{TARGET_PLAYLIST_NAME}' with {len(track_uris)} tracks!")
    print(f"Playlist ID: {playlist_id}")
    print(f"View at: https://open.spotify.com/playlist/{playlist_id}")

if __name__ == "__main__":
    main()
    