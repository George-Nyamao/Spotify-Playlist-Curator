from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

print("SPOTIPY_CLIENT_ID:", os.getenv("SPOTIPY_CLIENT_ID"))
print("SPOTIPY_CLIENT_SECRET:", "set" if os.getenv("SPOTIPY_CLIENT_SECRET") else "missing")
print("SPOTIPY_REDIRECT_URI:", os.getenv("SPOTIPY_REDIRECT_URI"))

print("RAPIDAPI_KEY:", "set" if os.getenv("RAPIDAPI_KEY") else "missing")
print("RAPIDAPI_HOST:", os.getenv("RAPIDAPI_HOST"))
