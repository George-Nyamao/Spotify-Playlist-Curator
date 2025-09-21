from dotenv import load_dotenv
import os
import requests

# Load .env
load_dotenv()

host = os.getenv("RAPIDAPI_HOST", "track-analysis.p.rapidapi.com")
key  = os.getenv("RAPIDAPI_KEY")
tid  = os.getenv("TEST_SPOTIFY_ID", "6iEvECKDbtcbfbTLNoQGe1")

url = f"https://{host}/pktx/spotify/{tid}"
headers = {"x-rapidapi-key": key, "x-rapidapi-host": host, "accept": "application/json"}
r = requests.get(url, headers=headers, timeout=20)

print("URL:", url)
print("Status:", r.status_code)
print("Body:", r.text[:400])
