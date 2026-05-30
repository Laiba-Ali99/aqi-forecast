import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
LAT     = os.getenv("LAT", "24.8607")
LON     = os.getenv("LON", "67.0011")
CITY    = os.getenv("CITY", "Karachi")
RAW_DATA_PATH = "data/raw/aqi_data.csv"


def fetch_air_pollution(lat: float, lon: float, api_key: str) -> dict:
    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={api_key}"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    if "list" not in data or not data["list"]:
        raise ValueError(f"Invalid API response: {data}")

    entry      = data["list"][0]
    components = entry["components"]

    return {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "aqi":      entry["main"]["aqi"],
        "co":       components.get("co",    0.0),
        "no2":      components.get("no2",   0.0),
        "o3":       components.get("o3",    0.0),
        "pm2_5":    components.get("pm2_5", 0.0),
        "pm10":     components.get("pm10",  0.0),
        "so2":      components.get("so2",   0.0),
    }


def append_to_csv(row: dict, path: str = RAW_DATA_PATH) -> pd.DataFrame:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_df = pd.DataFrame([row])
    if os.path.exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["datetime"], keep="last")
    else:
        combined = new_df
    combined.to_csv(path, index=False)
    return combined


def run():
    if not API_KEY:
        print("ERROR: API_KEY not set in .env file.")
        return
    print(f"Fetching AQI data for {CITY}...")
    try:
        row = fetch_air_pollution(float(LAT), float(LON), API_KEY)
        df  = append_to_csv(row)
        print(f"Saved. Total records: {len(df)}")
        print(f"  PM2.5 = {row['pm2_5']} | AQI = {row['aqi']}")
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
    except ValueError as e:
        print(f"Data error: {e}")


if __name__ == "__main__":
    run()
