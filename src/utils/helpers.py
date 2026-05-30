import os
from dotenv import load_dotenv

load_dotenv()


def get_env(key: str, default: str = None) -> str:
    val = os.getenv(key, default)
    if val is None:
        raise EnvironmentError(f"Missing required env var: {key}")
    return val


# AQI breakpoints (EPA standard, based on PM2.5 µg/m³)
AQI_BREAKPOINTS = [
    (0,   12.0,  "Good",                   "#00e400", "#155724"),
    (12.1, 35.4, "Moderate",               "#ffff00", "#856404"),
    (35.5, 55.4, "Unhealthy for Sensitive", "#ff7e00", "#7d3c00"),
    (55.5, 150.4,"Unhealthy",               "#ff0000", "#721c24"),
    (150.5,250.4,"Very Unhealthy",           "#8f3f97", "#4a235a"),
    (250.5,500.0,"Hazardous",                "#7e0023", "#4a0010"),
]

POLLUTANT_LIMITS = {
    "pm2_5": 35.4,
    "pm10":  154,
    "no2":   100,
    "o3":    70,
    "so2":   75,
    "co":    10000,
}


def aqi_category(pm25: float) -> dict:
    for lo, hi, label, bg, text in AQI_BREAKPOINTS:
        if lo <= pm25 <= hi:
            return {"label": label, "bg": bg, "text": text}
    return {"label": "Hazardous", "bg": "#7e0023", "text": "#4a0010"}


def is_hazardous(pm25: float) -> bool:
    return pm25 >= 55.5
