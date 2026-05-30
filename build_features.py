import pandas as pd
import numpy as np
import os

RAW_PATH       = "data/raw/aqi_data.csv"
PROCESSED_PATH = "data/processed/features.csv"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Basic time features
    df["hour"]        = df["datetime"].dt.hour
    df["day"]         = df["datetime"].dt.day
    df["month"]       = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)

    # Cyclical encodings (tells model that hour 23 is close to hour 0)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]        / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]       / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Lag features (past values as predictors)
    df["pm25_lag_1"] = df["pm2_5"].shift(1)
    df["pm25_lag_3"] = df["pm2_5"].shift(3)
    df["pm25_lag_6"] = df["pm2_5"].shift(6)

    # Rolling statistics
    df["pm25_rolling_mean_3"] = df["pm2_5"].rolling(3, min_periods=1).mean()
    df["pm25_rolling_mean_6"] = df["pm2_5"].rolling(6, min_periods=1).mean()
    df["pm25_rolling_std_3"]  = df["pm2_5"].rolling(3, min_periods=1).std().fillna(0)
    df["pm25_rolling_std_6"]  = df["pm2_5"].rolling(6, min_periods=1).std().fillna(0)
    df["pm25_rolling_min_6"]  = df["pm2_5"].rolling(6, min_periods=1).min()
    df["pm25_rolling_max_6"]  = df["pm2_5"].rolling(6, min_periods=1).max()

    # Change rate and momentum
    df["pm25_change"]     = df["pm2_5"].diff().fillna(0)
    df["pm25_change_pct"] = df["pm2_5"].pct_change().fillna(0).replace([np.inf,-np.inf],0)
    df["pm25_accel"]      = df["pm25_change"].diff().fillna(0)

    # Pollutant composite
    pollutants = ["co", "no2", "o3", "so2", "pm10"]
    available  = [p for p in pollutants if p in df.columns]
    df["pollutant_composite"] = df[available].apply(
        lambda row: row.fillna(0).mean(), axis=1)

    # Interaction features
    df["pm25_x_hour"]    = df["pm2_5"] * df["hour_sin"]
    df["pm25_x_weekend"] = df["pm2_5"] * df["is_weekend"]
    df["no2_o3_ratio"]   = df["no2"] / (df["o3"] + 0.001)

    # Target: next PM2.5 value (what we want to predict)
    df["target"] = df["pm2_5"].shift(-1)
    df = df.dropna(subset=["target"])
    df = df.fillna(df.median(numeric_only=True))
    return df


def run():
    if not os.path.exists(RAW_PATH):
        print(f"ERROR: {RAW_PATH} not found. Run fetch_data.py first.")
        return
    raw = pd.read_csv(RAW_PATH)
    print(f"Raw rows: {len(raw)}")
    features = engineer_features(raw)
    print(f"Feature rows: {len(features)} | Columns: {len(features.columns)}")
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    features.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved to {PROCESSED_PATH}")


if __name__ == "__main__":
    run()
