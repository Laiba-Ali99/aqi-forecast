import pandas as pd, numpy as np, joblib, json, os
from datetime import datetime, timedelta

MODEL_PATH    = "models/aqi_model.pkl"
FEATURES_PATH = "data/processed/features.csv"
METRICS_PATH  = "models/metrics.json"
DROP_COLS     = ["datetime", "target"]


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found. Run train_model.py first.")
    model        = joblib.load(MODEL_PATH)
    feature_cols = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            feature_cols = json.load(f).get("feature_cols")
    df = pd.read_csv(FEATURES_PATH).sort_values("datetime").reset_index(drop=True)
    return model, df, feature_cols


def build_future_row(base_row, h, prev_pm25, rolling_window):
    future_time = datetime.now() + timedelta(hours=h)
    row = base_row.copy()
    row["hour"]        = future_time.hour
    row["day"]         = future_time.day
    row["month"]       = future_time.month
    row["day_of_week"] = future_time.weekday()
    row["is_weekend"]  = int(future_time.weekday() >= 5)
    row["hour_sin"]    = np.sin(2 * np.pi * future_time.hour  / 24)
    row["hour_cos"]    = np.cos(2 * np.pi * future_time.hour  / 24)
    row["month_sin"]   = np.sin(2 * np.pi * future_time.month / 12)
    row["month_cos"]   = np.cos(2 * np.pi * future_time.month / 12)
    row["dow_sin"]     = np.sin(2 * np.pi * future_time.weekday() / 7)
    row["dow_cos"]     = np.cos(2 * np.pi * future_time.weekday() / 7)
    row["pm25_lag_1"]  = rolling_window[-1] if rolling_window else prev_pm25
    row["pm25_lag_3"]  = rolling_window[-3] if len(rolling_window) >= 3 else prev_pm25
    row["pm25_lag_6"]  = rolling_window[-6] if len(rolling_window) >= 6 else prev_pm25
    row["pm25_rolling_mean_3"] = np.mean(rolling_window[-3:]) if rolling_window else prev_pm25
    row["pm25_rolling_mean_6"] = np.mean(rolling_window[-6:]) if rolling_window else prev_pm25
    row["pm25_rolling_std_3"]  = np.std(rolling_window[-3:]) if len(rolling_window)>=2 else 0
    row["pm25_rolling_std_6"]  = np.std(rolling_window[-6:]) if len(rolling_window)>=2 else 0
    row["pm25_rolling_min_6"]  = min(rolling_window[-6:]) if rolling_window else prev_pm25
    row["pm25_rolling_max_6"]  = max(rolling_window[-6:]) if rolling_window else prev_pm25
    row["pm25_change"]  = prev_pm25-(rolling_window[-2] if len(rolling_window)>=2 else prev_pm25)
    row["pm25_x_hour"]   = prev_pm25 * row["hour_sin"]
    row["pm25_x_weekend"]= prev_pm25 * row["is_weekend"]
    return row


def predict_72h() -> list:
    model, df, feature_cols = load_artifacts()
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in DROP_COLS]
    base_row       = df.iloc[-1].copy()
    current_pm25   = float(base_row["pm2_5"])
    rolling_window = df["pm2_5"].tail(24).tolist()
    predictions    = []
    for h in range(1, 73):
        row       = build_future_row(base_row, h, current_pm25, rolling_window)
        available = [c for c in feature_cols if c in row.index]
        X_row     = pd.DataFrame([row[available].fillna(0)])
        pm25_pred = max(0.0, float(model.predict(X_row)[0]))
        future_time = datetime.now() + timedelta(hours=h)
        predictions.append({
            "hour":      h,
            "timestamp": future_time.strftime("%Y-%m-%d %H:%M"),
            "date":      future_time.strftime("%b %d"),
            "time":      future_time.strftime("%H:%M"),
            "pm2_5":     round(pm25_pred, 2),
            "day":       (h - 1) // 24 + 1,
        })
        rolling_window.append(pm25_pred)
        current_pm25 = pm25_pred
    return predictions


if __name__ == "__main__":
    preds = predict_72h()
    for p in preds[:10]:  # show first 10 hours
        print(f"Hour {p['hour']:>2}: {p['timestamp']}  PM2.5={p['pm2_5']}")
    print(f"Peak={max(p['pm2_5'] for p in preds):.2f}  ",
          f"Min={min(p['pm2_5'] for p in preds):.2f}  ",
          f"Avg={sum(p['pm2_5'] for p in preds)/72:.2f}")
