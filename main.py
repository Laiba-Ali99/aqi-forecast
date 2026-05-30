from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd, json, os
from predict      import predict_72h
from alert_engine import check_alerts, summarize_alerts

app = FastAPI(title="AQI Forecast API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

FEATURES_PATH = "data/processed/features.csv"
METRICS_PATH  = "models/metrics.json"
CITY          = os.getenv("CITY", "Karachi")


@app.get("/")
def home():
    return {"message": "AQI Forecast API running", "city": CITY, "docs": "/docs"}


@app.get("/forecast")
def forecast():
    try:
        predictions = predict_72h()
        alerts      = check_alerts(predictions)
        return {"city": CITY, "predictions": predictions,
                "alerts": alerts, "alert_summary": summarize_alerts(alerts)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/{day}")
def forecast_day(day: int):
    if day not in [1, 2, 3]:
        raise HTTPException(400, "Day must be 1, 2, or 3")
    predictions = predict_72h()
    day_preds   = predictions[(day-1)*24 : day*24]
    return {"city": CITY, "day": day, "predictions": day_preds,
            "alerts": check_alerts(day_preds),
            "avg_pm25":  round(sum(p["pm2_5"] for p in day_preds)/len(day_preds),2),
            "peak_pm25": round(max(p["pm2_5"] for p in day_preds), 2)}


@app.get("/current")
def current():
    df  = pd.read_csv(FEATURES_PATH)
    row = df.iloc[-1].to_dict()
    return {"city": CITY, "latest": {k: v for k,v in row.items() if k!="target"}}


@app.get("/metrics")
def metrics():
    with open(METRICS_PATH) as f: return json.load(f)
