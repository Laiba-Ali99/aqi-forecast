from src.utils.helpers import aqi_category

THRESHOLDS = {
    "sensitive": 12.1,   # Unhealthy for sensitive groups
    "unhealthy": 35.5,   # Unhealthy for all
    "hazardous": 55.5,   # Very unhealthy and above
}


def check_alerts(predictions: list) -> list:
    """Return alerts for any forecast period above thresholds."""
    alerts = []
    for p in predictions:
        pm25 = p["pm2_5"]
        if pm25 >= THRESHOLDS["hazardous"]:
            severity, icon = "critical", "🔴"
        elif pm25 >= THRESHOLDS["unhealthy"]:
            severity, icon = "warning", "🟠"
        elif pm25 >= THRESHOLDS["sensitive"]:
            severity, icon = "moderate", "🟡"
        else:
            continue
        cat = aqi_category(pm25)
        alerts.append({
            "hour":      p["hour"],
            "timestamp": p["timestamp"],
            "pm2_5":     pm25,
            "severity":  severity,
            "icon":      icon,
            "category":  cat["label"],
            "message":   f"{icon} {cat['label']} at {p.get('time','')} on"
                         f" {p.get('date','')} — PM2.5: {pm25:.1f} µg/m³",
        })
    return alerts


def summarize_alerts(alerts: list) -> dict:
    if not alerts:
        return {"count": 0, "worst": None, "hours_unhealthy": 0}
    worst = max(alerts, key=lambda a: a["pm2_5"])
    return {
        "count":           len(alerts),
        "worst":           worst,
        "hours_unhealthy": len([a for a in alerts if a["severity"] in ("warning","critical")]),
        "hours_critical":  len([a for a in alerts if a["severity"] == "critical"]),
    }
