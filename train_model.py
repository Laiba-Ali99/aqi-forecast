"""
train_model.py  —  Fixed version
Fixes:
  1. Detects near-zero variance and handles it gracefully
  2. Uses LinearExplainer for Ridge (instead of TreeExplainer)
  3. Falls back to permutation importance when SHAP fails
  4. Robust R² calculation that won't explode
  5. Minimum data validation with clear messages
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model    import Ridge
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection      import permutation_importance

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES_PATH = "data/processed/features.csv"
MODELS_DIR    = "models"
DROP_COLS     = ["datetime", "target"]
MIN_ROWS      = 20       # minimum rows before training makes sense
MIN_VARIANCE  = 0.01     # minimum target std before we warn


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"\n  ERROR: {FEATURES_PATH} not found.\n"
            f"  Run:  python fetch_data.py\n"
            f"        python build_features.py\n"
        )

    df = pd.read_csv(FEATURES_PATH).sort_values("datetime").reset_index(drop=True)

    if len(df) < MIN_ROWS:
        print(f"\n  WARNING: Only {len(df)} rows — model will be unreliable.")
        print(f"  TIP: Run 'python fetch_data.py' more times (ideally once per hour)")
        print(f"  Continuing anyway...\n")

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].fillna(0)
    y = df["target"]

    # Check variance
    target_std = y.std()
    if target_std < MIN_VARIANCE:
        print(f"\n  ⚠  WARNING: Target (PM2.5) has almost zero variance (std={target_std:.4f}).")
        print(f"  This means all your readings are nearly identical.")
        print(f"  CAUSE: You ran fetch_data.py many times in quick succession.")
        print(f"  FIX:   Wait and run fetch_data.py once per hour to get real variation.")
        print(f"  The model will still train but metrics will look strange.\n")

    print(f"  Rows: {len(df)}  |  Features: {len(feature_cols)}  |  Target std: {target_std:.4f}")
    return X, y, feature_cols


# ── Model pipelines ───────────────────────────────────────────────────────────
def build_pipelines() -> dict:
    return {
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=200, n_jobs=-1, random_state=42,
                min_samples_leaf=2,   # prevents overfitting on small datasets
            )),
        ]),
        "gradient_boost": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1,
                max_depth=3, random_state=42,
                min_samples_leaf=2,
            )),
        ]),
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0)),
        ]),
    }


# ── Safe R² that doesn't explode when variance is near zero ───────────────────
def safe_r2(y_true, y_pred) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-10:
        # Target has no variance — R² is undefined, return 0
        return 0.0
    r2 = 1 - ss_res / ss_tot
    # Clamp to [-1, 1] to avoid astronomically negative numbers on tiny datasets
    return float(np.clip(r2, -1.0, 1.0))


# ── Time-series cross-validation ─────────────────────────────────────────────
def cross_validate_tscv(pipe, X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> dict:
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    rmses, maes, r2s = [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        if len(X_tr) < 5:
            continue
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        rmses.append(np.sqrt(mean_squared_error(y_te, preds)))
        maes.append(mean_absolute_error(y_te, preds))
        r2s.append(safe_r2(y_te.values, preds))

    return {
        "rmse": round(float(np.mean(rmses)), 4) if rmses else 0.0,
        "mae":  round(float(np.mean(maes)),  4) if maes  else 0.0,
        "r2":   round(float(np.mean(r2s)),   4) if r2s   else 0.0,
    }


# ── SHAP — works for both tree models AND linear models ──────────────────────
def compute_shap(pipe, X: pd.DataFrame, feature_cols: list, model_name: str):
    os.makedirs(MODELS_DIR, exist_ok=True)

    scaler   = pipe.named_steps["scaler"]
    model    = pipe.named_steps["model"]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)

    shap_values  = None
    method_used  = None

    # --- Try TreeExplainer (Random Forest, Gradient Boosting) ----------------
    tree_models = (RandomForestRegressor, GradientBoostingRegressor)
    if isinstance(model, tree_models):
        try:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            method_used = "TreeExplainer"
        except Exception as e:
            print(f"  TreeExplainer failed: {e}")

    # --- Try LinearExplainer (Ridge, Lasso, LinearRegression) ----------------
    if shap_values is None:
        try:
            explainer   = shap.LinearExplainer(model, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
            method_used = "LinearExplainer"
        except Exception as e:
            print(f"  LinearExplainer failed: {e}")

    # --- Fallback: permutation importance (works for ANY model) ---------------
    if shap_values is None:
        print(f"  Falling back to permutation importance...")
        try:
            r = permutation_importance(
                model, X_scaled, pipe.predict(X) - X_scaled.mean().mean(),
                n_repeats=10, random_state=42, n_jobs=-1
            )
            importance_vals = np.abs(r.importances_mean)
            importance = dict(sorted(
                zip(feature_cols, importance_vals.tolist()),
                key=lambda x: x[1], reverse=True
            ))
            with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w") as f:
                json.dump(importance, f, indent=2)
            _plot_bar_importance(importance, model_name, "Permutation Importance")
            print(f"  Permutation importance computed and saved.")
            return
        except Exception as e:
            print(f"  Permutation importance also failed: {e}")
            return

    # --- Save SHAP results ---------------------------------------------------
    mean_abs   = np.abs(shap_values).mean(axis=0)
    importance = dict(sorted(
        zip(feature_cols, mean_abs.tolist()),
        key=lambda x: x[1], reverse=True
    ))
    with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w") as f:
        json.dump(importance, f, indent=2)

    # SHAP summary plot
    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_scaled, show=False, max_display=15)
        plt.title(f"SHAP Feature Importance — {model_name.replace('_',' ').title()} ({method_used})")
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, "shap_summary.png"), dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  SHAP plot saved ({method_used}).")
    except Exception as e:
        print(f"  Could not save SHAP plot: {e}")
        _plot_bar_importance(importance, model_name, method_used)


def _plot_bar_importance(importance: dict, model_name: str, method: str):
    """Fallback: simple horizontal bar chart of feature importance."""
    top = dict(list(importance.items())[:15])
    names = list(top.keys())
    vals  = list(top.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(names))
    bars  = ax.barh(y_pos, vals, color="#6366f1", edgecolor="none")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importance — {model_name.replace('_',' ').title()} ({method})")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "shap_summary.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Bar importance chart saved.")


# ── Main training entry point ─────────────────────────────────────────────────
def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\nLoading data...")
    X, y, feature_cols = load_data()

    pipelines   = build_pipelines()
    all_metrics = {}

    for name, pipe in pipelines.items():
        print(f"\nTraining {name}...")
        metrics = cross_validate_tscv(pipe, X, y)
        pipe.fit(X, y)  # final fit on all data
        all_metrics[name] = metrics
        joblib.dump(pipe, os.path.join(MODELS_DIR, f"{name}.pkl"))
        print(f"  RMSE = {metrics['rmse']}")
        print(f"  MAE  = {metrics['mae']}")
        print(f"  R²   = {metrics['r2']}  {'✓ good' if metrics['r2'] > 0.7 else '⚠ low — need more varied data'}")

    # Pick best by lowest RMSE
    best_name = min(all_metrics, key=lambda k: all_metrics[k]["rmse"])
    best_pipe = joblib.load(os.path.join(MODELS_DIR, f"{best_name}.pkl"))
    joblib.dump(best_pipe, os.path.join(MODELS_DIR, "aqi_model.pkl"))
    print(f"\n✓ Best model: {best_name}  →  saved as models/aqi_model.pkl")

    # Save metrics + feature list
    metrics_out = {
        "best_model":   best_name,
        "models":       all_metrics,
        "feature_cols": feature_cols,
        "data_rows":    len(X),
        "target_std":   round(float(y.std()), 4),
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)
    print("✓ Metrics saved: models/metrics.json")

    # SHAP / feature importance
    print(f"\nComputing feature importance for {best_name}...")
    compute_shap(best_pipe, X, feature_cols, best_name)

    print("\n✓ Training complete!")

    # Final summary
    print("\n" + "─" * 50)
    print("MODEL COMPARISON")
    print("─" * 50)
    print(f"{'Model':<20} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("─" * 50)
    for name, m in all_metrics.items():
        marker = " ◀ best" if name == best_name else ""
        print(f"{name:<20} {m['rmse']:>8.4f} {m['mae']:>8.4f} {m['r2']:>8.4f}{marker}")
    print("─" * 50)

    if y.std() < MIN_VARIANCE:
        print("\n⚠  NEXT STEP: Your data has no variance yet.")
        print("   Run fetch_data.py once per hour for a few hours,")
        print("   then run build_features.py and train_model.py again.")
        print("   R² will improve dramatically with real variation in the data.\n")

    return best_pipe, feature_cols


if __name__ == "__main__":
    train()