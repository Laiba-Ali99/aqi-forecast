# 🌫️ AQI Forecasting System (Karachi)

## 📌 Overview
End-to-end machine learning system that forecasts **72-hour PM2.5 air quality levels** for Karachi using real-time API data, feature engineering, and ML models with full automation and explainability.

---

## ⚙️ Features
- 🌍 Real-time AQI data collection (OpenWeatherMap API)
- 🧠 ML-based 72-hour PM2.5 forecasting
- 🔄 Automated data + retraining pipeline (GitHub Actions)
- 📊 Interactive Streamlit dashboard
- ⚡ FastAPI REST backend
- 🔍 Explainable AI using SHAP
- 🚨 AQI-based health alert system

---

## 🧠 Tech Stack
Python, Pandas, NumPy  
Scikit-learn, SHAP  
FastAPI, Uvicorn  
Streamlit, Plotly  
GitHub Actions  

---

## 📊 Project Pipeline
Data Collection → Feature Engineering → Model Training → Forecasting → Dashboard + API

---

## 🚀 How to Run

```bash id="run1"
pip install -r requirements.txt

# Step 1: Collect data
python fetch_data.py

# Step 2: Build features
python build_features.py

# Step 3: Train model
python train_model.py

# Step 4: Run API
uvicorn main:app --reload

# Step 5: Run dashboard
streamlit run app.py
