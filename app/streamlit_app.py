import json
from pathlib import Path
import sys

# Ensure project root is on sys.path so `import src.*` works when running from app/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from src.predict_utils import predict_crop
from src.fert_predict_utils import predict_fertilizer, load_artifacts as load_fert

st.set_page_config(page_title="AgroAssist ML Demo", layout="wide")
st.title("AgroAssist ML – Demo")
st.write("Pick a problem, compare models, and make predictions.")

# Sidebar
st.sidebar.header("Task")
mode = st.sidebar.selectbox("Select task", ["Crop Recommendation", "Fertilizer Recommendation"]) 

repo_root = Path(__file__).resolve().parents[1]
models_dir = repo_root / 'models'
fig_dir = repo_root / 'figures'

if mode == "Crop Recommendation":
    st.subheader("Predict Crop")
    with st.form("crop_form"):
        col1, col2, col3, col4 = st.columns(4)
        N = col1.number_input("N", min_value=0.0, value=90.0)
        P = col2.number_input("P", min_value=0.0, value=40.0)
        K = col3.number_input("K", min_value=0.0, value=40.0)
        temperature = col4.number_input("Temperature (°C)", value=24.5)
        col5, col6, col7 = st.columns(3)
        humidity = col5.number_input("Humidity (%)", value=80.0)
        ph = col6.number_input("pH", value=6.5)
        rainfall = col7.number_input("Rainfall (mm)", value=200.0)
        submitted = st.form_submit_button("Predict Crop")
    if submitted:
        try:
            label = predict_crop({
                "N": N, "P": P, "K": K,
                "temperature": temperature, "humidity": humidity,
                "ph": ph, "rainfall": rainfall,
            }, models_dir=models_dir)
            st.success(f"Recommended crop: {label}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    st.divider()
    st.subheader("Model Evaluation (Crop)")
    # Show existing figures if any
    crop_figs = [
        "crop_model_eval_metrics.csv",
        "report_rf.txt",
        "report_knn.txt",
        "report_svm.txt",
        "confusion_rf.png",
        "confusion_knn.png",
        "confusion_svm.png",
    ]
    for name in crop_figs:
        p = fig_dir / name
        if p.exists():
            if p.suffix == ".png":
                st.image(str(p), caption=name)
            else:
                with st.expander(name):
                    st.code(p.read_text(encoding="utf-8"))

else:
    st.subheader("Predict Fertilizer")
    # Load fertilizer artifacts to ensure availability
    try:
        load_fert(models_dir=models_dir)
    except Exception as e:
        st.warning(f"Note: Could not load fertilizer model yet. Train via notebook or the compare script. Error: {e}")

    with st.form("fert_form"):
        col1, col2, col3 = st.columns(3)
        Temparature = col1.number_input("Temperature (°C)", value=26.0)
        Humidity = col2.number_input("Humidity (%)", value=52.0)
        Moisture = col3.number_input("Moisture", value=38.0)
        col4, col5 = st.columns(2)
        Soil_Type = col4.selectbox("Soil Type", ["Sandy", "Loamy", "Black", "Red", "Clayey"])  # example list
        Crop_Type = col5.selectbox("Crop Type", ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy"])  # example list
        col6, col7, col8 = st.columns(3)
        Nitrogen = col6.number_input("Nitrogen", value=37.0)
        Potassium = col7.number_input("Potassium", value=0.0)
        Phosphorous = col8.number_input("Phosphorous", value=0.0)
        submitted = st.form_submit_button("Predict Fertilizer")
    if submitted:
        try:
            # Match input keys to dataset column names where possible
            x = {
                "Temparature": Temparature,
                "Humidity ": Humidity,
                "Moisture": Moisture,
                "Soil Type": Soil_Type,
                "Crop Type": Crop_Type,
                "Nitrogen": Nitrogen,
                "Potassium": Potassium,
                "Phosphorous": Phosphorous,
            }
            label = predict_fertilizer(x, models_dir=models_dir)
            st.success(f"Recommended fertilizer: {label}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.divider()
    st.subheader("Model Comparison (Fertilizer)")
    # Show comparison metrics and confusion matrices if available
    fert_files = [
        "fert_model_eval_metrics.csv",
        "report_fert_rf.txt",
        "report_fert_knn.txt",
        "report_fert_xgb.txt",
        "confusion_fert_rf.png",
        "confusion_fert_knn.png",
        "confusion_fert_xgb.png",
    ]
    for name in fert_files:
        p = fig_dir / name
        if p.exists():
            if p.suffix == ".png":
                st.image(str(p), caption=name)
            else:
                with st.expander(name):
                    st.code(p.read_text(encoding="utf-8"))

st.caption("Tip: Use src/train_fertilizer_compare.py to train and compare fertilizer models (rf/xgb/knn). Use src/train_crop.py for crop.")
