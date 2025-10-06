# AgroAssist-ML

Mini-project: Crop and Fertilizer Recommendation.

## What’s included
- Crop pipeline (train/evaluate/infer)
- Fertilizer pipeline (train/evaluate/infer) with multi-model comparison
- Basic Streamlit UI demo for both tasks

## Quickstart

1) Create and activate venv (Windows PowerShell)
```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

2) Train Crop
```powershell
python src/train_crop.py
```
Artifacts saved under `models/`.

3) Train & Compare Fertilizer Models (rf/xgb/knn)
```powershell
python src/train_fertilizer_compare.py
```
Outputs under `models/` and `figures/`.

4) Run Streamlit app
```powershell
streamlit run app/streamlit_app.py
```
Use the sidebar to switch between tasks and test predictions.

## Notes
- If XGBoost is not installed/available, the compare script will skip it gracefully.
- Fertilizer column names are matched as in the dataset (e.g., `Temparature`, `Humidity ` with a trailing space).
- To ensure artifacts consistently land in repo `models/`, run training scripts from repo root (not from inside notebooks).
