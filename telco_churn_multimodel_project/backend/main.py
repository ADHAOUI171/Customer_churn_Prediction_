from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.schemas import PredictRequest, PredictResponse
from backend.ml.registry import available_models, load_model, load_metrics, load_shap_summary, best_model_key
from backend.config import REPORTS_DIR, EDA_DIR

app = FastAPI(title="Telco Churn Multi-Model API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/models")
def get_models():
    models = [m.key for m in available_models()]
    metrics = load_metrics()
    best = best_model_key(metrics, split="validation", metric="recall")
    return {"models": models, "best_model": best}

@app.get("/metrics")
def get_metrics():
    metrics = load_metrics()
    best = best_model_key(metrics, split="validation", metric="recall")
    return {"best_model": best, **metrics}

@app.get("/plots/model_comparison")
def model_comparison_plot():
    path = REPORTS_DIR / "model_comparison.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Plot not found. Train models first.")
    return FileResponse(path)

@app.get("/eda/report")
def eda_report():
    path = EDA_DIR / "eda_report.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="EDA report not found. Train models first.")
    return FileResponse(path)

@app.get("/shap/summary")
def shap_summary():
    data = load_shap_summary()
    if not data:
        raise HTTPException(status_code=404, detail="SHAP summary not found. Train models first.")
    return data

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model = load_model(req.model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Convert to DataFrame with a single row
    import pandas as pd
    X = pd.DataFrame([req.features])

    try:
        proba = model.predict_proba(X)[:, 1][0]
        pred = int(proba >= 0.5)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return PredictResponse(model=req.model, churn_probability=float(proba), churn_prediction=pred)

# Serve EDA assets (plots) so the EDA report images resolve without 404s
app.mount("/eda", StaticFiles(directory=str(EDA_DIR), check_dir=False), name="eda")
