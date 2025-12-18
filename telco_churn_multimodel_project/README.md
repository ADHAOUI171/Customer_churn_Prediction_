# Telco Customer Churn — Multi-Model ML + EDA + FastAPI + Next.js

This project trains **multiple churn models** (Logistic Regression, Decision Tree, Random Forest, XGBoost) with:
- Train / Validation / Test split
- Proper preprocessing (impute + one-hot encode + scale numeric)
- **SMOTE applied only on training folds inside the pipeline**
- **GridSearchCV + cross-validation**
- Metrics: Accuracy / Precision / Recall
- Model comparison plot + per-model metrics
- **EDA report generated after cleaning + preprocessing inputs**

## 1) Project setup

### Python (recommended: 3.11)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Train all models + generate reports
Put the dataset CSV at:
`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

Then run:
```bash
python -m backend.ml.train_all --data_path data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Outputs:
- `models/<model_name>.joblib`
- `reports/metrics.json` (all models, val+test)
- `reports/model_comparison.png`
- `reports/eda/eda_report.html` + plots in `reports/eda/`

## 2) Backend (FastAPI)
```bash
uvicorn backend.main:app --reload
```

API:
- `GET /models` -> list models + best model (by validation recall)
- `GET /metrics` -> all metrics + best model
- `GET /eda` -> returns EDA report path info
- `POST /predict` -> predict with a chosen model

Example:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"model\":\"random_forest\",\"features\":{...}}"
```

## 3) Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev
```
Open: http://localhost:3000

You can:
- Select model from dropdown
- View validation + test metrics
- See model comparison image
- Paste JSON features and run inference

## Notes
- Run training before starting the backend for predictions.
- The backend auto-loads available models from the `models/` folder.
