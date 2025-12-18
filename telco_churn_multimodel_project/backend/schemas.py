from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    model: str = Field(..., description="Model key, e.g. logistic_regression, decision_tree, random_forest, xgboost")
    features: Dict[str, Any] = Field(..., description="Feature dict (no Churn column)")

class PredictResponse(BaseModel):
    model: str
    churn_probability: float
    churn_prediction: int
