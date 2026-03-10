from pathlib import Path

from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from app.model import load_model, Transaction, prepare_features


MODELS_FILE_PATH = Path(__file__).parent.parent / "models"
METADATA_FILE_PATH = MODELS_FILE_PATH / "metadata_v1_20260309.json"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model on startup...")
    model, metadata = load_model()
    app.state.model = model
    app.state.metadata = metadata
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def read_root():
    return {"status": "ok"}

@app.post("/predict")
def predict(transaction: Transaction, request: Request):
    model = request.app.state.model
    metadata = request.app.state.metadata
    if model is None:
        return {"error": "Model not loaded"}

    if metadata is None:
        return {"error": "Metadata not loaded"}

    features_df = prepare_features(transaction)
    prediction = model.predict(features_df)
    anomaly_score = model.decision_function(features_df)

    pred_val = int(prediction[0])
    score_val = float(anomaly_score[0])
    amount_ratio = float(features_df["amount_vs_cat_mean"].iloc[0])

    is_anomaly = bool(pred_val == -1)

    if not is_anomaly:
        reason = "Transaction appears normal"
    elif amount_ratio > 2.0:
        reason = f"Amount is {amount_ratio:.1f}x above your {transaction.category} average"
    elif score_val < 0.1:
        reason = "Significantly outside your normal spending patterns"
    else:
        reason = "Mildly unusual transaction"

    model_version = metadata.get("version", "unknown")

    return {
        "is_anomaly": bool(pred_val == -1),
        "anomaly_score": score_val,
        "model_version": model_version,
        "reason": reason
    }
