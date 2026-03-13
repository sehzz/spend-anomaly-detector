from pathlib import Path

from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from app.model import load_model, Transaction, prepare_features


MODELS_FILE_PATH = Path(__file__).parent.parent / "models"
METADATA_FILE_PATH = MODELS_FILE_PATH / "metadata_v1_20260309.json"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and metadata on startup and store them in the app state."""
    
    print("Loading model on startup...")
    model, metadata = load_model()
    app.state.model = model
    app.state.metadata = metadata
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def read_root():
    """Health check endpoint to verify the API is running and the model is loaded."""
    
    return {"status": "ok"}

@app.post("/predict")
def predict(transaction: Transaction, request: Request):
    """Predict whether a transaction is anomalous and provide an explanation."""
    
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

@app.post("/bulk_predict")
def bulk_predict(transactions: list[Transaction], request: Request):
    """Predict anomalies for a list of transactions."""
    
    return [predict(txn, request) for txn in transactions]