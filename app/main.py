"""
Spend Anomaly Detector — FastAPI Prediction Service

Endpoints:
    GET  /health        — Health check, returns active model version
    POST /predict       — Score a single transaction
    POST /bulk_predict  — Score a list of transactions
    POST /reload        — Hot-reload the latest trained model
    GET  /docs          — Interactive API documentation (Swagger UI)

Date format: All transaction_date fields must use YYYY-MM-DD format.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request

from app.model import Transaction, load_model, log_prediction, prepare_features

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
def read_root(request: Request):
    """
    Health check endpoint to verify the API is running and the model is loaded.
    
    Returns the current status and the active model version.
    
    Example response:
        {
            "status": "ok",
            "model_version": "v1_20260425"
        }
    """
    
    metadata = request.app.state.metadata
    model_version = metadata.get("version", "unknown")

    return {"status": "ok", "model_version": model_version}

@app.post("/predict")
def predict(transaction: Transaction, request: Request):
    """
    Predict whether a single transaction is anomalous.
    
    Accepts a transaction JSON payload, engineers the required features using
    historical data, and returns an anomaly prediction with a human-readable reason.
    
    Note: transaction_date must be in YYYY-MM-DD format.
    
    Example request:
        {
            "amount": 2200.00,
            "category": "subscriptions",
            "transaction_date": "2026-04-25"
        }
    
    Example response:
        {
            "is_anomaly": true,
            "anomaly_score": -0.134,
            "model_version": "v1_20260425",
            "reason": "Amount is 111.7x above your subscriptions average"
        }
    """
    
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

    log_prediction(
        amount=transaction.amount,
        category=transaction.category,
        transaction_date=transaction.transaction_date,
        is_anomaly=is_anomaly,
        anomaly_score=score_val,
        reason=reason,
        model_version=model_version
    )

    return {
        "is_anomaly": bool(pred_val == -1),
        "anomaly_score": score_val,
        "model_version": model_version,
        "reason": reason
    }

@app.post("/bulk_predict")
def bulk_predict(transactions: list[Transaction], request: Request):
    """
    Predict anomalies for a list of transactions in a single request.
    
    Accepts a list of transaction objects and returns a prediction for each.
    Used by MiniMon to score weekly transactions in bulk and generate alerts.
    
    Note: transaction_date must be in YYYY-MM-DD format for all transactions.
    
    Example request:
        [
            {
                "amount": 85.00,
                "category": "restaurants",
                "transaction_date": "2026-04-25"
            },
            {
                "amount": 3.50,
                "category": "transportation",
                "transaction_date": "2026-04-25"
            }
        ]
    
    Example response:
        [
            {
                "is_anomaly": true,
                "anomaly_score": -0.010,
                "model_version": "v1_20260425",
                "reason": "Amount is 5.5x above your restaurants average"
            },
            {
                "is_anomaly": false,
                "anomaly_score": 0.055,
                "model_version": "v1_20260425",
                "reason": "Transaction appears normal"
            }
        ]
    """
    
    return [predict(txn, request) for txn in transactions]

@app.post("/reload")
def reload_model(request: Request):
    """
    Hot-reload the latest trained model and metadata without restarting the server.
    
    Called automatically by the n8n retraining pipeline every Sunday after
    train.py completes. Picks up the latest .pkl file from the models/ directory.
    
    Example response:
        {
            "status": "model reloaded",
            "model_version": "v1_20260425"
        }
    """
    
    model, metadata = load_model()
    request.app.state.model = model
    request.app.state.metadata = metadata
    model_version = metadata.get("version", "unknown")

    return {"status": "model reloaded", "model_version": model_version}