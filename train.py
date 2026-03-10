import json

import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from datetime import datetime


CSV_OUTPUT_FILE = Path(__file__).parent / "data" / "processed_transactions.csv"
MODEL_PATH = Path(__file__).parent / "models"


def load_data():
    transactions_df = pd.read_csv(CSV_OUTPUT_FILE, parse_dates=['transaction_date'])
    return transactions_df

def select_features(df):
    cols_to_drop = [
        "id",
        "description",
        "transaction_date",
        "category",
        "amount"
    ]

    df = df.drop(columns=cols_to_drop, errors='ignore')
    return df

def train_model(df):
    forest = IsolationForest(contamination=0.05, random_state=42)
    forest.fit(df)
    return forest

def run():
    print("Running training script...")
    df = load_data()
    x = select_features(df)
    model = train_model(x)

    print("\nTraining completed.")

    df["anomaly_prediction"] = model.predict(x)
    df["anomaly_score"] = model.decision_function(x)

    print("\n--- Top Anomalies Detected ---")
    anomalies = df[df["anomaly_prediction"] == -1]
    display_columns = ['transaction_date', 'description', 'amount', 'category', 'anomaly_score']
    top_anomalies = anomalies[display_columns].sort_values(by='anomaly_score', ascending=True)
    print(top_anomalies.head(10))

    return model, df, list(x.columns)

def save_model():
    model, df, feature = run()

    now = datetime.now().strftime("%Y%m%d")
    version_name = f"v1_{now}"
    model_file = MODEL_PATH / f"model_{version_name}.pkl"
    metadata_file = MODEL_PATH / f"metadata_{version_name}.json"

    joblib.dump(model, model_file)

    row_count = len(df)
    anomaly_count = len(df[df["anomaly_prediction"] == -1])
    anomaly_rate = anomaly_count / row_count if row_count > 0 else 0

    metadata = {
        "version": version_name,
        "training_date": datetime.now().isoformat(),
        "row_count": row_count,
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate,
        "features": feature
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Model successfully saved to: {model_file}")
    print(f"Metadata successfully saved to: {metadata_file}")







if __name__ == "__main__":
    save_model()