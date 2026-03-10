import json
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

import joblib


MODELS_FILE_PATH = Path(__file__).parent.parent / "models"
CSV_OUTPUT_FILE = Path(__file__).parent.parent / "data" / "processed_transactions.csv"


class Transaction(BaseModel):
    amount: float
    category: str
    transaction_date: str


def get_latest_model_files(path: Path):    
    pkl_files = list(path.glob("*.pkl"))
    json_files = list(path.glob("*.json"))
    
    latest_pkl = max(pkl_files, key=lambda x: x.stat().st_mtime, default=None)
    latest_json = max(json_files, key=lambda x: x.stat().st_mtime, default=None)
    
    print(f"Latest Model: {latest_pkl}")
    print(f"Latest Metadata: {latest_json}")
    
    return latest_pkl, latest_json

def load_model():
    model_file, metadata_file = get_latest_model_files(MODELS_FILE_PATH)
    if not model_file:
        print("No model files found.")
        return None, None
    
    latest_model_file = model_file
    print(f"Loading model from: {latest_model_file}")
    model = joblib.load(latest_model_file)

    print(f"Loading metadata from: {metadata_file}")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    return model, metadata

def prepare_features(transaction: Transaction):
    history_df = pd.read_csv(CSV_OUTPUT_FILE)
    history_df['transaction_date'] = pd.to_datetime(history_df['transaction_date'])

    df = pd.DataFrame([transaction.model_dump()])
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    incoming_date = df['transaction_date'].iloc[0]
    incoming_amount = df['amount'].iloc[0]
    clean_category = df['category'].iloc[0].lower().strip()

    df['day_of_week'] = df['transaction_date'].dt.dayofweek
    df['week_of_month'] = df['transaction_date'].apply(lambda x: (x.day - 1) // 7 + 1)
    df['month'] = df['transaction_date'].dt.month
    df['amount_log'] = np.log1p(df['amount'])
    
    cat_mapping = history_df.drop_duplicates("category").set_index("category")["category_encoded"].to_dict()
    df['category_encoded'] = cat_mapping.get(clean_category, -1)

    cat_history = history_df[history_df['category'] == clean_category]
    if not cat_history.empty and cat_history['amount'].mean() > 0:
        historical_mean = cat_history['amount'].mean()
        df['amount_vs_cat_mean'] = incoming_amount / historical_mean
    
    else:
        df['amount_vs_cat_mean'] = 1.0

    mask_7d = (history_df['transaction_date'] >= incoming_date - pd.Timedelta(days=7)) & (history_df['transaction_date'] <= incoming_date)
    past_7d = history_df[mask_7d]
    df['rolling_7d_avg'] = past_7d['amount'].mean() if not past_7d.empty else incoming_amount

    mask_30d = (history_df['transaction_date'] >= incoming_date - pd.Timedelta(days=30)) & (history_df['transaction_date'] <= incoming_date)
    past_30d = history_df[mask_30d]
    df['rolling_30d_std'] = past_30d['amount'].std() if len(past_30d) > 1 else 0.0

    features_list = [
        'day_of_week', 
        'week_of_month', 
        'month',
        'category_encoded', 
        'amount_log', 
        'amount_vs_cat_mean', 
        'rolling_7d_avg', 
        'rolling_30d_std'
    ]
    
    return df[features_list]


if __name__ == "__main__":
    # print(load_model())
    # t = Transaction(
    #     amount=123.45, 
    #     category="clothing", 
    #     transaction_date="2026-03-10"
    # )
    # a = prepare_features(t)
    # print(a)

    get_latest_model_files(MODELS_FILE_PATH)