import json
import numpy as np
from pathlib import Path
import pandas as pd


TRANSACTION_FILE = Path(__file__).parent / "resources" / "transaction_data.json"
CATEGORIES_FILE = Path(__file__).parent / "resources" / "categories_data.json"
CSV_OUTPUT_FILE = Path(__file__).parent / "data" / "processed_transactions.csv"


def run():
    print("Running data pipeline...")
    
    transactions_data = get_data(TRANSACTION_FILE)
    categories_data = get_data(CATEGORIES_FILE)
    
    print("Processing DataFrames...")
    df_transactions = process_data(transactions_data)
    df_categories = process_data(categories_data)

    print("Merging data...")    
    merged_df = merge_df(df_transactions, df_categories)
    
    print("Cleaning data...")
    clean_df = clean_data(merged_df)
    
    final_df = engineer_features(clean_df)
    print("\nData pipeline completed.")
    print("\n--- Final Cleaned Head ---")
    print(final_df.head(5))
    save_to_csv(final_df)

    return final_df


def get_data(file):
    with open(file) as f:
        return json.load(f)
    
def process_data(data):
    df = pd.DataFrame(data)


    return df

def merge_df(df_transactions, df_categories):
    df = pd.merge(
        df_transactions, 
        df_categories, 
        left_on='category_id', 
        right_on='id', 
        how='left', 
        suffixes=('', '_cat'),
    )

    cols_to_drop = [
        'category_id', 
        'idx', 
        'parent_id', 
        'created_at', 
        'user_id',
        'user_id_cat',   
        'id_cat'
    ]

    df = df.drop(columns=cols_to_drop, errors='ignore')
    df = df.rename(columns={'name': 'category'})

    print("\nData pipeline completed.")

    return df

def clean_data(df):
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])

    df = df.sort_values(by='transaction_date', ascending=True).reset_index(drop=True)

    df['category'] = df['category'].str.lower().str.strip()

    return df

def engineer_features(df):
    df["day_of_week"] = df["transaction_date"].dt.day_of_week
    df["week_of_month"] = df["transaction_date"].apply(lambda x: (x.day - 1) // 7 + 1)
    df["month"] = df["transaction_date"].dt.month
    df['category'] = df['category'].str.lower().str.strip()    
    df["category_encoded"] = df["category"].astype("category").cat.codes
    df["amount_log"] = np.log1p(df["amount"])
    df["amount_vs_cat_mean"] = df["amount"] / df.groupby("category")["amount"].transform("mean")

    df = df.set_index('transaction_date')

    df['rolling_7d_avg'] = df['amount'].rolling('7D', min_periods=1).mean()
    df['rolling_30d_std'] = df['amount'].rolling('30D', min_periods=1).std().fillna(0)
    df = df.reset_index()

    return df

def save_to_csv(df):
    """Saves the fully processed DataFrame to a CSV file."""

    df.to_csv(CSV_OUTPUT_FILE, index=False)    
    print(f"Processed dataset saved to: {CSV_OUTPUT_FILE}")


if __name__ == "__main__":
    run()
