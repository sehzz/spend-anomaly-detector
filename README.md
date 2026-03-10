# spend-anomaly-detector
Production MLOps pipeline that detects anomalous personal transactions using Isolation Forest — with FastAPI, Docker, Grafana monitoring, and automated weekly retraining via n8n.

## What this project does
This project is an end-to-end machine learning pipeline that identifies unusual spending patterns in my personal financial data. It ingests raw transaction logs and category data, engineers time-series features, and outputs flagged anomalies to help catch accidental overcharges or out-of-character spending.

## Why I built this
I already use a custom database to track my daily expenses. While my current tracker has a basic, math-based alert system using static category thresholds, those hardcoded limits only help up to a certain point.

With Machine Learning becoming an industry standard, I wanted to move beyond simple threshold alerts and build a dynamic model using my own real-world data. I could have used off-the-shelf tools, but as a software engineer, building this custom pipeline from scratch was the perfect way to get hands-on experience with ML feature engineering and MLOps.

## Tech Stack
**Current stack:** pandas, numpy, pathlib, json, scikit-learn

## Project Status
Phase 1 Complete: Data ingestion and feature engineering.
Phase 2 Complete: Model training and versioning
Phase 3 In Progress: FastAPI prediction service
Phase 4: Docker deployment
Phase 5: Automated retraining via n8n
Phase 6: Grafana monitoring dashboard

## Phase 1: Data Pipeline & Feature Engineering
### What was done
In this phase, raw database exports are transformed into a clean, machine-learning-ready dataset. The pipeline performs the following automated steps:
-Merges relational transaction and category data.
-Cleans and normalizes text fields (e.g., standardizing "Clothing" and "clothing").
-Converts strings to proper datetime objects.
-Drops redundant database columns (like user IDs).
-Engineers 8 new statistical and time-series features for the ML model.

### How it works
Data Extraction -> Processing -> Storage

Data is fetched and saved inside the parent directory with the help of my MiniMon project (https://github.com/sehzz/MiniMon). Once the raw JSON data is available, this pipeline reads it into pandas DataFrames, applies the transformations, and saves the final output as a clean CSV file in the /data directory.

### Features engineered
**day_of_week**: Represents the day of the transaction (0=Mon, 6=Sun) to track weekly habits.

**week_of_month**: Indicates the week (1-5) to track pay-cycle or end-of-month behaviors.

**month**: Extracts the calendar month (1-12) to identify seasonal spending trends.

**category_encoded**: Converts text labels into unique integers for ML processing.

**amount_log**: Applies a log transformation (np.log1p) to safely squash massive outlier purchases.

**amount_vs_cat_mean**: Divides the transaction by the historical category average to flag unusually large purchases (e.g., spending 3x the norm on groceries).

**rolling_7d_avg**: Computes the average daily spending over the trailing 7 days for a short-term baseline.

**rolling_30d_std**: Measures the 30-day standard deviation to capture recent spending volatility.

## Phase 2: Model Training & Versioning
### What was done
In this phase, the engineered dataset is used to train an unsupervised machine learning model to detect financial outliers.

- Trained an Isolation Forest model using the 8 engineered features.

- Generated both binary anomaly predictions (1 or -1) and continuous anomaly scores for every transaction.

- Serialized the trained model and generated a version-controlled metadata file for downstream serving and dashboarding.

### How it works
Load CSV → Select Features → Train Model → Generate Predictions → Save Model + Metadata

The processed CSV from Phase 1 is loaded, and non-feature columns (like IDs, raw text, and dates) are stripped so the model only trains on the mathematical features. The data is fed into a Scikit-Learn Isolation Forest model configured with contamination=0.05 to flag the top ~5% most unusual transactions. Once trained, the model evaluates the dataset, appending an anomaly_prediction and anomaly_score back onto the original records. Finally, the model artifact and its training metrics are saved to disk.

### Model details
- **Algorithm:** Isolation Forest
- **Contamination:** 0.05 (expects ~5% anomalies)
- **Training rows:** 262
- **Anomalies detected:** 14 (5.3%)

### Why Isolation Forest?
The core reason is that we have no labelled data. I haven't manually tagged thousands of past transactions as "normal" or "anomalous," which renders supervised algorithms like Random Forest or Logistic Regression unusable.

Isolation Forest is perfect for this specific use case because:
- It handles mixed, multivariate patterns: My spending varies by category, day, and time of month, and this algorithm naturally isolates multi-dimensional outliers.

- It works well on small datasets: My initial training set only contained 262 rows, which is enough for Isolation Forest to establish a baseline.

- It gives you an anomaly score, not just a binary flag, which is much more useful for Grafana visualisation

### Versioning
To maintain strict version control for MLOps, the pipeline automatically saves the model artifact using a dated naming convention: model_v1_YYYYMMDD.pkl.

Alongside the .pkl file, it generates a metadata.json file. This acts as a snapshot of the model's health and context, containing the model version, training date, row count, anomaly count, anomaly rate, and the exact ordered list of features used.
