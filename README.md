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
Phase 1 Complete: Data ingestion and feature engineering. (More phases coming soon)

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
