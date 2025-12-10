# Stock Forecasting using Informer model

This repository provides a complete workflow for preparing data used in short-term stock forecasting models that combine market prices with financial news sentiment.

## Data Preparation Workflow

The data preparation process consists of two core scripts.

---

## 1. `sort_filtered_csv.py`

This script extracts and organizes news relevant to the Magnificent 7 stocks from the FNSPID dataset.

### Functions
- Identifies and filters news articles containing stock-specific keywords.
- Creates dedicated CSV files for each ticker.
- Sorts news chronologically for downstream processing.

### Inputs
- Raw `All_external.csv` from FNSPID.

### Outputs
- Filtered news files stored in the `filtered` folder.
- Date-sorted files stored in the `sorted_filtered` folder.

---

## 2. `generate_final_dataset.py`

This script merges price data with sorted news to produce the final training dataset.

### Functions
- Reads historical stock price CSVs.
- Reads sorted per-ticker news files.
- Computes sentiment scores using FinBERT.
- Merges all inputs into a unified, model-ready dataset.

### Outputs
- A consolidated CSV file containing:
  - Price features
  - Computed sentiment scores
  - Time-aligned records suitable for forecasting models

---

## Usage Summary

1. Run `sort_filtered_csv.py` to extract and sort relevant news.
2. Run `generate_final_dataset.py` to compute sentiment, combine prices and news, and produce the final training dataset.
3. Use the generated dataset for training forecasting models such as Informer or hybrid sentiment–technical models.
4. Python train.py to run training
5. python evaluate.py to run evaluation
---
