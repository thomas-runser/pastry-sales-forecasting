# Pastry Sales Forecasting

A demand-forecasting and production-planning project built from a real operational problem in a pastry business.

I currently work as a pastry chef and created this project to apply data science and machine-learning methods to a problem I encounter at work: deciding how many pastries should be produced for the next day while limiting both stockouts and waste.

The project is also part of my preparation to return to an IT career. It demonstrates how I approach an end-to-end problem, from cleaning difficult Excel data and translating business rules into code to validating models and producing practical recommendations.

The system predicts how many units of each product are likely to sell on the next business day, then converts that forecast into a recommended production quantity after accounting for stock carried over from the previous day.

The main business goal is to balance two risks:

- **Underproduction:** products sell out and potential sales are lost.
- **Overproduction:** products expire and become waste.

The repository uses synthetic data to reproduce the workflow. Real operational data is not included.

## Business context

The use case involves multiple pastry shops and products with short shelf lives.

Important inventory rules include:

- Products can be sold for two days.
- Stock from the previous day should be sold before the new batch: **FIFO** (first in, first out).
- Unsold old stock expires at the end of its second sales day and is treated as loss.
- Newly produced stock left at closing can become the next day's opening carryover.
- Seasonal products, weekends, holidays, weather, shop closures, and promotions can affect demand.
- Recorded production, stock, and loss values may contain human-entry errors, so the preparation workflow also creates diagnostic fields for review.

## What the project does

The workflow has two main stages.

### 1. Data preparation

`notebooks/01_data_preparation.ipynb`

This notebook:

- reads monthly production workbooks and daily loss files;
- extracts only valid daily product blocks from each monthly sheet;
- aligns production with the date on which it is sold;
- standardizes store, product, date, production, stock, sales, and loss fields;
- reconstructs demand from the most reliable available information;
- estimates FIFO stock age and identifies inconsistent inventory movements;
- adds product type, calendar, holiday, promotion, and weather features when available;
- saves a training dataset and a separate evaluation dataset.

The training dataset is kept separate from later actual results so the model cannot learn from the day it is asked to forecast.

### 2. Demand forecast and FIFO production policy

`notebooks/04_demand_model_ensemble_fifo.ipynb`

This notebook:

- builds leakage-safe lag and rolling-history features;
- performs chronological rolling validation;
- trains three regression models:
  - LightGBM with a Poisson objective;
  - histogram gradient boosting with a Poisson loss;
  - histogram gradient boosting with an absolute-error loss;
- combines the models using validation-based ensemble weights;
- tests several FIFO safety buffers;
- selects a buffer for each store-product combination, with a global fallback;
- retrains the selected ensemble on all historical training data;
- predicts demand for the selected forecast date;
- subtracts opening carryover to recommend production.

The core production rule is:

```text
Recommended production
= max(Forecast demand - Opening carryover, 0)
```

The FIFO simulation then evaluates the trade-off between stockouts and expired stock.

## Forecast output

The final forecast contains one row per date, store, and product. Important fields include:

| Column | Meaning |
|---|---|
| `BaseForecastDemand` | Weighted ensemble prediction before the safety buffer |
| `SafetyAdjustment` | Additional units selected through FIFO validation |
| `ForecastDemand` | Final demand target after the safety adjustment |
| `OpeningCarryover` | Stock available from the previous day |
| `RecommendedProduction` | Units recommended for the new batch |
| `ActualProduction` | Recorded production, when evaluation data exists |
| `ActualSales` | Recorded or reconstructed sales, when available |
| `ActualLoss` | Recorded loss, when available |
| `ExpectedFIFOLoss` | Loss expected if old stock was sold first |
| `LossReason` | Diagnostic comparison of recorded loss and FIFO expectation |

The notebook also saves validation metrics, ensemble weights, FIFO simulations, selected safety policies, the trained model bundle, and the final forecast CSV.

## Repository structure

```text
pastry-sales-forecasting/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   └── 04_demand_model_ensemble_fifo.ipynb
├── src/
│   ├── calendar_features.py
│   ├── create_synthetic_raw_files.py
│   ├── data_preparation.py
│   ├── demand_model.py
│   └── inventory_simulation.py
├── data/
│   ├── synthetic_raw/
│   └── processed/
├── outputs/
├── requirements.txt
└── README.md
```

## Running the project

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Start Jupyter

```bash
jupyter notebook
```

### 4. Run the notebooks in order

1. `notebooks/01_data_preparation.ipynb`
2. `notebooks/04_demand_model_ensemble_fifo.ipynb`

The included synthetic dataset allows the complete workflow to run without access to the original business files.

The forecast date is controlled by the `FORECAST_DATE` variable near the beginning of notebook `04`.

## Current limitations

- The quality of production recommendations depends on the accuracy of opening stock and historical demand reconstruction.
- Sales inferred from yen totals can be ambiguous during discounts or free-item promotions.
- FIFO mistakes, damage, transfers, and entry errors can produce the same inventory discrepancy; the system can flag inconsistencies but cannot always determine their exact cause.
- Weather is optional and is unavailable in the synthetic example.
- The current notebooks use a fixed historical training period and a manually selected forecast date.
- Recommendations should be reviewed by someone familiar with shop operations before being used for production.

## Project purpose

This project uses a real workplace problem to demonstrate an end-to-end data and machine-learning workflow. It combines:

- messy Excel-data cleaning;
- business-rule reconstruction;
- time-series feature engineering;
- chronological model validation;
- model ensembling;
- short-life FIFO inventory simulation;
- waste and stockout evaluation;
- practical next-day production recommendations.

The goal is not only to build a prediction model, but to connect machine learning to a concrete operational decision: **how much to produce tomorrow**.

It also represents my transition back toward an IT role by showing how I can turn domain knowledge from my current job into a structured technical solution.
