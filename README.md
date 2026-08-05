# Pastry Sales Forecasting

An end-to-end demand-forecasting, production-planning, and business-intelligence project built from a real operational problem in a pastry business.

I currently work as a pastry chef and created this project to apply data science and machine-learning methods to a question I encounter at work: **how many pastries should be produced for the next day while limiting both stockouts and waste?**

The project demonstrates the full workflow from cleaning difficult Excel data and translating business rules into code to validating forecasting models, simulating FIFO inventory, and presenting operational results in interactive Tableau dashboards.

The repository uses synthetic data that reproduces the original workflow. **Real operational data is not included.**

## Interactive Tableau dashboards

### Bakery Operations Dashboard

[Open the interactive Bakery Operations Dashboard on Tableau Public](https://public.tableau.com/app/profile/thomas.runser/viz/BakeryOperationsDashboard/BakeryOperationsDashboard)

[![Bakery Operations Dashboard](tableau/Bakery%20Operations%20Dashboard.png)](https://public.tableau.com/app/profile/thomas.runser/viz/BakeryOperationsDashboard/BakeryOperationsDashboard)

This dashboard provides an operational overview of:

- total sales;
- stockout rate;
- average closing stock;
- forecast mean absolute error;
- daily sales trends;
- weekday demand by product;
- shared store and product filters.

### Inventory and Forecast Risk Dashboard

[Open the interactive Inventory and Forecast Risk Dashboard on Tableau Public](https://public.tableau.com/app/profile/thomas.runser/viz/InventoryandForecastRisk/InventoryForecastRisk)

[![Inventory and Forecast Risk Dashboard](tableau/Inventory%20%26%20Forecast%20Risk.png)](https://public.tableau.com/app/profile/thomas.runser/viz/InventoryandForecastRisk/InventoryForecastRisk)

This dashboard demonstrates Tableau-specific interactivity:

- adjustable low-stock thresholds;
- adjustable forecast-error tolerance;
- KPI cards that update with parameter values;
- calculated 0/1 risk indicators aggregated as percentages;
- shared store and product filters;
- dashboard filter actions triggered by selecting a product.

The packaged workbook is available at `tableau/bakery_operations_dashboard.twbx`.

## Business context

The use case involves multiple pastry shops and products with short shelf lives.

Important inventory rules include:

- Products can be sold for two days.
- Stock from the previous day should be sold before the new batch: **FIFO** (first in, first out).
- Unsold old stock expires at the end of its second sales day and is treated as loss.
- Newly produced stock left at closing can become the next day's opening carryover.
- Seasonal products, weekends, holidays, weather, shop closures, and promotions can affect demand.
- Recorded production, stock, and loss values may contain human-entry errors, so the preparation workflow creates diagnostic fields for review.

The main business goal is to balance two risks:

- **Underproduction:** products sell out and potential sales are lost.
- **Overproduction:** products expire and become waste.

## Project workflow

The workflow has three main stages.

### 1. Data preparation

`notebooks/01_data_preparation.ipynb`

This notebook:

- reads monthly production workbooks and daily loss files;
- extracts valid daily product blocks from each monthly sheet;
- aligns production with the date on which it is sold;
- standardizes store, product, date, production, stock, sales, and loss fields;
- reconstructs demand from the most reliable available information;
- estimates FIFO stock age and identifies inconsistent inventory movements;
- adds product type, calendar, holiday, promotion, and weather features when available;
- saves separate training and evaluation datasets.

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

### 3. Tableau-ready public data model

The Tableau export contains two clean public CSV files:

- `tableau/bakery_tableau.csv` — operational fact table;
- `tableau/calendar.csv` — calendar dimension table.

They are generated from the deliberately messy synthetic Excel files by reusing the same production loader, product classification, loss merge, and inventory-sales reconstruction used in the data-preparation notebook.

`bakery_tableau.csv` contains one complete row per **date × store × product**. It includes forecast, production, sales, inventory, loss, event, promotion, and production-method fields. Calendar attributes are intentionally kept out of the fact table.

`calendar.csv` contains exactly one row per date, with year, quarter, month, weekday, weekend, Japanese holiday, day type, and season fields.

The Tableau relationship is:

```text
bakery_tableau.Date = calendar.Date
```

- `bakery_tableau`: **Many**
- `calendar`: **One**

The fact export validates that every row satisfies:

```text
OpeningStock + Production - Loss - Sales = ClosingStock
```

Do not interpret `Production - Sales` as waste. Sales may use opening stock, and newly produced stock remaining at closing can be carried into the next day.

Generate or refresh both public CSV files with:

```bash
python scripts/create_tableau_dataset.py
```

The command writes:

```text
tableau/bakery_tableau.csv
tableau/calendar.csv
```

See `docs/tableau_data_dictionary.md` for complete field definitions and the Tableau relationship setup.

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

The modelling notebook also saves validation metrics, ensemble weights, FIFO simulations, selected safety policies, the trained model bundle, and the final forecast CSV.

## Repository structure

```text
pastry-sales-forecasting/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 04_demand_model_ensemble_fifo.ipynb
│   └── 05_inventory_production_performance.ipynb
├── src/
│   ├── calendar_features.py
│   ├── create_synthetic_raw_files.py
│   ├── data_preparation.py
│   ├── demand_model.py
│   ├── inventory_simulation.py
│   └── tableau_dataset.py
├── scripts/
│   └── create_tableau_dataset.py
├── docs/
│   └── tableau_data_dictionary.md
├── data/
│   └── synthetic_raw/
├── tableau/
│   ├── bakery_operations_dashboard.twbx
│   ├── bakery_tableau.csv
│   ├── calendar.csv
│   ├── Bakery Operations Dashboard.png
│   └── Inventory & Forecast Risk.png
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
3. `notebooks/05_inventory_production_performance.ipynb`

The included synthetic dataset allows the workflow to run without access to the original business files.

The forecast date is controlled by the `FORECAST_DATE` variable near the beginning of notebook `04`.

### 5. Regenerate the Tableau datasets

```bash
python scripts/create_tableau_dataset.py
```

Open `tableau/bakery_operations_dashboard.twbx` with Tableau Public or Tableau Desktop to inspect the local workbook.

## Current limitations

- Production recommendations depend on the accuracy of opening stock and historical demand reconstruction.
- Sales inferred from yen totals can be ambiguous during discounts or free-item promotions.
- FIFO mistakes, damage, transfers, and entry errors can produce similar inventory discrepancies; the system can flag inconsistencies but cannot always determine their exact cause.
- Weather is optional and is unavailable in the synthetic example.
- The current notebooks use a fixed historical training period and a manually selected forecast date.
- The Tableau dashboards are descriptive tools and do not prove that an operational change caused an observed difference.
- Recommendations should be reviewed by someone familiar with shop operations before being used for production.

## Skills demonstrated

- Python and pandas data cleaning;
- messy Excel-data extraction;
- business-rule reconstruction;
- time-series feature engineering;
- chronological model validation;
- regression model ensembling;
- short-life FIFO inventory simulation;
- stockout and loss evaluation;
- relational fact and calendar data modelling;
- Tableau calculated fields and KPI cards;
- Tableau parameters and adjustable thresholds;
- shared filters and dashboard actions;
- stakeholder-facing dashboard design.

## Project purpose

This project connects machine learning to a concrete operational decision: **how much to produce tomorrow**.

It also represents my transition back toward an IT role by showing how I can turn domain knowledge from my current job into a structured technical solution and communicate the results to both technical and nontechnical audiences.
