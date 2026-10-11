# M5 Daily Demand Forecasting and Revenue-Based Pricing Demo

This project now uses the **M5 Forecasting Accuracy dataset** as its training data. It does not generate synthetic training data and does not download the Kaggle files automatically.

It trains an XGBoost regression model to predict the next day's unit sales for selected Walmart item/store series. It then scores 41 candidate prices between 80% and 120% of the current price, selecting the price with the highest **estimated daily revenue** (candidate price × predicted units). Optuna tunes model hyperparameters using a chronological validation period. FastAPI serves recommendations, while Prometheus and Grafana display request and feature-drift metrics.

> **What this can and cannot claim:** M5 includes historical unit sales and sell prices, but it does not include unit costs, competitor prices, inventory, or controlled price experiments. The recommendation is an observational revenue scenario. It is not a causal estimate of the effect of changing price, and it cannot optimize profit without cost data.

## Dataset and file placement

Download the M5 Forecasting Accuracy files from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data). You need a Kaggle account and must accept the competition rules. Extract these three CSV files directly into the project's data/raw/m5/ folder:

```text
Real-Time Dynamic Pricing Engine (Machine Learning) 3/
└── data/
    └── raw/
        └── m5/
            ├── calendar.csv
            ├── sales_train_validation.csv
            └── sell_prices.csv
```

Do not leave the files inside the downloaded ZIP. The trainer reads these exact filenames. The sales file is large; the loader reads it in chunks and melts only the selected item/store series. By default it chooses the 12 series with the highest unit sales before the final 56 days, keeping a local run manageable without using validation/test labels to choose the products. Increase the --series-limit option if your computer has enough memory and you want more series.

## Run from Anaconda Prompt on Windows

Open **Anaconda Prompt** and run these commands. The environment name below matches the pricing311 environment discussed in the setup steps:

```bat
conda activate pricing311
cd /d "C:\Users\Hp\Documents\ChatGPT\Real-Time Dynamic Pricing Engine (Machine Learning) 3"
python -m pip install -r requirements.txt
```

Before training, confirm the files are present:

```bat
dir data\raw\m5
```

Train the model:

```bat
python -m pricing.train --data-dir data\raw\m5 --series-limit 12 --trials 3 --output-dir models
```

For a more thorough hyperparameter search, raise --trials (for example, to 10). Training saves models\pricing_model.json and models\metadata.json. The metadata includes the validation/test metrics, feature schema, selected item/store IDs, and reference feature distributions for drift monitoring.

Start the API in that same prompt after training:

```bat
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), expand POST /recommend, and choose **Try it out**. The model only accepts item/store pairs included in the training run; these are recorded in models\metadata.json. Check model status at [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) and metrics at [http://127.0.0.1:8000/metrics](http://127.0.0.1:8000/metrics).

## Run from Spyder

1. In Anaconda Prompt, activate the environment and open Spyder from it:

   ```bat
   conda activate pricing311
   cd /d "C:\Users\Hp\Documents\ChatGPT\Real-Time Dynamic Pricing Engine (Machine Learning) 3"
   spyder
   ```

2. In Spyder, set the working directory to the project root. It must be the folder containing pricing, app, README.md, and requirements.txt.

3. In the IPython console, install packages into Spyder's selected Python environment if you have not already installed them:

   ```python
   import subprocess
   import sys
   subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
   ```

4. Confirm that the M5 files are in the right place:

   ```python
   from pathlib import Path
   [Path("data/raw/m5", name).exists() for name in
    ("calendar.csv", "sales_train_validation.csv", "sell_prices.csv")]
   ```

   The result should be [True, True, True].

5. Train the model from the Spyder console:

   ```ipython
   %run -m pricing.train --data-dir data/raw/m5 --series-limit 12 --trials 3 --output-dir models
   ```

   Wait until the training report prints and the two model files are saved.

6. Start the API in a **second IPython console** (click the plus/new console button first, so the training console is not occupied by the server):

   ```python
   import uvicorn
   uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
   ```

7. Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser to submit a recommendation.

If Spyder cannot import pricing, its working directory is not the project root. Set it to the folder whose path ends with Real-Time Dynamic Pricing Engine (Machine Learning) 3.

## Recommendation request

The API needs the recent history for one item/store pair. All sales and price history values must be from days before the requested date. Example shape:

```json
{
  "item_id": "FOODS_3_090",
  "store_id": "CA_1",
  "date": "2016-05-23",
  "current_price": 2.98,
  "price_7d_ago": 2.98,
  "price_ma_28d": 2.98,
  "sales_lag_1d": 4,
  "sales_lag_7d": 5,
  "sales_lag_28d": 3,
  "sales_ma_7d": 4.4,
  "sales_ma_28d": 3.9,
  "sales_std_28d": 1.7,
  "event_flag": 0,
  "snap_day": 0
}
```

The IDs and values above illustrate the request format; replace them with a pair and values in your downloaded M5 data. After training, you can list the pairs included in the model from Spyder:

```python
import json
with open("models/metadata.json", encoding="utf-8") as file:
    metadata = json.load(file)
metadata["series_examples"]
```

The API returns predicted units and estimated revenue at both the current and recommended prices, plus PSI drift scores when enough requests have accumulated.

## What the project does

1. **Load M5:** Read daily item/store sales, calendar dates/events/SNAP flags, and weekly sell prices from the three Kaggle CSVs.
2. **Select series:** Keep a configurable number of high-volume item/store series. The default is 12 for a laptop-friendly demonstration.
3. **Engineer daily features:** Create lagged sales (1, 7, and 28 days), 7-day and 28-day moving averages, 28-day sales variability, relative and recent price changes, weekday/month cycles, event flags, and SNAP-day indicators. Sales features use prior days only to avoid using the target day's sales as an input.
4. **Train and tune:** Use a chronological train/validation/test split. Optuna searches XGBoost parameters by validation MAE; final evaluation is on the later test period.
5. **Score price scenarios:** At serving time, predict demand for candidate prices and select the candidate with the largest estimated price-times-units revenue, constrained to ±20% of the current price.
6. **Serve and monitor:** FastAPI exposes the model. Prometheus scrapes API request, latency, recommendation-change, and feature PSI metrics. Grafana provides a dashboard.

## Skills, domain, and tools

- **Domain:** retail/e-commerce demand forecasting, revenue management, and dynamic pricing.
- **Python and pandas:** read the M5 CSVs and transform product/store/date records.
- **Time-series feature engineering:** create past-only lags, rolling demand averages, price comparisons, and calendar features.
- **Regression and XGBoost:** forecast non-negative daily units sold from historical data and context.
- **Optuna:** tune model hyperparameters against validation-period MAE.
- **Evaluation:** compare the forecast against actual held-out daily sales and a 7-day moving-average baseline using MAE, RMSE, and R².
- **FastAPI:** expose a JSON pricing recommendation endpoint.
- **Docker Compose:** run the trainer, API, Prometheus, and Grafana as services.
- **Prometheus/Grafana:** display service metrics and simple input-feature drift signals.
- **Spyder/Anaconda:** run and inspect the Python workflow interactively on Windows.

## Optional: Docker, Prometheus, and Grafana

Docker Desktop must be installed and running. From Anaconda Prompt or PowerShell in the project root:

```bat
docker compose up --build
```

Compose mounts data/raw/m5/ read-only into the trainer container. The first run trains the model, then starts the API and monitoring services:

- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Prometheus: [http://localhost:9090](http://localhost:9090)
- Grafana: [http://localhost:3000](http://localhost:3000) (demo login: admin / admin)

To stop the stack, press Ctrl+C, then run docker compose down. The M5 dataset stays in your project folder and the trained container model remains in the named Docker volume.

## Project layout

```text
app/
  main.py                 Daily demand and revenue-scenario API
  monitoring.py           Prometheus metrics and feature PSI monitor
pricing/
  m5_data.py              M5 CSV loading, filtering, and daily panel creation
  features.py             Daily M5 feature engineering and serving features
  train.py                Chronological evaluation, Optuna, XGBoost training
monitoring/
  prometheus.yml
  grafana/                Provisioned dashboard and Prometheus connection
data/raw/m5/              Put the three downloaded Kaggle CSVs here
models/                   Local saved model and training metadata
Dockerfile
docker-compose.yml
requirements.txt
```
