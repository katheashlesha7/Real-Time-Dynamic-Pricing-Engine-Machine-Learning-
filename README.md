# M5 Daily Demand Forecasting and Dynamic Pricing Demo

An end-to-end machine-learning demo that uses the M5 Forecasting Accuracy dataset to forecast daily product sales and compare price scenarios. It includes daily feature engineering, an Optuna-tuned XGBoost regressor, a FastAPI service, a Streamlit interface, and optional Prometheus/Grafana monitoring.

> **Important:** This is an educational demonstration, not a production pricing system. M5 contains historical sales and selling prices, but does not include unit costs, competitor prices, inventory, or controlled price experiments. The suggested price maximizes model-estimated revenue over a limited candidate range. Historical price associations do not establish that changing a price will cause the predicted sales change.

## How it works

1. Loads M5 daily sales, calendar events/SNAP flags, and weekly selling prices.
2. Selects a configurable number of high-volume item/store series. The default is 12 to keep local training manageable.
3. Builds features from past sales and price history: 1/7/28-day sales lags, 7/28-day moving averages, 28-day sales variability, price comparisons, weekday/month cycles, event flags, and SNAP-day indicators.
4. Splits data chronologically into training, validation, and test periods. Optuna tunes XGBoost against validation MAE.
5. Forecasts daily units and scores 41 candidate prices within ±20% of the current price. The selected scenario has the highest estimated daily revenue: price × predicted units.
6. Serves recommendations through FastAPI. Streamlit provides a form-based interface; Prometheus and Grafana are available with Docker Compose.

The model learns to forecast daily units sold. The price recommendation is a separate scenario comparison using those forecasts; it is not a causal price-elasticity estimate.

## Dataset

Download the M5 Forecasting Accuracy files from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data). A Kaggle account and acceptance of the competition rules are required.

Extract these three files into the repository:

```text
data/
└── raw/
    └── m5/
        ├── calendar.csv
        ├── sales_train_validation.csv
        └── sell_prices.csv
```

The trainer uses these three files. Other files in the Kaggle download, such as sales_train_evaluation.csv and sample_submission.csv, are not needed. Dataset files are excluded from Git; do not commit the Kaggle data to your repository.

## Windows setup with Anaconda

Open Anaconda Prompt and change to the repository root (the directory containing pricing, app, and requirements.txt). Create the environment once:

```bat
conda create -n pricing311 python=3.11 pip -y
conda activate pricing311
python -m pip install -r requirements.txt
```

If the pricing311 environment already exists, just activate it and install the requirements:

```bat
conda activate pricing311
python -m pip install -r requirements.txt
```

Confirm that the M5 files are present:

```bat
dir data\raw\m5
```

## Train the model

Run from the repository root:

```bat
python -m pricing.train --data-dir data\raw\m5 --series-limit 12 --trials 3 --output-dir models
```

The first run may take a while because the M5 sales and price files are large. Do not interrupt it while it is loading or training. A successful run saves:

- models/pricing_model.json — trained XGBoost model
- models/metadata.json — feature schema, selected item/store pairs, drift reference, and training/evaluation report

Use more Optuna trials for a longer search; for example, change --trials 3 to --trials 10. Increase --series-limit to train on more item/store series if your computer has enough memory.

## Run the API

After training, start FastAPI from the same repository root:

```bat
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Keep the terminal open while using the API. Check readiness at [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health). A ready model should report status ready, model_loaded true, and data_kind m5.

FastAPI's interactive page is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). The GET /health endpoint checks the service. The POST /recommend endpoint accepts one item/store context and returns a price scenario. Use an item/store pair from known_series in models/metadata.json.

## Run the Streamlit interface

Streamlit is the simpler form-based interface. Keep the API running in one Anaconda Prompt. Open a second Anaconda Prompt, activate the same environment, change to the repository root, and run:

```bat
conda activate pricing311
python -m streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501). Choose a trained item/store pair and an M5 date, inspect recent sales, optionally adjust the scenario price, and click **Get price recommendation**. The page finds the dataset under data/raw/m5 or data/m5 and reuses cached history after the first load.

If your CSV files are stored in another directory, set M5_DATA_DIR before launching the trainer or Streamlit. In Anaconda Prompt:

```bat
set "M5_DATA_DIR=C:\path\to\m5"
python -m pricing.train --data-dir "%M5_DATA_DIR%" --series-limit 12 --trials 3 --output-dir models
```

## Understanding a recommendation

- **Current price:** price supplied for the M5 item/store/date.
- **Suggested price:** the candidate price with the highest model-estimated daily revenue; candidates are limited to ±20% of the current price.
- **Predicted units:** model forecast for that price scenario.
- **Estimated revenue:** candidate price multiplied by predicted units. It is an estimate, not actual revenue or guaranteed uplift.
- **Drift PSI:** input-distribution drift scores appear after enough API requests. An empty object initially is normal; it does not mean an error.
- **Drift alert:** indicates whether a monitored feature exceeded the demo's alert threshold. It does not measure prediction accuracy.

An HTTP 200 response means the API processed the request. It does not prove the recommendation is commercially correct. A 422 response usually means an item/store pair or request field is invalid; a 503 response means the trained model is unavailable.

## Evaluation

Training uses a chronological split: the final 28 usable dates are held out for testing and the prior 28 for validation. The report includes MAE, RMSE, R², and a 7-day moving-average baseline for daily unit forecasts. These forecast metrics do not validate the causal effect of changing prices or real-world revenue lift.

The selected series are ranked using sales before the final 56 days, avoiding the use of validation/test sales for the series selection. The feature pipeline shifts demand lags and rolling averages so a day's sales are not used to predict that same day's target.

## Optional monitoring with Docker Compose

Install and start Docker Desktop. Put the three M5 files under data/raw/m5, then run from the repository root:

```bat
docker compose up --build
```

Compose trains the model, starts the API, and launches Prometheus and Grafana:

- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Prometheus: [http://localhost:9090](http://localhost:9090)
- Grafana: [http://localhost:3000](http://localhost:3000) (demo login: admin / admin)

Stop the services with Ctrl+C, then run docker compose down.

## Project structure

```text
app/
  main.py                 FastAPI model serving
  monitoring.py           Prometheus metrics and PSI drift monitor
pricing/
  m5_data.py              M5 CSV loading and daily panel creation
  features.py             Daily feature engineering
  train.py                Optuna tuning, evaluation, and model training
streamlit_app.py          Form-based user interface
monitoring/
  prometheus.yml
  grafana/                Provisioned datasource and dashboard
data/raw/m5/              Local M5 files (not committed)
models/                   Saved model and metadata (not committed)
```

## Tools and skills demonstrated

Python, pandas, time-series feature engineering, regression, XGBoost, Optuna, chronological model evaluation, FastAPI, Streamlit, Docker Compose, Prometheus, Grafana, Anaconda, and Spyder.
