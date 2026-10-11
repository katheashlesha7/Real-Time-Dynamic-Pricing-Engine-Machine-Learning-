"""Train an Optuna-tuned daily sales model from the M5 CSV dataset."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from pricing.features import FEATURE_COLUMNS, build_features, price_scenarios
from pricing.m5_data import load_m5_dataset


def reference_histograms(frame: pd.DataFrame) -> dict:
    result = {}
    quantiles = np.linspace(0, 1, 11)
    for name in FEATURE_COLUMNS:
        values = frame[name].to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, quantiles))
        result[name] = {
            "edges": edges.tolist(),
            "counts": np.histogram(values, bins=np.r_[-np.inf, edges[1:-1], np.inf])[0].tolist(),
            "n": int(len(values)),
        }
    return result


def _new_model(params: dict, seed: int) -> XGBRegressor:
    return XGBRegressor(
        **params,
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        n_jobs=max(1, min(4, os.cpu_count() or 1)),
        random_state=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/m5"))
    parser.add_argument(
        "--series-limit",
        type=int,
        default=12,
        help="Train on this many highest-unit-sales item/store series to limit RAM and runtime.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=int(os.getenv("OPTUNA_TRIALS", "10")))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    if args.series_limit < 1 or args.trials < 1:
        parser.error("--series-limit and --trials must be at least 1.")

    print(f"Loading M5 files from {args.data_dir.resolve()} (top {args.series_limit} item/store series)")
    panel, item_indices, store_indices = load_m5_dataset(args.data_dir, args.series_limit)
    frame = build_features(panel)
    frame = frame.dropna(subset=FEATURE_COLUMNS + ["sales_units"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No rows remain after feature engineering; check that M5 prices and dates align.")

    dates = pd.Index(frame["date"].drop_duplicates().sort_values())
    if len(dates) < 70:
        raise ValueError(f"Need at least 70 usable daily dates for train/validation/test; found {len(dates)}.")
    validation_start = pd.Timestamp(dates[-56])
    test_start = pd.Timestamp(dates[-28])
    train = frame[frame["date"] < validation_start]
    validation = frame[(frame["date"] >= validation_start) & (frame["date"] < test_start)]
    test = frame[frame["date"] >= test_start]
    if train.empty or validation.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition; use more M5 history.")
    x_train, y_train = train[FEATURE_COLUMNS], train["sales_units"]
    x_val, y_val = validation[FEATURE_COLUMNS], validation["sales_units"]

    def objective(trial: optuna.Trial) -> float:
        model = _new_model(
            {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 12),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 12, log=True),
            },
            args.seed,
        )
        model.fit(x_train, y_train, verbose=False)
        return float(mean_absolute_error(y_val, model.predict(x_val)))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        study_name="m5_daily_demand_regressor",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    fit_frame = frame[frame["date"] < test_start]
    final_model = _new_model(study.best_params, args.seed)
    final_model.fit(fit_frame[FEATURE_COLUMNS], fit_frame["sales_units"], verbose=False)
    predictions = np.maximum(final_model.predict(test[FEATURE_COLUMNS]), 0.0)
    current_predictions = predictions
    current_revenue = test["sell_price"].to_numpy(dtype=float) * current_predictions

    candidate_price_rows = []
    for _, row in test.iterrows():
        base = row[FEATURE_COLUMNS].astype(float).to_dict()
        current_price = float(row["sell_price"])
        candidates = np.linspace(current_price * 0.8, current_price * 1.2, 41)
        scenario_features = price_scenarios(base, candidates)
        scenario_units = np.maximum(final_model.predict(scenario_features), 0.0)
        best = int(np.argmax(candidates * scenario_units))
        candidate_price_rows.append(
            (float(candidates[best]), float(scenario_units[best]), float(candidates[best] * scenario_units[best]))
        )
    recommended = np.asarray([value[0] for value in candidate_price_rows])
    recommended_units = np.asarray([value[1] for value in candidate_price_rows])
    recommended_revenue = np.asarray([value[2] for value in candidate_price_rows])
    baseline_mae = float(mean_absolute_error(test["sales_units"], test["sales_ma_7d"]))

    report = {
        "data_kind": "m5",
        "dataset": "M5 Forecasting Accuracy",
        "target_definition": "next-day observed unit sales",
        "recommendation_objective": "estimated daily revenue (candidate price × predicted units), within ±20% of observed sell price",
        "causal_warning": "M5 is observational; price suggestions are revenue scenarios, not causal or profit-optimal estimates.",
        "source_files": ["calendar.csv", "sales_train_validation.csv", "sell_prices.csv"],
        "series_limit": int(args.series_limit),
        "series_count": int(frame[["item_id", "store_id"]].drop_duplicates().shape[0]),
        "series_examples": sorted(
            {f"{item}::{store}" for item, store in zip(panel["item_id"].astype(str), panel["store_id"].astype(str))}
        )[:10],
        "date_start": pd.Timestamp(frame["date"].min()).date().isoformat(),
        "date_end": pd.Timestamp(frame["date"].max()).date().isoformat(),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "train_start": pd.Timestamp(train["date"].min()).date().isoformat(),
        "validation_start": validation_start.date().isoformat(),
        "test_start": test_start.date().isoformat(),
        "test_mae_units": float(mean_absolute_error(test["sales_units"], predictions)),
        "test_rmse_units": float(np.sqrt(mean_squared_error(test["sales_units"], predictions))),
        "test_r2": float(r2_score(test["sales_units"], predictions)),
        "test_7day_moving_average_baseline_mae_units": baseline_mae,
        "test_mean_estimated_revenue_at_current_price": float(np.mean(current_revenue)),
        "test_mean_estimated_revenue_at_recommended_price": float(np.mean(recommended_revenue)),
        "test_mean_recommended_price": float(np.mean(recommended)),
        "test_mean_predicted_units_at_recommended_price": float(np.mean(recommended_units)),
        "best_validation_mae_units": float(study.best_value),
        "optuna_trials": int(args.trials),
        "best_params": study.best_params,
        "seed": args.seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_model.save_model(args.output_dir / "pricing_model.json")
    metadata = {
        "model_version": "xgb-m5-daily-v1",
        "feature_columns": FEATURE_COLUMNS,
        "item_indices": item_indices,
        "store_indices": store_indices,
        "known_series": sorted(
            {f"{item}::{store}" for item, store in zip(panel["item_id"].astype(str), panel["store_id"].astype(str))}
        ),
        "reference_histograms": reference_histograms(fit_frame[FEATURE_COLUMNS]),
        "report": report,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved model artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
