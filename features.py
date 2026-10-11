"""Daily, leakage-aware feature engineering for the m5 sales dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "item_index",
    "store_index",
    "current_price",
    "price_to_28d_avg_ratio",
    "price_change_7d_pct",
    "sales_lag_1d",
    "sales_lag_7d",
    "sales_lag_28d",
    "sales_ma_7d",
    "sales_ma_28d",
    "sales_std_28d",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "event_flag",
    "snap_day",
]


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add features known at the start of each day, using only prior sales."""
    required = {
        "item_id", "store_id", "date", "sell_price", "sales_units",
        "item_index", "store_index", "event_flag", "snap_day",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Missing required M5 columns: {', '.join(missing)}")

    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["item_id", "store_id", "date"])
    pieces = []
    for _, group in data.groupby(["item_id", "store_id"], sort=False):
        group = group.copy()
        group["current_price"] = group["sell_price"].astype(float)
        prior_sales = group["sales_units"].shift(1)
        group["sales_lag_1d"] = prior_sales
        group["sales_lag_7d"] = group["sales_units"].shift(7)
        group["sales_lag_28d"] = group["sales_units"].shift(28)
        group["sales_ma_7d"] = prior_sales.rolling(7, min_periods=7).mean()
        group["sales_ma_28d"] = prior_sales.rolling(28, min_periods=28).mean()
        group["sales_std_28d"] = prior_sales.rolling(28, min_periods=28).std().fillna(0)

        prior_price_mean = group["sell_price"].shift(1).rolling(28, min_periods=1).mean()
        group["price_to_28d_avg_ratio"] = (
            group["current_price"] / prior_price_mean.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        price_7d_ago = group["sell_price"].shift(7)
        group["price_change_7d_pct"] = (
            (group["current_price"] - price_7d_ago) / price_7d_ago.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        weekday_angle = 2 * np.pi * group["date"].dt.dayofweek / 7
        month_angle = 2 * np.pi * (group["date"].dt.month - 1) / 12
        group["weekday_sin"] = np.sin(weekday_angle)
        group["weekday_cos"] = np.cos(weekday_angle)
        group["month_sin"] = np.sin(month_angle)
        group["month_cos"] = np.cos(month_angle)
        pieces.append(group)

    return pd.concat(pieces, ignore_index=True)


def features_from_context(context: dict, item_indices: dict, store_indices: dict) -> dict[str, float]:
    """Create the exact model feature vector from an API daily context."""
    if context["item_id"] not in item_indices:
        raise ValueError(f"Unknown item_id: {context['item_id']}")
    if context["store_id"] not in store_indices:
        raise ValueError(f"Unknown store_id: {context['store_id']}")

    timestamp = pd.Timestamp(context["date"])
    current_price = float(context["current_price"])
    price_7d_ago = float(context["price_7d_ago"])
    prior_price_mean = float(context["price_ma_28d"])
    price_ratio = current_price / max(prior_price_mean, 1e-8)
    price_change = (current_price - price_7d_ago) / max(price_7d_ago, 1e-8)
    weekday_angle = 2 * np.pi * timestamp.dayofweek / 7
    month_angle = 2 * np.pi * (timestamp.month - 1) / 12
    values = {
        "item_index": float(item_indices[context["item_id"]]),
        "store_index": float(store_indices[context["store_id"]]),
        "current_price": current_price,
        "price_to_28d_avg_ratio": price_ratio,
        "price_change_7d_pct": price_change,
        "sales_lag_1d": float(context["sales_lag_1d"]),
        "sales_lag_7d": float(context["sales_lag_7d"]),
        "sales_lag_28d": float(context["sales_lag_28d"]),
        "sales_ma_7d": float(context["sales_ma_7d"]),
        "sales_ma_28d": float(context["sales_ma_28d"]),
        "sales_std_28d": float(context["sales_std_28d"]),
        "weekday_sin": float(np.sin(weekday_angle)),
        "weekday_cos": float(np.cos(weekday_angle)),
        "month_sin": float(np.sin(month_angle)),
        "month_cos": float(np.cos(month_angle)),
        "event_flag": float(context["event_flag"]),
        "snap_day": float(context["snap_day"]),
    }
    return {name: values[name] for name in FEATURE_COLUMNS}


def price_scenarios(base_features: dict[str, float], candidate_prices) -> pd.DataFrame:
    """Adjust price-derived fields consistently for candidate-price scoring."""
    current = float(base_features["current_price"])
    prior_mean = current / max(float(base_features["price_to_28d_avg_ratio"]), 1e-8)
    prior_week = current / (1.0 + float(base_features["price_change_7d_pct"]))
    if prior_week <= 0:
        prior_week = current
    rows = []
    for candidate in candidate_prices:
        row = dict(base_features)
        row["current_price"] = float(candidate)
        row["price_to_28d_avg_ratio"] = float(candidate) / max(prior_mean, 1e-8)
        row["price_change_7d_pct"] = (float(candidate) - prior_week) / max(prior_week, 1e-8)
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype=float)
