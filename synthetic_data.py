"""Reproducible hourly synthetic retail panel for local demos and training."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_history(
    *, products: int = 12, days: int = 120, seed: int = 42, end: str | None = None
) -> pd.DataFrame:
    """Generate demand, competition, inventory, and observed sales by product-hour."""
    rng = np.random.default_rng(seed)
    end_time = pd.Timestamp(end or pd.Timestamp.now(tz="UTC").floor("h"))
    if end_time.tzinfo is None:
        end_time = end_time.tz_localize("UTC")
    else:
        end_time = end_time.tz_convert("UTC")
    timestamps = pd.date_range(end=end_time, periods=days * 24, freq="h")
    rows: list[dict] = []

    for product_num in range(products):
        product_id = f"SKU-{product_num + 1:03d}"
        msrp = float(rng.uniform(18, 240))
        cost = msrp * float(rng.uniform(0.42, 0.72))
        base_demand = float(rng.uniform(4, 34))
        true_elasticity = float(rng.uniform(0.75, 1.8))
        competitor = msrp * float(rng.uniform(0.92, 1.08))
        price = msrp * float(rng.uniform(0.94, 1.06))
        inventory = int(base_demand * rng.uniform(36, 72))

        for index, timestamp in enumerate(timestamps):
            if index and index % 168 == 0:
                inventory += int(base_demand * rng.uniform(48, 96))
            if rng.random() < 0.0015:
                inventory = max(0, int(inventory * rng.uniform(0.2, 0.65)))

            # Competitor prices drift and react to occasional promotions.
            competitor *= float(np.exp(rng.normal(0, 0.006)))
            if rng.random() < 0.004:
                competitor *= float(rng.uniform(0.88, 0.96))
            competitor = float(np.clip(competitor, msrp * 0.72, msrp * 1.32))

            # The current price is the price in force before this hour's sales.
            if index % 24 == 0 or rng.random() < 0.08:
                price += float(rng.normal(0, msrp * 0.025))
                if rng.random() < 0.035:
                    price *= float(rng.uniform(0.88, 0.96))
            price = float(np.clip(price, cost * 1.04, msrp * 1.45))

            hour_factor = 1 + 0.30 * np.sin(2 * np.pi * (timestamp.hour - 9) / 24)
            weekend_factor = 1.12 if timestamp.dayofweek >= 5 else 1.0
            slow_trend = 1 + 0.10 * np.sin(2 * np.pi * index / (24 * 35) + product_num)
            demand_level = max(0.1, base_demand * hour_factor * weekend_factor * slow_trend)
            own_price_factor = (price / msrp) ** (-true_elasticity)
            comp_factor = float(np.exp(np.clip(1.0 * (competitor - price) / competitor, -0.8, 0.5)))
            expected_demand = max(0.0, demand_level * own_price_factor * comp_factor)
            requested = int(rng.poisson(expected_demand))
            sales = min(requested, inventory)

            rows.append(
                {
                    "product_id": product_id,
                    "timestamp": timestamp,
                    "current_price": price,
                    "unit_cost": cost,
                    "competitor_price": competitor,
                    "inventory_units": inventory,
                    "sales_units": sales,
                    "requested_units": requested,
                }
            )
            inventory = max(0, inventory - sales)

    return pd.DataFrame(rows).sort_values(["product_id", "timestamp"]).reset_index(drop=True)

