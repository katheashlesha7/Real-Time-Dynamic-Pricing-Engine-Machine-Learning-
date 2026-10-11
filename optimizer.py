"""Business-rule-constrained profit target used for the synthetic demo."""

from __future__ import annotations

import numpy as np
import pandas as pd


def price_bounds(current_price: float, unit_cost: float, competitor_price: float) -> tuple[float, float]:
    """Return the shared floor and ceiling applied during labeling and serving."""
    current = max(float(current_price), 0.01)
    cost = max(float(unit_cost), 0.01)
    competitor = max(float(competitor_price), 0.01)
    floor = max(cost * 1.02, current * 0.70)
    ceiling = max(floor, min(current * 1.50, competitor * 1.30, cost * 3.5))
    return floor, ceiling


def optimal_price(
    current_price: float,
    unit_cost: float,
    competitor_price: float,
    inventory_units: float,
    demand_baseline: float,
    elasticity: float,
    *,
    grid_size: int = 41,
) -> float:
    """Maximize expected one-hour gross profit on a bounded candidate grid.

    Demand follows a power curve with a bounded competitor-attractiveness term.
    This is a transparent synthetic oracle, not a claim of causal identification.
    """
    current = max(float(current_price), 0.01)
    cost = max(float(unit_cost), 0.01)
    competitor = max(float(competitor_price), 0.01)
    floor, ceiling = price_bounds(current, cost, competitor)
    candidates = np.linspace(floor, ceiling, grid_size)
    gross_profit = expected_gross_profit(
        candidates,
        current,
        cost,
        competitor,
        inventory_units,
        demand_baseline,
        elasticity,
    )
    return float(candidates[int(np.argmax(gross_profit))])


def expected_gross_profit(
    price: float | np.ndarray,
    current_price: float | np.ndarray,
    unit_cost: float | np.ndarray,
    competitor_price: float | np.ndarray,
    inventory_units: float | np.ndarray,
    demand_baseline: float | np.ndarray,
    elasticity: float | np.ndarray,
) -> float | np.ndarray:
    """Evaluate the demo's expected one-hour gross profit at one or more prices."""
    candidate = np.maximum(np.asarray(price, dtype=float), 0.01)
    current = np.maximum(np.asarray(current_price, dtype=float), 0.01)
    cost = np.maximum(np.asarray(unit_cost, dtype=float), 0.01)
    competitor = np.maximum(np.asarray(competitor_price, dtype=float), 0.01)
    stock = np.maximum(np.asarray(inventory_units, dtype=float), 0.0)
    baseline = np.maximum(np.asarray(demand_baseline, dtype=float), 0.0)
    price_elasticity = np.clip(np.asarray(elasticity, dtype=float), 0.25, 3.0)

    price_response = np.power(candidate / current, -price_elasticity)
    competitor_response = np.exp(
        np.clip(1.15 * (competitor - candidate) / competitor, -1.4, 0.7)
    )
    expected_demand = baseline * price_response * competitor_response
    expected_sales = np.minimum(expected_demand, stock)
    profit = (candidate - cost) * expected_sales
    return float(profit) if profit.ndim == 0 else profit


def add_optimal_price_targets(features: pd.DataFrame) -> pd.DataFrame:
    """Append the synthetic oracle's per-hour recommended price."""
    data = features.copy()
    data["target_optimal_price"] = [
        optimal_price(
            row.current_price,
            row.unit_cost,
            row.competitor_price,
            row.inventory_units,
            row.demand_ma_168h,
            row.price_elasticity_168h,
        )
        for row in data.itertuples()
    ]
    return data
