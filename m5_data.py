"""Load the official M5 CSV files without melting the full sales matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMNS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _top_sales_rows(sales_path: Path, series_limit: int, chunk_rows: int = 256) -> pd.DataFrame:
    header = pd.read_csv(sales_path, nrows=0).columns.tolist()
    day_columns = [column for column in header if column.startswith("d_")]
    if not day_columns:
        raise ValueError("sales_train_validation.csv has no d_1, d_2, ... sales columns.")
    ids = [column for column in ID_COLUMNS if column in header]
    if not {"item_id", "store_id"}.issubset(ids):
        raise ValueError("M5 sales file must include item_id and store_id columns.")

    # Choose the subset using earlier history only; keep the final 56 days
    # untouched for validation and test rather than selecting on those labels.
    selection_days = day_columns[:-56] if len(day_columns) > 56 else day_columns
    candidates = pd.DataFrame()
    for chunk in pd.read_csv(
        sales_path,
        usecols=ids + day_columns,
        dtype={name: "string" for name in ids},
        chunksize=chunk_rows,
    ):
        chunk["_selection_sales"] = chunk[selection_days].sum(axis=1)
        candidates = pd.concat([candidates, chunk], ignore_index=True)
        candidates = candidates.nlargest(series_limit, "_selection_sales")

    if candidates.empty:
        raise ValueError("No product-store rows were found in the M5 sales file.")
    return candidates.drop(columns="_selection_sales").reset_index(drop=True)


def _prices_for_selected_series(prices_path: Path, selected: pd.DataFrame) -> pd.DataFrame:
    selected_pairs = set(zip(selected["item_id"].astype(str), selected["store_id"].astype(str)))
    pieces = []
    for chunk in pd.read_csv(
        prices_path,
        usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
        dtype={"store_id": "string", "item_id": "string"},
        chunksize=250_000,
    ):
        pairs = zip(chunk["item_id"].astype(str), chunk["store_id"].astype(str))
        mask = np.fromiter((pair in selected_pairs for pair in pairs), dtype=bool, count=len(chunk))
        if mask.any():
            pieces.append(chunk.loc[mask])
    if not pieces:
        raise ValueError("No matching rows found in sell_prices.csv for selected products/stores.")
    prices = pd.concat(pieces, ignore_index=True)
    prices["wm_yr_wk"] = prices["wm_yr_wk"].astype(int)
    return prices.drop_duplicates(["item_id", "store_id", "wm_yr_wk"])


def load_m5_dataset(data_dir: str | Path, series_limit: int = 12) -> tuple[pd.DataFrame, dict, dict]:
    """Return the top-selling item/store daily panel, item ids and store ids.

    Required files are calendar.csv, sales_train_validation.csv, and
    sell_prices.csv from the m5 Forecasting Accuracy competition download.
    The series limit controls memory and training time; selection uses sales
    from before the final 56 days so validation and test labels do not drive it.
    """
    root = Path(data_dir)
    expected = ["calendar.csv", "sales_train_validation.csv", "sell_prices.csv"]
    missing = [name for name in expected if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing m5 file(s) in {root.resolve()}: {', '.join(missing)}. "
            "Download and extract the three files into this directory."
        )
    if series_limit < 1:
        raise ValueError("series_limit must be at least 1.")

    selected = _top_sales_rows(root / "sales_train_validation.csv", series_limit)
    id_columns = [name for name in ID_COLUMNS if name in selected.columns]
    days = [name for name in selected.columns if name.startswith("d_")]
    sales = selected.melt(
        id_vars=id_columns,
        value_vars=days,
        var_name="d",
        value_name="sales_units",
    )
    sales["sales_units"] = pd.to_numeric(sales["sales_units"], errors="coerce").fillna(0)

    calendar = pd.read_csv(root / "calendar.csv", parse_dates=["date"])
    calendar_columns = ["d", "date", "wm_yr_wk", "event_name_1", "event_name_2"]
    optional_snap_columns = [name for name in ("snap_CA", "snap_TX", "snap_WI") if name in calendar]
    calendar = calendar[[name for name in calendar_columns + optional_snap_columns if name in calendar]].copy()
    calendar["event_flag"] = (
        calendar.get("event_name_1", pd.Series(index=calendar.index, dtype=object)).notna()
        | calendar.get("event_name_2", pd.Series(index=calendar.index, dtype=object)).notna()
    ).astype(float)
    sales = sales.merge(calendar, on="d", how="left", validate="many_to_one")

    prices = _prices_for_selected_series(root / "sell_prices.csv", selected)
    panel = sales.merge(
        prices,
        on=["item_id", "store_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )
    panel = panel.sort_values(["item_id", "store_id", "date"])
    # Prices are weekly. Carry a known price across any absent week within a
    # listed series, and remove pre-launch days that have no historical price.
    panel["sell_price"] = panel.groupby(["item_id", "store_id"], sort=False)["sell_price"].ffill()
    panel = panel.dropna(subset=["sell_price", "date"]).copy()
    if panel.empty:
        raise ValueError("No daily M5 rows with a matching sell price were produced.")

    panel["snap_day"] = 0.0
    for state in ("CA", "TX", "WI"):
        name = f"snap_{state}"
        if name in panel.columns:
            state_rows = panel["state_id"].astype(str) == state
            panel.loc[state_rows, "snap_day"] = panel.loc[state_rows, name].fillna(0).astype(float)
    item_values = sorted(panel["item_id"].astype(str).unique())
    store_values = sorted(panel["store_id"].astype(str).unique())
    item_indices = {value: index for index, value in enumerate(item_values)}
    store_indices = {value: index for index, value in enumerate(store_values)}
    panel["item_index"] = panel["item_id"].astype(str).map(item_indices).astype(int)
    panel["store_index"] = panel["store_id"].astype(str).map(store_indices).astype(int)
    panel["item_id"] = panel["item_id"].astype(str)
    panel["store_id"] = panel["store_id"].astype(str)
    return panel.reset_index(drop=True), item_indices, store_indices
