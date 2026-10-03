"""
pstone_data.py  --  load the cleaned data and cut walk-forward windows.

Rules from the context file that this file enforces:
  * each asset keeps its own trading days (no forward-fill, no common calendar);
  * training data for a test year = all rows strictly before 1 January of that year;
  * an asset enters a test year only with >= 3 calendar years of earlier history;
  * data files are only read, never written.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import pstone_config as cfg


def load_manifest(data_dir: Path | str = cfg.DATA_DIR) -> pd.DataFrame:
    data_dir = Path(data_dir)
    path = data_dir / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"manifest.csv not found in {data_dir}. Point --data-dir (or the PSTONE_DATA "
            f"environment variable) at the folder 'by_class'."
        )
    return pd.read_csv(path)


def asset_info(asset_id: str, data_dir: Path | str = cfg.DATA_DIR) -> dict:
    m = load_manifest(data_dir)
    row = m[m["asset_id"] == asset_id]
    if row.empty:
        raise KeyError(f"asset '{asset_id}' is not in manifest.csv")
    return row.iloc[0].to_dict()


def apply_exclusions(df: pd.DataFrame, asset_id: str, path: Path = cfg.EXCLUSIONS_FILE) -> pd.DataFrame:
    """
    Remove documented bad-tick rows (data_exclusions.csv: asset_id,date,reason).
    The bad day's row is dropped and its log return is added to the next row, so the price level
    is unchanged and no zero return is invented (log returns add up across the gap).
    """
    if not Path(path).exists():
        return df
    ex = pd.read_csv(path, comment="#")
    dates = pd.to_datetime(ex.loc[ex["asset_id"] == asset_id, "date"])
    if dates.empty:
        return df
    df = df.copy()
    for d in sorted(dates):
        hit = df.index[df["date"] == d]
        if len(hit) == 0:
            continue
        i = hit[0]
        if i + 1 < len(df):
            df.loc[i + 1, "log_ret"] += df.loc[i, "log_ret"]
            if "gap_days" in df:
                df.loc[i + 1, "gap_days"] += df.loc[i, "gap_days"]
            if "simple_ret" in df:
                df.loc[i + 1, "simple_ret"] = np.expm1(df.loc[i + 1, "log_ret"])
        df = df.drop(index=i).reset_index(drop=True)
    return df


def load_close(asset_id: str, data_dir: Path | str = cfg.DATA_DIR, exclusions: bool = True) -> pd.DataFrame:
    """Close-to-close view of one asset: date, close, log_ret, gap_days, (peg_dev), ..."""
    data_dir = Path(data_dir)
    info = asset_info(asset_id, data_dir)
    path = data_dir / info["asset_class"] / "close" / f"{asset_id}.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].duplicated().any():
        raise ValueError(f"{asset_id}: duplicate dates in {path}")
    if exclusions:
        df = apply_exclusions(df, asset_id)
    df.attrs["asset_id"] = asset_id
    df.attrs["asset_class"] = info["asset_class"]
    return df


def rows_per_week(asset_class: str) -> int:
    return 7 if asset_class in cfg.SEVEN_DAY_CLASSES else 5


def walk_forward_split(df: pd.DataFrame, year: int) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Expanding window: train = rows before 1 Jan of `year`, test = rows in `year`.
    Returns None if the asset has < 3 years of history before the test year or no test rows.
    """
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year + 1}-01-01")
    train = df[df["date"] < start]
    test = df[(df["date"] >= start) & (df["date"] < end)]
    if train.empty or test.empty:
        return None
    if (start - train["date"].min()).days < cfg.MIN_HISTORY_DAYS:
        return None
    return train.reset_index(drop=True), test.reset_index(drop=True)


def eligible_years(df: pd.DataFrame, years=cfg.TEST_YEARS) -> list[int]:
    return [y for y in years if walk_forward_split(df, y) is not None]


def load_panel(name: str, data_dir: Path | str = cfg.DATA_DIR) -> pd.DataFrame:
    """A group panel (log returns on common dates). Index = date (or week_ending)."""
    path = Path(data_dir) / "panels" / cfg.PANELS[name]
    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else "week_ending"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    return df.drop(columns=[c for c in ["gap_days"] if c in df.columns])


def split_panel(panel: pd.DataFrame, year: int) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year + 1}-01-01")
    train = panel[panel.index < start]
    test = panel[(panel.index >= start) & (panel.index < end)]
    if train.empty or test.empty:
        return None
    if (start - train.index.min()).days < cfg.MIN_HISTORY_DAYS:
        return None
    return train, test
