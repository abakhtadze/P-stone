"""
pstone_leveraged.py  --  Phase 2 layer: build a synthetic leveraged ETF from SYNTHETIC index paths.

For each synthetic index path we apply the ETF's rule, close to close:
    lev_return = L * simple_return  -  daily_cost
    daily_cost = (expense + (L - 1) * (rf + spread)) * calendar_days / 365
and chain the days (volatility decay appears by itself). This is the same idea as
leveraged_engine.py (which also rebuilds open/high/low); use that file for full candles
and for fitting the spread. Real TQQQ/SPXL are only used to TEST the result.

NOTE: the spread fitted on real SPY/SPXL data came out negative (the engine has no dividend
term), so fix that before treating a fitted spread as validated.
SYNTHETIC DATA: nothing here is a real price.
"""
from __future__ import annotations

import numpy as np


def leveraged_paths(index_log_ret: np.ndarray, leverage: float = 3.0, expense: float = 0.0095,
                    rf: float | np.ndarray = 0.02, spread: float = 0.005, calendar_days: float = 1.4,
                    start_price: float = 100.0, floor: float = 1e-6) -> np.ndarray:
    """
    index_log_ret : (n_paths, T) synthetic index log returns
    rf            : yearly rate as a fraction (number, or array of length T)
    calendar_days : average calendar days per row (about 1.4 for a 5-day market: weekends included)
    Returns synthetic leveraged-ETF price paths (n_paths, T).
    """
    simple = np.expm1(np.asarray(index_log_ret, dtype=float))
    cost = (expense + (leverage - 1.0) * (np.asarray(rf) + spread)) * calendar_days / 365.0
    growth = np.maximum(1.0 + leverage * simple - cost, floor)        # a fund cannot go below zero
    return start_price * np.cumprod(growth, axis=-1)
