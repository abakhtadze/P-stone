"""
pstone_gate.py  --  the quality gate: Fidelity and Novelty.

(Utility, the third gate, is in pstone_utility.py.)

FIDELITY  Does the synthetic data have the same statistics as the REAL TRAINING data?
          For each statistic we compute it on every synthetic path, then check whether the real
          value falls inside the 5%-95% band of the synthetic values. The band rule is a
          PROVISIONAL pass line: the numeric thresholds are for the team to agree after the
          first real run.

NOVELTY   Are the synthetic paths new, or copies of history? We cut standardised windows of 64
          rows and measure the distance from each synthetic window to its nearest real TRAINING
          window. As a yardstick we measure the same for held-out real windows (the last 20% of
          the training years) against the earlier real windows.

All statistics are computed on the training window only, as the context file requires.
SYNTHETIC DATA is never presented as real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import pstone_config as cfg

CORE_METRICS = ["excess_kurtosis", "q01", "q99", "acf1_ret", "acf1_sq", "acf1_abs", "leverage_corr"]


def _acf(x: np.ndarray, lag: int) -> np.ndarray:
    """Autocorrelation at `lag`, along the last axis. x: (..., T)."""
    xc = x - x.mean(axis=-1, keepdims=True)
    num = np.sum(xc[..., :-lag] * xc[..., lag:], axis=-1)
    den = np.sum(xc ** 2, axis=-1)
    return num / np.where(den == 0, np.nan, den)


def path_metrics(r: np.ndarray) -> dict[str, np.ndarray]:
    """Statistical fingerprint of each path. r: (n_paths, T) or (T,) of log returns."""
    r = np.atleast_2d(np.asarray(r, dtype=float))
    sd = r.std(axis=-1)
    z = (r - r.mean(axis=-1, keepdims=True)) / np.where(sd == 0, np.nan, sd)[:, None]
    q = np.quantile(r, [0.01, 0.05, 0.95, 0.99], axis=-1)
    r1, sq1 = r[:, :-1], r[:, 1:] ** 2
    lev = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(r1, sq1)])
    return {
        "std": sd,
        "excess_kurtosis": np.mean(z ** 4, axis=-1) - 3.0,
        "q01": q[0], "q05": q[1], "q95": q[2], "q99": q[3],
        "acf1_ret": _acf(r, 1),
        "acf1_sq": _acf(r ** 2, 1),
        "acf5_sq": _acf(r ** 2, 5),
        "acf1_abs": _acf(np.abs(r), 1),
        "leverage_corr": lev,
    }


def peg_metrics(price: np.ndarray) -> dict[str, np.ndarray]:
    """Extra statistics for pegged assets. price: (n_paths, T)."""
    price = np.atleast_2d(price)
    dev = price - 1.0
    return {
        "peg_mean_abs_dev": np.mean(np.abs(dev), axis=-1),
        "peg_end_abs_dev": np.abs(dev[:, -1]),
        "peg_share_off_1c": np.mean(np.abs(dev) > 0.01, axis=-1),
    }


def fidelity_table(real_ret: np.ndarray, synth_ret: np.ndarray,
                   real_price: np.ndarray | None = None, synth_price: np.ndarray | None = None,
                   band: tuple[float, float] = (0.05, 0.95)) -> pd.DataFrame:
    """
    One row per statistic: real value, synthetic median and band, and whether the real value
    is inside the band. Pass real_price/synth_price (stablecoins) to add the peg statistics.
    """
    real = {k: float(v[0]) for k, v in path_metrics(real_ret).items()}
    syn = path_metrics(synth_ret)
    if real_price is not None and synth_price is not None:
        real.update({k: float(v[0]) for k, v in peg_metrics(real_price).items()})
        syn.update(peg_metrics(synth_price))
    rows = []
    for k, rv in real.items():
        sv = syn[k][np.isfinite(syn[k])]
        lo, med, hi = np.quantile(sv, [band[0], 0.5, band[1]]) if len(sv) else (np.nan,) * 3
        rows.append({"metric": k, "real": rv, "synth_median": med, "synth_p05": lo, "synth_p95": hi,
                     "inside_band": bool(lo <= rv <= hi) if np.isfinite(lo) else False,
                     "core": k in CORE_METRICS or k.startswith("peg_")})
    return pd.DataFrame(rows)


def fidelity_summary(tab: pd.DataFrame) -> dict:
    core = tab[tab["core"]]
    return {"core_metrics": len(core), "core_inside_band": int(core["inside_band"].sum()),
            "fidelity_pass_rate": float(core["inside_band"].mean()) if len(core) else np.nan}


# ----------------------------------------------------------------------------
# Novelty
# ----------------------------------------------------------------------------
def _windows(x: np.ndarray, size: int, stride: int) -> np.ndarray:
    idx = np.arange(0, len(x) - size + 1, stride)
    return np.stack([x[i: i + size] for i in idx]) if len(idx) else np.empty((0, size))


def novelty_report(train_ret: np.ndarray, synth_ret: np.ndarray, window: int = cfg.NOVELTY_WINDOW,
                   stride_synth: int = cfg.NOVELTY_STRIDE_SYNTH, max_synth_windows: int = 1500,
                   seed: int = 0) -> dict:
    """
    Nearest-real-window distance on returns standardised (median/MAD) with TRAINING statistics.

    synth_to_real   : synthetic window -> nearest real training window
    holdout_to_real : held-out real window (last 20% of training) -> nearest earlier real window
    near_copy_share : share of synthetic windows closer to a real window than the 5th percentile
                      of the holdout distances. Values far above 5% suggest copying.
    """
    train_ret = np.asarray(train_ret, dtype=float)
    # robust scaling (median and MAD): a few extreme days must not shrink every other window to ~0
    mu = float(np.median(train_ret))
    sd = float(1.4826 * np.median(np.abs(train_ret - mu))) or float(train_ret.std())
    zt = (train_ret - mu) / sd
    cut = int(len(zt) * 0.8)
    real_fit = _windows(zt[:cut], window, 1)
    holdout = _windows(zt[cut + window:], window, 5)           # purge gap of one window
    real_all = _windows(zt, window, 1)
    syn = np.atleast_2d((np.asarray(synth_ret, dtype=float) - mu) / sd)
    rng = np.random.default_rng(seed)
    wins = np.concatenate([_windows(p, window, stride_synth) for p in syn[: min(len(syn), 200)]])
    if len(wins) > max_synth_windows:
        wins = wins[rng.choice(len(wins), max_synth_windows, replace=False)]

    def nn(a, b):
        return cdist(a, b).min(axis=1) / np.sqrt(window)       # per-row RMS distance

    d_syn = nn(wins, real_all)
    d_hold = nn(holdout, real_fit) if len(holdout) and len(real_fit) else np.array([np.nan])
    thr = np.nanquantile(d_hold, 0.05)
    q = lambda d: np.nanquantile(d, [0.05, 0.5, 0.95])
    return {
        "synth_to_real_p05": q(d_syn)[0], "synth_to_real_median": q(d_syn)[1], "synth_to_real_p95": q(d_syn)[2],
        "holdout_to_real_p05": q(d_hold)[0], "holdout_to_real_median": q(d_hold)[1], "holdout_to_real_p95": q(d_hold)[2],
        "near_copy_share": float(np.mean(d_syn < thr)),
        "n_synth_windows": int(len(wins)), "n_holdout_windows": int(len(holdout)),
    }
