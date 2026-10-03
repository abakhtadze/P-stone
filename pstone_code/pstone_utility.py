"""
pstone_utility.py  --  the Utility gate: do synthetic "parallel histories" help a forecaster?

Target   next-week realised volatility  = sqrt(sum of squared log returns over the next
         5 rows (7 for crypto)), modelled in logs.
Model    HistGradientBoostingRegressor (a higher-capacity forecaster).
Four arms (same forecaster, same target, same test year):
    A  real only
    B  real + standard regularisation (L2, early stopping, larger leaves)
    C  real + synthetic            <- must beat B, not just A
    D  synthetic only, tested on real
Hyperparameters of each arm are chosen on a chronological validation block (the last 20% of
the training window). For arms C and D the generator used during selection is fitted on the
first 80% only, so the validation block never leaks into the synthetic data; the final models
are refitted on the full training window with a generator fitted on the full training window.

Comparison: Diebold-Mariano test (Harvey small-sample correction) on squared forecast errors,
C versus B and C versus A. Also reported: how stable the chosen hyperparameters are.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

import pstone_config as cfg
import pstone_data as D
from pstone_models import fit_generator

ARMS = ["A_real", "B_real_reg", "C_real_synth", "D_synth_only"]


# ----------------------------------------------------------------------------
# features and target
# ----------------------------------------------------------------------------
def make_features(r: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From a log-return series r build (X, y, t_index). Row t uses information up to and including t;
    the target is log of the realised volatility over rows t+1 .. t+h.
    """
    r = np.asarray(r, dtype=float)
    n = len(r)
    s = pd.Series(r)
    sq = s ** 2
    feats = {
        "rv_h": np.sqrt(sq.rolling(h).sum()),
        "rv_4h": np.sqrt(sq.rolling(4 * h).sum() / 4.0),
        "rv_12h": np.sqrt(sq.rolling(12 * h).sum() / 12.0),
        "abs_r": s.abs(),
        "r": s,
        "neg_share_h": (s < 0).rolling(h).mean().astype(float),
        "min_h": s.rolling(h).min(),
    }
    X = pd.DataFrame(feats)
    fut = np.sqrt(sq[::-1].rolling(h).sum()[::-1].shift(-1))      # sum of squares over t+1..t+h
    y = np.log(fut.replace(0, np.nan))
    X = X.assign(rv_h=np.log(X["rv_h"].replace(0, np.nan)), rv_4h=np.log(X["rv_4h"].replace(0, np.nan)),
                 rv_12h=np.log(X["rv_12h"].replace(0, np.nan)))
    ok = X.notna().all(axis=1) & y.notna()
    return X[ok].to_numpy(), y[ok].to_numpy(), np.flatnonzero(ok.to_numpy())


def _paths_to_xy(paths: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for p in paths:
        X, y, _ = make_features(p, h)
        xs.append(X); ys.append(y)
    return np.vstack(xs), np.concatenate(ys)


# ----------------------------------------------------------------------------
# configurations
# ----------------------------------------------------------------------------
def config_grid(regularised: bool, max_iter: int) -> list[dict]:
    grid = []
    for depth in (3, 6):
        for lr in (0.03, 0.1):
            c = dict(max_depth=depth, learning_rate=lr, max_iter=max_iter)
            if regularised:
                c.update(l2_regularization=1.0, min_samples_leaf=40, early_stopping=True,
                         validation_fraction=0.15, n_iter_no_change=15)
            else:
                c.update(l2_regularization=0.0, min_samples_leaf=5, early_stopping=False)
            grid.append(c)
    return grid


def _fit(cfg_: dict, X, y, w=None, seed=0):
    m = HistGradientBoostingRegressor(random_state=seed, **cfg_)
    m.fit(X, y, sample_weight=w)
    return m


def _mse(m, X, y) -> float:
    return float(np.mean((m.predict(X) - y) ** 2))


# ----------------------------------------------------------------------------
# Diebold-Mariano
# ----------------------------------------------------------------------------
def diebold_mariano(loss_ref: np.ndarray, loss_new: np.ndarray, h: int) -> dict:
    """
    H1: `new` has LOWER expected loss than `ref`. d = loss_ref - loss_new (positive = new is better).
    Newey-West variance with h-1 lags (forecasts overlap over h rows) and Harvey-Leybourne-Newbold correction.
    """
    d = np.asarray(loss_ref) - np.asarray(loss_new)
    n = len(d)
    if n < 10:
        return {"dm_stat": np.nan, "p_one_sided": np.nan, "mean_loss_diff": float(np.mean(d)) if n else np.nan}
    dbar = d.mean()
    dc = d - dbar
    gam = [np.dot(dc[: n - k], dc[k:]) / n for k in range(h)]
    var = (gam[0] + 2.0 * sum(gam[1:])) / n
    if var <= 0:
        return {"dm_stat": np.nan, "p_one_sided": np.nan, "mean_loss_diff": float(dbar)}
    dm = dbar / np.sqrt(var)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {"dm_stat": float(dm), "p_one_sided": float(1.0 - stats.t.cdf(dm, df=n - 1)),
            "mean_loss_diff": float(dbar)}


# ----------------------------------------------------------------------------
# the four arms for one asset and one test year
# ----------------------------------------------------------------------------
def run_four_arms(asset_df: pd.DataFrame, year: int, generator: str = "garch_fhs", n_paths: int = 20,
                  max_iter: int = 150, seed: int = cfg.SEED, val_frac: float = 0.2) -> dict | None:
    """Returns a dict with test losses per arm and the chosen hyperparameters, or None if not eligible."""
    split = D.walk_forward_split(asset_df, year)
    if split is None:
        return None
    train, test = split
    cls = asset_df.attrs.get("asset_class", "equity_index")
    h = D.rows_per_week(cls)
    rng = np.random.default_rng(seed + year)

    r_train = train["log_ret"].to_numpy()
    # Test features need history before the test year: use the last rows of the training window as warm-up.
    warm = 12 * h + 5
    r_test_full = np.concatenate([r_train[-warm:], test["log_ret"].to_numpy()])
    Xte, yte, tix = make_features(r_test_full, h)
    keep = tix >= warm - 1                       # rows whose "now" lies inside the test year
    Xte, yte = Xte[keep], yte[keep]
    if len(yte) < 30:
        return None

    # real training and validation block (chronological, with a purge gap of h rows)
    Xtr, ytr, _ = make_features(r_train, h)
    nv = int(len(ytr) * val_frac)
    Xa, ya = Xtr[: len(ytr) - nv - h], ytr[: len(ytr) - nv - h]
    Xv, yv = Xtr[len(ytr) - nv:], ytr[len(ytr) - nv:]

    # synthetic data: one generator on the inner window (for selection), one on the full window (final)
    inner_len = len(r_train) - nv - h
    gen_in = fit_generator(generator, train.iloc[:inner_len])
    gen_full = fit_generator(generator, train)
    Xs_in, ys_in = _paths_to_xy(gen_in.sample_log_returns(n_paths, inner_len, rng), h)
    Xs_full, ys_full = _paths_to_xy(gen_full.sample_log_returns(n_paths, len(r_train), rng), h)
    w_syn_in = np.full(len(ys_in), 1.0 / n_paths)           # total weight of synthetic = one history
    w_syn_full = np.full(len(ys_full), 1.0 / n_paths)

    out = {"asset": asset_df.attrs.get("asset_id"), "asset_class": cls, "year": year, "generator": generator,
           "h": h, "n_test": len(yte), "n_paths": n_paths}
    losses, chosen = {}, {}
    for arm in ARMS:
        regularised = arm != "A_real" and arm != "D_synth_only"
        if arm == "D_synth_only":
            regularised = True
        grid = config_grid(regularised, max_iter)

        def train_set(final: bool):
            if arm in ("A_real", "B_real_reg"):
                return (Xtr, ytr, None) if final else (Xa, ya, None)
            xs, ys, ws = (Xs_full, ys_full, w_syn_full) if final else (Xs_in, ys_in, w_syn_in)
            if arm == "D_synth_only":
                return xs, ys, None
            return (np.vstack([Xtr, xs]), np.concatenate([ytr, ys]), np.concatenate([np.ones(len(ytr)), ws])) if final \
                else (np.vstack([Xa, xs]), np.concatenate([ya, ys]), np.concatenate([np.ones(len(ya)), ws]))

        Xt, yt, wt = train_set(final=False)
        scores = [_mse(_fit(c, Xt, yt, wt, seed), Xv, yv) for c in grid]
        best = int(np.argmin(scores))
        Xf, yf, wf = train_set(final=True)
        model = _fit(grid[best], Xf, yf, wf, seed)
        pred = model.predict(Xte)
        losses[arm] = (pred - yte) ** 2
        chosen[arm] = best
    out["mse"] = {a: float(np.mean(l)) for a, l in losses.items()}
    out["chosen_config"] = chosen
    out["dm_C_vs_B"] = diebold_mariano(losses["B_real_reg"], losses["C_real_synth"], h)
    out["dm_C_vs_A"] = diebold_mariano(losses["A_real"], losses["C_real_synth"], h)
    out["losses"] = losses
    return out


def flatten_result(res: dict) -> dict:
    """One flat row for the results table."""
    row = {k: res[k] for k in ["asset", "asset_class", "year", "generator", "h", "n_test", "n_paths"]}
    for a in ARMS:
        row[f"mse_{a}"] = res["mse"][a]
        row[f"cfg_{a}"] = res["chosen_config"][a]
    for tag in ("C_vs_B", "C_vs_A"):
        d = res[f"dm_{tag}"]
        row[f"dm_stat_{tag}"], row[f"dm_p_{tag}"], row[f"mean_loss_diff_{tag}"] = d["dm_stat"], d["p_one_sided"], d["mean_loss_diff"]
    row["C_beats_B"] = res["mse"]["C_real_synth"] < res["mse"]["B_real_reg"]
    row["C_beats_A"] = res["mse"]["C_real_synth"] < res["mse"]["A_real"]
    return row


def summarise_utility(df: pd.DataFrame) -> pd.DataFrame:
    """Per asset class: how often C beats B and A, mean MSE change, and hyperparameter stability."""
    rows = []
    for (cls, gen), g in df.groupby(["asset_class", "generator"]):
        rows.append({
            "asset_class": cls, "generator": gen, "n_asset_years": len(g),
            "share_C_beats_B": g["C_beats_B"].mean(), "share_C_beats_A": g["C_beats_A"].mean(),
            "median_pct_mse_change_C_vs_B": 100 * np.median(g["mse_C_real_synth"] / g["mse_B_real_reg"] - 1),
            "share_DM_p_below_0.10_C_vs_B": (g["dm_p_C_vs_B"] < 0.10).mean(),
            "distinct_configs_B_per_asset": g.groupby("asset")["cfg_B_real_reg"].nunique().mean(),
            "distinct_configs_C_per_asset": g.groupby("asset")["cfg_C_real_synth"].nunique().mean(),
        })
    return pd.DataFrame(rows)
