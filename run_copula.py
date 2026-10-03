"""
run_copula.py  --  Wave 3: multi-asset generator (GJR-GARCH per asset + Student-t copula) on the
group panels. Gate: synthetic cross-asset correlations (and joint crashes) match the real ones.

Examples
    python run_copula.py --panels us_equity_daily --years 2022 --n-paths 100
    python run_copula.py                      # all four panels, all test years

Panels never include ETFs, leveraged ETFs or stablecoins (near-duplicates / pegs), as the context
file requires. The cross_asset_weekly panel uses weekly returns, so cross-region links are fair.
"""
from __future__ import annotations

import argparse
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import pstone_config as cfg
import pstone_data as D
from pstone_models import TCopulaFHS


def pair_corr(paths: np.ndarray, i: int, j: int) -> np.ndarray:
    a, b = paths[:, :, i], paths[:, :, j]
    a = a - a.mean(axis=1, keepdims=True); b = b - b.mean(axis=1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))


def joint_crash_freq(x: np.ndarray, q: float, ref_q: np.ndarray) -> np.ndarray:
    """Share of rows where ALL assets are below their own training q-quantile. x: (..., T, d)."""
    return np.mean(np.all(x < ref_q, axis=-1), axis=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(cfg.DATA_DIR))
    ap.add_argument("--out", default=str(cfg.OUT_DIR / "copula"))
    ap.add_argument("--panels", nargs="+", default=list(cfg.PANELS))
    ap.add_argument("--years", nargs="+", type=int, default=cfg.TEST_YEARS)
    ap.add_argument("--n-paths", type=int, default=200)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pair_rows, tail_rows, t0 = [], [], time.time()
    for name in a.panels:
        panel = D.load_panel(name, a.data_dir)
        cols = list(panel.columns)
        for year in a.years:
            sp = D.split_panel(panel, year)
            if sp is None:
                continue
            train, _ = sp
            try:
                gen = TCopulaFHS().fit(train.to_numpy(), cols)
                paths = gen.sample_log_returns(a.n_paths, len(train), np.random.default_rng(a.seed + year))
            except Exception as e:
                print(f"  ! {name} {year}: {e}"); continue
            real = train.to_numpy()
            for i, j in combinations(range(len(cols)), 2):
                rc = float(np.corrcoef(real[:, i], real[:, j])[0, 1])
                sc = pair_corr(paths, i, j)
                lo, med, hi = np.quantile(sc, [0.05, 0.5, 0.95])
                pair_rows.append(dict(panel=name, year=year, pair=f"{cols[i]}-{cols[j]}", real_corr=rc,
                                      synth_median=med, synth_p05=lo, synth_p95=hi, inside_band=bool(lo <= rc <= hi),
                                      abs_diff=abs(rc - med)))
            q05 = np.quantile(real, 0.05, axis=0)
            rj = float(np.mean(np.all(real < q05, axis=1)))
            sj = joint_crash_freq(paths, 0.05, q05)
            tail_rows.append(dict(panel=name, year=year, n_assets=len(cols), nu=gen.nu, real_joint_crash=rj,
                                  synth_median=float(np.median(sj)), synth_p05=float(np.quantile(sj, 0.05)),
                                  synth_p95=float(np.quantile(sj, 0.95)),
                                  inside_band=bool(np.quantile(sj, 0.05) <= rj <= np.quantile(sj, 0.95))))
            print(f"{name:20s} {year}  t-copula nu={gen.nu}  max |corr diff| = "
                  f"{max(r['abs_diff'] for r in pair_rows if r['panel'] == name and r['year'] == year):.3f}")
    if not pair_rows:
        print("Nothing was run."); return
    pairs, tails = pd.DataFrame(pair_rows), pd.DataFrame(tail_rows)
    pairs.to_csv(out / "copula_pair_correlations.csv", index=False)
    tails.to_csv(out / "copula_joint_crashes.csv", index=False)
    summ = pairs.groupby("panel").agg(pairs=("pair", "nunique"), share_inside_band=("inside_band", "mean"),
                                      mean_abs_diff=("abs_diff", "mean"), max_abs_diff=("abs_diff", "max"))
    print("\nCorrelation match per panel (real correlation inside the synthetic 5-95% band):")
    print(summ.round(3).to_string())
    print("\nJoint-crash frequency (all assets below their 5% quantile on the same day):")
    print(tails.groupby("panel")[["real_joint_crash", "synth_median"]].mean().round(4).to_string())
    print(f"\nWritten to {out}  ({time.time() - t0:.0f}s). All paths are SYNTHETIC.")


if __name__ == "__main__":
    main()
