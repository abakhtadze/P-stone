"""
run_utility.py  --  Wave 2 Utility gate: the four arms (A real, B real+regularisation,
C real+synthetic, D synthetic only) on next-week realised volatility, per asset and test year,
with Diebold-Mariano tests and a per-asset-class summary.

Examples
    python run_utility.py --assets NDX --years 2022 --n-paths 10 --max-iter 100     # quick check
    python run_utility.py                                                           # NDX DJI IBEX EURUSD, all years
    python run_utility.py --tier1                                                   # all Tier-1 except stablecoins

Stablecoins are skipped: the context file excludes them from pooled return-based training.
Leveraged ETFs are run as their own series and never pooled with their index.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

import pstone_config as cfg
import pstone_data as D
import pstone_utility as U


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(cfg.DATA_DIR))
    ap.add_argument("--out", default=str(cfg.OUT_DIR / "utility"))
    ap.add_argument("--assets", nargs="+", default=None)
    ap.add_argument("--tier1", action="store_true")
    ap.add_argument("--years", nargs="+", type=int, default=cfg.TEST_YEARS)
    ap.add_argument("--generator", default="garch_fhs", choices=["garch_fhs", "bootstrap", "random_walk"])
    ap.add_argument("--n-paths", type=int, default=20, help="synthetic paths used for training (each as long as the training window)")
    ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--no-exclusions", action="store_true", help="ignore data_exclusions.csv (use the raw cleaned data)")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    a = ap.parse_args()

    if not a.no_exclusions:
        print("Applying documented data exclusions from data_exclusions.csv (use --no-exclusions to switch off).")
    assets = a.assets or (cfg.TIER1 if a.tier1 else ["NDX", "DJI", "IBEX", "EURUSD"])
    assets = [x for x in assets if x not in cfg.STABLECOINS]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for asset in assets:
        df = D.load_close(asset, a.data_dir, exclusions=not a.no_exclusions)
        for year in D.eligible_years(df, a.years):
            try:
                res = U.run_four_arms(df, year, a.generator, a.n_paths, a.max_iter, a.seed)
            except Exception as e:
                print(f"  ! {asset} {year}: {e}"); continue
            if res is None:
                continue
            row = U.flatten_result(res); rows.append(row)
            print(f"{asset:7s} {year}  MSE  A {row['mse_A_real']:.4f}  B {row['mse_B_real_reg']:.4f}  "
                  f"C {row['mse_C_real_synth']:.4f}  D {row['mse_D_synth_only']:.4f}   "
                  f"C<B: {row['C_beats_B']}  (DM p={row['dm_p_C_vs_B']:.2f})")
    if not rows:
        print("Nothing was run."); return
    df_res = pd.DataFrame(rows)
    df_res.to_csv(out / "utility_results.csv", index=False)
    summ = U.summarise_utility(df_res)
    summ.to_csv(out / "utility_summary_by_class.csv", index=False)
    print("\nPer asset class (C must beat B, not just A):")
    print(summ.round(3).to_string(index=False))
    print(f"\nWritten to {out}  ({time.time() - t0:.0f}s).")
    print("Lower MSE is better. Report the result as it is, including when C does not beat B.")


if __name__ == "__main__":
    main()
