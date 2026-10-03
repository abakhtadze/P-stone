"""
run_wave1.py  --  Waves 0-2 (single-asset generators): Fidelity + Novelty for every asset,
test year and generator, written to outputs/wave1/.

Examples
    python run_wave1.py                                  # core five, all test years, all generators
    python run_wave1.py --assets NDX --years 2022 --n-paths 200
    python run_wave1.py --tier1                          # all 16 Tier-1 assets
    python run_wave1.py --save-paths 20                  # also save 20 labelled synthetic paths per run

Rules: fit on rows before 1 January of the test year only; same test years for every asset;
stablecoins are tested with the return-based generators (expected to FAIL the peg checks)
and with peg_ar1 (the fix). Synthetic output is labelled and never written into the data folder.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pstone_config as cfg
import pstone_data as D
import pstone_gate as G
from pstone_models import fit_generator, prices_from_log_returns


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(cfg.DATA_DIR))
    ap.add_argument("--out", default=str(cfg.OUT_DIR / "wave1"))
    ap.add_argument("--assets", nargs="+", default=None, help="asset ids (default: the core five)")
    ap.add_argument("--tier1", action="store_true", help="use all Tier-1 assets")
    ap.add_argument("--years", nargs="+", type=int, default=cfg.TEST_YEARS)
    ap.add_argument("--generators", nargs="+", default=["random_walk", "bootstrap", "garch_fhs"])
    ap.add_argument("--n-paths", type=int, default=cfg.DEFAULT_N_PATHS)
    ap.add_argument("--save-paths", type=int, default=0, help="save this many synthetic paths per run (labelled)")
    ap.add_argument("--no-exclusions", action="store_true", help="ignore data_exclusions.csv (use the raw cleaned data)")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    a = ap.parse_args()

    if not a.no_exclusions:
        print("Applying documented data exclusions from data_exclusions.csv (use --no-exclusions to switch off).")
    assets = a.assets or (cfg.TIER1 if a.tier1 else cfg.CORE_FIVE)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fid_rows, nov_rows, par_rows = [], [], []
    t0 = time.time()
    for asset in assets:
        df = D.load_close(asset, a.data_dir, exclusions=not a.no_exclusions)
        cls = df.attrs["asset_class"]
        gens = list(a.generators) + (["peg_ar1"] if asset in cfg.STABLECOINS else [])
        for year in D.eligible_years(df, a.years):
            train, _ = D.walk_forward_split(df, year)
            real_r = train["log_ret"].to_numpy()
            p_last = float(train["close"].iloc[-1])
            for g_name in gens:
                try:
                    gen = fit_generator(g_name, train)
                    rng = np.random.default_rng(a.seed + year)
                    r = gen.sample_log_returns(a.n_paths, len(train), rng)
                    is_peg = cls == "stablecoin"
                    tab = G.fidelity_table(
                        real_r, r,
                        real_price=train["close"].to_numpy() if is_peg else None,
                        synth_price=prices_from_log_returns(r, p_last) if is_peg else None)
                    s = G.fidelity_summary(tab)
                    nv = G.novelty_report(real_r, r, seed=a.seed)
                except Exception as e:                                   # keep going, record the failure
                    print(f"  ! {asset} {year} {g_name}: {e}")
                    continue
                base = dict(asset=asset, asset_class=cls, year=year, generator=g_name,
                            n_train=len(train), n_paths=a.n_paths, synthetic=True)
                tab.insert(0, "year", year); tab.insert(0, "generator", g_name)
                tab.insert(0, "asset", asset); tab.insert(0, "asset_class", cls)
                fid_rows.append(tab)
                nov_rows.append({**base, **nv})
                if hasattr(gen, "garch"):
                    par_rows.append({**base, **gen.garch.params})
                if a.save_paths:
                    np.savez_compressed(out / f"SYNTHETIC_{asset}_{year}_{g_name}_seed{a.seed}.npz",
                                        log_ret=r[: a.save_paths], asset=asset, test_year=year, generator=g_name,
                                        seed=a.seed, note="SYNTHETIC DATA - not real prices")
                print(f"{asset:7s} {year} {g_name:12s} fidelity {s['core_inside_band']}/{s['core_metrics']} core metrics in band"
                      f" | near-copy share {nv['near_copy_share']:.2f}")

    if not fid_rows:
        print("Nothing was run (no eligible asset-years). Check --assets and --years."); return
    fid = pd.concat(fid_rows, ignore_index=True)
    nov = pd.DataFrame(nov_rows)
    fid.to_csv(out / "fidelity_by_metric.csv", index=False)
    nov.to_csv(out / "novelty.csv", index=False)
    if par_rows:
        pd.DataFrame(par_rows).to_csv(out / "garch_parameters.csv", index=False)

    core = fid[fid["core"]]
    summ = (core.groupby(["asset_class", "asset", "generator"])["inside_band"].mean().rename("fidelity_pass_rate").reset_index()
            .merge(nov.groupby(["asset", "generator"])["near_copy_share"].mean().reset_index(), on=["asset", "generator"]))
    summ.to_csv(out / "summary_by_asset.csv", index=False)
    wide = summ.pivot_table(index="asset", columns="generator", values="fidelity_pass_rate").round(2)
    print("\nFidelity pass rate (share of core metrics inside the synthetic 5-95% band), averaged over test years:")
    print(wide.to_string())
    print("\nNovelty: near-copy share (should be near 0.05 or below; high = copies of history):")
    print(summ.pivot_table(index="asset", columns="generator", values="near_copy_share").round(2).to_string())
    print(f"\nWritten to {out}  ({time.time() - t0:.0f}s). All paths are SYNTHETIC.")


if __name__ == "__main__":
    main()
