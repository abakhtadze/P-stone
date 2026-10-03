# P Stone synthetic data code

Implements the plan in `P_STONE_DATA_CONTEXT.md` and the algorithm guide: a generator ladder
(random walk, block bootstrap, **GJR-GARCH Filtered Historical Simulation** as the main model,
a mean-reverting model for stablecoins, a t-copula for several assets) and the quality gate
**Fidelity, Novelty, Utility**. All output is SYNTHETIC and labelled so. No API key is used or needed.

## 1. Where to put the files

Make one folder, for example `pstone_project`, and put in it:

```
pstone_project/
  pstone_config.py  pstone_data.py  pstone_models.py  pstone_gate.py
  pstone_utility.py  pstone_leveraged.py  data_exclusions.csv
  run_wave1.py  run_utility.py  run_copula.py  test_pstone.py  README.md
  data_new/                      <- unzip your data_new zip here
    by_class/
      manifest.csv
      equity_index/ fx/ stablecoin/ crypto/ ... panels/
```

The code reads `data_new/by_class` by default. If your data is elsewhere, either pass
`--data-dir /path/to/by_class` to every run script, or set the environment variable `PSTONE_DATA`.
The code only reads the data, never writes to it. Results go to `outputs/` next to the code.

## 2. Set up (VS Code terminal, Mac)

Open the `pstone_project` folder in VS Code, open a terminal (Terminal > New Terminal), then:

```
python3 -m pip install pandas numpy scipy scikit-learn
python3 -m unittest -v test_pstone
```

Expect 21 tests to pass ("OK"). One test loads every file in your data folder.

## 3. Run, in this order (matches the roadmap waves)

| Step | Command | What it does | Writes to |
|---|---|---|---|
| Quick check | `python3 run_wave1.py --assets NDX --years 2022 --n-paths 100` | One asset, one year, a few seconds | `outputs/wave1/` |
| Wave 1 | `python3 run_wave1.py` | Core five (NDX, DJI, IBEX, EURUSD, USDC) x 2020-2025 x random walk, bootstrap, GARCH-FHS (+ peg model for USDC). Fidelity + Novelty | `outputs/wave1/` |
| Wave 2 | `python3 run_wave1.py --tier1` and `python3 run_utility.py --tier1` | All Tier-1 assets, then the four arms | `outputs/wave1/`, `outputs/utility/` |
| Wave 3 | `python3 run_copula.py` | t-copula on the four group panels, correlation and joint-crash match | `outputs/copula/` |

Useful options: `--assets A B C`, `--years 2022 2023`, `--n-paths N` (default 1000 for Wave 1; the
utility test uses far fewer paths per fit), `--save-paths 20` (saves labelled synthetic paths),
`--no-exclusions`, `--seed`.

## 4. What each file does

| File | Role |
|---|---|
| `pstone_config.py` | Paths, test years (2020-2025), asset lists, seeds. Change settings here. |
| `pstone_data.py` | Loads close files, walk-forward split (train = rows before 1 Jan of the test year, needs 3+ years of history), group panels, documented exclusions. |
| `pstone_models.py` | The generators: `RandomWalk`, `BlockBootstrap`, `GJRGarchFHS` (main), `PegAR1`, `TCopulaFHS`. |
| `pstone_gate.py` | Fidelity (fat tails, volatility clustering, return autocorrelation, leverage effect, peg statistics) and Novelty (nearest real window). |
| `pstone_utility.py` | The four arms A/B/C/D, hyperparameter selection, Diebold-Mariano test, summaries. |
| `pstone_leveraged.py` | Phase 2: leveraged-ETF price paths from synthetic index paths (close to close). |
| `data_exclusions.csv` | Log of removed bad ticks (see section 6). |
| `test_pstone.py` | 21 automatic checks on a fake market with known rules. |
| `example_results/` | Output of my own runs (300 paths / quick utility settings) so you can see the format. |

## 5. How to read the outputs

* `fidelity_by_metric.csv`: per asset, year, generator and statistic: real value, synthetic median
  and 5-95% band, `inside_band`. **Provisional pass line**: real value inside the band. The team still
  has to agree numeric thresholds after the first real run.
* `novelty.csv`: distance from synthetic windows to the nearest real training window versus the
  same distance for held-out real windows. `near_copy_share` should be near 0.05 or below;
  a high value means copies of history.
* `summary_by_asset.csv`: fidelity pass rate and near-copy share per asset and generator.
* `utility_results.csv`: test MSE for arms A-D, Diebold-Mariano p-values, chosen hyperparameters.
  **C must beat B**, not just A. `utility_summary_by_class.csv` aggregates per asset class.
* `copula_*.csv`: real versus synthetic pairwise correlations and joint-crash frequency.

## 6. Things to know (deviations, findings, limits)

1. **GARCH is estimated with scipy, not the `arch` package**, so the code runs with no extra install
   and could be tested here. It is the same model as `arch_model(..., vol="GARCH", p=1, o=1, q=1, dist="t")`
   fitted on `100 * log_ret`. If `arch` is installed, the block-length rule uses it automatically.
2. **FHS stability fix.** Reusing real residuals makes the simulated process more persistent than the
   fitted t-model says (falls are bigger than rises), and some paths exploded. When the empirical
   persistence is above 0.995 the code scales alpha, gamma, beta down together and resets omega so the long-run
   variance equals the training variance. Parameters are in `garch_parameters.csv`
   (`adjusted_for_simulation`). Volatility is also capped at 5x the sample level.
3. **Block bootstrap length.** The context file says Politis-White on the training returns. Returns have
   almost no autocorrelation, which gives ~1-day blocks and erases volatility clustering, so the default
   uses the larger of the lengths found on returns and on squared returns. Use `BlockBootstrap(block_basis="returns")`
   for the literal rule. Please confirm this choice with Elena.
4. **Bad ticks in EURUSD.** The cleaned file still contains 2008-10-08 (+9.6%, then -9.5% next day) and
   2008-09-08 (+5.3%, then -6.2%). These look like Yahoo bad ticks missed by the cleaning. They inflate
   kurtosis (25.9 on the 2022 training window) and made GARCH fidelity on EURUSD look bad (0.29). The code
   removes them via `data_exclusions.csv` (the bad day's return is added to the next day, so the price
   level is unchanged) and prints a notice. **They are proposed, not confirmed: check against `flags.csv`
   with Tomas**, and switch off with `--no-exclusions` if you disagree. `flags.csv` was not in the zip.
5. **Stablecoins.** Return-based generators drift off $1 and fail the peg checks, as the context file predicts.
   `peg_ar1` fixes the drift but undershoots USDC's day-to-day noise (the early USDC years were noisier than
   later ones, and one set of parameters cannot reproduce both). A regime term or a recent-window fit would be the next step.
6. **Novelty is a heuristic.** For series with regime changes (USDC) the held-out yardstick is unreliable; read the
   distance quantiles, not only the share.
7. **Not included:** the generative ML rung (VAE/GAN/diffusion, Anano's part) because it needs PyTorch, which is not
   installed here and cannot be tested; daily candles (Wave 5); intraday data. The scaling, windowing and novelty
   code in `pstone_gate.py` can be reused for it.
8. **Leveraged layer.** `pstone_leveraged.py` is a simple close-to-close version for synthetic paths. For candles and
   spread fitting keep using `leveraged_engine.py`. The negative fitted spread on real SPY/SPXL (about -1.66%) is still unresolved.

## 7. What my runs showed (core five, 300 paths, exclusions on; see `example_results/`)

* **Block bootstrap** passes Fidelity almost perfectly (it reuses real data) but **fails Novelty**: 74-77% of its windows
  are near-copies for NDX, DJI, IBEX and EURUSD. This is the gate doing its job.
* **GJR-GARCH FHS** is new data (near-copy share 0.00-0.51) with fidelity pass rates of 0.71 (DJI), 0.74 (NDX), 0.79 (EURUSD), 0.98 (IBEX).
  It tends to undershoot overall volatility and the extreme tails in some years.
* **Random walk** fails Fidelity everywhere (0.00-0.14), as it should.
* **Utility (quick settings: 10 synthetic paths, 100 trees):** equity indices - C beat B in 33% of asset-years
  and was worse in median (+3.4% MSE); FX - C beat B in 67% of asset-years with a median -1.3% MSE, but no
  Diebold-Mariano result below 0.10 for FX. Not conclusive. Re-run with more paths and trees before drawing conclusions,
  and report the result either way.
