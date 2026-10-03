"""
pstone_config.py  --  shared settings for the P Stone synthetic-data code.

Everything that the team may want to change lives here, so the other files
contain no hidden numbers. The values follow P_STONE_DATA_CONTEXT.md.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Where the cleaned data lives (the folder that contains manifest.csv).
# Default: <project>/data_new/by_class. Override with the environment variable
# PSTONE_DATA or the --data-dir option of the run scripts.
DATA_DIR = Path(os.environ.get("PSTONE_DATA", ROOT / "data_new" / "by_class"))

# Documented removals of proven data errors (see data_exclusions.csv). Never edits the data files.
EXCLUSIONS_FILE = ROOT / "data_exclusions.csv"

# Everything we write goes here. We never write into the data folder.
OUT_DIR = ROOT / "outputs"

# Walk-forward evaluation (context file, section 9)
TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
MIN_HISTORY_DAYS = 3 * 365          # an asset enters a test year only with >= 3 years of history

# Assets
CORE_FIVE = ["NDX", "DJI", "IBEX", "EURUSD", "USDC"]            # Wave 1
TIER1 = ["SPX", "NDX", "DJI", "IBEX", "SPY", "QQQ", "SPXL", "TQQQ",
         "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "BTC", "ETH", "USDC", "USDT"]
STABLECOINS = ["USDC", "USDT"]
ONE_PER_FAMILY = ["SPX", "NDX", "IBEX", "N225", "AAPL", "EURUSD", "USDJPY",
                  "GBPUSD", "USDCHF", "BTC", "ETH", "GLD", "USO", "TLT"]

# Asset classes that trade 7 days a week (a "week" is 7 rows, not 5)
SEVEN_DAY_CLASSES = ["crypto", "stablecoin"]

# Group panels for copulas (built by the data pipeline in data/by_class/panels)
PANELS = {
    "us_equity_daily": "us_equity_daily.csv",
    "fx_daily": "fx_daily.csv",
    "crypto_daily": "crypto_daily.csv",
    "cross_asset_weekly": "cross_asset_weekly.csv",
}

# Generators
DEFAULT_N_PATHS = 1000              # paths per asset and window (context file, section 9)
SEED = 12345

# Novelty check
NOVELTY_WINDOW = 64                 # rows per window (context file, section 8.4)
NOVELTY_STRIDE_SYNTH = 25           # spacing between sampled synthetic windows
