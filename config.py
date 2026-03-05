"""Configuration file for directory paths and global constants.

Put your ROOT_DIR here and the rest of the paths will be derived automatically.
This file is imported by all other modules so that you only have to change
one location when migrating the project or pushing to GitHub.
"""
from pathlib import Path

# --- USER SETTINGS ---------------------------------------------------------
ROOT_DIR = Path(r"C:/Users/j/python/Shocks_analysis_project")  # <-- edit once
# --------------------------------------------------------------------------

SCRIPTS_DIR = ROOT_DIR / "scripts"

DATA_DIR = ROOT_DIR / "data"
DAILY_DATA_DIR = DATA_DIR / "daily_data"
INTRADAY_DIR = DATA_DIR / "1min_data"

MAIN_DIR = ROOT_DIR / "main_dataframe"
GAP_SLICE_DIR = ROOT_DIR / "gap_trades_slices"

# output files
UNFILTERED_FILE = MAIN_DIR / "daily_ohlc_unfiltered.csv"
PROCESSED_FILE = MAIN_DIR / "daily_ohlc_processed.csv"
GAP_UP_FILE = MAIN_DIR / "gap_up_trades.csv"
GAP_DOWN_FILE = MAIN_DIR / "gap_down_trades.csv"
SIGNIFICANT_GAPS_FILE = MAIN_DIR / "SIGNIFICANT_GAPS_final.csv"

# earnings and news files
EARNINGS_MASTER_FILE = MAIN_DIR / "earnings_master.csv"
EARNINGS_WITH_SUE_SUR_FILE = MAIN_DIR / "earnings_with_sue_sur.csv"
NEWS_CLASSIFIED_FILE = MAIN_DIR / "news_classified.csv"

# processed earnings outputs
EARNINGS_WITH_SUE_SUR_FILE = MAIN_DIR / "earnings_with_sue_sur.csv"
EARNINGS_STATS_FILE = MAIN_DIR / "earnings_descriptive_stats.csv"
EARNINGS_OUTLIERS_FILE = MAIN_DIR / "earnings_outliers.csv"

# gap+earnings joined outputs
GAP_EARNINGS_JOINED_FILE = MAIN_DIR / "gap_earnings_joined.csv"
GAP_EARNINGS_UP_FILE = MAIN_DIR / "gap_earnings_up.csv"
GAP_EARNINGS_DOWN_FILE = MAIN_DIR / "gap_earnings_down.csv"

# CAR results directory
CAR_RESULTS_DIR = MAIN_DIR / "car_results"

# CAR results files
CAR_RAW_RESULTS_FILE = CAR_RESULTS_DIR / "CAR_raw_results.csv"
CAR_EXTREME_AUDIT_FILE = CAR_RESULTS_DIR / "CAR_extreme_audit.csv"
CAR_QUARTILES_RESULTS_FILE = CAR_RESULTS_DIR / "CAR_quartiles_results.csv"
CAR_OUTLIERS_FILE = CAR_RESULTS_DIR / "CAR_outliers.csv"

# convenient helpers --------------------------------------------------------
def ensure_directories():
    """Create any directories that are expected by the workflow."""
    for p in (MAIN_DIR, GAP_SLICE_DIR):
        p.mkdir(parents=True, exist_ok=True)


# make it easy to switch between Windows and other OSes if paths are strings
def to_path(p):
    return Path(p)
