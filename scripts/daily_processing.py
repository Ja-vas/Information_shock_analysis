"""Utilities for combining and filtering the daily OHLC text files.

Functions here are basically the contents of the first two cells from the
original notebook, but parameterised and written as reusable functions.  The
"config" module provides default file paths so that the notebook itself only
has one import line.
"""
from pathlib import Path
import glob
import pandas as pd
from tqdm import tqdm

from config import DAILY_DATA_DIR, UNFILTERED_FILE, PROCESSED_FILE


def merge_unfiltered(start_date: str = "2015-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
    """Read every daily text file and concatenate them without quality filters.

    The only filtering performed is on the date range specified.  The resulting
    dataframe is also saved to ``UNFILTERED_FILE``.

    Returns
    -------
    pd.DataFrame
        Combined data with columns [Date,Open,High,Low,Close,Volume,Ticker].
    """
    daily_pattern = str(DAILY_DATA_DIR / "*_full_1day_adjsplitdiv.txt")
    files = glob.glob(daily_pattern)
    print(f"Found {len(files)} daily files to merge")

    frames = []
    for f in tqdm(files, desc="Merging tickers"):
        try:
            ticker = Path(f).stem.split("_")[0]
            df = pd.read_csv(f, header=None,
                             names=["Date", "Open", "High", "Low", "Close", "Volume"])
            df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
            df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()
            if not df.empty:
                df["Ticker"] = ticker
                frames.append(df)
        except Exception as exc:
            print(f"error processing {f}: {exc}")

    if frames:
        result = pd.concat(frames, ignore_index=True)
        UNFILTERED_FILE.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(UNFILTERED_FILE, index=False)
        print(f"saved unfiltered file to {UNFILTERED_FILE} with {len(result)} rows")
        return result
    else:
        print("no data merged")
        return pd.DataFrame()


def merge_processed(start_date: str = "2005-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
    """Read, filter and add liquidity metrics, then save to ``PROCESSED_FILE``.

    Uses the same directory as ``merge_unfiltered`` so we simply loop the files
    again and apply thresholds on prices, volume, and USD volume.  A rolling
    50‑day average usd_volume is computed to use as a liquidity watchdog.
    """
    pattern = str(DAILY_DATA_DIR / "*_full_1day_adjsplitdiv.txt")
    files = glob.glob(pattern)
    print(f"Searching in: {DAILY_DATA_DIR}")
    print(f"Found {len(files)} files matching the pattern.")

    frames = []
    for f in files:
        try:
            ticker = Path(f).stem.split("_")[0]
            df = pd.read_csv(f, header=None,
                             names=["Date", "Open", "High", "Low", "Close", "Volume"])
            df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
            numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
            df["Ticker"] = ticker

            mask = (
                (df["Open"] >= 0) & (df["High"] >= 0) & (df["Low"] >= 0) &
                (df["Close"] < 50000) & (df["Close"] > 0.1) & (df["Volume"] >= 1) &
                (df["Date"] >= start_date) & (df["Date"] <= end_date)
            )
            df = df[mask].copy()
            if not df.empty:
                df["usd_volume"] = df["Close"] * df["Volume"]
                df["mean_usd_vol"] = df["usd_volume"].rolling(window=50, min_periods=1).mean()
                df = df[df["usd_volume"] >= 50000]
                if not df.empty:
                    frames.append(df)
        except Exception as exc:
            print(f"Error processing {ticker}: {exc}")

    if frames:
        result = pd.concat(frames, ignore_index=True)
        PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(PROCESSED_FILE, index=False)
        print(f"saved filtered data to {PROCESSED_FILE} ({len(result)} rows)")
        return result
    else:
        print("no processed data")
        return pd.DataFrame()
