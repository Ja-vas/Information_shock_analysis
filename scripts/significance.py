"""Analyze intraday gap files and flag significant events."""
from pathlib import Path
import glob
import pandas as pd
from tqdm import tqdm

from config import (
    MAIN_DIR,
    GAP_SLICE_DIR,
    SIGNIFICANT_GAPS_FILE,
)

# default thresholds and windows
WINDOWS = {30: 10, 60: 20}
CHUNK_SAVE = 200
THRESHOLDS = {"prem1": 1, "prem5": 1, "prem30": 2}


def run_analysis(
    daily_ohlc_file: Path | str = None,
    gap_base_dir: Path | str = None,
    output_file: Path | str = None,
    windows=None,
    thresholds=None,
    chunk_save=None,
) -> None:
    """Main entrypoint replicating the notebook's run_analysis logic."""
    if daily_ohlc_file is None:
        daily_ohlc_file = MAIN_DIR / "daily_ohlc_processed.csv"
    if gap_base_dir is None:
        gap_base_dir = GAP_SLICE_DIR
    if output_file is None:
        output_file = SIGNIFICANT_GAPS_FILE
    windows = windows or WINDOWS
    thresholds = thresholds or THRESHOLDS
    chunk_save = chunk_save or CHUNK_SAVE

    daily_idx = pd.read_csv(daily_ohlc_file, parse_dates=["Date"])
    daily_idx["usd_volume"] = daily_idx["Close"] * daily_idx["Volume"]
    daily_idx = daily_idx.sort_values(["Ticker", "Date"])
    for w, minp in windows.items():
        daily_idx[f"rolling_mean_{w}"] = (
            daily_idx.groupby("Ticker")["usd_volume"]
            .transform(lambda s: s.shift(1).rolling(window=w, min_periods=minp).mean())
        )
    daily_idx = daily_idx.set_index(["Ticker", "Date"])

    search = str(Path(gap_base_dir) / "**" / "*.csv")
    files = glob.glob(search, recursive=True)
    files = [f for f in files if Path(f).name != Path(output_file).name]
    print(f" Found {len(files)} potential gap files.")

    results = []
    saved = 0
    for fp in tqdm(files, desc="Analyzing gaps"):
        fname = Path(fp).name
        try:
            base = fname.replace(".csv", "")
            parts = base.split("_")
            date_str = parts[-1]
            ticker = parts[-2]
            gap_date = pd.to_datetime(date_str)
        except Exception:
            continue

        try:
            idf = pd.read_csv(fp)
            idf["Datetime"] = pd.to_datetime(idf["Datetime"])
            idf["USD"] = idf["Close"] * idf["Volume"]
            times = idf["Datetime"].dt.time
            t04, t0930, t0931, t0935, t10 = [
                pd.to_datetime(t).time()
                for t in ["04:00:00", "09:30:00", "09:31:00", "09:35:00", "10:00:00"]
            ]
            prem = idf.loc[(times >= t04) & (times < t0930), "USD"].sum()
            o1 = idf.loc[(times >= t0930) & (times < t0931), "USD"].sum()
            o5 = idf.loc[(times >= t0930) & (times < t0935), "USD"].sum()
            o30 = idf.loc[(times >= t0930) & (times < t10), "USD"].sum()
            p1, p5, p30 = prem + o1, prem + o5, prem + o30
            drow = daily_idx.loc[(ticker, gap_date)]
            m30, m60 = drow[f"rolling_mean_30"], drow[f"rolling_mean_60"]
            rec = {
                "Ticker": ticker,
                "Date": date_str,
                "Type": "up" if "up" in fname.lower() else "down",
                "P1_USD": p1,
                "P5_USD": p5,
                "P30_USD": p30,
                "mean30": m30,
                "mean60": m60,
                "Sig30_P1": p1 >= thresholds["prem1"] * m30 if pd.notna(m30) else False,
                "Sig30_P5": p5 >= thresholds["prem5"] * m30 if pd.notna(m30) else False,
                "Sig30_P30": p30 >= thresholds["prem30"] * m30 if pd.notna(m30) else False,
            }
            if rec["Sig30_P1"] or rec["Sig30_P5"] or rec["Sig30_P30"]:
                results.append(rec)
        except (KeyError, Exception):
            continue

        if len(results) >= chunk_save:
            mode = "w" if saved == 0 else "a"
            pd.DataFrame(results).to_csv(output_file, index=False, mode=mode, header=(mode == "w"))
            saved += len(results)
            results = []

    if results:
        mode = "w" if saved == 0 else "a"
        pd.DataFrame(results).to_csv(output_file, index=False, mode=mode, header=(mode == "w"))
        saved += len(results)
    print(f"\nSaved {saved} significant gaps to {output_file}")
