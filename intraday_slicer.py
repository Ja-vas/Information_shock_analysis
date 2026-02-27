"""Generate per-gap day intraday CSVs for drift analysis."""
from pathlib import Path
import glob
import pandas as pd
from tqdm import tqdm

from config import INTRADAY_DIR, GAP_SLICE_DIR


def slice_intraday(gap_files: list[str] | None = None) -> None:
    """Read every gap row and save a slice of the 1‑minute file for that day.

    Parameters
    ----------
    gap_files : list[str] | None
        List of paths to the combined gap_up and gap_down csvs.  If None will
        read the default files from ``config.GAP_UP_FILE`` and ``GAP_DOWN_FILE``.
    """
    # load the combined gap table if user did not supply explicit list
    if gap_files is None:
        from config import GAP_UP_FILE, GAP_DOWN_FILE
        gap_files = [str(GAP_UP_FILE), str(GAP_DOWN_FILE)]

    # build DataFrame of all gaps
    frames = []
    for gf in gap_files:
        if Path(gf).exists():
            df = pd.read_csv(gf, parse_dates=["Date"])
            frames.append(df)
    if not frames:
        raise FileNotFoundError("no gap files found to slice")
    all_trades = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Ticker", "Date"])
    all_trades["Date"] = pd.to_datetime(all_trades["Date"]).dt.date

    # create output directories if they don't exist
    for side in ("gap_up", "gap_down"):
        (GAP_SLICE_DIR / side).mkdir(parents=True, exist_ok=True)

    missing = []
    tickers = all_trades["Ticker"].unique()
    pbar = tqdm(tickers, desc="tickers")
    for ticker, grp in all_trades.groupby("Ticker"):
        first_letter = str(ticker)[0].upper()
        pattern = str(INTRADAY_DIR / f"stock_{first_letter}_full_1min*")
        matches = glob.glob(pattern)
        if not matches:
            missing.append((ticker, "folder not found"))
            pbar.update(1)
            continue
        folder = matches[0]
        file_path = Path(folder) / f"{ticker}_full_1min_adjsplitdiv.txt"
        if not file_path.exists():
            missing.append((ticker, "file missing"))
            pbar.update(1)
            continue
        try:
            df = pd.read_csv(file_path, header=None,
                             names=["Datetime", "Open", "High", "Low", "Close", "Volume"],
                             parse_dates=["Datetime"], engine="c")
            df["Date_only"] = df["Datetime"].dt.date
            for _, row in grp.iterrows():
                day = row["Date"]
                direction = row.get("Direction", "")
                prev_close = row.get("Prev_Close", None)
                slice_df = df[df["Date_only"] == day].copy()
                if slice_df.empty:
                    missing.append((ticker, day, "date not found"))
                    continue
                slice_df["Ticker"] = ticker
                slice_df["Direction"] = direction
                slice_df["Prev_Close"] = prev_close
                out_name = f"{direction}_{ticker}_{day}.csv"
                out_path = GAP_SLICE_DIR / direction / out_name
                slice_df.drop(columns=["Date_only"]).to_csv(out_path, index=False)
        except Exception as exc:
            missing.append((ticker, str(exc)))
        pbar.update(1)
    pbar.close()
    if missing:
        pd.DataFrame(missing, columns=["Ticker", "Date", "Reason"]).to_csv(
            GAP_SLICE_DIR / "missing_data_summary.csv", index=False
        )
