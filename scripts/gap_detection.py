"""Mark overnight gap ups/downs and apply quality filters."""
from pathlib import Path
import pandas as pd

from config import UNFILTERED_FILE, GAP_UP_FILE, GAP_DOWN_FILE


def detect_gaps(min_volume=500, min_usd_volume=30000,
                gap_up_pct=1.06, gap_down_pct=0.94,
                overnight_max_days=5) -> pd.DataFrame:
    """Load the unfiltered daily file, flag gaps, and save two csvs.

    Returns full dataframe with gap columns included for further inspection.
    """
    df = pd.read_csv(UNFILTERED_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"])  # make shift reliable

    df["Prev_Close"] = df.groupby("Ticker")["Close"].shift(1)
    df["Prev_Date"] = df.groupby("Ticker")["Date"].shift(1)
    df["Days_Diff"] = (df["Date"] - df["Prev_Date"]).dt.days

    df["usd_volume"] = df["Close"] * df["Volume"]
    df["mean_usd_vol_50"] = df.groupby("Ticker")["usd_volume"].transform(
        lambda x: x.rolling(50, min_periods=1).mean()
    )

    is_overnight = df["Days_Diff"] <= overnight_max_days
    df["Is_Gap_Up"] = is_overnight & (df["Open"] >= gap_up_pct * df["Prev_Close"])
    df["Is_Gap_Down"] = is_overnight & (df["Open"] <= gap_down_pct * df["Prev_Close"])

    quality = (
        (df["Close"] > 0.5)
        & (df["Close"] < 50000)
        & (df["usd_volume"] >= min_usd_volume)
        & (df["Volume"] >= min_volume)
    )

    gap_up = df[df["Is_Gap_Up"] & quality].copy()
    gap_up["Direction"] = "Gap_Up"
    gap_down = df[df["Is_Gap_Down"] & quality].copy()
    gap_down["Direction"] = "Gap_Down"

    GAP_UP_FILE.parent.mkdir(parents=True, exist_ok=True)
    gap_up.to_csv(GAP_UP_FILE, index=False)
    gap_down.to_csv(GAP_DOWN_FILE, index=False)
    print(f"wrote {len(gap_up)} gap ups and {len(gap_down)} gap downs")
    return df
