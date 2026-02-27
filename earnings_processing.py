"""
earnings_processing.py
Reads earnings_master.csv, computes SUE/SUR with winsorization,
and outputs descriptive statistics + outlier tables.

No logic changes from original code.
"""

import pandas as pd
import numpy as np
from config import EARNINGS_MASTER_FILE, EARNINGS_WITH_SUE_SUR_FILE, EARNINGS_STATS_FILE, EARNINGS_OUTLIERS_FILE


def winsorize(s, lo=0.005, hi=0.995):
    """Winsorize series to [lo, hi] quantiles."""
    if s.dropna().empty:
        return s
    lower = s.quantile(lo)
    upper = s.quantile(hi)
    return s.clip(lower, upper)


def stats_row(s, name):
    """Generate a single row of descriptive statistics."""
    clean = s.dropna()
    n_total = len(s)
    n_clean = len(clean)
    if n_clean == 0:
        return {
            "Variable": name,
            "N": 0,
            "NaN": n_total,
            "Mean": np.nan,
            "Median": np.nan,
            "Std": np.nan,
            "Min": np.nan,
            "P1": np.nan,
            "P10": np.nan,
            "P90": np.nan,
            "Max": np.nan,
        }
    return {
        "Variable": name,
        "N": n_clean,
        "NaN": n_total - n_clean,
        "Mean": clean.mean(),
        "Median": clean.median(),
        "Std": clean.std(),
        "Min": clean.min(),
        "P1": clean.quantile(0.01),
        "P10": clean.quantile(0.10),
        "P90": clean.quantile(0.90),
        "Max": clean.max(),
    }


def process_earnings():
    """Load earnings, compute SUE/SUR, winsorize, save outputs."""
    print(f"loading {EARNINGS_MASTER_FILE}")
    earn = pd.read_csv(EARNINGS_MASTER_FILE, parse_dates=["Date"])
    print(f"{len(earn):,} rows | {earn['Ticker'].nunique():,} tickers")

    # forecast errors
    earn["FE_EPS"] = earn["Used_EPS"] - earn["Consensus_EPS"]
    earn["FE_Rev"] = earn["Actual_Revenue"] - earn["Consensus_Revenue"]

    # rolling std, 8 quarters (2 years) back, min_periods=4
    print("computing rolling standard deviations...")
    earn["std_FE_EPS"] = earn.groupby("Ticker")["FE_EPS"].transform(
        lambda x: x.shift(1).rolling(window=8, min_periods=4).std()
    )
    earn["std_FE_Rev"] = earn.groupby("Ticker")["FE_Rev"].transform(
        lambda x: x.shift(1).rolling(window=8, min_periods=4).std()
    )

    # SUE / SUR raw
    earn["SUE_EPS_raw"] = np.where(
        earn["std_FE_EPS"] > 1e-8, earn["FE_EPS"] / earn["std_FE_EPS"], np.nan
    )
    earn["SUR_Rev_raw"] = np.where(
        earn["std_FE_Rev"] > 1e-8, earn["FE_Rev"] / earn["std_FE_Rev"], np.nan
    )

    # winsorize
    print("winsorizing...")
    earn["SUE_EPS"] = winsorize(earn["SUE_EPS_raw"])
    earn["SUR_Rev"] = winsorize(earn["SUR_Rev_raw"])
    earn["EPS_Surprise_win"] = winsorize(earn["EPS_Surprise"])
    earn["Revenue_Surprise_win"] = winsorize(earn["Revenue_Surprise"])

    # descriptive stats
    stats_list = [
        ("EPS_Surprise", "EPS Surprise (%) – raw"),
        ("Revenue_Surprise", "Revenue Surprise (%) – raw"),
        ("EPS_Surprise_win", "EPS Surprise (%) – winsorized"),
        ("Revenue_Surprise_win", "Revenue Surprise (%) – winsorized"),
        ("SUE_EPS_raw", "SUE (EPS) – raw"),
        ("SUE_EPS", "SUE (EPS) – winsorized 0.5–99.5%"),
        ("SUR_Rev_raw", "SUR (Revenue) – raw"),
        ("SUR_Rev", "SUR (Revenue) – winsorized 0.5–99.5%"),
    ]
    stats = [stats_row(earn[col], name) for col, name in stats_list]
    stats_df = pd.DataFrame(stats)

    print("\n" + "=" * 120)
    print("DESCRIPTIVE STATISTICS FOR EARNINGS SURPRISE & SUE")
    print("=" * 120)
    print(stats_df.round(6).to_string(index=False))
    print("=" * 120)

    # outliers: top 5 and bottom 5 for each variable
    OUTLIER_VARS = [
        "EPS_Surprise",
        "Revenue_Surprise",
        "SUE_EPS_raw",
        "SUR_Rev_raw",
    ]
    outlier_rows = []
    for var in OUTLIER_VARS:
        if var not in earn.columns:
            continue
        s = earn[var]
        non_na = s.notna()
        sub = earn.loc[non_na, ["Ticker", "Date", var]].copy()
        if sub.empty:
            continue
        bottom5 = sub.nsmallest(5, var).assign(Variable=var, Side="Bottom")
        top5 = sub.nlargest(5, var).assign(Variable=var, Side="Top")
        outlier_rows.append(bottom5)
        outlier_rows.append(top5)

    if outlier_rows:
        outliers_df = pd.concat(outlier_rows, ignore_index=True)
        outliers_df = outliers_df[["Variable", "Side", "Ticker", "Date"] + OUTLIER_VARS]
    else:
        outliers_df = pd.DataFrame(columns=["Variable", "Side", "Ticker", "Date"] + OUTLIER_VARS)

    # save outputs
    earn.to_csv(EARNINGS_WITH_SUE_SUR_FILE, index=False)
    stats_df.to_csv(EARNINGS_STATS_FILE, index=False)
    outliers_df.to_csv(EARNINGS_OUTLIERS_FILE, index=False)

    print("\n" + "=" * 120)
    print(f"Saved earnings master with SUE/SUR:  {EARNINGS_WITH_SUE_SUR_FILE}")
    print(f"Saved descriptive stats:              {EARNINGS_STATS_FILE}")
    print(f"Saved outliers:                       {EARNINGS_OUTLIERS_FILE}")
    print("=" * 120)

    return earn


if __name__ == "__main__":
    process_earnings()
