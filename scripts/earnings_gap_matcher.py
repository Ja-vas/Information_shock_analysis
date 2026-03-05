"""
earnings_gap_matcher.py
Matches earnings announcement dates to gap trading days.
If earnings were announced on the gap day or the day before,
marks them as "earnings-driven" gaps.

Logic:
  - earnings_master.csv has only dates (no times)
  - Earnings can be announced after market close (previous day) or before market open (same day)
  - Match on the gap date OR gap date - 1 day
  - Separate results into gap_up and gap_down strategies
"""

import pandas as pd
from config import (
    SIGNIFICANT_GAPS_FILE,
    EARNINGS_WITH_SUE_SUR_FILE,
    GAP_EARNINGS_JOINED_FILE,
    GAP_EARNINGS_UP_FILE,
    GAP_EARNINGS_DOWN_FILE,
)


def match_earnings_to_gaps():
    """
    Load gaps and earnings, match by date +/- 1 day.
    Save joined CSVs for each strategy.
    """
    print(f"Loading gaps from {SIGNIFICANT_GAPS_FILE}")
    gaps = pd.read_csv(SIGNIFICANT_GAPS_FILE, parse_dates=["Date"])
    print(f"  {len(gaps):,} gap events")

    # some gap files don't include boolean flags, infer from "Type" if available
    if "Type" in gaps.columns:
        gaps["Is_Gap_Up"] = gaps["Type"].str.lower() == "up"
        gaps["Is_Gap_Down"] = gaps["Type"].str.lower() == "down"
    # if the booleans already exist, this is a no-op

    print(f"Loading earnings from {EARNINGS_WITH_SUE_SUR_FILE}")
    earn = pd.read_csv(EARNINGS_WITH_SUE_SUR_FILE, parse_dates=["Date"])
    print(f"  {len(earn):,} earnings events")

    # Create matchers: earnings on gap date OR gap date - 1
    gaps["Date_Gap"] = gaps["Date"]
    gaps["Date_Gap_Minus1"] = gaps["Date"] - pd.Timedelta(days=1)

    # wide join: each gap matched to all earnings on date or date-1
    merged_list = []

    for _, gap_row in gaps.iterrows():
        ticker = gap_row["Ticker"]
        gap_date = gap_row["Date_Gap"]
        gap_date_minus1 = gap_row["Date_Gap_Minus1"]

        # find earnings for this ticker on gap_date or gap_date - 1
        earn_subset = earn[
            (earn["Ticker"] == ticker)
            & ((earn["Date"] == gap_date) | (earn["Date"] == gap_date_minus1))
        ]

        if len(earn_subset) > 0:
            # match each earnings record to this gap
            for _, earn_row in earn_subset.iterrows():
                merged_row = {**gap_row.to_dict(), **earn_row.to_dict()}
                merged_row["Days_Before_Gap"] = (gap_date - earn_row["Date"]).days
                merged_list.append(merged_row)

    if merged_list:
        merged = pd.DataFrame(merged_list)
        print(f"\nMatched {len(merged):,} earnings records to {merged['Ticker'].nunique():,} tickers")
    else:
        print("\nNo earnings matched to gaps – creating empty dataframe")
        merged = pd.DataFrame()

    if not merged.empty:
        # Separate by strategy
        up = merged[merged["Is_Gap_Up"] == True].copy()
        down = merged[merged["Is_Gap_Down"] == True].copy()

        print(f"  → Gap-up earnings events:   {len(up):,}")
        print(f"  → Gap-down earnings events: {len(down):,}")

        # breakdown by significance windows for each side
        def print_sig_counts(df_side, label):
            print(f"\n  {label} significance counts:")
            for sig in ["Sig30_P1", "Sig30_P5", "Sig30_P30"]:
                if sig in df_side.columns:
                    cnt = int(df_side[sig].sum())
                    print(f"    {sig}: {cnt}")
                else:
                    print(f"    {sig}: <missing column>")

        print_sig_counts(up, "Gap-up")
        print_sig_counts(down, "Gap-down")

        # Save
        merged.to_csv(GAP_EARNINGS_JOINED_FILE, index=False)
        up.to_csv(GAP_EARNINGS_UP_FILE, index=False)
        down.to_csv(GAP_EARNINGS_DOWN_FILE, index=False)

        print(f"\nSaved:")
        print(f"  {GAP_EARNINGS_JOINED_FILE}")
        print(f"  {GAP_EARNINGS_UP_FILE}")
        print(f"  {GAP_EARNINGS_DOWN_FILE}")

        return merged, up, down
    else:
        return merged, pd.DataFrame(), pd.DataFrame()


if __name__ == "__main__":
    match_earnings_to_gaps()
