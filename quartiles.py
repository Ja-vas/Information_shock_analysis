"""Rolling quartile classification of earnings metrics for gap events.

The analysis needs to partition gap trades into four bins (quartiles) based on
various earnings-related metrics (EPS surprise, revenue surprise, SUE, SUR and
combined SUE+SUR).  Importantly, the quartiles must be computed using *only
information available prior to the gap day* to avoid look‑ahead bias.

To keep things simple with the small sample size we implement an **expanding
window**: each row's quartile is determined by comparing its metric to all
previous rows of the same significance window (1m / 5m / 30m).  An optional
lookback parameter can restrict the comparison to the most recent N previous
events.

The supplied dataframe is expected to come from
`earnings_gap_matcher.match_earnings_to_gaps()` (i.e. has revenue/earnings
metrics and the Sig30_P1/P5/P30 boolean flags).

The function adds new columns:
    Q_EPS_Surprise_1m, Q_Revenue_Surprise_1m, Q_SUE_EPS_1m, etc.

where each quartile value is 1..4 (or NaN if insufficient history).

Usage example:

    from quartiles import add_earnings_quartiles
    merged, up, down = match_earnings_to_gaps()
    merged = add_earnings_quartiles(merged)
    merged.to_csv("gap_earnings_with_quartiles.csv", index=False)

"""

from __future__ import annotations

from typing import List, Optional
import pandas as pd
import numpy as np


SIGNAL_WINDOWS = {"Sig30_P1": "1m", "Sig30_P5": "5m", "Sig30_P30": "30m"}
METRICS = [
    "EPS_Surprise",
    "Revenue_Surprise",
    "SUE_EPS",
    "SUR_Rev",
]


def _assign_quartiles_for_series(series: pd.Series, lookback: Optional[int]) -> pd.Series:
    """Return rolling quartiles for a pandas Series using past values only.

    ``series`` should already be sorted chronologically.  The returned Series has
    the same index and contains integers 1..4 or NaN when there is no prior
    data.  ``lookback`` (if not None) bounds the number of prior rows considered.
    """
    result = pd.Series(index=series.index, dtype=float)
    prev_vals: List[float] = []
    for idx, val in series.items():
        if prev_vals and pd.notna(val):
            arr = np.array(prev_vals)
            # compute proportion strictly less than current value
            prop = (arr < val).sum() / len(arr)
            quart = int(prop * 4) + 1  # map [0,1) into 1..4
            if quart > 4:
                quart = 4
            result.at[idx] = quart
        else:
            result.at[idx] = np.nan
        # append current value for future rows (use lookback)
        if pd.notna(val):
            prev_vals.append(val)
            if lookback is not None and len(prev_vals) > lookback:
                prev_vals.pop(0)
    return result


def add_earnings_quartiles(
    df: pd.DataFrame, lookback: Optional[int] = None
) -> pd.DataFrame:
    """Attach rolling quartile columns to a gap/earnings DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by ``match_earnings_to_gaps`` or similar.  Must include
        a ``Date`` column (datetime64), the columns listed in :data:`METRICS`,
        and boolean flags ``Sig30_P1``, ``Sig30_P5``, ``Sig30_P30``.
    lookback : int or None
        Number of past events to include when computing quartiles; ``None`` means
        use all prior events (expanding window).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with new quartile columns added.  For each metric and
        signal window a column named ``Q_<metric>_<window>`` is created, where
        ``<window>`` is one of ``1m``, ``5m`` or ``30m``.
    """
    out = df.copy()
    if "Date" not in out.columns:
        raise ValueError("DataFrame must contain a 'Date' column")
    # determine which window applies to each row.  we only need the flag values
    # during processing so we will not actually store the string label in a
    # numeric column (which caused dtype errors previously).  instead we just
    # use the boolean masks directly later.
    # (the old _sig_window column is no longer needed)
    # if some rows have no flag they will simply be ignored below.

    # sort by date to guarantee chronological order
    out = out.sort_values("Date").reset_index(drop=True)

    # compute quartiles separately for each signal window using masks
    for flag, label in SIGNAL_WINDOWS.items():
        if flag not in out.columns:
            continue
        window_mask = out[flag] == True
        if not window_mask.any():
            continue
        sub_idx = out.index[window_mask]
        # for each metric compute quartiles on the subset
        for metric in METRICS:
            if metric not in out.columns:
                continue
            colname = f"Q_{metric}_{label}"
            out[colname] = np.nan
            metric_series = out.loc[sub_idx, metric]
            quart_series = _assign_quartiles_for_series(metric_series, lookback)
            out.loc[sub_idx, colname] = quart_series
        # combined metric SUE+SUR
        if "SUE_EPS" in out.columns and "SUR_Rev" in out.columns:
            comb = out.loc[sub_idx, "SUE_EPS"].fillna(0) + out.loc[sub_idx, "SUR_Rev"].fillna(0)
            comb_series = pd.Series(comb, index=sub_idx)
            quart_comb = _assign_quartiles_for_series(comb_series, lookback)
            out.loc[sub_idx, f"Q_SUE_SUR_{label}"] = quart_comb

    return out


if __name__ == "__main__":
    # quick smoke test when run as script
    from earnings_gap_matcher import match_earnings_to_gaps

    merged, up, down = match_earnings_to_gaps()
    df = add_earnings_quartiles(merged)
    print(df[["Ticker", "Date"] + [c for c in df.columns if c.startswith("Q_")]].head())
    df.to_csv("gap_earnings_with_quartiles.csv", index=False)
