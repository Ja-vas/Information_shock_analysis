"""Helper functions for CAR calculation with statistical outlier detection.

Key design:
- CAR values are NOT capped or winsorized (they are cumulative returns)
- Outlier detection via z-score, IQR, and percentile rank for inspection
- Store individual event-level results with symbol/date for manual review
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import time
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# OUTLIER DETECTION & DIAGNOSTICS
# ============================================================================

def detect_outliers(arr, method='zscore', threshold=3.0):
    """Identify outlier indices using z-score or IQR.
    
    Args:
        arr: array of numeric values (NaNs ignored).
        method: 'zscore' or 'iqr'.
        threshold: z-score cutoff (default 3.0) or IQR multiplier (1.5 standard).
    
    Returns:
        Tuple (outlier_mask, stats_dict) where outlier_mask is boolean array
        and stats_dict contains diagnostic info.
    """
    a = np.array(arr, dtype=float)
    mask_valid = ~np.isnan(a)
    
    if mask_valid.sum() < 2:
        return np.zeros_like(a, dtype=bool), {}
    
    valid_vals = a[mask_valid]
    
    if method == 'zscore':
        mean = np.nanmean(valid_vals)
        std = np.nanstd(valid_vals)
        if std == 0:
            return np.zeros_like(a, dtype=bool), {'mean': mean, 'std': std, 'n_outliers': 0}
        z = np.abs((valid_vals - mean) / std)
        out_mask = z > threshold
    else:  # iqr
        q1 = np.nanpercentile(valid_vals, 25)
        q3 = np.nanpercentile(valid_vals, 75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        out_mask = (valid_vals < lower) | (valid_vals > upper)
    
    full_mask = np.zeros_like(a, dtype=bool)
    full_mask[mask_valid] = out_mask
    
    return full_mask, {
        'n_total': len(a),
        'n_valid': mask_valid.sum(),
        'n_outliers': out_mask.sum(),
        'pct_outliers': 100.0 * out_mask.sum() / max(mask_valid.sum(), 1),
        'min': np.nanmin(valid_vals),
        'q1': np.nanpercentile(valid_vals, 25),
        'median': np.nanmedian(valid_vals),
        'q3': np.nanpercentile(valid_vals, 75),
        'max': np.nanmax(valid_vals),
        'mean': np.nanmean(valid_vals),
        'std': np.nanstd(valid_vals)
    }


def get_percentile_rank(value, arr):
    """Return percentile rank (0-100) of a value in an array."""
    if pd.isna(value):
        return np.nan
    valid = arr[~pd.isna(arr)]
    if len(valid) == 0:
        return np.nan
    return 100.0 * (valid < value).sum() / len(valid)


def format_outlier_table(arr, metric_name=""):
    """Create a diagnostic DataFrame for extreme values in array."""
    a = np.array(arr, dtype=float)
    valid_mask = ~np.isnan(a)
    valid_vals = a[valid_mask]
    
    if len(valid_vals) == 0:
        return pd.DataFrame()
    
    # Find indices of extreme values
    sorted_idx = np.argsort(np.abs(valid_vals))[::-1]  # sort by absolute value, descending
    
    extremes = []
    for idx in sorted_idx[:20]:  # top 20 by magnitude
        raw_idx = np.where(valid_mask)[0][idx]
        val = a[raw_idx]
        pct_rank = get_percentile_rank(val, valid_vals)
        z_score = (val - np.mean(valid_vals)) / (np.std(valid_vals) + 1e-10)
        
        extremes.append({
            'Index': raw_idx,
            'Value': val,
            'Pct_Rank': pct_rank,
            'Z_Score': z_score
        })
    
    return pd.DataFrame(extremes)


# ============================================================================
# CAR CALCULATION (no winsorization)
# ============================================================================

def build_daily_lookup(daily_df):
    """Create lookup dictionaries for daily closing prices and dates."""
    daily_lookup = {}
    daily_grouped_dates = {}
    
    for sym, g in daily_df.groupby("symbol"):
        g_sorted = g.sort_values("Date")
        arr = g_sorted["Date"].values
        daily_grouped_dates[sym] = arr
        for d, price in zip(g_sorted["Date"], g_sorted["Close"]):
            daily_lookup[(sym, pd.to_datetime(d))] = float(price)
    
    return daily_lookup, daily_grouped_dates


def find_trading_index(dates_arr, target_date_ts):
    """Find index of target date in sorted dates array."""
    if dates_arr is None or len(dates_arr) == 0:
        return None
    
    t64 = np.datetime64(pd.to_datetime(target_date_ts))
    idx = np.searchsorted(dates_arr, t64)
    
    if idx < len(dates_arr) and dates_arr[idx] == t64:
        return int(idx)
    if idx - 1 >= 0 and dates_arr[idx - 1] == t64:
        return int(idx - 1)
    
    return None


def load_intraday_file(path, filter_after_open=True):
    """Load intraday CSV file with optional 9:30 AM filtering."""
    # Try several strategies to read large/irregular intraday files robustly
    parsers = [
        {"kwargs": {"parse_dates": ["Datetime"], "low_memory": False}},
        {"kwargs": {"sep": "\t", "parse_dates": ["Datetime"], "low_memory": False}},
        {"kwargs": {"sep": ",", "parse_dates": ["Datetime"], "low_memory": False}},
        {"kwargs": {"engine": "python", "delim_whitespace": True, "low_memory": False}},
        {"kwargs": {"engine": "python", "sep": None, "low_memory": False}},
    ]

    df = None
    for p in parsers:
        try:
            df = pd.read_csv(path, **p["kwargs"])
            if df is None or df.shape[1] == 0:
                df = None
                continue
            break
        except Exception:
            df = None
            continue

    if df is None:
        # try reading without header (many intraday files have no header)
        try:
            df = pd.read_csv(path, header=None,
                             names=["Datetime", "Open", "High", "Low", "Close", "Volume"],
                             parse_dates=[0], low_memory=False)
        except Exception:
            return None


    # Normalize column names to handle case differences
    cols_map = {c.lower(): c for c in df.columns}

    # Ensure we have a Datetime column; try to build/convert if necessary
    dt_col = None
    if "datetime" in cols_map:
        dt_col = cols_map["datetime"]
    elif "date" in cols_map and "time" in cols_map:
        # combine date + time
        try:
            df["Datetime"] = pd.to_datetime(df[cols_map["date"]].astype(str) + " " + df[cols_map["time"]].astype(str))
            dt_col = "Datetime"
        except Exception:
            dt_col = None
    elif "date" in cols_map:
        try:
            df["Datetime"] = pd.to_datetime(df[cols_map["date"]])
            dt_col = "Datetime"
        except Exception:
            dt_col = None
    else:
        # try any column that looks like a datetime by attempting conversion
        for c in df.columns:
            try:
                tmp = pd.to_datetime(df[c])
                df["Datetime"] = tmp
                dt_col = "Datetime"
                break
            except Exception:
                continue

    if dt_col is None or "Open" not in cols_map and "open" not in cols_map:
        # ensure open exists (case-insensitive)
        return None

    # standardize Open column name
    open_col = cols_map.get("open", cols_map.get("Open", None))
    if open_col is None:
        # try to locate a column likely to be the open price by heuristics
        for c in df.columns:
            if c.lower() in ("o","open","price","bid"):
                open_col = c
                break

    if open_col is None:
        return None

    # ensure Datetime column is datetime dtype
    if not np.issubdtype(df["Datetime"].dtype, np.datetime64):
        try:
            df["Datetime"] = pd.to_datetime(df["Datetime"], errors='coerce')
        except Exception:
            return None

    df = df[["Datetime", open_col]].dropna().sort_values("Datetime").reset_index(drop=True)
    df = df.rename(columns={open_col: "Open"})

    if filter_after_open:
        try:
            df = df[df["Datetime"].dt.time >= time(9, 30)].reset_index(drop=True)
        except Exception:
            # if dt parsing failed for some rows, just return full df
            pass

    return df


# Simple intraday file cache to avoid reloading the same CSV repeatedly
_intraday_cache = {}
def load_intraday_cached(path, filter_after_open=True):
    """Cached wrapper around `load_intraday_file`.

    Stores DataFrame or None in module cache keyed by path string.
    """
    key = str(path)
    if key in _intraday_cache:
        return _intraday_cache[key]
    df = load_intraday_file(path, filter_after_open=filter_after_open)
    _intraday_cache[key] = df
    return df


def cap_daily_returns(daily_df, close_col='Close', date_col='Date', symbol_col='symbol',
                      upper=10.0, lower=-0.95, inplace=False):
    """Cap daily returns per symbol and produce capped close values.

    Adds columns:
      - prev_close
      - daily_return_raw
      - daily_return_capped
      - Close_capped

    Returns a DataFrame (copied unless inplace=True).
    `upper` and `lower` are decimal returns (10.0 = +1000%, -0.95 = -95%).
    """
    if not inplace:
        df = daily_df.copy()
    else:
        df = daily_df

    # ensure date column is datetime
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])

    # ensure symbol column exists (some datasets use 'Ticker')
    if symbol_col not in df.columns and 'Ticker' in df.columns:
        df[symbol_col] = df['Ticker'].astype(str).str.upper()

    # compute prev_close per symbol
    df = df.sort_values([symbol_col, date_col]).reset_index(drop=True)
    df['prev_close'] = df.groupby(symbol_col)[close_col].shift(1)

    # compute raw daily return
    df['daily_return_raw'] = (df[close_col] / df['prev_close']) - 1.0

    # cap
    df['daily_return_capped'] = df['daily_return_raw'].clip(lower=lower, upper=upper)

    # compute capped close price based on prev_close * (1 + capped_return)
    df['Close_capped'] = df['prev_close'] * (1.0 + df['daily_return_capped'])

    return df


def calculate_car(entry_price, exit_price):
    """Calculate single event return (as decimal, not percent).
    
    Args:
        entry_price: Entry price (e.g., open at +1 min)
        exit_price: Exit price (e.g., close +N days later)
    
    Returns:
        Return as decimal (e.g., 0.05 for +5%, 3.0 for +300%)
    """
    if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0 or exit_price <= 0:
        return np.nan
    
    return (exit_price / entry_price) - 1.0


def get_exit_price_for_window(symbol, event_date, window_days, daily_lookup, daily_grouped):
    """Get closing price for specified time window after event."""
    dates_arr = daily_grouped.get(symbol)
    if dates_arr is None:
        return np.nan
    
    pos = find_trading_index(dates_arr, event_date)
    if pos is None:
        return np.nan
    
    exit_pos = pos + window_days
    if exit_pos >= len(dates_arr):
        return np.nan
    
    exit_date = pd.to_datetime(dates_arr[exit_pos])
    return daily_lookup.get((symbol, exit_date), np.nan)


def compute_car_for_event(symbol, event_date, entry_price, windows, 
                          daily_lookup, daily_grouped):
    """Compute CAR for a single event across multiple windows.
    
    Returns:
        Dict with CAR values for each window (as decimals, not percent)
    """
    result = {}
    
    for window_name, window_days in windows.items():
        exit_price = get_exit_price_for_window(
            symbol, event_date, window_days, daily_lookup, daily_grouped
        )
        car = calculate_car(entry_price, exit_price)
        result[window_name] = car
    
    return result


# ============================================================================
# QUARTILE ASSIGNMENT
# ============================================================================

def prepare_quartile_data(df, metric_col, symbol_col="symbol", date_col="Date",
                          rolling=True, window=50, min_periods=2):
    """Assign quartiles with optional rolling (in-sample) cutoffs to avoid look-ahead.

    Args:
        df: input DataFrame
        metric_col: column to compute quartiles on
        symbol_col: grouping column (symbol)
        date_col: date column name
        rolling: if True compute rolling (in-sample) quartiles per symbol
        window: lookback window size (default 50)
        min_periods: minimum prior observations required to compute quartile (default 4)
                    If fewer than min_periods prior observations exist, Quartile will be NaN.
    """
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])

    df["Quartile"] = np.nan

    if not rolling:
        mask = df[metric_col].notna()
        if mask.sum() == 0:
            return df
        try:
            df.loc[mask, "Quartile"] = pd.qcut(df.loc[mask, metric_col], 4,
                                                 labels=["Q1", "Q2", "Q3", "Q4"])
        except ValueError:
            df.loc[mask, "Quartile"] = pd.cut(df.loc[mask, metric_col], 4,
                                                labels=["Q1", "Q2", "Q3", "Q4"])
        return df

    # Rolling quartiles
    grouped = df.sort_values([symbol_col, date_col]).groupby(symbol_col, sort=False)

    def assign_rolling_quartiles(group):
        vals = group[metric_col].values
        quarters = np.array([np.nan] * len(group), dtype=object)
        for i in range(len(group)):
            # require at least `min_periods` prior observations
            if i == 0:
                continue
            start = max(0, i - window)
            prior = vals[start:i]
            prior = prior[~pd.isna(prior)]
            if len(prior) < min_periods:
                quarters[i] = np.nan
                continue
            try:
                bins = pd.qcut(prior, 4, retbins=True, duplicates='drop')[1]
                cur = vals[i]
                if pd.isna(cur):
                    quarters[i] = np.nan
                else:
                    idx = np.digitize([cur], bins[1:-1], right=True)[0]
                    quarters[i] = f"Q{idx+1}"
            except Exception:
                quarters[i] = np.nan
        group = group.copy()
        group["Quartile"] = quarters
        return group

    out = grouped.apply(assign_rolling_quartiles).reset_index(drop=True)
    return out


def prepare_quartile_data_global(df, metric_col, date_col="Date", rolling=True, window=100, min_periods=2):
    """Assign quartiles using a global (cross-sectional) rolling window across all symbols.

    For each event (row) sorted by `Date`, the function looks back at the prior `window`
    metric values across all symbols and computes quartile bins; the current row is
    then assigned to one of Q1..Q4 based on those bins. This matches the workflow where
    quartile cutoffs are determined from the cross-sectional distribution of prior events
    (not symbol-specific).

    Args:
        df: DataFrame containing at least `date_col` and `metric_col`.
        metric_col: column with numeric metric values.
        date_col: date column name.
        rolling: if True, compute rolling cutoffs; if False compute global quartiles on full sample.
        window: lookback window size (number of prior events to use).
        min_periods: minimum prior observations required to assign quartile.

    Returns:
        DataFrame with a new `Quartile` column aligned to input order.
    """
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])

    df = df.sort_values(date_col).reset_index(drop=True)
    vals = pd.to_numeric(df[metric_col], errors='coerce').values
    quartiles = np.array([np.nan] * len(df), dtype=object)

    if not rolling:
        mask = ~np.isnan(vals)
        if mask.sum() == 0:
            df['Quartile'] = quartiles
            return df
        try:
            bins = pd.qcut(vals[mask], 4, retbins=True, duplicates='drop')[1]
            # assign
            idxs = np.digitize(vals, bins[1:-1], right=True)
            for i in range(len(idxs)):
                if np.isnan(vals[i]):
                    quartiles[i] = np.nan
                else:
                    quartiles[i] = f"Q{idxs[i]+1}"
        except Exception:
            quartiles[mask] = pd.cut(vals[mask], 4, labels=["Q1","Q2","Q3","Q4"]).astype(object)
        df['Quartile'] = quartiles
        return df

    # rolling global quartiles: for each row use prior `window` metric values
    for i in range(len(df)):
        start = max(0, i - window)
        prior = vals[start:i]
        prior = prior[~np.isnan(prior)]
        if len(prior) < min_periods:
            quartiles[i] = np.nan
            continue
        try:
            bins = pd.qcut(prior, 4, retbins=True, duplicates='drop')[1]
            cur = vals[i]
            if np.isnan(cur):
                quartiles[i] = np.nan
            else:
                idx = np.digitize([cur], bins[1:-1], right=True)[0]
                quartiles[i] = f"Q{idx+1}"
        except Exception:
            quartiles[i] = np.nan

    df['Quartile'] = quartiles
    return df


# ============================================================================
# T-TEST & STATISTICS
# ============================================================================

def apply_ttest(group_values, window_name=""):
    """Run one-sample t-test on returns (values in decimal, NOT percent).
    
    Returns:
        Dict with t-stat, p-value, significance level (***/**/*), and descriptives
    """
    from scipy import stats
    
    if len(group_values) < 2 or np.isnan(group_values).all():
        return {"t_stat": np.nan, "p_value": np.nan, "sig_level": ""}
    
    values_clean = np.array(group_values)[~np.isnan(group_values)]
    if len(values_clean) < 2:
        return {"t_stat": np.nan, "p_value": np.nan, "sig_level": ""}
    
    # Convert to percent for reporting (but t-test stays on same scale)
    values_pct = values_clean * 100.0
    t_stat, p_value = stats.ttest_1samp(values_pct, 0)
    
    if p_value < 0.01:
        sig_level = "***"
    elif p_value < 0.05:
        sig_level = "**"
    elif p_value < 0.10:
        sig_level = "*"
    else:
        sig_level = ""
    
    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "sig_level": sig_level,
        "mean_pct": np.nanmean(values_pct),      # ← in percent
        "median_pct": np.nanmedian(values_pct),  # ← in percent
        "std_pct": np.nanstd(values_pct),        # ← in percent
        "min_pct": np.nanmin(values_pct),
        "max_pct": np.nanmax(values_pct),
        "n": len(values_clean)
    }


def compute_descriptive_stats(group_data, windows):
    """Compute mean, median, std for each window."""
    stats_dict = {}
    
    for window_name in windows.keys():
        if window_name in group_data.columns:
            values = group_data[window_name].dropna()
            if len(values) > 0:
                stats_dict[window_name] = {
                    "N": len(values),
                    "Mean": values.mean(),
                    "Median": values.median(),
                    "Std": values.std(),
                    "Min": values.min(),
                    "Max": values.max()
                }
    
    return stats_dict
