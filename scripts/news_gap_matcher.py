"""Match classified news articles to earnings gaps.

This module joins news data with significant gaps based on ticker and date.
Output is used for downstream analysis of market reactions to news.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def match_news_to_gaps(gaps_df: pd.DataFrame, news_df: pd.DataFrame,
                       gap_ticker_col: str = "symbol",
                       gap_date_col: str = "Date",
                       news_ticker_col: str = "Ticker",
                       news_date_col: str = "Date",
                       time_tolerance_days: int = 1) -> pd.DataFrame:
    """Match news articles to gaps on same day or within tolerance.
    
    Args:
        gaps_df: DataFrame with significant gaps
        news_df: DataFrame with classified news
        gap_ticker_col: Column name for ticker in gaps (default: "symbol")
        gap_date_col: Column name for date in gaps (default: "Date")
        news_ticker_col: Column name for ticker in news (default: "Ticker")
        news_date_col: Column name for date in news (default: "Date")
        time_tolerance_days: Days allow between gap and news (default: 1)
    
    Returns:
        Matched DataFrame with gap and news info
    """
    logger.info(f"Matching {len(gaps_df)} gaps with {len(news_df)} news items...")
    
    # Ensure date columns are datetime
    gaps_df = gaps_df.copy()
    news_df = news_df.copy()
    
    gaps_df[gap_date_col] = pd.to_datetime(gaps_df[gap_date_col], errors='coerce')
    news_df[news_date_col] = pd.to_datetime(news_df[news_date_col], errors='coerce')
    
    # Standardize ticker format
    gaps_df[gap_ticker_col] = gaps_df[gap_ticker_col].astype(str).str.upper().str.strip()
    news_df[news_ticker_col] = news_df[news_ticker_col].astype(str).str.upper().str.strip()
    
    # First pass: exact date match
    news_df["Date_only"] = news_df[news_date_col].dt.date
    gaps_df["Date_only"] = gaps_df[gap_date_col].dt.date
    
    matched = gaps_df.merge(
        news_df,
        left_on=[gap_ticker_col, "Date_only"],
        right_on=[news_ticker_col, "Date_only"],
        how="left",
        suffixes=("_gap", "_news")
    )
    
    # Second pass: fuzzy date matching for unmatched gaps
    unmatched = matched[matched[news_ticker_col].isna()].copy()
    
    if len(unmatched) > 0 and time_tolerance_days > 0:
        logger.info(f"Attempting fuzzy date matching for {len(unmatched)} unmatched gaps...")
        
        new_matches = []
        for _, gap_row in unmatched.iterrows():
            ticker = gap_row[gap_ticker_col]
            gap_date = gap_row["Date_only"]
            
            # Find news for same ticker within tolerance
            candidate_news = news_df[
                (news_df[news_ticker_col] == ticker) &
                ((news_df["Date_only"] - gap_date).abs() <= pd.Timedelta(days=time_tolerance_days))
            ].copy()
            
            if len(candidate_news) > 0:
                # Keep closest match
                candidate_news["days_diff"] = (candidate_news["Date_only"] - gap_date).abs()
                closest = candidate_news.loc[candidate_news["days_diff"].idxmin()]
                new_matches.append({
                    "gap_id": gap_row.get("gap_id", None),
                    "match_type": "fuzzy_date",
                    **closest.to_dict()
                })
        
        if new_matches:
            logger.info(f"Found {len(new_matches)} fuzzy matches")
    
    # Clean up
    matched = matched.drop(columns=["Date_only"], errors='ignore')
    gaps_df = gaps_df.drop(columns=["Date_only"], errors='ignore')
    
    logger.info(f"Total matched records: {matched[news_ticker_col].notna().sum()}")
    
    return matched


def aggregate_news_per_gap(matched_df: pd.DataFrame,
                           ticker_col: str = "symbol",
                           date_col: str = "Date",
                           category_col: str = "news_category") -> pd.DataFrame:
    """Aggregate news articles per gap (count by category).
    
    When multiple news items match one gap, create category counts.
    
    Args:
        matched_df: Output from match_news_to_gaps()
        ticker_col: Ticker column
        date_col: Date column
        category_col: News category column
    
    Returns:
        DataFrame with gap as row, category counts as columns
    """
    logger.info("Aggregating news per gap by category...")
    
    # Group by gap (ticker + date)
    grouped = matched_df.groupby([ticker_col, date_col]).agg({
        category_col: lambda x: x.value_counts().to_dict()
    }).reset_index()
    
    # Expand category counts to separate columns
    category_counts = grouped[category_col].apply(pd.Series).fillna(0).astype(int)
    category_counts.columns = [f"news_{col}_count" for col in category_counts.columns]
    
    result = grouped[[ticker_col, date_col]].reset_index(drop=True)
    result = pd.concat([result, category_counts], axis=1)
    
    logger.info(f"Created {len(result)} gap records with category counts")
    
    return result


def match_gaps_to_news_file(gaps_path: Path, news_path: Path, output_path: Path,
                            gap_ticker_col: str = "symbol") -> None:
    """Load gaps and news from files and save matched result.
    
    Args:
        gaps_path: Path to significant gaps CSV
        news_path: Path to classified news CSV
        output_path: Path to save matched output
        gap_ticker_col: Ticker column in gaps file
    """
    logger.info(f"Loading gaps from: {gaps_path}")
    gaps_df = pd.read_csv(gaps_path, parse_dates=["Date"], low_memory=False)
    
    logger.info(f"Loading news from: {news_path}")
    news_df = pd.read_csv(news_path, parse_dates=["Date"], low_memory=False)
    
    # Match
    matched = match_news_to_gaps(gaps_df, news_df, gap_ticker_col=gap_ticker_col)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"Saved matched data to: {output_path}")


def main():
    """Example: Match gaps in SIG_SPLITS to classified news."""
    from config import MAIN_DIR
    
    # Paths (update as needed)
    gaps_file = MAIN_DIR / "SIG_SPLITS" / "Sig30_P30.csv"
    news_file = MAIN_DIR / "news_classified.csv"
    output_file = MAIN_DIR / "gaps_with_news.csv"
    
    if not gaps_file.exists():
        logger.error(f"Gaps file not found: {gaps_file}")
        return
    
    if not news_file.exists():
        logger.error(f"News file not found: {news_file}")
        return
    
    match_gaps_to_news_file(gaps_file, news_file, output_file)
    logger.info("News-to-gaps matching complete!")


if __name__ == "__main__":
    main()
