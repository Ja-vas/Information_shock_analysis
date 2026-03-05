"""Helper functions for hierarchical news category assignment and analysis."""

import pandas as pd
import numpy as np


# Strict hierarchical order for news categories
HIERARCHY_ORDER = [
    "news_Guidance___Outlook_count",          # 1. Highest priority
    "news_M&A___Deal_count",                  # 2.
    "news_Product_News_count",                # 3.
    "news_Regulatory_Approval_count",         # 4.
    "news_Analyst___Rating_count",            # 5.
    "news_Financing___Capital_count",         # 6.
    "news_Management___Legal_count",          # 7.
    "news_Dividends___Buybacks_count"         # 8. Lowest priority
]

CATEGORY_LABELS = {
    "news_Guidance___Outlook_count": "Guidance & Outlook",
    "news_M&A___Deal_count": "M&A & Deals",
    "news_Product_News_count": "Product News",
    "news_Regulatory_Approval_count": "Regulatory Approval",
    "news_Analyst___Rating_count": "Analyst Ratings",
    "news_Financing___Capital_count": "Financing & Capital",
    "news_Management___Legal_count": "Management & Legal",
    "news_Dividends___Buybacks_count": "Dividends & Buybacks",
    "No_news": "No News"
}


def pick_category(row, hierarchy=None):
    """Assign single category using strict hierarchical order.
    
    For each row, finds the first category in the hierarchy with count > 0.
    
    Args:
        row: DataFrame row or dict with category columns
        hierarchy: List of category column names (default: HIERARCHY_ORDER)
    
    Returns:
        Category column name or "No_news"
    """
    if hierarchy is None:
        hierarchy = HIERARCHY_ORDER
    
    for cat in hierarchy:
        if row.get(cat, 0) > 0:
            return cat
    
    return "No_news"


def apply_hierarchical_categorization(df, hierarchy=None):
    """Apply hierarchical category assignment to entire DataFrame.
    
    Args:
        df: DataFrame with news count columns
        hierarchy: List of category columns (default: HIERARCHY_ORDER)
    
    Returns:
        DataFrame with added 'main_category' column
    """
    if hierarchy is None:
        hierarchy = HIERARCHY_ORDER
    
    df = df.copy()
    df["main_category"] = df.apply(lambda row: pick_category(row, hierarchy), axis=1)
    
    return df


def count_events_by_category(df, category_col="main_category"):
    """Count events in each category.
    
    Args:
        df: DataFrame with category column
        category_col: Name of category column
    
    Returns:
        Series with counts per category
    """
    return df[category_col].value_counts().sort_index()


def get_pretty_category_name(category_key):
    """Get human-readable name for category.
    
    Args:
        category_key: Raw category column name (e.g., "news_M&A___Deal_count")
    
    Returns:
        Pretty name (e.g., "M&A & Deals")
    """
    return CATEGORY_LABELS.get(category_key, category_key)


def categorize_news_hierarchy_table(category_counts, df_reference=None):
    """Create summary table of categories with their counts and ranks.
    
    Args:
        category_counts: Series or dict with counts per category
        df_reference: Optional reference DataFrame to get additional stats
    
    Returns:
        DataFrame with category info and ranking
    """
    if isinstance(category_counts, dict):
        category_counts = pd.Series(category_counts)
    
    # Sort by count descending
    sorted_counts = category_counts.sort_values(ascending=False)
    
    result = pd.DataFrame({
        "Category": sorted_counts.index,
        "Pretty_Name": [get_pretty_category_name(c) for c in sorted_counts.index],
        "Count": sorted_counts.values,
        "Percent": (sorted_counts.values / sorted_counts.sum() * 100).round(2)
    })
    
    # Add hierarchy rank if available
    result["Hierarchy_Rank"] = result["Category"].apply(
        lambda c: HIERARCHY_ORDER.index(c) + 1 if c in HIERARCHY_ORDER else 999
    )
    
    return result.reset_index(drop=True)


def filter_by_category(df, category, category_col="main_category"):
    """Filter DataFrame to specific category.
    
    Args:
        df: DataFrame with category column
        category: Category to filter to
        category_col: Name of category column
    
    Returns:
        Filtered DataFrame
    """
    return df[df[category_col] == category].copy()


def group_by_category(df, category_col="main_category", agg_func="size"):
    """Group data by category and apply aggregation.
    
    Args:
        df: DataFrame with category column
        category_col: Name of category column
        agg_func: Aggregation function or dict of {col: func}
    
    Returns:
        Grouped results
    """
    return df.groupby(category_col).agg(agg_func)


def compute_category_stats(df, value_cols, category_col="main_category"):
    """Compute statistics for value columns grouped by category.
    
    Args:
        df: DataFrame with category and value columns
        value_cols: List of value column names (e.g., CAR columns)
        category_col: Name of category column
    
    Returns:
        DataFrame with statistics per category and value column
    """
    results = []
    
    for category, group_df in df.groupby(category_col):
        for col in value_cols:
            if col in group_df.columns:
                values = group_df[col].dropna()
                if len(values) > 0:
                    results.append({
                        "Category": category,
                        "Pretty_Category": get_pretty_category_name(category),
                        "Value_Col": col,
                        "N": len(values),
                        "Mean": values.mean(),
                        "Median": values.median(),
                        "Std": values.std(),
                        "Min": values.min(),
                        "Max": values.max()
                    })
    
    return pd.DataFrame(results) if results else pd.DataFrame()


def test_category_significance(df, value_cols, category_col="main_category"):
    """Run t-tests for each category and value column.
    
    Args:
        df: DataFrame with category and value columns
        value_cols: List of value column names
        category_col: Name of category column
    
    Returns:
        DataFrame with t-test results and significance flags
    """
    from scipy import stats as scipy_stats
    
    results = []
    
    for category, group_df in df.groupby(category_col):
        for col in value_cols:
            if col in group_df.columns:
                values = group_df[col].dropna()
                if len(values) >= 2:
                    t_stat, p_val = scipy_stats.ttest_1samp(values, 0)
                    
                    if p_val < 0.01:
                        sig_mark = "***"
                    elif p_val < 0.05:
                        sig_mark = "**"
                    elif p_val < 0.10:
                        sig_mark = "*"
                    else:
                        sig_mark = ""
                    
                    results.append({
                        "Category": category,
                        "Pretty_Category": get_pretty_category_name(category),
                        "Value_Col": col,
                        "N": len(values),
                        "Mean": values.mean(),
                        "t_stat": t_stat,
                        "p_value": p_val,
                        "Significance": sig_mark
                    })
    
    return pd.DataFrame(results) if results else pd.DataFrame()


def create_category_cross_matrix(df, metric_col="Metric", category_col="main_category"):
    """Create crosstab of metric categories.
    
    Args:
        df: DataFrame
        metric_col: Name of metric column
        category_col: Name of category column
    
    Returns:
        Crosstab DataFrame
    """
    if metric_col not in df.columns or category_col not in df.columns:
        return pd.DataFrame()
    
    return pd.crosstab(
        df[metric_col],
        df[category_col],
        margins=True,
        margins_name="Total"
    )
