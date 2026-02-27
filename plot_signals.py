"""Simple helpers to visualize price and signals for a single ticker."""
import pandas as pd
import matplotlib.pyplot as plt

from config import MAIN_DIR


def plot_ticker(ticker: str, signal_col: str = "Sig30_P5") -> None:
    daily_df = pd.read_csv(MAIN_DIR / "daily_ohlc_processed.csv", parse_dates=["Date"])
    sig_df = pd.read_csv(MAIN_DIR / "SIGNIFICANT_GAPS_final.csv", parse_dates=["Date"])
    gap_ups = pd.read_csv(MAIN_DIR / "gap_up_trades.csv", parse_dates=["Date"])
    gap_downs = pd.read_csv(MAIN_DIR / "gap_down_trades.csv", parse_dates=["Date"])

    valid_ups = sig_df.merge(gap_ups[["Ticker", "Date"]], on=["Ticker", "Date"], how="inner")
    valid_downs = sig_df.merge(gap_downs[["Ticker", "Date"]], on=["Ticker", "Date"], how="inner")

    price_data = daily_df[daily_df["Ticker"] == ticker].sort_values("Date")
    ticker_ups = valid_ups[(valid_ups["Ticker"] == ticker) & (valid_ups[signal_col] == True)]
    ticker_downs = valid_downs[(valid_downs["Ticker"] == ticker) & (valid_downs[signal_col] == True)]

    if price_data.empty:
        print(f"No data for {ticker}")
        return

    plt.figure(figsize=(15, 8))
    plt.plot(price_data["Date"], price_data["Close"], label="Close Price", color="black", linewidth=1.2, alpha=0.8)
    if not ticker_ups.empty:
        ups_plot = ticker_ups.merge(price_data[["Date", "Close"]], on="Date")
        plt.scatter(ups_plot["Date"], ups_plot["Close"], color="lime", label="Significant Gap Up", s=120,
                    edgecolors="black", marker="^", zorder=5)
    if not ticker_downs.empty:
        downs_plot = ticker_downs.merge(price_data[["Date", "Close"]], on="Date")
        plt.scatter(downs_plot["Date"], downs_plot["Close"], color="red", label="Significant Gap Down", s=120,
                    edgecolors="black", marker="v", zorder=5)
    plt.title(f"{ticker} Price Action & Valid Shocks\n(Filter: {signal_col})", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
