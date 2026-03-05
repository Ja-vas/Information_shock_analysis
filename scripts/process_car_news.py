import pandas as pd
from pathlib import Path
import sys

# Attempt to load from config, with a safe fallback
try:
    from config import MAIN_DIR, CAR_RESULTS_DIR
    MAIN_DIR = Path(MAIN_DIR)
    CAR_RESULTS_DIR = Path(CAR_RESULTS_DIR)
except ImportError:
    ROOT_DIR = Path(__file__).resolve().parents[1]
    MAIN_DIR = ROOT_DIR / "main_dataframe"
    CAR_RESULTS_DIR = MAIN_DIR / "car_results"

def process_car_raw():
    print("1. Loading datasets...")
    car_data_path = CAR_RESULTS_DIR / "CAR_raw_results.csv"
    news_classified_path = MAIN_DIR / "news_classified.csv"

    if not car_data_path.exists():
        raise FileNotFoundError(f"Missing CAR data at {car_data_path}")
    if not news_classified_path.exists():
        raise FileNotFoundError(f"Missing News data at {news_classified_path}")

    car_data = pd.read_csv(car_data_path, low_memory=False)
    news_classified = pd.read_csv(news_classified_path, low_memory=False)

    print("2. Formatting Dates and Tickers for a bulletproof merge...")
    
    # 100% guarantee identical Date strings (e.g., '2016-08-15')
    car_data['Date'] = pd.to_datetime(car_data['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    news_classified['Date_gap'] = pd.to_datetime(news_classified['Date_gap'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    # Drop rows where dates were completely invalid
    car_data = car_data.dropna(subset=['Date'])
    news_classified = news_classified.dropna(subset=['Date_gap'])
    
    # 100% guarantee identical Tickers with NO hidden spaces
    car_data['Ticker'] = car_data.get('Ticker', car_data.get('Symbol')).astype(str).str.upper().str.strip()
    news_classified['symbol'] = news_classified['symbol'].astype(str).str.upper().str.strip()

    print("3. Applying strict hierarchy to resolve multiple news articles...")
    
    HIERARCHY_ORDER = [
        "Guidance",     # 1: Guidance / Outlook
        "M&A",          # 2: M&A / Deal
        "Product",      # 3: Product News
        "Regulatory",   # 4: Regulatory Approval
        "Analyst",      # 5: Analyst / Rating
        "Financing",    # 6: Financing / Capital
        "Management",   # 7: Management / Legal
        "Dividend"      # 8: Dividends / Buybacks
    ]

    def pick_top_category(categories):
        valid_cats = [str(c).strip() for c in categories if pd.notna(c) and str(c).strip() != ""]
        
        if not valid_cats:
            return "No_news"
            
        # Search against strict hierarchy
        for keyword in HIERARCHY_ORDER:
            for cat in valid_cats:
                if keyword.lower() in cat.lower():
                    return keyword # Return EXACTLY the clean keyword (e.g. "Guidance")
                    
        # Filter out "full" and "market" noise
        for cat in valid_cats:
            cat_lower = cat.lower()
            if "full" not in cat_lower and "market" not in cat_lower:
                return cat
                
        return "No_news"

    # Group by Gap (Symbol + Date) and find the top category
    grouped_news = news_classified.groupby(['symbol', 'Date_gap'])['news_category_groq'].apply(pick_top_category).reset_index()
    grouped_news.rename(columns={'news_category_groq': 'main_category'}, inplace=True)
    
    print(f" -> Condensed {len(news_classified)} articles into {len(grouped_news)} unique gap days with news.")

    print("4. Merging CAR returns with top news category...")
    
    # The Merge! Now that formatting is identical, this will map perfectly.
    merged_data = pd.merge(
        car_data, 
        grouped_news, 
        left_on=['Ticker', 'Date'], 
        right_on=['symbol', 'Date_gap'], 
        how='left'
    )

    merged_data['main_category'] = merged_data['main_category'].fillna('No_news')
    
    if 'symbol' in merged_data.columns:
        merged_data.drop(columns=['symbol', 'Date_gap'], inplace=True, errors='ignore')

    print("5. Splitting and Saving results...")
    dir_col = 'Direction' if 'Direction' in merged_data.columns else 'Type'
    
    car_news_results_down = merged_data[merged_data[dir_col].astype(str).str.lower() == "down"]
    car_news_results_up = merged_data[merged_data[dir_col].astype(str).str.lower() == "up"]

    car_news_results_down_path = CAR_RESULTS_DIR / "CAR_news_results_down.csv"
    car_news_results_up_path = CAR_RESULTS_DIR / "CAR_news_results_up.csv"

    car_news_results_down.to_csv(car_news_results_down_path, index=False, encoding="utf-8-sig")
    car_news_results_up.to_csv(car_news_results_up_path, index=False, encoding="utf-8-sig")

    # Quick audit print to verify the merge worked
    no_news_count = (merged_data['main_category'] == 'No_news').sum()
    has_news_count = len(merged_data) - no_news_count
    print(f"\n[AUDIT] Total events: {len(merged_data)} | Categorized with News: {has_news_count} | No News: {no_news_count}")

    print(f" Saved Gap DOWN results ({len(car_news_results_down)} rows)")
    print(f" Saved Gap UP results ({len(car_news_results_up)} rows)")

if __name__ == "__main__":
    process_car_raw()