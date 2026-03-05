# Information Shock Analysis

This project is a quantitative study of stock prices reaction to sudden information shocks. It specifically looks at **positive significant gaps** (with higher than average pre-market volume) and tests whether the market underreacts to the news behind them.

Building on my previous work with **Post-Earnings Announcement Drift (PEAD)**, this version adds a LLM classifier to determine the "why" behind a price jump, allowing us to distinguish between 3 main categories: earnings and revenue surprises, informative firm-specific news, and "no-news" price shocks.

---

## What's in this repo?

**1. Data pipeline & cleaning**
* Scripts to process **16M+ rows** of daily OHLCV data and news text for more than 6,000 US stocks.
* Cleaning process: handling of stock splits, inactive stocks, liquidity filters and winsorizing outliers (0.01/99.99 percentiles) 
* Calculation and winsorizing of SUE (Earnings) and SUR (Revenue) by scaling surprises against an 8-quarter rolling standard deviation.

**2. Identification of "Significant gaps"**
To isolate moves with real price shock, gaps are filtered by pre-market and early-session volume. A gap is "Significant" if:
* **The price jump is ≥ 6%**
* **It has relative volume:** The dollar volume in the first 1 or 5 minutes exceeds the 30-day average daily volume, or exceeds 2x the average daily volume within the first 30 minutes.

**3. Hierarchical news classification**
Using **Llama-3 (via Groq API)**, 19,000+ news articles were categorized. To handle events with multiple news stories, the system uses a priority hierarchy to assign a single dominant catalyst:
1. Guidance / Outlook
2. M&A / Deal
3. Product News
4. Regulatory Approval
5. Analyst / Rating
6. Financing / Capital
7. Management / Legal
8. Dividends / Buybacks

**4. Empirical analysis**
* **Cumulative abnormal returns (CAR):** Measuring price performance across T+0, T+5, T+10, T+22, and T+60 windows.
* **Significance testing:** One-sample t-tests to determine if the CAR for specific news categories or surprise quartiles is statistically different from zero.

---

## How it all works

**Before running any notebooks, edit `config.py`:**

### Data Pipeline (`01_main_workflow.ipynb`)
- Load raw OHLCV data
- Apply quality filters (liquidity, outliers)
- Calculate standardized earning surprises (SUE, SUR)
- Detect significant gaps (≥6% with volume confirmation)
- Match gaps to earnings announcements
- Output: `SIGNIFICANT_GAPS_final.csv`

### News Classification (Preprocessing)
Before running analysis notebooks, classify news articles:

**Step A: Download FNSPID News**
```bash
# Clone FNSPID dataset
git clone https://github.com/Zdong104/FNSPID_Financial_News_Dataset.git
# Extract news CSV and place in main_dataframe/
```

**Step B: Categorize News with Groq API**
```python
from scripts.groq_news_categorizer import main_categorize_news
from config import MAIN_DIR

input_file = MAIN_DIR / "raw_news.csv"  # FNSPID input
output_file = MAIN_DIR / "news_classified.csv"
main_categorize_news(input_file, output_file)
```

**Step C: Match News to Gaps**
```python
from scripts.news_gap_matcher import match_gaps_to_news_file
from config import MAIN_DIR

gaps_file = MAIN_DIR / "SIGNIFICANT_GAPS_final.csv"
news_file = MAIN_DIR / "news_classified.csv"
output_file = MAIN_DIR / "gaps_with_news.csv"
match_gaps_to_news_file(gaps_file, news_file, output_file)
```

**Requirements:**
- Groq API key (set as `GROQ_API_KEY` environment variable)
- Install: `pip install groq`


### CAR Quartiles & Categories (`02_car_analysis.ipynb`)
- Compute Cumulative Abnormal Returns across windows (+0d, +5d, +10d, +22d, +60d)
- Assign each earnigns gap to quartiles
- Run t-tests on returns
- Visualize CAR profiles and significance

### Backtesting (`03_backtest_workflow.ipynb`)
- Design trading strategies based on gaps
- Measure performance: returns, Sharpe ratio, max drawdown
---

If you wish to run this pipeline using your own data providers, ensure your raw files are structured with the following columns before running the preprocessing scripts. Dummy datasets is located in `data/sample/`.

**1. 1-Minute OHLCV Data + Daily data (Optional)**
* `Timestamp` (format: YYYY-MM-DD HH:MM:SS)
* `Open`, `High`, `Low`, `Close` 
* `Volume` 
* `Ticker` 

**2. Earnings & Fundamentals**
* `Ticker` 
* `Earnings_Date` (format: YYYY-MM-DD)
* `EPS` 
* `Consensus` 
* `Revenue` 
* `Revenue_Consensus` 

**3. News Headlines**
* `Ticker`
* `Date` (format: YYYY-MM-DD HH:MM:SS)
* `Headline` 
* `Body` (optional but recommended for LLM classification)

---

## Requirements

* Python 3.9+ (tested on 3.12)  
* pandas  
* numpy  
* scipy  
* beautifulsoup4  
* selenium  
* requests (for APIs)  
* groq (for LLM API)  
* matplotlib/seaborn (for plots)

---

## License
MIT
