# Information shock analysis               WORK-IN-PROGRESS


This project is a quantitative study of stock prices reaction to sudden information shocks. It specifically looks at **"Significant Gaps"** (price jumps >6% with higher than average pre-market volume) and tests whether the market underreacts to the news behind them.

Building on my previous work with **Post-Earnings Announcement Drift (PEAD)**, this version adds a LLM classifier to determine the "why" behind a price jump, allowing us to distinguish between 3 main categories: earnings and revenue surprises, informative firm-specific news, and "no-news" price shocks.




### What’s in this repo?

**1. Data pipeline & cleaning**
* Scripts to process **16M+ rows** of OHLCV data and news text for ~10,000 US stocks.
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
* **Volume & gap profile:**  Comparing how initial gap size and volume intensity correlate with subsequent drift or reversion.

---

### 📁 Repository Structure

* `data/` – Sample datasets and CSV exports (raw high-frequency data excluded).
* `scripts/` – Core Python logic for cleaning, LLM API calls and CAR calculation.
* `notebooks/` – Research workflow, summary statistics and figures.
* `results/` – Exported charts and tables.


---


### Requirements
*Python 3.9+ (tested on 3.12)  
*pandas  
*numpy  
*scipy  
*beautifulsoup4  
*selenium  
*requests (for APIs)  
*groq (for LLM API)  
*matplotlib/seaborn (for plots)

