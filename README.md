# Information shock analysis


This project is a quantitative study of stock prices reaction to sudden information shocks. It specifically looks at **"Significant Gaps"** (price jumps >6% with higher than average pre-market volume) and tests whether the market underreacts to the news behind them.

Building on my previous work with **Post-Earnings Announcement Drift (PEAD)**, this version adds a LLM classifier to determine the "why" behind a price jump, thus allowing us to distinguish between 3 main categories: earnings and revenue surprises, informative news, and "no-news" noise.




### What’s in this repo?

**1. Data pipeline & cleaning**
* Scripts to process **16M+ rows** of OHLCV data and news text for ~10,000 US stocks.
* Automated cleaning: handling stock splits, winsorizing outliers (0.01/99.99 percentiles) and enforcing dollar volume liquidity thresholds.
* Standardized Surprises: Calculation and winsorizing of SUE (Earnings) and SUR (Revenue) by scaling surprises against an 8-quarter rolling standard deviation.

**2. Identification of "Significant gaps"**
To isolate moves with real price shock, gaps are filtered by pre-market and early-session volume. A gap is "Significant" if:
* **The price jump is ≥ 6%**
* **Relative Volume:** The dollar volume in the first 1 or 5 minutes exceeds the 30-day average daily volume (1x multiplier), or exceeds 2x the average daily volume within the first 30 minutes.

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

* `data/` – Sample datasets and CSV exports (raw 16M row data excluded).
* `scripts/` – Core Python logic for cleaning, LLM API calls and CAR calculation.
* `notebooks/` – Step-by-step research flow, t-test results and visualizations.
* `results/` – Exported charts and tables showing drift stats and volume profiles.

---

### How to use this project

1. **Find Significant gaps:** Run the filtering scripts to identify stocks meeting the 6% price and 1x/2x volume thresholds.
2. **Classify the news:** Run the LLM script to assign a catalyst based on the priority hierarchy.
3. **Analyze the surprise:** Use the SUE/SUR scripts to calculate fundamental surprises and bucket the "Earnings gaps" into quartiles.
4. **Run the svent study:** Use the notebooks to calculate CAR and run t-tests for the different news categories (Earnings vs. News vs. No-News).
5. **Compare entry windows:** Analyze how results change depending on the entry time (+1m, +5m, or +30m after open).

---

### Requirements
* Python 3.9+
* pandas / numpy / scipy
* groq (for LLM classification)
* matplotlib / seaborn

---

### Requirements
* Python 3.9+
* pandas / numpy / scipy
* groq (for LLM classification)
* matplotlib / seaborn
