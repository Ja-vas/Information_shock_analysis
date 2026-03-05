"""Categorize financial news using Groq LLM API.

This module uses Groq's Llama model to classify news articles into
predefined categories. It batch-processes news data and exports
categorized results to CSV.

Data source: FNSPID Financial News Dataset
https://github.com/Zdong104/FNSPID_Financial_News_Dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import time
import logging

try:
    from groq import Groq
except ImportError:
    raise ImportError(
        "groq package required. Install with: pip install groq"
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# News category hierarchy (priority order)
CATEGORY_HIERARCHY = [
    "Guidance / Outlook",
    "M&A / Deal",
    "Product News",
    "Regulatory Approval",
    "Analyst / Rating",
    "Financing / Capital",
    "Management / Legal",
    "Dividends / Buybacks"
]

CATEGORIZATION_PROMPT = """You are a financial news analyst. Classify the following news headline and body into ONE category from this list:

1. Guidance / Outlook - Forward guidance, outlook changes, future expectations
2. M&A / Deal - Mergers, acquisitions, partnerships, joint ventures
3. Product News - New products, launches, product updates
4. Regulatory Approval - FDA approval, regulatory action, licensing
5. Analyst / Rating - Analyst upgrades/downgrades, price targets
6. Financing / Capital - Fundraising, debt, equity offerings, dividends, buybacks
7. Management / Legal - Executive changes, lawsuits, legal issues
8. Dividends / Buybacks - Special dividends, share buyback programs

NEWS HEADLINE: {headline}

NEWS BODY: {body}

INSTRUCTIONS:
- Return ONLY the category number (1-8)
- If multiple categories apply, choose the PRIMARY one
- If no clear match, return the closest category
- Do NOT include explanation, just the number

RESPONSE:"""


class GroqNewsCategorizer:
    """Categorize financial news using Groq LLM."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-70b-versatile"):
        """Initialize Groq client.
        
        Args:
            api_key: Groq API key (from environment if not provided)
            model: Model to use (default: llama-3.1-70b-versatile)
        """
        self.client = Groq(api_key=api_key) if api_key else Groq()
        self.model = model
        self.category_map = {str(i): cat for i, cat in enumerate(CATEGORY_HIERARCHY, 1)}
        self.api_calls = 0
        self.tokens_used = 0
    
    def categorize_single(self, headline: str, body: str = "") -> Dict[str, str]:
        """Categorize a single news item.
        
        Args:
            headline: News headline
            body: News body (optional)
        
        Returns:
            Dict with 'category', 'category_num', and 'raw_response'
        """
        prompt = CATEGORIZATION_PROMPT.format(
            headline=headline,
            body=body[:500] if body else ""  # Limit body to 500 chars
        )
        
        try:
            message = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
                max_tokens=10
            )
            
            self.api_calls += 1
            response_text = message.content[0].text.strip()
            
            # Extract category number
            cat_num = response_text.split()[0] if response_text else "9"
            category = self.category_map.get(cat_num, "Unknown")
            
            return {
                "category": category,
                "category_num": cat_num,
                "raw_response": response_text
            }
        
        except Exception as e:
            logger.error(f"Error categorizing news: {e}")
            return {
                "category": "Error",
                "category_num": "0",
                "raw_response": str(e)
            }
    
    def categorize_batch(self, df: pd.DataFrame, 
                        headline_col: str = "headline",
                        body_col: str = "body",
                        batch_size: int = 10,
                        delay_seconds: float = 1.0) -> pd.DataFrame:
        """Categorize batch of news items.
        
        Args:
            df: DataFrame with news articles
            headline_col: Column name for headlines
            body_col: Column name for body text
            batch_size: Number of items per batch
            delay_seconds: Delay between API calls (rate limiting)
        
        Returns:
            DataFrame with added categorization columns
        """
        df = df.copy()
        categories = []
        category_nums = []
        
        total = len(df)
        logger.info(f"Categorizing {total} news items...")
        
        for idx, row in df.iterrows():
            headline = row.get(headline_col, "")
            body = row.get(body_col, "")
            
            result = self.categorize_single(headline, body)
            categories.append(result["category"])
            category_nums.append(result["category_num"])
            
            # Log progress
            if (idx + 1) % batch_size == 0:
                logger.info(f"Processed {idx + 1}/{total} articles")
            
            # Rate limiting
            time.sleep(delay_seconds)
        
        df["news_category"] = categories
        df["news_category_num"] = category_nums
        
        logger.info(f"Completed {self.api_calls} API calls")
        return df
    
    def categorize_file(self, input_path: Path, output_path: Path,
                       headline_col: str = "headline",
                       body_col: str = "body") -> None:
        """Categorize news from CSV file and save results.
        
        Args:
            input_path: Path to input CSV
            output_path: Path to output CSV
            headline_col: Column name for headlines
            body_col: Column name for body text
        """
        logger.info(f"Loading news from: {input_path}")
        df = pd.read_csv(input_path, low_memory=False)
        
        logger.info(f"Loaded {len(df)} articles")
        
        # Categorize
        df_categorized = self.categorize_batch(df, headline_col, body_col)
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_categorized.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved categorized news to: {output_path}")


def main_categorize_news(input_file: Path, output_file: Path) -> None:
    """Main function to categorize news from FNSPID dataset.
    
    Args:
        input_file: Path to raw news CSV from FNSPID
        output_file: Path to save categorized results
    """
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    categorizer = GroqNewsCategorizer()
    categorizer.categorize_file(input_file, output_file)
    
    logger.info(f"News categorization complete!")
    logger.info(f"Output saved to: {output_file}")


if __name__ == "__main__":
    # Example usage
    from config import MAIN_DIR
    
    input_csv = MAIN_DIR / "raw_news.csv"  # Replace with actual FNSPID data path
    output_csv = MAIN_DIR / "news_classified.csv"
    
    if input_csv.exists():
        main_categorize_news(input_csv, output_csv)
    else:
        logger.info("To use this script:")
        logger.info("1. Download FNSPID Financial News Dataset")
        logger.info("2. Place raw news CSV in main_dataframe/ folder")
        logger.info("3. Update input_csv path above")
        logger.info("4. Set GROQ_API_KEY environment variable")
        logger.info("5. Run this script")
