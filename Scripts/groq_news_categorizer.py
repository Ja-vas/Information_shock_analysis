import pandas as pd
import openai
import time
import json
from tqdm.auto import tqdm
from pathlib import Path
import os

# ====================== CONFIG ==============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Set GROQ_API_KEY environment variable!")

client = openai.OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# Models in fallback order (no decommissioned ones)
MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-27b-it",
]
current_model_idx = 0

# ====================== CATEGORIES ==============================
CATEGORIES = [
    "Earnings",
    "Guidance / Outlook",
    "M&A / Deal",
    "Product News",
    "Regulatory Approval",
    "Analyst / Rating",
    "Management / Legal",
    "Macro / Market",
    "Dividends / Buybacks",
    "Financing / Capital",
    "Fluff / Other"
]

# ====================== PROMPT ==============================
SINGLE_PROMPT = """You are a senior equity analyst.

Classify the following financial news into EXACTLY ONE category from this list.
You MUST return the category EXACTLY as written below (including punctuation):

{category_list}

Return ONLY this JSON:
{{"category": "<exact category from the list above>"}}

Examples:
→ "Apple beats Q4 earnings" → {{"category": "Earnings"}}
→ "FDA approves Pfizer drug" → {{"category": "Regulatory Approval"}}


Text:
{item_text}

"""

# ====================== FILES & SETTINGS ==============================
INPUT_FILE  = "C:/Users/j-vas/1JUPYTERLAB/diplomka/stats_output/sig_gaps_with_news_matches.csv"
OUTPUT_FILE = "gaps_with_news_GROQ_classified.csv"
FAILED_FILE = "gaps_with_news_GROQ_failed.csv"

MAX_TEXT_CHARS = 15000   # ~7-8k tokens, safe
MIN_SECONDS_BETWEEN_CALLS = 2.0

# ====================== LOAD DATA ==============================
print("Loading data...")
df = pd.read_csv(INPUT_FILE, parse_dates=['Date_gap'], low_memory=False)
df['text'] = df['Article'].fillna(df.get('Article_title', '')).astype(str)
df['gap_id'] = df['symbol'] + "_" + df['Date_gap'].dt.strftime('%Y-%m-%d')
df = df.drop_duplicates(subset=['gap_id', 'text']).reset_index(drop=True)

print(f"→ {len(df):,} unique news items across {df['gap_id'].nunique():,} gaps")

# Resume logic — skip gaps that already have at least one valid classification
if Path(OUTPUT_FILE).exists():
    print("Resuming from previous output...")
    done = pd.read_csv(OUTPUT_FILE, parse_dates=['Date_gap'])
    if 'news_category_groq' not in done.columns:
        done['news_category_groq'] = ""
    done['gap_id'] = done['symbol'] + "_" + done['Date_gap'].dt.strftime('%Y-%m-%d')

    completed_gaps = set(done[done['news_category_groq'].isin(CATEGORIES)]['gap_id'])
    df = df[~df['gap_id'].isin(completed_gaps)]
    print(f"→ Skipping {len(completed_gaps)} already classified gaps")

df_to_process = df.copy()
print(f"→ {len(df_to_process):,} items to classify")

# ====================== HELPER: APPEND ROW TO CSV ==============================
def append_row_to_csv(row_df: pd.DataFrame, filepath: str):
    mode = 'a' if Path(filepath).exists() else 'w'
    header = not Path(filepath).exists()
    row_df.to_csv(filepath, mode=mode, header=header, index=False)

# ====================== CLASSIFY SINGLE ITEM ==============================
def classify_single(text: str) -> str:
    """
    Try to classify a single text. Returns:
    - valid category (one of CATEGORIES) on success
    - "FAILED" if all models either error out or return invalid categories
    """
    global current_model_idx
    safe_text = text[:MAX_TEXT_CHARS]

    prompt = SINGLE_PROMPT.format(
        category_list="\n".join(CATEGORIES),
        item_text=safe_text
    )

    start_idx = current_model_idx
    # we will try each model at most once for this item
    while current_model_idx < len(MODELS):
        model = MODELS[current_model_idx]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content.strip()

            # Defensive JSON parsing
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                print(f"{model} → invalid JSON: {content[:120]}...")
                current_model_idx += 1
                continue

            cat = data.get("category")

            if cat in CATEGORIES:
                return cat
            else:
                print(f"{model} → wrong category: '{cat}'")
                current_model_idx += 1

        except openai.RateLimitError as e:
            msg = str(e).lower()
            # HARD limit -> switch model
            if any(x in msg for x in ["daily", "quota", "tpd", "hard"]):
                print(f"{model} → HARD rate limit, switching model")
                current_model_idx += 1
            else:
                # soft limit, w8 and retry
                print(f"{model} → soft limit, sleeping {MIN_SECONDS_BETWEEN_CALLS}s")
                time.sleep(MIN_SECONDS_BETWEEN_CALLS)
        except Exception as e:
            
            print(f"Error on {model}: {e}")
            current_model_idx += 1

    # All models failed
    print("ALL MODELS FAILED for this item")
    current_model_idx = start_idx
    return "FAILED"

# ====================== MAIN LOOP ==============================
print(f"\nStarting 1-by-1 classification of {len(df_to_process)} items...")

last_call_time = 0.0

for idx in tqdm(range(len(df_to_process)), desc="Items", unit="item"):
    row = df_to_process.iloc[idx].copy()

    # rate limiting
    now = time.time()
    elapsed = now - last_call_time
    if elapsed < MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
    last_call_time = time.time()

    category = classify_single(row['text'])
    row['news_category_groq'] = category

    output_row = row.drop(labels=['text']).to_frame().T

    # Save
    if category == "FAILED":
        append_row_to_csv(output_row, FAILED_FILE)
    else:
        append_row_to_csv(output_row, OUTPUT_FILE)

print("\n" + "="*80)
print("GROQ 1-BY-1 CLASSIFICATION COMPLETE")
print("="*80)

# ====================== FINAL STATS ==============================
if Path(OUTPUT_FILE).exists():
    final = pd.read_csv(OUTPUT_FILE, parse_dates=['Date_gap'])
    final['gap_id'] = final['symbol'] + "_" + final['Date_gap'].dt.strftime('%Y-%m-%d')

    print("Category distribution:")
    print(final['news_category_groq'].value_counts(dropna=False))

    total = final['gap_id'].nunique()
    earnings = final[final['news_category_groq'] == 'Earnings']['gap_id'].nunique()
    guidance = final[final['news_category_groq'] == 'Guidance / Outlook']['gap_id'].nunique()

    print(f"\nUnique gaps with news      : {total:,}")
    print(f"└── Earnings-related      : {earnings:,} ({earnings/total:.1%})")
    print(f"└── Guidance-related      : {guidance:,} ({guidance/total:.1%})")
    print(f"└── Non-E/G news-driven   : {total - earnings - guidance:,}")
else:
    print("No successful classifications written to OUTPUT_FILE.")

if Path(FAILED_FILE).exists():
    failed_df = pd.read_csv(FAILED_FILE)
    print(f"\nFailed items → {FAILED_FILE} ({len(failed_df):,} rows)")

