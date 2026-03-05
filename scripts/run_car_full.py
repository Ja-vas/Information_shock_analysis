# filepath: [run_car_full.py](http://_vscodecontentref_/6)
import argparse
import sys
from pathlib import Path
from datetime import time

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis_car import (
    load_intraday_file,          # unchanged helper from before
    build_daily_lookup,
    compute_car_for_event        # returns decimal return, not pct
)

# ------------------------------------------------------------------
# constants / tuning
# ------------------------------------------------------------------
PRICE_FLOOR_DEFAULT = 1.00      # minimum entry‑price to keep an event
EXTREME_ABS_PCT   = 1000.0      # log anything larger than ±1000%
SAVE_EVERY        = 1000

WINDOWS   = {
    "That_day": 0, "plus_1d": 1, "plus_5d": 5,
    "plus_10d": 10, "plus_22d": 22, "plus_60d": 60,
}
ENTRY_IDX = {"plus_1": 1, "plus_5": 6, "plus_30": 31}


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def compute_return_pct(entry_price: float,
                       exit_price: float,
                       direction: str) -> float:
    """
    Long/up: (exit-entry)/entry ; short/down: opposite.
    Returns percentage (not decimal).
    """
    if entry_price <= 0 or exit_price <= 0:
        return np.nan
    ret = (exit_price - entry_price) / entry_price
    if str(direction).lower() in ("down", "short"):
        ret = -ret
    return ret * 100.0


def extract_day_frame(idf: pd.DataFrame, event_date: pd.Timestamp) -> pd.DataFrame:
    """Return only the rows on the gap date (09:30+)."""
    if "Datetime" in idf.columns:
        idf["Datetime"] = pd.to_datetime(idf["Datetime"])
        df = idf[idf["Datetime"].dt.date == event_date]
        df = df[df["Datetime"].dt.time >= time(9, 30)]
        return df.sort_values("Datetime")
    elif isinstance(idf.index, pd.DatetimeIndex):
        df = idf[idf.index.date == event_date]
        return df.between_time("09:30", "16:00").sort_index()
    else:
        return pd.DataFrame()


# ------------------------------------------------------------------
# main routine
# ------------------------------------------------------------------
def run(side: str,
        out_dir: Path | None,
        price_floor: float = PRICE_FLOOR_DEFAULT):
    MAIN_DF = ROOT / "main_dataframe"
    DATA_1MIN = ROOT / "data" / "1min_data"
    OUT_DIR = Path(out_dir) if out_dir else MAIN_DF / "car_results"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gaps_path = MAIN_DF / "SIGNIFICANT_GAPS_final.csv"
    daily_path = MAIN_DF / "daily_ohlc_capped.csv"
    if not gaps_path.exists():
        raise SystemExit(f"Missing {gaps_path}")
    if not daily_path.exists():
        raise SystemExit(f"Missing {daily_path}")

    gaps = pd.read_csv(gaps_path, parse_dates=["Date"])
    if side != "both":
        if side == "up":
            mask = gaps.get("Is_Gap_Up",
                            gaps.get("Type", "").str.lower() == "up") == True
        else:
            mask = gaps.get("Is_Gap_Down",
                            gaps.get("Type", "").str.lower() == "down") == True
        gaps = gaps[mask].copy()

    gaps["Side"] = gaps.get("Type", "all")
    if "Is_Gap_Up" in gaps.columns:
        gaps.loc[gaps["Is_Gap_Up"] == True, "Side"] = "up"
    if "Is_Gap_Down" in gaps.columns:
        gaps.loc[gaps["Is_Gap_Down"] == True, "Side"] = "down"

    print(f"Processing {len(gaps):,} gap events (side={side})")

    daily = pd.read_csv(daily_path, parse_dates=["Date"])
    if "symbol" not in daily.columns and "Ticker" in daily.columns:
        daily = daily.rename(columns={"Ticker": "symbol"})
    daily_lookup, daily_grouped = build_daily_lookup(daily)

    # build quick index of 1‑min files
    intraday_index: dict[str, list[Path]] = {}
    for p in DATA_1MIN.rglob("*"):
        if not p.is_file():
            continue
        stem = p.stem
        for part in stem.split("_"):
            token = "".join([c for c in part if c.isalpha()])
            if token.isalpha() and 1 <= len(token) <= 6:
                intraday_index.setdefault(token.upper(), []).append(p)
                break
    print(f"Intraday index built: {len(intraday_index)} symbols")

    output_file = OUT_DIR / "CAR_raw_results.csv"
    processed_set = set()

    if output_file.exists():
        print(f"\n[RESUME] Found existing results file: {output_file.name}")
        try:
            # 1. Read existing file (bypassing completely broken lines)
            df_existing = pd.read_csv(output_file, low_memory=False, on_bad_lines='skip')

            if not df_existing.empty and 'Symbol' in df_existing.columns and 'Date' in df_existing.columns:
                # 2. Drop any rows that have missing data due to a mid-save interruption
                df_existing = df_existing.dropna(subset=['Symbol', 'Date', 'Horizon'])

                # 3. OVERWRITE the file with the clean data to fix the half-written line at the bottom
                df_existing.to_csv(output_file, index=False)

                # 4. Memorize what we have already done so we can skip it
                df_existing['Date'] = pd.to_datetime(df_existing['Date']).dt.date
                processed_set = set(zip(df_existing['Symbol'].astype(str).str.upper(), df_existing['Date']))

                print(f"[RESUME] Cleaned corrupted lines. Skipping {len(processed_set)} already-processed events...\n")
            else:
                print("[WARNING] Existing file is empty or missing columns. Deleting and starting fresh...")
                output_file.unlink() # Physically delete the bad file!

        except Exception as e:
            print(f"[WARNING] Fatal corruption in backup file ({e}). Deleting and starting fresh...")
            output_file.unlink() # Physically delete the bad file!

    results: list[dict] = []
    audit: list[dict] = []
    count = 0

    for _, row in gaps.iterrows():
        count += 1
        sym = str(row.get("Ticker", row.get("symbol", ""))).upper()
        event_date = pd.to_datetime(row["Date"]).date()
        if (sym, event_date) in processed_set:
            if count % SAVE_EVERY == 0:
                print(f" ... skipped {count}/{len(gaps)}")
            continue

        files = intraday_index.get(sym, [])
        if not files:
            files = list(DATA_1MIN.rglob(f"*{sym}*"))
        if not files:
            continue
        intraday_path = files[0]

        idf = load_intraday_file(intraday_path)
        if idf is None or idf.empty:
            continue
        day_df = extract_day_frame(idf, event_date)
        if day_df.empty or len(day_df) <= max(ENTRY_IDX.values()):
            continue

        for horizon_label, entry_idx in ENTRY_IDX.items():
            if len(day_df) <= entry_idx:
                continue
            entry_price = day_df.iloc[entry_idx]["Open"]
            if pd.isna(entry_price) or entry_price < price_floor:
                continue    # penny‑stock or missing price

            # compute CAR; helper returns *decimal* return
            car_dict = compute_car_for_event(
                sym,
                pd.to_datetime(event_date),
                entry_price,
                WINDOWS,
                daily_lookup,
                daily_grouped,
            )
            # convert once to percent
            car_dict = {k: (np.nan if pd.isna(v) else v * 100.0)
                        for k, v in car_dict.items()}

            if str(row.get("Side")).lower() == "down":
                car_dict = {k: (-v if not pd.isna(v) else v)
                            for k, v in car_dict.items()}

            out = row.to_dict()
            out.update({
                "Horizon": horizon_label,
                "Symbol": sym,
                "Entry_Price": entry_price,
                "Direction": row.get("Side", "all"),
            })
            for w_name, car_val in car_dict.items():
                out[f"CAR_{w_name}"] = car_val
                if pd.notna(car_val) and abs(car_val) > EXTREME_ABS_PCT:
                    audit.append({
                        "Ticker": sym,
                        "Date": event_date,
                        "Horizon": horizon_label,
                        "Window": w_name,
                        "Entry_Price": entry_price,
                        "CAR_pct": car_val,
                    })

            results.append(out)

        if count % SAVE_EVERY == 0:
            print(f" ... processed {count}/{len(gaps)} events; {len(results)} valid rows")
            if results:
                pd.DataFrame(results).to_csv(
                    output_file,
                    mode="a",
                    header=not output_file.exists(),
                    index=False,
                )
                results = []

    if results:
        pd.DataFrame(results).to_csv(
            output_file,
            mode="a",
            header=not output_file.exists(),
            index=False,
        )

    if audit:
        pd.DataFrame(audit).to_csv(OUT_DIR / "CAR_extreme_audit.csv", index=False)

    print(f"\nRun complete. {len(results):,} rows written to {output_file}")
    if audit:
        print(f"Audit file with {len(audit):,} extreme returns at {OUT_DIR/'CAR_extreme_audit.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["up", "down", "both"], default="both")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--price-floor", type=float,
                        default=PRICE_FLOOR_DEFAULT,
                        help="discard events with entry price below this value")
    args = parser.parse_args()
    run(args.side, args.out_dir, price_floor=args.price_floor)