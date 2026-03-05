import pandas as pd
import numpy as np
from pathlib import Path
from config import MAIN_DIR
from scripts.analysis_car import apply_ttest

def load_and_winsorize_car_data():
    print("1. Loading raw CARs and Earnings...")
    car_df = pd.read_csv(MAIN_DIR / "car_results" / "CAR_raw_results.csv", low_memory=False)
    car_df['Date'] = pd.to_datetime(car_df['Date']).dt.date

    # Winsorization
    print(" -> Winsorizing CAR data to neutralize extreme split-glitches...")
    car_cols = ['CAR_That_day', 'CAR_plus_1d', 'CAR_plus_5d', 'CAR_plus_10d', 'CAR_plus_22d', 'CAR_plus_60d']

    for col in car_cols:
        if col in car_df.columns:
            lower_bound = car_df[col].quantile(0.005)
            upper_bound = car_df[col].quantile(0.995)
            car_df[col] = car_df[col].clip(lower=lower_bound, upper=upper_bound)

    return car_df

def merge_car_and_earnings(car_df):
    earn_df = pd.read_csv(MAIN_DIR / "gap_earnings_joined.csv", low_memory=False)
    earn_df['Date'] = pd.to_datetime(earn_df['Date']).dt.date

    if 'Ticker' not in car_df.columns and 'Symbol' in car_df.columns:
        car_df['Ticker'] = car_df['Symbol']

    print("2. Merging datasets on Ticker and Date...")
    merged_df = pd.merge(car_df, earn_df, on=['Ticker', 'Date'], how='inner', suffixes=('_car', '_earn'))
    print(f" -> Merged successfully. Total overlapping events: {len(merged_df)}")

    return merged_df

def process_quartiles(merged_df):
    METRICS = ['EPS_Surprise_win', 'Revenue_Surprise_win', 'SUE_EPS', 'SUR_Rev']
    WINDOWS = ['That_day', 'plus_1d', 'plus_5d', 'plus_10d', 'plus_22d', 'plus_60d']

    dir_col = 'Direction_car' if 'Direction_car' in merged_df.columns else 'Direction'
    directions = [d for d in merged_df[dir_col].dropna().unique() if d in ['up', 'down']]

    for direction in directions:
        df_dir = merged_df[merged_df[dir_col] == direction].copy()
        if df_dir.empty: continue

        print(f"\n3. Processing Quartiles for GAP {direction.upper()}...")
        q_records = []

        for horizon in df_dir['Horizon'].unique():
            df_hz = df_dir[df_dir['Horizon'] == horizon].copy()

            for metric in METRICS:
                if metric not in df_hz.columns:
                    continue

                valid_df = df_hz.dropna(subset=[metric]).copy()
                if len(valid_df) < 10:
                    continue

                try:
                    valid_df['Quartile'] = pd.qcut(valid_df[metric], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
                except Exception as e:
                    continue

                grouped = valid_df.groupby('Quartile', observed=True)

                for q, group in grouped:
                    row_data = {
                        'Horizon': horizon, 
                        'Direction': direction, 
                        'Metric': metric, 
                        'Quartile': q, 
                        'N_gaps': len(group)
                    }

                    for w in WINDOWS:
                        car_col = f"CAR_{w}" 
                        if car_col not in group.columns: continue

                        vals = group[car_col].dropna().values
                        if len(vals) < 2: 
                            row_data[f"{w}_mean"] = ""
                            continue

                        mean_val = np.mean(vals)
                        ttest_res = apply_ttest(vals / 100.0) 

                        sig = ttest_res['sig_level'] if ttest_res['sig_level'] else ""
                        row_data[f"{w}_mean"] = f"{mean_val:.4f}{sig}"
                        row_data[f"{w}_pvalue"] = ttest_res['p_value']

                    q_records.append(row_data)

        # Save the final safely-capped quartile data
        q_path = MAIN_DIR / f'CAR_quartiles_results_{direction}.csv'
        pd.DataFrame(q_records).to_csv(q_path, index=False)
        print(f" -> Saved: {q_path.name}")

    print("\nDONE! Your Quartiles are now clean, capped, and ready to chart.")