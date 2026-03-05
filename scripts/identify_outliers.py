import pandas as pd
import numpy as np
from pathlib import Path
from config import MAIN_DIR

def identify_outliers():
    # Define the exact path to your raw backup data
    file_path = Path(MAIN_DIR) / "car_results" / "CAR_raw_results.csv"

    # Load the data
    print(f"Loading data from {file_path.name}...")
    df = pd.read_csv(file_path, low_memory=False)

    # Define the exact CAR columns to check for outliers
    car_cols = [
        'CAR_That_day', 'CAR_plus_1d', 'CAR_plus_5d', 
        'CAR_plus_10d', 'CAR_plus_22d', 'CAR_plus_60d'
    ]

    print("Calculating Z-scores to find outliers...")

    # Create an empty mask of "False" for every row
    outlier_mask = pd.Series(False, index=df.index)

    # Check each column for values that are more than 3 Standard Deviations from the mean
    for col in car_cols:
        if col in df.columns:
            # Get mean and standard deviation (ignoring NaNs)
            col_mean = df[col].mean()
            col_std = df[col].std()
            
            # Calculate the absolute Z-score for every row in this column
            z_scores = np.abs((df[col] - col_mean) / col_std)
            
            # If the Z-score is > 3, flag it as True (it's an outlier!)
            # The '|' means OR. If a stock is an outlier in ANY time window, we catch it.
            outlier_mask = outlier_mask | (z_scores > 3)

    # Filter the main dataframe to ONLY keep the outliers
    outliers_df = df[outlier_mask].copy()

    print("-" * 40)
    print(f"Total events analyzed: {len(df):,}")
    print(f"Extreme Outliers found: {len(outliers_df):,}")
    print("-" * 40)

    # Let's look at the craziest ones! 
    # Sorting by absolute value of 60-day CAR to see the biggest boom/busts
    if 'CAR_plus_60d' in outliers_df.columns:
        outliers_df['Abs_CAR_60d'] = outliers_df['CAR_plus_60d'].abs()
        top_outliers = outliers_df.sort_values(by='Abs_CAR_60d', ascending=False).drop(columns=['Abs_CAR_60d'])
    else:
        top_outliers = outliers_df

    # Display the top 15 craziest movers
    print(top_outliers[['Ticker', 'Date', 'Direction', 'Horizon', 'Entry_Price', 'CAR_That_day', 'CAR_plus_5d', 'CAR_plus_60d']].head(15))

    # Optional: Save these to a CSV to inspect in Excel later
    # outliers_df.to_csv(file_path.parent / "CAR_EXTREME_OUTLIERS.csv", index=False)
    # print("Saved extreme outliers to CAR_EXTREME_OUTLIERS.csv")