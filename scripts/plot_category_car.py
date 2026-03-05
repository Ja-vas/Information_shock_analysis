import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from matplotlib.patches import Patch

def plot_news_categories(main_dir=None, out_subdir='charts_news_categories', verbose=True, save_files=True, show=False):
    if main_dir is None:
        try:
            from config import MAIN_DIR
            main_dir = Path(MAIN_DIR)
        except Exception:
            raise ValueError('main_dir not provided and config.MAIN_DIR not importable')
            
    main_dir = Path(main_dir)
    results_dir = main_dir / "car_results"
    out_dir = main_dir / out_subdir
    
    if save_files:
        out_dir.mkdir(parents=True, exist_ok=True)

    CAR_COLS = ["CAR_That_day", "CAR_plus_1d", "CAR_plus_5d", "CAR_plus_10d", "CAR_plus_22d", "CAR_plus_60d"]
    
    # Beautiful shade gradient from light red (That day) to deep dark red (60d)
    RED_SHADES = {
        "CAR_That_day": "#FF9999", 
        "CAR_plus_1d": "#FF4D4D", 
        "CAR_plus_5d": "#E60000",
        "CAR_plus_10d": "#B30000", 
        "CAR_plus_22d": "#800000", 
        "CAR_plus_60d": "#4D0000"
    }

    # Matches the exact strings produced by process_car_news.py
    RAW_CATEGORIES = [
        "No_news", "Guidance", "M&A", "Product", "Regulatory", 
        "Analyst", "Financing", "Management", "Dividend"
    ]

    CATEGORY_LABELS = {
        "No_news": "No news", 
        "Guidance": "Guidance/Outlook", 
        "M&A": "M&A/Deal",
        "Product": "Product News", 
        "Regulatory": "Regulatory Approval",
        "Analyst": "Analyst Rating", 
        "Financing": "Financing/Capital",
        "Management": "Management/Legal", 
        "Dividend": "Dividends/Buybacks"
    }

    def stars_from_p(p):
        if pd.isna(p): return ""
        if p < 0.01: return "***"
        elif p < 0.05: return "**"
        elif p < 0.10: return "*"
        return ""

    for direction in ['up', 'down']:
        file_path = results_dir / f"CAR_news_results_{direction}.csv"
        
        if not file_path.exists():
            print(f"Skipping GAP {direction.upper()} - File not found: {file_path.name}")
            continue
            
        print(f"\nLoading {file_path.name}...")
        df_full = pd.read_csv(file_path, low_memory=False)
        
        # =========================================================
        # ON-THE-FLY WINSORIZATION
        # Neutralize the 7000% split-glitches BEFORE calculating averages!
        # =========================================================
        for col in CAR_COLS:
            if col in df_full.columns:
                lower_bound = df_full[col].quantile(0.01)
                upper_bound = df_full[col].quantile(0.99)
                df_full[col] = df_full[col].clip(lower=lower_bound, upper=upper_bound)
        # =========================================================

        horizons = df_full['Horizon'].unique() if 'Horizon' in df_full.columns else ['All']
        
        for horizon in horizons:
            df = df_full[df_full['Horizon'] == horizon].copy() if horizon != 'All' else df_full.copy()
            df = df[df["main_category"].isin(RAW_CATEGORIES)]
            if df.empty: continue

            # 1. Calculate Mean CARs per category
            mean_car = df.groupby("main_category")[CAR_COLS].mean().reindex(RAW_CATEGORIES)
            
            # 2. Calculate p-values via T-Test
            p_values = pd.DataFrame(index=RAW_CATEGORIES, columns=CAR_COLS, dtype=float)

            for cat in RAW_CATEGORIES:
                sub = df[df["main_category"] == cat]
                for col in CAR_COLS:
                    if col not in sub.columns: continue
                    vals = sub[col].dropna()
                    # Apply T-test only if we have enough samples
                    if len(vals) >= 2:
                        _, p_val = stats.ttest_1samp(vals, 0)
                        p_values.loc[cat, col] = p_val

            # Apply pretty labels for the chart
            mean_car.index = [CATEGORY_LABELS.get(x, x) for x in mean_car.index]
            p_values.index = [CATEGORY_LABELS.get(x, x) for x in p_values.index]

            # ---------------- PLOT ----------------
            plt.figure(figsize=(15, 7.5))
            bar_width = 0.12
            x = np.arange(len(mean_car))

            # Center the bars over the tick mark
            offset_start = -((len(CAR_COLS) * bar_width) / 2) + (bar_width / 2)

            for i, car_col in enumerate(CAR_COLS):
                for j, cat in enumerate(mean_car.index):
                    value = mean_car.loc[cat, car_col] # NO MULTIPLYING BY 100! 
                    p_val = p_values.loc[cat, car_col]
                    stars = stars_from_p(p_val)

                    significant = stars != ""
                    color = RED_SHADES[car_col] if significant else "#DCEEFF" # Light blue for insignificant
                    alpha = 0.9 if significant else 0.6

                    xpos = x[j] + offset_start + (i * bar_width)

                    plt.bar(
                        xpos, value, width=bar_width, color=color, alpha=alpha,
                        edgecolor="black", linewidth=0.6
                    )

                    # Add stars directly above/below the bars
                    if stars:
                        offset = 0.05 * abs(value) if abs(value) > 1 else 0.2
                        y = value + offset if value >= 0 else value - offset
                        plt.text(
                            xpos, y, stars, ha="center", va="bottom" if value >= 0 else "top",
                            fontsize=11, fontweight="bold", color="black"
                        )

            title_hz = str(horizon).replace('plus_', '+') + " mins" if horizon != 'All' else 'All'
            plt.title(f"Cumulative Abnormal Returns by News Category (GAP {direction.upper()}) — Entry: {title_hz}", fontsize=16, fontweight="bold", pad=20)
            plt.ylabel("CAR (%)", fontsize=14, fontweight="bold")
            plt.axhline(0, color="black", linewidth=1.2)
            
            plt.xticks(x, mean_car.index, rotation=35, ha="right", fontsize=12)
            plt.grid(True, axis="y", alpha=0.3, linestyle="--")

            # Custom Legend
            legend_elements = [Patch(facecolor="#DCEEFF", edgecolor="black", alpha=0.6, label="Not significant")] + \
                              [Patch(facecolor=RED_SHADES[c], edgecolor="black", label=c.replace("CAR_", "").replace("_", " ")) for c in CAR_COLS]

            plt.legend(handles=legend_elements, title="CAR Window / Significance", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=11)
            plt.tight_layout()

            # Save Output
            fname = f"CAR_Category_{direction.upper()}_{str(horizon).replace(' ', '').replace('+', '')}"
            
            if save_files:
                plt.savefig(out_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
                plt.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
                plt.close()
                if verbose: print(f" -> Saved Chart: {fname}.png")
            else:
                if show: plt.show()
                else: plt.close()

if __name__ == "__main__":
    plot_news_categories()