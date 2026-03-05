import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

def make_quartile_charts(direction='up', main_dir=None, out_subdir='charts_quartiles_redblue', verbose=True, save_files=True, show=False):
    """
    Creates quartile CAR charts (red shades for quartiles Q1-Q4, light blue for insignificant).
    Reads directly from CAR_quartiles_results_{direction}.csv
    """
    direction = direction.lower()
    if direction not in ['up', 'down']:
        raise ValueError("Direction must be either 'up' or 'down'")

    # === PATHS ===
    if main_dir is None:
        try:
            from config import MAIN_DIR
            main_dir = Path(MAIN_DIR)
        except Exception:
            raise ValueError('main_dir not provided and config.MAIN_DIR not importable')
    
    main_dir = Path(main_dir)
    out_dir = main_dir / out_subdir

    if save_files:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Automatically looks for CAR_quartiles_results_up/down.csv
    csv_path = main_dir / f'CAR_quartiles_results_{direction}.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path.name} in {main_dir}")

    if verbose:
        print(f"Loading data from {csv_path.name}...")
        
    df = pd.read_csv(csv_path, low_memory=False)

    if 'Quartile' not in df.columns or 'Metric' not in df.columns:
        raise KeyError(f"CSV {csv_path.name} is missing 'Quartile' or 'Metric' columns!")

    # === WINDOW MAPPING ===
    windows = [
        ("That day\n(gap→close)", "That_day_mean"),
        ("+1d", "plus_1d_mean"),
        ("+5d", "plus_5d_mean"),
        ("+10d", "plus_10d_mean"),
        ("+22d", "plus_22d_mean"),
        ("+60d", "plus_60d_mean")
    ]
    
    window_labels = [w[0] for w in windows]
    window_cols = [w[1] for w in windows]

    # === COLORS ===
    QUARTILE_COLORS = ["#FFCCCC", "#FF6666", "#CC0000", "#990000"]
    LIGHT_BLUE = "#DCEEFF"

    if verbose:
        print(f"Creating charts for GAP {direction.upper()}...")

    # === MAIN LOOP ===
    horizons = df["Horizon"].unique() if "Horizon" in df.columns else ["All"]
    
    for horizon in horizons:
        data = df[df["Horizon"] == horizon].copy() if horizon != "All" else df.copy()
        if data.empty: continue

        data["Quartile"] = pd.Categorical(data["Quartile"], categories=["Q1", "Q2", "Q3", "Q4"], ordered=True)
        data = data.sort_values(["Metric", "Quartile"])
        metrics = data["Metric"].unique()

        fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 4.5 * len(metrics)), sharex=True)
        if len(metrics) == 1: axes = [axes]

        title_hz = str(horizon).replace('plus_', '+') + " mins" if horizon != "All" else "All"
        fig.suptitle(
            f"Cumulative Abnormal Returns by Quartile (GAP {direction.upper()}) — Entry: {title_hz}\n"
            "Event windows: same-day (open→close), post-event [+1d, +Xd]",
            fontsize=18, fontweight="bold", y=0.98 if len(metrics) > 1 else 1.05
        )

        x = np.arange(len(window_labels))
        width = 0.18

        # === PLOT EACH METRIC ===
        for ax_idx, metric in enumerate(metrics):
            ax = axes[ax_idx]
            sub = data[data["Metric"] == metric]

            for _, row in sub.iterrows():
                try:
                    q_idx = ["Q1", "Q2", "Q3", "Q4"].index(row["Quartile"])
                except ValueError:
                    continue 

                values, stars, colors = [], [], []

                for col in window_cols:
                    if col not in sub.columns:
                        values.append(0.0); stars.append(""); colors.append(LIGHT_BLUE); continue
                        
                    val_str = str(row[col])
                    if val_str == "nan" or val_str.strip() == "":
                        values.append(0.0); stars.append(""); colors.append(LIGHT_BLUE); continue
                    
                    # 1. Extract stars
                    if "***" in val_str: star = "***"
                    elif "**" in val_str: star = "**"
                    elif "*" in val_str: star = "*"
                    else: star = ""
                        
                    # 2. Extract true number
                    clean_num = val_str.replace("***", "").replace("**", "").replace("*", "").strip()
                    try:
                        num_val = float(clean_num)
                    except ValueError:
                        num_val = 0.0
                    
                    # BUG FIX: No more multiplying by 100! Your data is already in % form.

                    values.append(num_val)
                    stars.append(star)
                    colors.append(QUARTILE_COLORS[q_idx] if star else LIGHT_BLUE)

                bars = ax.bar(x + (q_idx - 1.5) * width, values, width, color=colors, edgecolor="black", linewidth=0.9)

                for bar, star, val in zip(bars, stars, values):
                    if star:
                        y_pos = val + (0.05 * abs(val)) if val >= 0 else val - (0.1 * abs(val))
                        ax.text(
                            bar.get_x() + bar.get_width() / 2, y_pos, star,
                            ha="center", va="bottom" if val >= 0 else "top",
                            fontsize=12, fontweight="bold", color="black"
                        )

            ax.set_title(metric, fontsize=15, fontweight="bold", pad=18)
            ax.set_ylabel("CAR (%)", fontsize=12, fontweight="bold")
            ax.axhline(0, color="black", linewidth=1.2)
            ax.grid(True, axis="y", alpha=0.3, linestyle="--")

        legend_elements = [Patch(facecolor=LIGHT_BLUE, label="Not significant (no stars)")] + \
                          [Patch(facecolor=QUARTILE_COLORS[i], label=f"Quartile Q{i+1}") for i in range(4)]

        fig.legend(handles=legend_elements, title="Quartile / Significance", bbox_to_anchor=(1.02, 0.5), loc="center")

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(window_labels, fontsize=12)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        fname = f"CAR_Quartiles_{direction.upper()}_{str(horizon).replace(' ', '').replace('+', '')}"
        
        if save_files:
            plt.savefig(out_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
            plt.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
            plt.close()
            if verbose: print(f" -> Saved: {out_dir / fname}.png")
        else:
            if show: plt.show()
            else: plt.close()

    if verbose and save_files:
        print(f"✅ Finished GAP {direction.upper()}! Charts saved to {out_dir}")
        
    return out_dir