"""Utility for splitting the final significance file into per-strategy CSVs."""
from pathlib import Path
import pandas as pd

from config import SIGNIFICANT_GAPS_FILE


def split_file(input_file: Path | str = None, output_dir: Path | str = None) -> None:
    if input_file is None:
        input_file = SIGNIFICANT_GAPS_FILE
    if output_dir is None:
        from config import MAIN_DIR
        output_dir = Path(MAIN_DIR) / "SIG_SPLITS"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(input_file).exists():
        print(f"❌ Error: {input_file} not found.")
        return

    df = pd.read_csv(input_file, parse_dates=["Date"])
    sig_cols = [c for c in df.columns if c.startswith("Sig")]
    print(f"Loaded {len(df)} rows. Splitting into {len(sig_cols)} groups")
    for col in sig_cols:
        sub = df[df[col] == True].copy()
        out_path = output_dir / f"{col}.csv"
        sub.to_csv(out_path, index=False)
        print(f" saved {len(sub)} rows to {out_path}")
