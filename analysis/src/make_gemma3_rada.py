# Usage:
#   python make_gemma3_radar.py --csv path/to/your.csv --out gemma3_radar.png
#
# Optional fixed rings instead of autoscale:
#   python make_gemma3_radar.py --csv your.csv --out out.png --rings 30 55 5
#
# Requires: pandas, numpy, matplotlib

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from pathlib import Path
import sys

METRICS = [
    "Power_Distance",
    "Individualism",
    "Masculinity",
    "Uncertainty_Avoidance",
    "Long_Term_Orientation",
    "Indulgence",
]
LABELS_EN = [
    "Power\nDistance",
    "Individualism",
    "Masculinity",
    "Uncertainty\nAvoidance",
    "Long-Term\nOrientation",
    "Indulgence",
]

def load_gemma(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"model_id", *METRICS}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"CSV is missing required columns: {sorted(missing)}")

    gemma = df[df["model_id"].str.startswith("google_gemma-3-", na=False)].copy()
    if gemma.empty:
        sys.exit("No Gemma 3 rows found (model_id should start with 'google_gemma-3-').")

    # Short model label: 1B, 4B, 12B, 27B
    gemma["short_name"] = (
        gemma["model_id"].str.extract(r"gemma-3-(.*?)-it")[0].str.replace("b", "B", regex=False)
    )
    return gemma

def compute_angles(n: int):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close loop
    return angles

def auto_ticks(values: np.ndarray):
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    lower = 5 * np.floor((vmin - 2) / 5.0)
    upper = 5 * np.ceil((vmax + 2) / 5.0)
    ticks = np.arange(lower + 5, upper, 5)
    return ticks, lower, upper

def fixed_ticks(start: float, stop: float, step: float):
    ticks = np.arange(start, stop, step)
    return ticks, start, stop - 0.0001  # small epsilon so top ring renders nicely

def plot_radar(gemma: pd.DataFrame, out_path: str, rings=None):
    angles = compute_angles(len(METRICS))

    # Prepare ticks
    all_vals = gemma[METRICS].to_numpy().ravel()
    if rings is None:
        ticks, ymin, ymax = auto_ticks(all_vals)
    else:
        start, stop, step = rings
        ticks, ymin, ymax = fixed_ticks(start, stop, step)

    # Colors + the SAME styles/markers as grayscale version
    colors     = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # blue/orange/green/red
    linestyles = ["solid", "dashed", "dashdot", (0, (1, 1))]   # dotted as tuple
    markers    = ["o", "s", "^", "x"]

    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)

    for idx, (_, row) in enumerate(gemma.iterrows()):
        vals = row[METRICS].astype(float).tolist()
        vals += vals[:1]
        ax.plot(
            angles, vals, linewidth=2.2,
            label=row["short_name"],
            linestyle=linestyles[idx % len(linestyles)],
            marker=markers[idx % len(markers)],
            markersize=5,
            color=colors[idx % len(colors)],
        )

    # Axis labels: bigger, bold, white halo, nudged outward
    ax.set_xticks(angles[:-1])
    lbls = ax.set_xticklabels(LABELS_EN, fontsize=14, fontweight="bold")
    for lbl in lbls:
        lbl.set_y(lbl.get_position()[1] - 0.05)
        lbl.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground="white"),
            path_effects.Normal()
        ])

    # Ring labels only (no per-point numbers)
    ax.set_rgrids(ticks, angle=90, fontsize=10)
    ax.set_ylim(ymin, ymax)
    ax.set_yticklabels([f"{int(t)}" for t in ticks])

    ax.grid(True, linestyle=":", linewidth=0.8, color="lightgray")
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.1), title="Model", frameon=False)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Gemma 3 radar chart from CSV.")
    parser.add_argument("--csv", required=True, help="Path to CSV file.")
    parser.add_argument("--out", required=True, help="Output image path (e.g., radar.png).")
    parser.add_argument(
        "--rings", nargs=3, type=float, metavar=("START", "STOP", "STEP"),
        help="Optional fixed rings (e.g., --rings 30 60 5)."
    )
    args = parser.parse_args()

    gemma = load_gemma(args.csv)
    rings = tuple(args.rings) if args.rings else None
    plot_radar(gemma, args.out, rings=rings)

if __name__ == "__main__":
    main()
