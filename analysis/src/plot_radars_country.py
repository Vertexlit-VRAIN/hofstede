#!/usr/bin/env python3
# country_radars_main.py
# Generate country-specific radar charts (China, USA, Europe) using ALL models from each country.

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

# -----------------------------
# Global config / constants
# -----------------------------
DIMENSIONS = [
    "Power_Distance",
    "Individualism",
    "Masculinity",
    "Uncertainty_Avoidance",
    "Long_Term_Orientation",
    "Indulgence",
]
SIZE_ORDER = {"XS": 0, "S": 1, "M": 2, "L": 3, "XL": 4}

def use_paper_style(dpi=600):
    mpl.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def save_figure(out_path_base):
    """Save both PNG (for quick view) and PDF (vector for paper)."""
    out_path_base = Path(out_path_base)
    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path_base.with_suffix(".png").as_posix(), bbox_inches="tight")
    plt.savefig(out_path_base.with_suffix(".pdf").as_posix(), bbox_inches="tight")

# -----------------------------
# Radar helpers
# -----------------------------
def _angles(n):
    a = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    a += a[:1]
    return a

def _english_multiline_labels():
    return [
        "Power\nDistance",
        "Individualism",
        "Masculinity",
        "Uncertainty\nAvoidance",
        "Long-Term\nOrientation",
        "Indulgence",
    ]

def _auto_ring_ticks(values):
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    lower = 5 * np.floor((vmin - 2) / 5.0)
    upper = 5 * np.ceil((vmax + 2) / 5.0)
    ticks = np.arange(lower + 5, upper, 5)
    return ticks, lower, upper

def _size_sort_key(size):
    return SIZE_ORDER.get(str(size), 999)

def _normalize_country(name: str) -> str:
    """Map any 'Europe-*' to 'Europe' so it groups together."""
    if isinstance(name, str) and name.startswith("Europe"):
        return "Europe"
    return name

# -----------------------------
# Core plotting function
# -----------------------------
def plot_country_radar_all_models(
    profiles_df: pd.DataFrame,
    country_name: str,
    out_path_base: Path,
    hofstede_df: pd.DataFrame | None = None,
    show_country_ref: bool = False,
):
    """
    Create ONE radar for <country_name> using ALL models whose country_ref maps to that country.
    Optionally overlay Hofstede's country polygon if provided.
    The radial range (rings) is computed from BOTH the models' values and the country reference
    (when overlay is enabled and available).
    """
    df = profiles_df.copy()
    df["country_norm"] = df["country_ref"].map(_normalize_country)

    sub = df[df["country_norm"] == country_name].copy()
    if sub.empty:
        print(f"[plot_country_radar_all_models] No rows for {country_name}; skipped.")
        return

    # Compact legend label: short model id + size
    def _short_model(mid):
        if isinstance(mid, str) and "_" in mid:
            return mid.split("_", 1)[-1]
        return str(mid)

    sub["legend_label"] = sub.apply(
        lambda r: f"{_short_model(r['model_id'])} ({r['size']})", axis=1
    )
    sub["__rank"] = sub["size"].map(_size_sort_key)
    sub = sub.sort_values("__rank")  # size-sorted legend

    angles = _angles(len(DIMENSIONS))
    fig = plt.figure(figsize=(10, 9))
    ax = plt.subplot(111, polar=True)

    # --- Collect values for autoscaling (models + optional country ref) ---
    all_vals = sub[DIMENSIONS].to_numpy().ravel()

    ref_vals_closed = None
    if show_country_ref and hofstede_df is not None:
        hof = hofstede_df.copy()
        hof["country_norm"] = hof["country"].map(_normalize_country)
        row = hof[hof["country_norm"] == country_name]
        if not row.empty:
            ref_vals = row.iloc[0][DIMENSIONS].astype(float).to_numpy()
            # include ref values in the autoscale pool
            all_vals = np.concatenate([all_vals, ref_vals])
            # prepare closed loop for plotting
            ref_vals_closed = ref_vals.tolist()
            ref_vals_closed += ref_vals_closed[:1]

    # --- Axis scale (now based on models + ref if present) ---
    ticks, rmin, rmax = _auto_ring_ticks(all_vals)
    ax.set_rgrids(ticks, angle=90, fontsize=11)
    ax.set_ylim(rmin, rmax)
    ax.set_yticklabels([f"{int(t)}" for t in ticks])

    # --- Optional: draw country reference polygon (after scale set) ---
    if ref_vals_closed is not None:
        ax.plot(angles, ref_vals_closed, linewidth=2.2, color="#7f7f7f", label=f"{country_name} (ref)")
        ax.fill(angles, ref_vals_closed, alpha=0.10, color="#7f7f7f")

    # Colors + SAME styles/markers we standardized
    colors     = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                  "#9467bd", "#8c564b", "#e377c2", "#17becf", "#7f7f7f"]
    linestyles = ["solid", "dashed", "dashdot", (0, (1, 1))]  # dotted tuple
    markers    = ["o", "s", "^", "x"]

    # Plot each model
    for i, (_, row) in enumerate(sub.iterrows()):
        vals = row[DIMENSIONS].astype(float).tolist()
        vals += vals[:1]
        ax.plot(
            angles, vals, linewidth=2.6,
            label=row["legend_label"],
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markersize=7,
            color=colors[i % len(colors)],
        )

    # Axis labels: English, multi-line, bold + halo, nudged outward
    ax.set_xticks(angles[:-1])
    xt = ax.set_xticklabels(
        ["Power\nDistance", "Individualism", "Masculinity",
         "Uncertainty\nAvoidance", "Long-Term\nOrientation", "Indulgence"],
        fontsize=15, fontweight="bold", linespacing=1.2
    )
    for lbl in xt:
        lbl.set_y(lbl.get_position()[1] - 0.05)
        lbl.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])

    ax.grid(True, linestyle=":", linewidth=0.8, color="lightgray")

    # Legend at bottom, big, sorted (plot order already size-sorted)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        title=f"Models — {country_name} (sorted by size)",
        frameon=False,
        fontsize=14,
        title_fontsize=15,
        markerscale=1.6,
        ncol=max(3, min(5, len(sub))),
    )

    plt.subplots_adjust(left=0.08, right=0.92, bottom=0.22, top=0.96)
    save_figure(out_path_base)
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Create country radar charts (China/USA/Europe) using ALL models per country.")
    ap.add_argument("--model_profiles_csv", required=True, help="CSV with columns: model_id, size, country_ref, and the 6 dimensions.")
    ap.add_argument("--hofstede_csv", required=False, help="Optional Hofstede country CSV with columns: country and the 6 dimensions.")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--overlay_country_ref", action="store_true", help="Overlay the country reference polygon (Hofstede) on each radar.")
    ap.add_argument("--dpi", type=int, default=600, help="Figure DPI (default: 600).")
    args = ap.parse_args()

    use_paper_style(dpi=args.dpi)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    profiles = pd.read_csv(args.model_profiles_csv)
    hofstede = pd.read_csv(args.hofstede_csv) if args.hofstede_csv else None

    for cn in ["China", "USA", "Europe"]:
        plot_country_radar_all_models(
            profiles_df=profiles,
            country_name=cn,
            out_path_base=out_dir / f"radar_allmodels_{cn.lower()}",
            hofstede_df=hofstede,
            show_country_ref=args.overlay_country_ref,
        )

    print(f"Figures written to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
