import argparse
from pathlib import Path
import math
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from sklearn.decomposition import PCA
from scipy.stats import t as tdist

# -----------------------------
# Global config for paper figs
# -----------------------------
DIMENSIONS = ["Power_Distance","Individualism","Masculinity","Uncertainty_Avoidance","Long_Term_Orientation","Indulgence"]


def use_paper_style():
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
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
# Helpers
# -----------------------------
def radar_factory(num_vars):
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    return angles

def wrap_labels(labels, max_chars=18):
    """Soft-wrap tick labels for readability."""
    out = []
    for lab in labels:
        if len(lab) <= max_chars:
            out.append(lab)
        else:
            parts = []
            current = ""
            for word in lab.replace("_"," ").split():
                if len(current) + 1 + len(word) <= max_chars:
                    current = (current + " " + word).strip()
                else:
                    parts.append(current)
                    current = word
            if current:
                parts.append(current)
            out.append("\n".join(parts))
    return out

def dynamic_legend(ax, n_items, loc="upper left", anchor=(1.02, 1.0)):
    # Put the legend outside on the right; choose columns based on count
    ncol = 1
    if n_items > 12:
        ncol = 2
    if n_items > 24:
        ncol = 3
    leg = ax.legend(loc=loc, bbox_to_anchor=anchor, frameon=True, ncol=ncol, borderaxespad=0.0)
    # Slightly thinner frame
    if leg:
        leg.get_frame().set_linewidth(0.8)

def text_halo_kwargs():
    return dict(path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

def jitter(values, scale=0.08, rng=None):
    rng = np.random.default_rng(rng)
    return np.array(values) + rng.normal(0, scale, size=len(values))

def _size_sort_key(s):
    # Ordena por número si existe (p.ej., 2B < 7B < 27B), luego por texto
    digits = ''.join([c for c in str(s) if c.isdigit()])
    return (digits == "", int(digits or 0), str(s))

def _ci95(mean, sd, n):
    if n is None or n < 2 or sd is None or np.isnan(sd):
        return (np.nan, np.nan)
    df = n - 1
    tcrit = tdist.ppf(0.975, df)
    half = tcrit * (sd / np.sqrt(n))
    return (float(mean - half), float(mean + half))

# -----------------------------
# Plots
# -----------------------------
def plot_radar(country_name, country_vec, models_df, out_path):
    labels = DIMENSIONS
    angles = radar_factory(len(labels))
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(6.2, 6.2))
    ax = plt.subplot(111, polar=True)

    # Reference (country)
    values_c = country_vec.values.tolist()
    values_c += values_c[:1]
    ax.plot(angles, values_c, linewidth=2.2, label=f"{country_name} (ref)")
    ax.fill(angles, values_c, alpha=0.12)

    # Models
    n_lines = 1
    for _, row in models_df.iterrows():
        vals = [row[d] for d in labels]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.1, alpha=0.9,
                label=f"{row['model_id']} ({row['size']})")
        n_lines += 1

    # Axes formatting
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(wrap_labels(labels, max_chars=16))
    ax.set_ylim(0, 100)
    ax.set_rlabel_position(0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_title(f"Radar: {country_name}", pad=18)

    dynamic_legend(ax, n_items=n_lines, loc="upper left", anchor=(1.05, 1.02))
    plt.tight_layout()
    save_figure(out_path)
    plt.close(fig)

def heatmap_abs_deltas(deltas_df, out_path):
    cols = [f"AbsDelta_{d}" for d in DIMENSIONS]
    M = deltas_df[cols].values

    fig_h = max(3.2, 0.38*len(deltas_df))
    fig = plt.figure(figsize=(8.6, fig_h))
    ax = plt.gca()
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("|Δ|", rotation=90, labelpad=10)

    ax.set_yticks(range(len(deltas_df)))
    ax.set_yticklabels([f"{r.model_id}-{r.size}-{r.country_ref}" for r in deltas_df.itertuples()], fontsize=8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_title("Heatmap |Δ| por dimensión")
    ax.grid(False)

    plt.tight_layout()
    save_figure(out_path)
    plt.close(fig)

def pca_map(profiles_df, hofstede_df, out_path):
    models = profiles_df.copy()
    models["label"] = models["model_id"] + "-" + models["size"] + "@" + models["country_ref"]
    models["type"] = "model"
    countries = hofstede_df.copy()
    countries["label"] = countries["country"]
    countries["type"] = "country"

    common_cols = DIMENSIONS
    X = pd.concat([models[common_cols], countries[common_cols]], ignore_index=True).fillna(0.0)
    labels = pd.concat([models["label"], countries["label"]], ignore_index=True)
    types = pd.concat([pd.Series(["model"]*len(models)), pd.Series(["country"]*len(countries))], ignore_index=True)

    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X.values)

    fig = plt.figure(figsize=(7.6, 6.4))
    ax = plt.gca()

    # Plot by type
    for typ, ms in [("country", 60), ("model", 30)]:
        idx = (types == typ).values
        sc = ax.scatter(coords[idx,0], coords[idx,1], s=ms, label=typ, alpha=0.85, edgecolors="none")

    # Labels with white halo for readability
    for (x, y, lbl) in zip(coords[:,0], coords[:,1], labels):
        ax.text(x, y, " " + lbl + " ",
                fontsize=8, va="center", ha="center",
                bbox=dict(facecolor="white", alpha=0.6, boxstyle="round,pad=0.2", lw=0),
                **text_halo_kwargs())

    var = pca.explained_variance_ratio_
    ax.set_title(f"Mapa de proximidad (PCA 2D) – Var exp: PC1={var[0]:.2f}, PC2={var[1]:.2f}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Legend outside to avoid overlap
    dynamic_legend(ax, n_items=2, loc="upper left", anchor=(1.02, 1.0))

    plt.tight_layout()
    save_figure(out_path)
    plt.close(fig)

def scatter_size_vs_mae(metrics_df, out_path, show_trend=True):
    fig = plt.figure(figsize=(6.6, 5.2))
    ax = plt.gca()

    # Order sizes by a sensible key (try numeric if present)
    sizes = sorted(metrics_df["size"].unique(), key=lambda s: (''.join([c for c in s if c.isdigit()]) == "", int(''.join([c for c in s if c.isdigit()]) or 0), s))
    size_to_x = {s: i for i, s in enumerate(sizes)}

    xs = [size_to_x[s] for s in metrics_df["size"]]
    xj = jitter(xs, scale=0.10, rng=0)  # jittered x to avoid point overlap

    ax.scatter(xj, metrics_df["MAE"], alpha=0.85, s=32)

    # Draw medians per size
    medians = metrics_df.groupby("size")["MAE"].median().reindex(sizes)
    ax.plot(range(len(sizes)), medians.values, lw=1.6, linestyle="-", marker="o", alpha=0.9, label="Mediana MAE")

    if show_trend and len(metrics_df) >= 3:
        # Simple linear trend over the raw jitter-free x positions
        coeffs = np.polyfit(xs, metrics_df["MAE"].values, deg=1)
        xline = np.linspace(min(xs), max(xs), 100)
        yline = np.polyval(coeffs, xline)
        ax.plot(xline, yline, lw=1.2, linestyle="--", alpha=0.8, label="Tendencia (OLS)")

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes, rotation=0)
    ax.set_ylabel("MAE")
    ax.set_xlabel("Tamaño de modelo")
    ax.set_title("Tamaño de modelo vs MAE (por país)")

    dynamic_legend(ax, n_items=2, loc="upper left", anchor=(1.02, 1.0))
    plt.tight_layout()
    save_figure(out_path)
    plt.close(fig)

def bland_altman(deltas_df, out_dir):
    # Using Δ histograms per dimension; annotate mean and std; auto-bin
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for d in DIMENSIONS:
        diffs = deltas_df[f"Delta_{d}"].dropna().values
        if diffs.size == 0:
            continue
        mu = float(np.mean(diffs))
        sd = float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0

        fig = plt.figure(figsize=(6.2, 4.4))
        ax = plt.gca()
        ax.hist(diffs, bins="auto", alpha=0.9)
        ax.axvline(mu, linestyle="-", linewidth=1.5, alpha=0.9, label=f"Media = {mu:.2f}")
        ax.axvline(mu + 1.96*sd, linestyle="--", linewidth=1.2, alpha=0.9, label=f"±1.96·SD ({mu+1.96*sd:.2f})")
        ax.axvline(mu - 1.96*sd, linestyle="--", linewidth=1.2, alpha=0.9)

        ax.set_title(f"Distribución Δ ({d})")
        ax.set_xlabel("Δ (Modelo - País)")
        ax.set_ylabel("Frecuencia")
        dynamic_legend(ax, n_items=2, loc="upper left", anchor=(1.02, 1.0))
        plt.tight_layout()
        save_figure(Path(out_dir) / f"bland_altman_{d}")
        plt.close(fig)

def corr_matrices(profiles_df, out_dir):
    # Correlación entre perfiles de modelos (6D)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    X = profiles_df.copy()
    X["row"] = X["model_id"] + "-" + X["size"] + "@" + X["country_ref"]
    M = X.set_index("row")[DIMENSIONS]
    C = M.T.corr()

    fig_w = max(6.5, min(12, 0.30*len(C) + 4))
    fig = plt.figure(figsize=(fig_w, fig_w*0.85))
    ax = plt.gca()
    im = ax.imshow(C.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Pearson r", rotation=90, labelpad=10)

    ax.set_xticks(range(len(C)))
    ax.set_xticklabels(C.index, rotation=90, fontsize=7)
    ax.set_yticks(range(len(C)))
    ax.set_yticklabels(C.index, fontsize=7)
    ax.set_title("Matriz de correlación entre perfiles de modelos")
    ax.grid(False)

    plt.tight_layout()
    save_figure(Path(out_dir) / "correlation_matrix_models")
    plt.close(fig)

def gemma_dimensions_by_size(profiles_df, out_path_base, model_name="gemma"):
    """
    Figura de barras agrupadas: dimensiones (x) × tamaños (barras) para Gemma.
    Agregación: media por país; error bars: IC95%.
    """
    needed = {"model_id", "size", "country_ref", *DIMENSIONS}
    missing = needed - set(profiles_df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en profiles_df: {missing}")

    # Filtrar gemma (insensible a mayúsculas)
    gem = profiles_df[profiles_df["model_id"].astype(str).str.contains(model_name, case=False, na=False)].copy()
    if gem.empty:
        print(f"[gemma_dimensions_by_size] No se encontraron filas para model_id ~ '{model_name}'.")
        return

    # Largo: (model_id, size, country_ref, dimension, score)
    long = gem.melt(id_vars=["model_id","size","country_ref"],
                    value_vars=DIMENSIONS, var_name="dimension", value_name="score").dropna(subset=["score"])

    # Agregado por (size, dimension)
    grp = (long
           .groupby(["size","dimension"], as_index=False)
           .agg(mean=("score","mean"),
                sd=("score","std"),
                n=("score","size")))

    # Orden de tamaños estable
    sizes = sorted(grp["size"].unique(), key=_size_sort_key)
    size_to_idx = {s:i for i,s in enumerate(sizes)}

    # Preparar estructura por dimensión
    dims = DIMENSIONS
    means = np.zeros((len(dims), len(sizes)), dtype=float) * np.nan
    lows  = np.zeros_like(means) * np.nan
    highs = np.zeros_like(means) * np.nan

    for i, dim in enumerate(dims):
        sub = grp[grp["dimension"] == dim]
        for _, r in sub.iterrows():
            j = size_to_idx[r["size"]]
            lo, hi = _ci95(r["mean"], r["sd"], r["n"])
            means[i, j] = r["mean"]
            lows[i, j]  = lo
            highs[i, j] = hi

    # Plot
    fig = plt.figure(figsize=(9.0, 4.8))
    ax = plt.gca()

    x = np.arange(len(dims))
    k = len(sizes)
    width = min(0.8 / max(k, 1), 0.22)  # ancho de cada barra
    offsets = (np.arange(k) - (k-1)/2) * (width + 0.02)

    for j, size in enumerate(sizes):
        y = means[:, j]
        yerr = np.vstack([y - lows[:, j], highs[:, j] - y])
        ax.bar(x + offsets[j], y, width=width, label=str(size))
        # Error bars: sólo donde hay datos válidos
        valid = np.isfinite(y) & np.isfinite(yerr).all(axis=0)
        if valid.any():
            ax.errorbar((x + offsets[j])[valid], y[valid],
                        yerr=yerr[:, valid],
                        fmt="none", linewidth=1.0, capsize=3, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(wrap_labels(dims, max_chars=16))
    ax.set_ylim(0, 100)
    ax.set_ylabel("Puntuación (0–100)")
    ax.set_title("Gemma: puntuaciones por dimensión y tamaño del modelo\n(media ± IC95% sobre países)")

    # Leyenda fuera (no se solapa)
    dynamic_legend(ax, n_items=len(sizes), loc="upper left", anchor=(1.02, 1.0))

    plt.tight_layout()
    save_figure(out_path_base)
    plt.close(fig)



# -----------------------------
# Main
# -----------------------------
def main(args):
    use_paper_style()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    profiles = pd.read_csv(args.model_profiles_csv)
    metrics = pd.read_csv(args.metrics_csv)
    deltas = pd.read_csv(args.deltas_csv)
    hofstede = pd.read_csv(args.hofstede_csv)

    for country in hofstede["country"].dropna().unique():
        ref_row = hofstede[hofstede["country"] == country].iloc[0]
        subset = profiles[profiles["country_ref"] == country]
        if subset.empty:
            continue
        outp = out_dir / f"radar_{country.replace(' ','_')}"
        plot_radar(
            country,
            ref_row[DIMENSIONS],
            subset,
            outp.as_posix()
        )

    heatmap_abs_deltas(deltas, (out_dir/"heatmap_abs_deltas").as_posix())
    pca_map(profiles, hofstede, (out_dir/"pca_map").as_posix())
    scatter_size_vs_mae(metrics, (out_dir/"size_vs_mae").as_posix())
    gemma_out = (out_dir / "gemma_dimensions_by_size").as_posix()
    gemma_dimensions_by_size(profiles, gemma_out)
    bland_altman(deltas, out_dir / "bland_altman")
    corr_matrices(profiles, out_dir)

    print("Figures created in:", out_dir.as_posix())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_profiles_csv", required=True)
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--deltas_csv", required=True)
    ap.add_argument("--hofstede_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(args)
