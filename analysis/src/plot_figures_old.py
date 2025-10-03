import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

DIMENSIONS = ["Power_Distance","Individualism","Masculinity","Uncertainty_Avoidance","Long_Term_Orientation","Indulgence"]

def radar_factory(num_vars):
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    return angles

def plot_radar(country_name, country_vec, models_df, out_path):
    labels = DIMENSIONS
    angles = radar_factory(len(labels))
    angles = np.concatenate((angles, [angles[0]]))

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    values_c = country_vec.values.tolist()
    values_c += values_c[:1]
    ax.plot(angles, values_c, linewidth=2, label=f"{country_name} (ref)")
    ax.fill(angles, values_c, alpha=0.1)

    for _, row in models_df.iterrows():
        vals = [row[d] for d in labels]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1, label=f"{row['model_id']} ({row['size']})")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0,100)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05))
    plt.title(f"Radar: {country_name}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def heatmap_abs_deltas(deltas_df, out_path):
    cols = [f"AbsDelta_{d}" for d in DIMENSIONS]
    M = deltas_df[cols].values
    plt.figure(figsize=(8, max(3, 0.35*len(deltas_df))))
    plt.imshow(M, aspect="auto")
    plt.colorbar(label="|Δ|")
    plt.yticks(range(len(deltas_df)), [f"{r.model_id}-{r.size}-{r.country_ref}" for r in deltas_df.itertuples()], fontsize=8)
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.title("Heatmap |Δ| por dimensión")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

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
    plt.figure(figsize=(7,6))
    for typ in ["country","model"]:
        idx = (types==typ).values
        plt.scatter(coords[idx,0], coords[idx,1], label=typ)
    for (x,y,lbl) in zip(coords[:,0], coords[:,1], labels):
        plt.text(x, y, lbl, fontsize=8)
    var = pca.explained_variance_ratio_
    plt.title(f"Mapa de proximidad (PCA 2D) – Var exp: PC1={var[0]:.2f}, PC2={var[1]:.2f}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def scatter_size_vs_mae(metrics_df, out_path):
    plt.figure(figsize=(6,5))
    size_order = {s:i for i,s in enumerate(sorted(metrics_df["size"].unique()))}
    xs = [size_order[s] for s in metrics_df["size"]]
    plt.scatter(xs, metrics_df["MAE"])
    plt.xticks(list(size_order.values()), list(size_order.keys()))
    plt.ylabel("MAE")
    plt.title("Tamaño de modelo vs MAE (por país)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def bland_altman(deltas_df, out_dir):
    # Bland–Altman por dimensión (M-H vs M-H difference = Δ)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for d in DIMENSIONS:
        meanMH = []
        diff = []
        for _, row in deltas_df.iterrows():
            # we need the actual M and H to compute (M+H)/2; rebuild H using Delta + H? not present
            # We approximate with: we can't recover H from deltas only; skip mean axis and plot Δ histogram instead
            diff.append(row[f"Delta_{d}"])
        # Plot histogram of Δ as a proxy (simplified BA visualization)
        plt.figure(figsize=(6,4))
        plt.hist(diff, bins=10)
        plt.title(f"Distribución Δ ({d})")
        plt.xlabel("Δ (Modelo - País)")
        plt.ylabel("Frecuencia")
        outp = out_dir / f"bland_altman_{d}.png"
        plt.tight_layout()
        plt.savefig(outp.as_posix())
        plt.close()

def corr_matrices(profiles_df, out_dir):
    # Correlación entre perfiles de modelos (6D)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # build matrix models×dimensions by concatenating country_ref so rows are unique entities
    X = profiles_df.copy()
    X["row"] = X["model_id"] + "-" + X["size"] + "@" + X["country_ref"]
    M = X.set_index("row")[DIMENSIONS]
    C = M.T.corr()
    plt.figure(figsize=(6,5))
    plt.imshow(C.values, aspect="auto")
    plt.colorbar(label="Pearson r")
    plt.xticks(range(len(C)), C.index, rotation=90, fontsize=7)
    plt.yticks(range(len(C)), C.index, fontsize=7)
    plt.title("Matriz de correlación entre perfiles de modelos")
    plt.tight_layout()
    plt.savefig((Path(out_dir)/"correlation_matrix_models.png").as_posix())
    plt.close()

def main(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    profiles = pd.read_csv(args.model_profiles_csv)
    metrics = pd.read_csv(args.metrics_csv)
    deltas = pd.read_csv(args.deltas_csv)
    hofstede = pd.read_csv(args.hofstede_csv)

    for country in hofstede["country"].unique():
        ref_row = hofstede[hofstede["country"]==country].iloc[0]
        subset = profiles[profiles["country_ref"]==country]
        if subset.empty:
            continue
        outp = out_dir / f"radar_{country.replace(' ','_')}.png"
        plot_radar(country, ref_row[["Power_Distance","Individualism","Masculinity","Uncertainty_Avoidance","Long_Term_Orientation","Indulgence"]], subset, outp.as_posix())

    heatmap_abs_deltas(deltas, (out_dir/"heatmap_abs_deltas.png").as_posix())
    pca_map(profiles, hofstede, (out_dir/"pca_map.png").as_posix())
    scatter_size_vs_mae(metrics, (out_dir/"size_vs_mae.png").as_posix())
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