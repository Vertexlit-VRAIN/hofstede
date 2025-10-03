
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

DIMENSIONS = ["Power_Distance","Individualism","Masculinity","Uncertainty_Avoidance","Long_Term_Orientation","Indulgence"]

def proximity_maps(profiles_csv, hofstede_csv, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    prof = pd.read_csv(profiles_csv)
    hof = pd.read_csv(hofstede_csv)

    models = prof.copy()
    models["label"] = models["model_id"] + "-" + models["size"] + "@" + models["country_ref"]
    models["type"] = "model"
    countries = hof.copy()
    countries["label"] = countries["country"]
    countries["type"] = "country"

    X = pd.concat([models[DIMENSIONS], countries[DIMENSIONS]], ignore_index=True).fillna(0.0).values
    labels = pd.concat([models["label"], countries["label"]], ignore_index=True).tolist()
    types = pd.concat([pd.Series(["model"]*len(models)), pd.Series(["country"]*len(countries))], ignore_index=True).tolist()

    pca = PCA(n_components=2, random_state=0).fit(X)
    Xp = pca.transform(X)
    plt.figure(figsize=(7,6))
    for typ in ["country","model"]:
        idx = [i for i,t in enumerate(types) if t==typ]
        plt.scatter(Xp[idx,0], Xp[idx,1], label=typ)
    for (x,y,lbl) in zip(Xp[:,0], Xp[:,1], labels):
        plt.text(x,y,lbl, fontsize=8)
    var = pca.explained_variance_ratio_
    plt.title(f"Mapa de proximidad – PCA (PC1={var[0]:.2f}, PC2={var[1]:.2f})")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(); plt.tight_layout()
    plt.savefig((out/"map_pca.png").as_posix()); plt.close()

    mds = MDS(n_components=2, dissimilarity="euclidean", random_state=0)
    Xm = mds.fit_transform(X)
    plt.figure(figsize=(7,6))
    for typ in ["country","model"]:
        idx = [i for i,t in enumerate(types) if t==typ]
        plt.scatter(Xm[idx,0], Xm[idx,1], label=typ)
    for (x,y,lbl) in zip(Xm[:,0], Xm[:,1], labels):
        plt.text(x,y,lbl, fontsize=8)
    plt.title("Mapa de proximidad – MDS (Euclidiana 6D)")
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2"); plt.legend(); plt.tight_layout()
    plt.savefig((out/"map_mds.png").as_posix()); plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles_csv", required=True)
    ap.add_argument("--hofstede_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    proximity_maps(args.profiles_csv, args.hofstede_csv, args.out_dir)
