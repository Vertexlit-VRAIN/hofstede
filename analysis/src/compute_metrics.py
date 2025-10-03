import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import euclidean, cityblock, chebyshev, canberra, braycurtis

DIMENSIONS = ["Power_Distance","Individualism","Masculinity","Uncertainty_Avoidance","Long_Term_Orientation","Indulgence"]

def scale_row(row):
    # If score_scaled is already present, keep it; otherwise scale from raw + min/max
    if "score_scaled" in row and pd.notnull(row["score_scaled"]):
        return row["score_scaled"]
    denom = (row["scale_max"] - row["scale_min"]) if (row["scale_max"] - row["scale_min"]) != 0 else 1.0
    return 100.0 * (row["score_raw"] - row["scale_min"]) / denom

def load_hofstede(path):
    ref = pd.read_csv(path)
    assert set(DIMENSIONS).issubset(ref.columns), "Missing Hofstede columns"
    return ref

def pivot_model_means(df):
    # aggregate to means per model_id × size × country_ref × dimension
    agg = (df
           .assign(score_scaled=df.apply(scale_row, axis=1))
           .groupby(["model_id","size","params_log","country_ref","dimension"], dropna=False)
           .agg(mean_score=("score_scaled","mean"),
                sd=("score_scaled","std"),
                n=("score_scaled","size"))
           .reset_index())
    # Pivot to 6D vectors
    mat = agg.pivot_table(index=["model_id","size","params_log","country_ref"],
                          columns="dimension", values="mean_score").reset_index()
    for d in DIMENSIONS:
        if d not in mat.columns:
            mat[d] = np.nan
    return agg, mat[["model_id","size","params_log","country_ref"]+DIMENSIONS]

def vector_metrics(model_vec, country_vec, weights=None):
    m = model_vec.values.astype(float)
    h = country_vec.values.astype(float)
    diffs = m - h
    ad = np.abs(diffs)
    if weights is None:
        w = np.ones_like(m, dtype=float)
    else:
        w = np.array([weights.get(d,1.0) for d in model_vec.index], dtype=float)
    w = w / w.sum()  # normalize weights to sum 1

    mae = float((w*ad).sum())
    rmse = float(np.sqrt((w*(diffs**2)).sum()))
    l2 = float(np.linalg.norm(diffs))
    l1 = float(np.sum(ad))
    l_inf = float(np.max(ad))
    # cosine similarity and distance
    m_norm = np.linalg.norm(m); h_norm = np.linalg.norm(h)
    cos_sim = float(np.dot(m, h) / (m_norm * h_norm)) if m_norm>0 and h_norm>0 else np.nan
    cos_dist = float(1.0 - cos_sim) if np.isfinite(cos_sim) else np.nan
    # Pearson & Spearman
    try:
        pear_r, _ = pearsonr(m, h)
    except Exception:
        pear_r = np.nan
    try:
        spear_r, _ = spearmanr(m, h)
    except Exception:
        spear_r = np.nan
    r2 = float(pear_r**2) if np.isfinite(pear_r) else np.nan
    # Other distances
    canb = float(canberra(m, h))
    bray = float(braycurtis(m, h))
    bias = float(np.mean(diffs))
    # standardized errors (divide by 100 to keep scale-free)
    z_err = (diffs/100.0).tolist()
    return diffs, mae, rmse, l2, l1, l_inf, cos_sim, cos_dist, pear_r, spear_r, r2, canb, bray, bias, z_err

def main(args):
    data_path = Path(args.models_csv)
    hofstede_path = Path(args.hofstede_csv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # optional weights json
    weights = None
    if args.weights_json and Path(args.weights_json).exists():
        import json
        with open(args.weights_json, "r", encoding="utf-8") as f:
            weights = json.load(f)

    df = pd.read_csv(data_path)
    ref = load_hofstede(hofstede_path)

    # aggregate to model profiles (6D)
    agg, profiles = pivot_model_means(df)

    results = []
    diffs_rows = []
    for _, row in profiles.iterrows():
        country = row["country_ref"]
        ref_row = ref[ref["country"]==country]
        if ref_row.empty:
            continue
        ref_vec = ref_row.iloc[0][DIMENSIONS]
        model_vec = row[DIMENSIONS]
        diffs, mae, rmse, l2, l1, l_inf, cos, cosd, pr, sr, r2, canb, bray, bias, zerr = vector_metrics(model_vec, ref_vec, weights=weights)
        res = {
            "model_id": row["model_id"],
            "size": row["size"],
            "params_log": row["params_log"],
            "country_ref": country,
            "MAE": mae,
            "RMSE": rmse,
            "L2": l2,
            "L1": l1,
            "L_inf": l_inf,
            "Cosine": cos,
            "CosineDist": cosd,
            "Pearson_r": pr,
            "Spearman_rho": sr,
            "R2_from_Pearson": r2,
            "Canberra": canb,
            "BrayCurtis": bray,
            "Bias": bias
        }
        results.append(res)
        diffr = {"model_id": row["model_id"], "size": row["size"], "params_log": row["params_log"], "country_ref": country}
        for (d, val, z) in zip(DIMENSIONS, diffs, zerr):
            diffr[f"Delta_{d}"] = float(val)
            diffr[f"AbsDelta_{d}"] = float(abs(val))
            diffr[f"Zerr_{d}"] = float(z)
        diffs_rows.append(diffr)

    metrics_df = pd.DataFrame(results).sort_values(["country_ref","model_id","size"])
    deltas_df = pd.DataFrame(diffs_rows).sort_values(["country_ref","model_id","size"])

    # Aggregates for comparisons
    same_model_size = (metrics_df
                       .groupby(["model_id","country_ref","size"], as_index=False)
                       .agg(MAE=("MAE","mean"), L2=("L2","mean"), Cosine=("Cosine","mean")))

    all_models_rank = (metrics_df
                       .sort_values(["country_ref","MAE","L2","CosineDist"], ascending=[True,True,True,True]))

    country_groups = {
        "Europe": ["Europe-France"],
        "ChinaMean": ["China"],
        "USAMean": ["USA"]
    }
    rows = []
    for grp, countries in country_groups.items():
        sub = metrics_df[metrics_df["country_ref"].isin(countries)]
        if sub.empty:
            continue
        g = (sub
             .groupby(["model_id","size"], as_index=False)
             .agg(MAE=("MAE","mean"),
                  L2=("L2","mean"),
                  L1=("L1","mean"),
                  L_inf=("L_inf","mean"),
                  Cosine=("Cosine","mean"),
                  CosineDist=("CosineDist","mean")))
        g["comparison_group"] = grp
        rows.append(g)
    country_comparisons = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    usa_compare = metrics_df[metrics_df["country_ref"]=="USA"].copy()

    # Save outputs
    profiles.to_csv(out_dir/"model_profiles_6D.csv", index=False)
    metrics_df.to_csv(out_dir/"metrics_per_model_country.csv", index=False)
    deltas_df.to_csv(out_dir/"per_dimension_deltas.csv", index=False)
    same_model_size.to_csv(out_dir/"same_model_different_sizes.csv", index=False)
    all_models_rank.to_csv(out_dir/"all_models_rank_by_country.csv", index=False)
    country_comparisons.to_csv(out_dir/"country_group_comparisons.csv", index=False)
    usa_compare.to_csv(out_dir/"usa_models_comparison.csv", index=False)

    print("Wrote:")
    for f in ["model_profiles_6D.csv","metrics_per_model_country.csv","per_dimension_deltas.csv",
              "same_model_different_sizes.csv","all_models_rank_by_country.csv",
              "country_group_comparisons.csv","usa_models_comparison.csv"]:
        print(" -", (out_dir/f).as_posix())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_csv", type=str, required=True, help="Path to models_results CSV")
    ap.add_argument("--hofstede_csv", type=str, required=True, help="Path to Hofstede reference CSV")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory for outputs")
    ap.add_argument("--weights_json", type=str, default="", help="Optional JSON file with weights per dimension")
    args = ap.parse_args()
    main(args)