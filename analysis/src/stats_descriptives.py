
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

DIMENSIONS = ["Power_Distance","Individualism","Masculinity","Uncertainty_Avoidance","Long_Term_Orientation","Indulgence"]

def _alpha_cronbach(df_items):
    k = df_items.shape[1]
    if k < 2 or df_items.shape[0] < 2:
        return np.nan
    item_vars = df_items.var(axis=0, ddof=1)
    total_var = df_items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    alpha = (k/(k-1))*(1 - item_vars.sum()/total_var)
    return float(alpha)

def _icc_two_way_random_absolute(df_wide):
    n, k = df_wide.shape
    if n < 2 or k < 2:
        return np.nan
    X = df_wide.values
    mpt = X.mean()
    msr = n * (df_wide.mean(axis=1) - mpt)**2
    msr = msr.sum() / (n-1)
    msc = k * (df_wide.mean(axis=0) - mpt)**2
    msc = msc.sum() / (k-1)
    e = X - df_wide.mean(axis=1).values[:,None] - df_wide.mean(axis=0).values[None,:] + mpt
    mse = (e**2).sum() / ((n-1)*(k-1))
    icc2k = (msr - mse) / (msr + (msc - mse)/n)
    return float(icc2k)

def ci_mean_t(mean, sd, n, alpha=0.05):
    if n is None or n < 2 or sd is None or np.isnan(sd):
        return (np.nan, np.nan)
    from scipy.stats import t as tdist
    df = n - 1
    tcrit = tdist.ppf(1-alpha/2, df)
    half = tcrit * (sd / np.sqrt(n))
    return (float(mean - half), float(mean + half))

def main(args):
    df = pd.read_csv(args.models_csv)
    if "score_scaled" not in df.columns or df["score_scaled"].isna().all():
        denom = (df["scale_max"] - df["scale_min"]).replace(0,1)
        df["score_scaled"] = 100.0 * (df["score_raw"] - df["scale_min"]) / denom

    group_cols = ["model_id","size","country_ref","dimension"]
    desc = (df
            .groupby(group_cols, dropna=False)
            .agg(mean=("score_scaled","mean"),
                 sd=("score_scaled","std"),
                 N=("score_scaled","size"))
            .reset_index())
    ci = desc.apply(lambda r: pd.Series(ci_mean_t(r["mean"], r["sd"], r["N"]), index=["CI95_low","CI95_high"]), axis=1)
    desc = pd.concat([desc, ci], axis=1)

    # Alpha
    alpha_rows = []
    for (model_id,size,country), sub in df.groupby(["model_id","size","country_ref"], dropna=False):
        for dim, dsub in sub.groupby("dimension"):
            wide = dsub.pivot_table(index=["run_id"], columns="item_id", values="score_scaled", aggfunc="mean")
            a = _alpha_cronbach(wide) if wide.shape[1] >= 2 else np.nan
            alpha_rows.append({"model_id":model_id,"size":size,"country_ref":country,"dimension":dim,"alpha":a})
    alpha_df = pd.DataFrame(alpha_rows)

    # ICC(2,k)
    icc_rows = []
    for (model_id,size,country), sub in df.groupby(["model_id","size","country_ref"], dropna=False):
        for dim, dsub in sub.groupby("dimension"):
            wide = dsub.pivot_table(index="item_id", columns="run_id", values="score_scaled", aggfunc="mean")
            icc = _icc_two_way_random_absolute(wide) if (wide.shape[0] >= 2 and wide.shape[1] >= 2) else np.nan
            icc_rows.append({"model_id":model_id,"size":size,"country_ref":country,"dimension":dim,"ICC_2k":icc})
    icc_df = pd.DataFrame(icc_rows)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    desc_full = desc.merge(alpha_df, on=["model_id","size","country_ref","dimension"], how="left") \
                    .merge(icc_df, on=["model_id","size","country_ref","dimension"], how="left")
    desc_full.to_csv(out_dir/"table1_descriptives.csv", index=False)
    print("Wrote:", (out_dir/"table1_descriptives.csv").as_posix())

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(args)
