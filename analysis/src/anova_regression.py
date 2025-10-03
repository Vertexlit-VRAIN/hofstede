
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy.stats import kruskal, spearmanr

def main(args):
    metrics = pd.read_csv(args.metrics_csv)
    deltas = pd.read_csv(args.deltas_csv)
    abs_cols = [c for c in deltas.columns if c.startswith("AbsDelta_")]
    m = deltas.melt(id_vars=["model_id","size","country_ref","params_log"], value_vars=abs_cols,
                    var_name="dim", value_name="abs_delta")
    m["dimension"] = m["dim"].str.replace("AbsDelta_","", regex=False)

    anova_rows = []
    for dim, sub in m.groupby("dimension"):
        try:
            model = ols("abs_delta ~ C(size)", data=sub).fit()
            aov = sm.stats.anova_lm(model, typ=2)
            eta_sq = aov.loc["C(size)","sum_sq"]/aov["sum_sq"].sum()
            anova_rows.append({"dimension":dim,
                               "F": aov.loc["C(size)","F"],
                               "p": aov.loc["C(size)","PR(>F)"],
                               "eta_sq": eta_sq,
                               "N": len(sub)})
        except Exception:
            anova_rows.append({"dimension":dim,"F":np.nan,"p":np.nan,"eta_sq":np.nan,"N":len(sub)})
    anova_df = pd.DataFrame(anova_rows)

    kw_rows = []
    for dim, sub in m.groupby("dimension"):
        groups = [g["abs_delta"].dropna().values for _, g in sub.groupby("size")]
        if len(groups) >= 2 and all(len(g)>1 for g in groups):
            H, p = kruskal(*groups)
        else:
            H, p = (np.nan, np.nan)
        kw_df_row = {"dimension":dim,"H":H,"p":p,"N":len(sub)}
        kw_rows.append(kw_df_row)
    kw_df = pd.DataFrame(kw_rows)

    reg_rows = []
    metrics = metrics.dropna(subset=["MAE"]).copy()
    if "params_log" not in metrics.columns:
        metrics["params_log"] = np.nan
    formula = "MAE ~ params_log + C(country_ref)"
    try:
        model = ols(formula, data=metrics).fit(cov_type="cluster", cov_kwds={"groups":metrics["model_id"]})
        params = model.params
        conf = model.conf_int()
        reg_rows.append({
            "term":"params_log",
            "beta": params.get("params_log", np.nan),
            "ci_low": conf.loc["params_log",0] if "params_log" in conf.index else np.nan,
            "ci_high": conf.loc["params_log",1] if "params_log" in conf.index else np.nan,
            "p": model.pvalues.get("params_log", np.nan),
            "R2": model.rsquared
        })
    except Exception:
        reg_rows.append({"term":"params_log","beta":np.nan,"ci_low":np.nan,"ci_high":np.nan,"p":np.nan,"R2":np.nan})

    try:
        rho, p = spearmanr(metrics["params_log"], metrics["MAE"], nan_policy="omit")
    except Exception:
        rho, p = (np.nan, np.nan)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    anova_df.to_csv(out/"table3_anova_size_by_dimension.csv", index=False)
    kw_df.to_csv(out/"table3_kruskal_size_by_dimension.csv", index=False)
    pd.DataFrame(reg_rows).to_csv(out/"table3_regression_mae_paramslog_countryFE.csv", index=False)
    pd.DataFrame([{"spearman_rho":rho, "p":p}]).to_csv(out/"table3_spearman_mae_paramslog.csv", index=False)
    print("Wrote:", (out/"table3_anova_size_by_dimension.csv").as_posix())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--deltas_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(args)
