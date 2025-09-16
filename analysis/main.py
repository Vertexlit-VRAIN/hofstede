#!/usr/bin/env python3
"""
Per-model Hofstede metrics with *two* agreement notions:
- agreement_pct_global: mode over all runs in the group
- agreement_pct_avg_question: average of per-question agreements

Inputs
------
1) validated_data.jsonl  (must have: index, domain, dimension_code)
2) results_root/         (one folder per model, each with 3 JSONL runs)
   <ModelX>/
     inference_results_run_1.jsonl
     inference_results_run_2.jsonl
     inference_results_run_3.jsonl

Outputs (per model)
-------------------
out_dir/<ModelX>/
  metrics_by_question.csv
  metrics_by_domain.csv
  metrics_by_dimension.csv
"""

import os
import re
import json
import argparse
from glob import glob
from typing import List, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

DECIMALS_STD = 2
DECIMALS_AGR = 1
DECIMALS_MEAN = 2

RESULTS_PATTERN = r"inference_results_run_(\d+)\.jsonl"

# ---------- I/O helpers ----------

def read_validated(validated_path: str) -> pd.DataFrame:
    df = pd.read_json(validated_path, lines=True)
    needed = ["index", "domain", "dimension_code"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"'validated_data.jsonl' missing column: {c}")
    return df[needed].copy()

def normalize_results_df(df: pd.DataFrame, inferred_run: int) -> pd.DataFrame:
    rename_map = {
        "question_index": "original_index",
        "question_id": "original_index",
        "index": "original_index",
        "run": "inference_run",
        "level": "selected_level",
        "score": "selected_level",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    if "original_index" not in df.columns or "selected_level" not in df.columns:
        raise ValueError("Results file missing 'original_index' or 'selected_level'")
    df["original_index"] = pd.to_numeric(df["original_index"], errors="coerce").astype("Int64")
    df["selected_level"] = pd.to_numeric(df["selected_level"], errors="coerce").astype(float)
    if "inference_run" not in df.columns or df["inference_run"].isna().all():
        df["inference_run"] = inferred_run
    return df

def read_model_results(model_dir: str) -> pd.DataFrame:
    files = sorted(glob(os.path.join(model_dir, "inference_results_run_*.jsonl")))
    if not files:
        raise ValueError(f"No run files in {model_dir}")
    dfs = []
    for f in files:
        m = re.search(RESULTS_PATTERN, os.path.basename(f))
        run_num = int(m.group(1)) if m else None
        df = pd.read_json(f, lines=True)
        df = normalize_results_df(df, run_num)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# ---------- metrics ----------

def agreement_pct(series: pd.Series) -> float:
    counts = series.value_counts(dropna=True)
    total = counts.sum()
    if total == 0:
        return np.nan
    return float(counts.max() / total * 100.0)

def level_distribution(levels: pd.Series) -> Dict[str, int]:
    out = {f"level_{i}_count": 0 for i in range(1, 6)}
    for k, v in levels.value_counts(dropna=True).items():
        try:
            k_int = int(k)
            if 1 <= k_int <= 5:
                out[f"level_{k_int}_count"] = int(v)
        except Exception:
            pass
    return out

def level_mode(levels: pd.Series) -> float:
    if levels.dropna().empty:
        return np.nan
    return float(levels.mode().iloc[0])

def level_mean(levels: pd.Series) -> float:
    if levels.dropna().empty:
        return np.nan
    return float(levels.mean())

def compute_per_question(df: pd.DataFrame) -> pd.DataFrame:
    """Per-question metrics for a single model."""
    grp = df.groupby("original_index")["selected_level"]
    base = grp.agg(
        n_runs="count",
        std="std",
        min="min",
        max="max"
    ).reset_index()
    base["range"] = (base["max"] - base["min"]).fillna(0)
    base["agreement_pct"] = grp.apply(agreement_pct).values

    # level distribution + mode/mean
    dist_rows = []
    for qid, sub in df.groupby("original_index"):
        levels = sub["selected_level"]
        row = {"original_index": qid}
        row.update(level_distribution(levels))
        row["level_mode"] = level_mode(levels)
        row["level_mean"] = level_mean(levels)
        dist_rows.append(row)
    dist_df = pd.DataFrame(dist_rows)

    out = pd.merge(base, dist_df, on="original_index", how="left")
    out["std"] = out["std"].fillna(0).round(DECIMALS_STD)
    out["range"] = out["range"].round(0).astype(int)
    out["agreement_pct"] = out["agreement_pct"].round(DECIMALS_AGR)
    out["level_mean"] = out["level_mean"].round(DECIMALS_MEAN)
    out["level_mode"] = out["level_mode"].round(0).astype("Int64")
    return out.drop(columns=["min", "max"])

def _summarize_over_questions(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    For domain/dimension groups:
      - agreement_pct_global: mode over all runs in the group
      - agreement_pct_avg_question: average of per-question agreement
      - std_avg_question: average of per-question std
      - range_avg_question: average of per-question range
      - level_1..5_count: summed across runs in the group
      - level_mode_global/level_mean_global: over all runs (reference)
    """
    # global aggregation (all runs in the group)
    grp_runs = df.groupby(group_cols)["selected_level"]
    global_base = grp_runs.agg(
        n_runs="count",
        std_global="std",
        min="min",
        max="max"
    ).reset_index()
    global_base["range_global"] = (global_base["max"] - global_base["min"]).fillna(0)
    global_base["agreement_pct_global"] = grp_runs.apply(agreement_pct).values

    # per-question agreement, then average within the group
    per_q = compute_per_question(df)  # columns: original_index, std, range, agreement_pct, counts...
    # attach group labels to per-question rows
    join_cols = ["original_index"] + group_cols
    df_q = df[["original_index"] + group_cols].drop_duplicates()
    per_q = per_q.merge(df_q, on="original_index", how="left")

    grp_q = per_q.groupby(group_cols)
    averaged = grp_q.agg(
        n_questions=("original_index", "nunique"),
        agreement_pct_avg_question=("agreement_pct", "mean"),
        std_avg_question=("std", "mean"),
        range_avg_question=("range", "mean"),
    ).reset_index()

    # level distribution (sum counts)
    count_cols = [f"level_{i}_count" for i in range(1, 6)]
    sum_counts = grp_q[count_cols].sum().reset_index()
    sum_counts.columns = group_cols + count_cols

    # global mode/mean (over all runs in group)
    global_mode_mean = grp_runs.apply(level_mode).reset_index(name="level_mode_global")
    global_mode_mean["level_mean_global"] = grp_runs.mean().values

    # merge all
    out = global_base.drop(columns=["min","max"]).merge(
        averaged, on=group_cols, how="left"
    ).merge(
        sum_counts, on=group_cols, how="left"
    ).merge(
        global_mode_mean, on=group_cols, how="left"
    )

    # formatting
    out["std_global"] = out["std_global"].fillna(0).round(DECIMALS_STD)
    out["range_global"] = out["range_global"].round(0).astype(int)
    out["agreement_pct_global"] = out["agreement_pct_global"].round(DECIMALS_AGR)
    out["agreement_pct_avg_question"] = out["agreement_pct_avg_question"].round(DECIMALS_AGR)
    out["std_avg_question"] = out["std_avg_question"].round(DECIMALS_STD)
    out["range_avg_question"] = out["range_avg_question"].round(0).astype(int)
    out["level_mean_global"] = out["level_mean_global"].round(DECIMALS_MEAN)
    out["level_mode_global"] = out["level_mode_global"].round(0).astype("Int64")

    return out

# ---------- main ----------

def main(validated_path: str, results_root: str, out_dir: str):
    validated = read_validated(validated_path)

    model_dirs = [d for d in sorted(glob(os.path.join(results_root, "*"))) if os.path.isdir(d)]
    if not model_dirs:
        raise ValueError(f"No model folders found under: {results_root}")

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        print(f"Processing model: {model_name}")

        # load this model's 3 runs and join domain/dimension
        runs_df = read_model_results(model_dir)
        df = runs_df.merge(validated, left_on="original_index", right_on="index", how="left")

        # sanity check: all rows should have domain & dimension_code
        if df["domain"].isna().any() or df["dimension_code"].isna().any():
            missing = df[df["domain"].isna() | df["dimension_code"].isna()][["original_index"]].drop_duplicates()
            raise ValueError(f"{model_name}: missing domain/dimension for indices:\n{missing.head(20)}")

        # outputs folder per model
        mdir = os.path.join(out_dir, model_name)
        os.makedirs(mdir, exist_ok=True)

        # 1) per-question (agreement here will be 100/66.7/33.3 with 3 runs)
        by_question = compute_per_question(df)
        by_question.to_csv(os.path.join(mdir, "metrics_by_question.csv"), index=False)

        # 2) domain-level: both agreement metrics
        by_domain = _summarize_over_questions(df, ["domain"])
        by_domain.to_csv(os.path.join(mdir, "metrics_by_domain.csv"), index=False)

        # 3) dimension-level: both agreement metrics
        by_dimension = _summarize_over_questions(df, ["dimension_code"])
        by_dimension.to_csv(os.path.join(mdir, "metrics_by_dimension.csv"), index=False)

        print(f"  Saved: {mdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Per-model Hofstede metrics with dual agreement definitions.")
    ap.add_argument("--validated_path", required=True, help="Path to validated_data.jsonl")
    ap.add_argument("--results_root", required=True, help="Root folder containing one subfolder per model")
    ap.add_argument("--out_dir", required=True, help="Directory to write per-model CSVs")
    args = ap.parse_args()
    main(args.validated_path, args.results_root, args.out_dir)
