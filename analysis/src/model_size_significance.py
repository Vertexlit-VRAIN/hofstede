#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General analysis of whether MODEL SIZE produces statistically significant differences
in Hofstede dimensions across MULTIPLE model families.

It automatically detects families and sizes from model_id strings like:
  - google/gemma-3-1b-it, google/gemma-3-27b-it
  - Qwen/Qwen3-0.6B, Qwen/Qwen3-32B
  - ibm-granite/granite-3.3-2b-instruct, ibm-granite/granite-3.3-8b-instruct
  - mistralai/Mistral-7B-Instruct-v0.3
  - meta-llama/Llama-3.1-8B-Instruct
  - microsoft/phi-4  (interpreted as family=phi, size=4.0)

It runs a repeated-measures design using items as blocks across sizes, per family × dimension.
The script is robust to missing overlap across sizes:
  - Writes an overlap table of shared items for each size pair within a family × dimension.
  - Chooses the best trio of sizes (max shared items) for a Friedman test if enough complete items.
  - Always runs pairwise Wilcoxon for size pairs with sufficient shared items.

Input CSV (minimum):
    model_id, dimension, item_id, score_raw
Optional:
    scale_min, scale_max, run_id, prompt_id, size, params_log, time_stamp, country_ref

Outputs:
  - size_family_summary.csv           (what families and sizes were detected)
  - size_summary_descriptives.csv     (means, sd, se, CI95 by family × dimension × size)
  - size_friedman_results.csv         (best-trio Friedman per family × dimension)
  - size_pairwise_wilcoxon.csv        (pairwise tests per family × dimension)
  - size_overlap_counts.csv           (shared items per size pair)
  - size_agg_item_level.csv           (wide matrix for QA)

Usage:
    python analyze_any_family_model_size.py \
      --input models_results.csv \
      --outdir ./out \
      --min_blocks 10 \
      --min_pairs 10 \
      --min_sizes_friedman 3

Dependencies: pandas, numpy, scipy
"""

import argparse
import itertools
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# -----------------------------
# 1) CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to input CSV")
    p.add_argument("--outdir", required=True, help="Directory for outputs")
    p.add_argument("--min_blocks", type=int, default=10,
                   help="Minimum complete items required to run Friedman on a trio (default 10)")
    p.add_argument("--min_pairs", type=int, default=10,
                   help="Minimum paired items required to run Wilcoxon for a size pair (default 10)")
    p.add_argument("--min_sizes_friedman", type=int, default=3,
                   help="Minimum number of distinct sizes to attempt a Friedman test (default 3)")
    return p.parse_args()


# -----------------------------
# 2) Parsing utilities
# -----------------------------
# Precompile a few helpful patterns
RE_TOKENIZE = re.compile(r"[\/_\-]+")
RE_FLOAT_B = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")
RE_FLOAT_STANDALONE = re.compile(r"(?:^|[^0-9.])(\d+(?:\.\d+)?)(?:[^0-9.]|$)")
RE_GEMMA = re.compile(r"gemma\s*[-_]?3", re.IGNORECASE)
RE_QWEN3 = re.compile(r"qwen\s*[-_]?qwen3", re.IGNORECASE)
RE_DEEPSEEK_QWEN = re.compile(r"deepseek.", re.IGNORECASE)
RE_MISTRAL = re.compile(r"mistral", re.IGNORECASE)
RE_LLAMA = re.compile(r"llama", re.IGNORECASE)
RE_GRANITE = re.compile(r"granite", re.IGNORECASE)
RE_PHI = re.compile(r"\bphi\b", re.IGNORECASE)


def normalize_model_id(mid: str) -> str:
    if not isinstance(mid, str):
        return ""
    return mid.strip()


def guess_family(model_id: str) -> Optional[str]:
    s = model_id.lower()
    if RE_GEMMA.search(s):
        return "gemma"
    if RE_QWEN3.search(s):
        return "qwen3"
    if RE_DEEPSEEK_QWEN.search(s):
        return "deepseek-qwen"
    if RE_GRANITE.search(s):
        return "granite"
    if RE_MISTRAL.search(s):
        return "mistral"
    if RE_LLAMA.search(s):
        return "llama"
    if RE_PHI.search(s):
        return "phi"
    return None


def guess_size_b(model_id: str) -> Optional[float]:
    """
    Heuristics to get a numeric size in billions from model_id.
    Priority:
      1) number followed by 'B' (e.g., 0.6B, 7B, 27B)
      2) for cases like 'phi-4' or 'Llama-3.1-8B-Instruct', handle gracefully
    """
    s = model_id
    # 1) Common pattern: digits (possibly decimal) followed by 'B'
    m = RE_FLOAT_B.search(s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    # 2) If family implies a size token without explicit 'B' (e.g., 'phi-4')
    # Try to pick the last numeric token in the id as "size"
    nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", s)]
    if nums:
        # In entries like 'Llama-3.1-8B-Instruct', the '8' will be captured by (1) above.
        # For 'phi-4', this returns 4.0
        return float(nums[-1])
    return None


def extract_family_and_size(model_id: str) -> Tuple[Optional[str], Optional[float]]:
    mid = normalize_model_id(model_id)
    fam = guess_family(mid)
    size_b = guess_size_b(mid)
    if fam is None or size_b is None:
        return None, None
    return fam, size_b


# -----------------------------
# 3) Stats helpers
# -----------------------------
def scale_to_0_100(df: pd.DataFrame) -> pd.Series:
    if "score_raw" not in df.columns:
        raise ValueError("Missing 'score_raw' column.")
    if {"scale_min", "scale_max"}.issubset(df.columns):
        rng = (df["scale_max"] - df["scale_min"]).replace(0, np.nan)
        scaled = 100 * (df["score_raw"] - df["scale_min"]) / rng
        scaled = scaled.fillna(df["score_raw"]).clip(0, 100)
        return scaled
    return df["score_raw"].clip(0, 100)


def ci95_halfwidth_from_se(se: float) -> float:
    return 1.96 * se if se is not None and np.isfinite(se) else np.nan


def kendalls_w(chi2: float, n_blocks: int, k: int) -> float:
    if n_blocks <= 0 or k <= 1 or not np.isfinite(chi2):
        return np.nan
    return chi2 / (n_blocks * (k - 1))


def rank_biserial_from_wilcoxon(T_stat: float, n_eff: int) -> Optional[float]:
    if n_eff is None or n_eff <= 0 or not np.isfinite(T_stat):
        return np.nan
    denom = n_eff * (n_eff + 1) / 2.0
    if denom == 0:
        return np.nan
    return 1.0 - (2.0 * T_stat / denom)


def holm_adjust(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    p_sorted = np.array([pvals[i] for i in order])
    adj_sorted = np.minimum.accumulate(((m - np.arange(m)) * p_sorted)[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.clip(adj_sorted, 0, 1)
    return adj.tolist()


def best_trio_with_max_complete(blocks_wide: pd.DataFrame,
                                sizes: Sequence[float]) -> Tuple[Optional[Tuple[float, float, float]], int]:
    best = None
    best_n = 0
    for trio in itertools.combinations(sizes, 3):
        n_complete = blocks_wide[list(trio)].notna().all(axis=1).sum()
        if n_complete > best_n:
            best = trio
            best_n = n_complete
    return best, best_n


# -----------------------------
# 4) Main
# -----------------------------
def main():
    args = parse_args()
    outdir = args.outdir

    # Load data
    df = pd.read_csv(args.input)
    required = {"model_id", "dimension", "item_id", "score_raw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Extract family and size
    fam_size = df["model_id"].apply(extract_family_and_size)
    df["family"] = [x[0] for x in fam_size]
    df["size_b"] = [x[1] for x in fam_size]

    # Keep only rows with both family and size
    df = df[df["family"].notna() & df["size_b"].notna()].copy()
    if df.empty:
        raise ValueError("No rows with recognizable family and size. Check model_id formatting.")

    # Types
    df["item_id"] = df["item_id"].astype(str)
    df["size_b"] = df["size_b"].astype(float)

    # Scale scores to [0,100]
    df["score"] = scale_to_0_100(df)

    # Family-size summary
    famsum = (
        df.groupby(["family", "size_b"])
          .agg(n_rows=("item_id", "count"), n_dimensions=("dimension", "nunique"))
          .reset_index().sort_values(["family", "size_b"])
    )
    famsum.to_csv(f"{outdir}/size_family_summary.csv", index=False)

    # Descriptives per family × dimension × size
    desc = (df.groupby(["family", "dimension", "size_b"])
              .agg(n=("score", "count"),
                   mean=("score", "mean"),
                   sd=("score", "std"))
              .reset_index())
    desc["se"] = desc["sd"] / np.sqrt(desc["n"].replace(0, np.nan))
    desc["ci95_halfwidth"] = desc["se"].apply(ci95_halfwidth_from_se)
    desc = desc.sort_values(["family", "dimension", "size_b"])
    desc.to_csv(f"{outdir}/size_summary_descriptives.csv", index=False)

    # Build long → wide by block keys
    keys = ["family", "dimension", "item_id"]
    extra_keys = [c for c in ["prompt_id", "run_id"] if c in df.columns]
    agg_keys = keys + extra_keys + ["size_b"]

    agg = (df.groupby(agg_keys, dropna=False)
             .agg(score=("score", "mean"))
             .reset_index())

    wide = agg.pivot_table(index=keys + extra_keys, columns="size_b", values="score")
    wide.reset_index().to_csv(f"{outdir}/size_agg_item_level.csv", index=False)

    # Prepare outputs
    friedman_rows = []
    pairwise_rows = []
    overlap_rows = []

    if wide.empty:
        print("[WARN] Paired matrix is empty. No inferential tests possible.")
    else:
        mi = wide.index
        lvl_names = mi.names
        try:
            idx_family = lvl_names.index("family")
            idx_dim = lvl_names.index("dimension")
        except ValueError:
            raise RuntimeError("Index is missing 'family' or 'dimension'.")

        facets = sorted(set((ix[idx_family], ix[idx_dim]) for ix in mi))

        for family, dim in facets:
            sub_idx = [ix for ix in mi if ix[idx_family] == family and ix[idx_dim] == dim]
            sub = wide.loc[sub_idx]

            # Sizes with any data in this facet
            sizes_here = [c for c in sub.columns
                          if isinstance(c, (int, float, np.floating)) and sub[c].notna().any()]
            sizes_here = sorted(sizes_here)
            if len(sizes_here) < 2:
                friedman_rows.append({
                    "family": family, "dimension": dim, "k_sizes": len(sizes_here),
                    "n_blocks": 0, "friedman_chi2": np.nan, "friedman_p": np.nan,
                    "kendalls_W": np.nan, "sizes_tested": "|".join(str(s) for s in sizes_here)
                })
                continue

            # Overlap counts for diagnostics
            for sa, sb in itertools.combinations(sizes_here, 2):
                n_pair = sub[[sa, sb]].notna().all(axis=1).sum()
                overlap_rows.append({
                    "family": family, "dimension": dim,
                    "size_a": sa, "size_b": sb,
                    "n_paired_items": int(n_pair),
                })

            # Friedman on the best trio if enough sizes and overlap
            trio, n_complete = (None, 0)
            if len(sizes_here) >= max(args.min_sizes_friedman, 3):
                trio, n_complete = best_trio_with_max_complete(sub, sizes_here)

            if trio is not None and n_complete >= args.min_blocks:
                blocks = sub[list(trio)].dropna(how="any")
                try:
                    fried = stats.friedmanchisquare(*[blocks[s].values for s in trio])
                    chi2, pval = float(fried.statistic), float(fried.pvalue)
                except Exception:
                    chi2, pval = np.nan, np.nan
                W = kendalls_w(chi2, blocks.shape[0], 3) if np.isfinite(chi2) else np.nan
                friedman_rows.append({
                    "family": family, "dimension": dim,
                    "k_sizes": 3, "n_blocks": int(blocks.shape[0]),
                    "friedman_chi2": chi2, "friedman_p": pval,
                    "kendalls_W": W, "sizes_tested": "|".join(str(s) for s in trio),
                })
            else:
                friedman_rows.append({
                    "family": family, "dimension": dim,
                    "k_sizes": len(sizes_here), "n_blocks": 0,
                    "friedman_chi2": np.nan, "friedman_p": np.nan,
                    "kendalls_W": np.nan, "sizes_tested": "|".join(str(s) for s in sizes_here),
                })

            # Pairwise Wilcoxon for any pair with enough overlap
            pvals_tmp = []
            tmp_rows = []
            for sa, sb in itertools.combinations(sizes_here, 2):
                paired = sub[[sa, sb]].dropna(how="any")
                n_pairs = int(paired.shape[0])
                if n_pairs < args.min_pairs:
                    tmp_rows.append((sa, sb, np.nan, np.nan, np.nan, n_pairs))
                    pvals_tmp.append(1.0)
                    continue
                a = paired[sa].values
                b = paired[sb].values
                diffs = a - b
                n_eff = int(np.sum(np.abs(diffs) > 0))  # exclude exact ties
                if n_eff == 0:
                    T, p, r_rb = np.nan, np.nan, np.nan
                else:
                    try:
                        w = stats.wilcoxon(a, b, zero_method="wilcox",
                                           alternative="two-sided", correction=False)
                        T, p = float(w.statistic), float(w.pvalue)
                        r_rb = rank_biserial_from_wilcoxon(T, n_eff)
                    except Exception:
                        T, p, r_rb = np.nan, np.nan, np.nan
                tmp_rows.append((sa, sb, T, p, r_rb, n_pairs))
                pvals_tmp.append(p if np.isfinite(p) else 1.0)

            # Holm correction within this facet
            if len(tmp_rows) > 0:
                p_adj = holm_adjust(pvals_tmp)
                for i, (sa, sb, T, p, r_rb, n_pairs) in enumerate(tmp_rows):
                    pairwise_rows.append({
                        "family": family, "dimension": dim,
                        "size_a": sa, "size_b": sb,
                        "wilcoxon_T": T, "p_raw": p, "p_holm": p_adj[i],
                        "rank_biserial_r": r_rb, "n_pairs": n_pairs,
                    })

    # Save outputs
    pd.DataFrame(famsum).to_csv(f"{outdir}/size_family_summary.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(f"{outdir}/size_overlap_counts.csv", index=False)
    pd.DataFrame(friedman_rows).to_csv(f"{outdir}/size_friedman_results.csv", index=False)
    pd.DataFrame(pairwise_rows).to_csv(f"{outdir}/size_pairwise_wilcoxon.csv", index=False)

    print("Done ✓")
    print(f"- Family/size summary:  {outdir}/size_family_summary.csv")
    print(f"- Descriptives:         {outdir}/size_summary_descriptives.csv")
    print(f"- Friedman results:     {outdir}/size_friedman_results.csv")
    print(f"- Pairwise Wilcoxon:    {outdir}/size_pairwise_wilcoxon.csv")
    print(f"- Overlap counts:       {outdir}/size_overlap_counts.csv")
    print(f"- Paired wide matrix:   {outdir}/size_agg_item_level.csv")
    print("\nNotes:")
    print("  • If a family has many sizes but little shared overlap, Friedman may not run.")
    print("  • Adjust thresholds with --min_blocks and --min_pairs as needed.")
    print("  • Families are inferred from model_id; check size_family_summary.csv to verify detection.")


if __name__ == "__main__":
    main()
