#!/usr/bin/env python3
"""
======================================================================
CULTURAL BIAS INDEX (CBI*) COMPUTATION — WITH ARGPARSE
======================================================================

HOW TO EXECUTE:

    python compute_cbi.py \
        --models model_profiles_6D.csv \
        --references hofstede_reference.csv \
        --output Cultural_Bias_Index_Full.csv

DESCRIPTION:
------------
This script computes two variants of the Cultural Bias Index (CBI*):

1. Neutral CBI:
   Measures deviation from a universal neutrality point (M=50, D=50).

   \[
   \text{CBI}_{neutral}^{*} = \frac{1}{Dn} \sum_{k=1}^{n} |x_{ik} - M|
   \]

2. Country-relative CBI:
   Measures deviation relative to each model’s country reference profile.
   Normalization adjusts dynamically as

   \[
   D = \max(|0 - M_{country}|, |100 - M_{country}|)
   \]

   ensuring results are bounded in [0,1].

The output CSV includes both indices and a per-country summary.

Author: <your name>
Date: 2025-10-20
======================================================================
"""

import argparse
import pandas as pd

# ============================================================
# PARSE COMMAND-LINE ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser(
    description="Compute Cultural Bias Index (CBI*) for models using Hofstede dimensions."
)
parser.add_argument("--models", required=True, help="Path to model_profiles_6D.csv")
parser.add_argument("--references", required=True, help="Path to hofstede_reference.csv")
parser.add_argument("--output", default="Cultural_Bias_Index_Full.csv", help="Output CSV file path")

args = parser.parse_args()

# ============================================================
# CONFIGURATION
# ============================================================
DIMENSIONS = [
    "Power_Distance",
    "Individualism",
    "Masculinity",
    "Uncertainty_Avoidance",
    "Long_Term_Orientation",
    "Indulgence",
]
M_global = 50
D_global = 50
n = len(DIMENSIONS)

# ============================================================
# LOAD DATA
# ============================================================
df_models = pd.read_csv(args.models)
df_ref = pd.read_csv(args.references)

# ============================================================
# CLEAN MODEL NAMES (optional)
# ============================================================
def clean_model_name(name: str) -> str:
    name = name.replace("_", " ").replace("-", " ")
    parts = name.split()
    parts = [p.capitalize() for p in parts if p.lower() not in ["ai", "instruct", "it", "v0.3", "r1", "distill"]]
    return " ".join(parts)

df_models["Readable_Model"] = df_models["model_id"].apply(clean_model_name)

# ============================================================
# 1. NEUTRAL CBI (M=50, D=50)
# ============================================================
df_models["CBI_neutral*"] = (df_models[DIMENSIONS].sub(M_global).abs().sum(axis=1)) / (D_global * n)

# ============================================================
# 2. COUNTRY-RELATIVE CBI (M_country, dynamic D)
# ============================================================
merged = pd.merge(df_models, df_ref, left_on="country_ref", right_on="country", suffixes=("_model", "_ref"))

def compute_country_relative_cbi(row):
    diffs = []
    for dim in DIMENSIONS:
        model_val = row[f"{dim}_model"]
        ref_val = row[f"{dim}_ref"]
        D = max(abs(0 - ref_val), abs(100 - ref_val))
        diffs.append(abs(model_val - ref_val) / D)
    return sum(diffs) / n

merged["CBI_country_relative*"] = merged.apply(compute_country_relative_cbi, axis=1)

# ============================================================
# 3. FINAL OUTPUT
# ============================================================
result = merged[[
    "Readable_Model",
    "country_ref",
    "CBI_neutral*",
    "CBI_country_relative*"
]].sort_values("CBI_country_relative*", ascending=False)

# Save
result.to_csv(args.output, index=False)
print(f"✅ CBI values computed and saved to: {args.output}")

# ============================================================
# 4. SUMMARY BY COUNTRY
# ============================================================
summary = result.groupby("country_ref")[["CBI_neutral*", "CBI_country_relative*"]].mean().reset_index()
print("\nAverage CBI per country:")
print(summary.to_string(index=False))
