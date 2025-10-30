#!/usr/bin/env python3
"""
======================================================================
CULTURAL BIAS INDEX (CBI*) — MULTI-COUNTRY REFERENCE VERSION
======================================================================

HOW TO EXECUTE:

    python compute_cbi_multicountry.py \
        --models model_profiles_6D.csv \
        --references hofstede_reference.csv \
        --output Cultural_Bias_Index_MultiCountry.csv \
        --countries "Europe-France,China,USA"

DESCRIPTION:
------------
This script extends the computation of the Cultural Bias Index (CBI*) to allow
comparisons against multiple country reference profiles simultaneously.

For each model:
  1. Computes a neutral CBI (deviation from M=50, D=50).
  2. Computes a country-relative CBI for each country in --countries list.
     Normalization D is dynamically calculated per Hofstede dimension.

The result is a table of CBI metrics for multiple reference cultures, allowing
direct cross-cultural comparison of models’ inferred profiles.

Example output columns:
    Readable_Model | country_ref | CBI_neutral | CBI_France | CBI_China | CBI_USA

Author: <your name>
Date: 2025-10-28
======================================================================
"""

import argparse
import pandas as pd

# ============================================================
# PARSE COMMAND-LINE ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser(
    description="Compute Cultural Bias Index (CBI*) for multiple country references."
)
parser.add_argument("--models", required=True, help="Path to model_profiles_6D.csv")
parser.add_argument("--references", required=True, help="Path to hofstede_reference.csv")
parser.add_argument("--output", default="Cultural_Bias_Index_MultiCountry.csv", help="Output CSV file path")
parser.add_argument("--countries", required=True, help="Comma-separated list of reference countries (e.g. 'Europe-France,China,USA')")
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
countries = [c.strip() for c in args.countries.split(",")]

# ============================================================
# FILTER MODELS OF INTEREST
# ============================================================
MODELS_TO_KEEP = [
    "ibm-granite_granite-3.3-2b-instruct",
    "mistralai_Mistral-7B-Instruct-v0.3",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B"
]
df_models = df_models[df_models["model_id"].isin(MODELS_TO_KEEP)].copy()

# ============================================================
# CLEAN MODEL NAMES
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
df_models["CBI_neutral"] = (df_models[DIMENSIONS].sub(M_global).abs().sum(axis=1)) / (D_global * n)

# ============================================================
# 2. COUNTRY-RELATIVE CBI for each requested country
# ============================================================
def compute_country_relative(model_row, ref_row):
    """Compute country-relative CBI for a single model vs a single country reference."""
    diffs = []
    for dim in DIMENSIONS:
        model_val = model_row[dim]
        ref_val = ref_row[dim]
        D = max(abs(0 - ref_val), abs(100 - ref_val))
        diffs.append(abs(model_val - ref_val) / D)
    return sum(diffs) / n

for country in countries:
    if country not in df_ref["country"].values:
        print(f"⚠️ Country '{country}' not found in reference table. Skipping.")
        continue

    ref_row = df_ref[df_ref["country"] == country].iloc[0]
    df_models[f"CBI_{country.replace('-', '_')}"] = df_models.apply(lambda r: compute_country_relative(r, ref_row), axis=1)

# ============================================================
# 3. FINAL OUTPUT
# ============================================================
cols_to_export = ["Readable_Model", "country_ref", "CBI_neutral"] + [f"CBI_{c.replace('-', '_')}" for c in countries]
result = df_models[cols_to_export]

# Save
result.to_csv(args.output, index=False)
print(f"✅ Multi-country CBI computed and saved to: {args.output}")

# ============================================================
# 4. SUMMARY OUTPUT
# ============================================================
print("\nAverage CBI by model:")
print(result.to_string(index=False))
