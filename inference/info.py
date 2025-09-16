#!/usr/bin/env python3
"""
Scan Hofstede dataset to extract unique domains and dimension codes with counts.

Works with CSV, JSON, JSONL.
Requires pandas.
"""

import os
import json
import argparse
import pandas as pd

def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in [".json", ".jsonl"]:
        # Try JSONL
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if ext == ".jsonl" or (text and "\n" in text and text.lstrip().startswith("{")):
            records = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    line = line.strip("`")
                    obj = json.loads(line)
                records.append(obj)
            df = pd.DataFrame.from_records(records)
        else:
            data = json.loads(text)
            if isinstance(data, list):
                df = pd.DataFrame.from_records(data)
            elif isinstance(data, dict) and "rows" in data:
                df = pd.DataFrame.from_records(data["rows"])
            else:
                df = pd.DataFrame([data])
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return df

def main(input_path: str):
    df = read_any(input_path)
    if "domain" not in df.columns or "dimension_code" not in df.columns:
        raise ValueError("Dataset must contain 'domain' and 'dimension_code' columns")

    print("\nUnique Domains:")
    print(df["domain"].value_counts())

    print("\nUnique Dimension Codes:")
    print(df["dimension_code"].value_counts())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input_path", help="Path to dataset (CSV/JSON/JSONL)")
    args = p.parse_args()
    main(args.input_path)
