#!/usr/bin/env python3
# eval_metrics_folder.py
import json
import sys
import pathlib
import collections
import csv

def iter_jsonl(folder):
    """Iterate through all JSONL files in the given folder."""
    folder = pathlib.Path(folder)
    for path in folder.glob("*.jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    obj["__file__"] = str(path)
                    obj["__line__"] = ln
                    yield obj
                except json.JSONDecodeError as e:
                    yield {"__file__": str(path), "__line__": ln, "__parse_error__": str(e)}

def check_format_compliance(rec):
    """Validates that model_output_parsed follows the strict Hofstede prompt rules."""
    parsed = rec.get("model_output_parsed")
    if parsed is None:
        return False, "missing model_output_parsed"
    if not isinstance(parsed, list):
        return False, "not a list"
    if len(parsed) != 5:
        return False, f"expected 5 items, got {len(parsed)}"
    seen_levels = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            return False, f"item {i} not a dict"
        if set(item.keys()) != {"statement", "level"}:
            return False, f"item {i} keys mismatch"
        if not isinstance(item["statement"], str) or not item["statement"]:
            return False, f"item {i} empty statement"
        if not isinstance(item["level"], int):
            return False, f"item {i} level not int"
        seen_levels.append(item["level"])
    if seen_levels != [1, 2, 3, 4, 5]:
        return False, f"levels invalid: {seen_levels}"
    return True, ""

def normalize_eval(e):
    if not e:
        return None
    e = str(e).strip().lower()
    mapping = {"good": "good", "correct": "good", "pass": "good",
               "bad": "bad", "incorrect": "bad", "fail": "bad"}
    return mapping.get(e, e)

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_metrics_folder.py <folder_path> [--out-csv output.csv]")
        sys.exit(1)

    folder = pathlib.Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory.")
        sys.exit(1)

    out_csv = None
    if "--out-csv" in sys.argv:
        i = sys.argv.index("--out-csv")
        if i + 1 < len(sys.argv):
            out_csv = sys.argv[i + 1]

    counts = collections.Counter()
    format_ok = format_bad = 0
    error_present = parse_fail = total = 0
    per_record = []

    for rec in iter_jsonl(folder):
        total += 1
        if "__parse_error__" in rec:
            parse_fail += 1
            per_record.append({
                "file": rec["__file__"], "line": rec["__line__"],
                "evaluation": "", "error_present": "parse_error",
                "format_ok": False, "format_reason": rec["__parse_error__"]
            })
            continue

        ev = normalize_eval(rec.get("evaluation"))
        if ev:
            counts[ev] += 1
        if rec.get("error") is not None:
            error_present += 1

        ok, reason = check_format_compliance(rec)
        if ok:
            format_ok += 1
        else:
            format_bad += 1

        per_record.append({
            "file": rec["__file__"], "line": rec["__line__"],
            "evaluation": ev or "", "error_present": "yes" if rec.get("error") else "no",
            "format_ok": ok, "format_reason": "" if ok else reason
        })

    print(f"\nFolder: {folder}")
    print(f"Total records: {total}")
    print(f"Parse errors: {parse_fail}")
    print(f"Records with error field: {error_present}")
    print(f"Format OK: {format_ok}")
    print(f"Format violations: {format_bad}")

    good, bad = counts["good"], counts["bad"]
    total_eval = good + bad
    if total_eval:
        print(f"Accuracy (good/total): {good}/{total_eval} = {good/total_eval:.3f}")
    else:
        print("No evaluation labels found.")

    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=per_record[0].keys())
            writer.writeheader()
            writer.writerows(per_record)
        print(f"Saved detailed CSV report to {out_csv}")

if __name__ == "__main__":
    main()
