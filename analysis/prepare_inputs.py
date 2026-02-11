#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_inputs.py

Genera los CSV de entrada para tu pipeline analysis_v2:
  1) data/hofstede_reference.csv     (con China, USA, Europe-France)
  2) data/models_results_<stamp>.csv (a partir de tus JSONL por modelo + validated_data.jsonl)

USO:
------
# a) Solo crear hofstede_reference.csv
python prepare_inputs.py --write_hofstede --out_dir analysis_v2/data

# b) Convertir tus JSONL a models_results.csv
python prepare_inputs.py \
  --convert_jsonl \
  --models_root /ruta/a/tus_modelos \
  --validated_jsonl /ruta/a/validated_data.jsonl \
  --out_csv analysis_v2/data/models_results_real.csv

# c) Hacer ambas cosas (si no existe hofstede_reference.csv lo crea)
python prepare_inputs.py \
  --write_hofstede \
  --convert_jsonl \
  --models_root /ruta/a/tus_modelos \
  --validated_jsonl /ruta/a/validated_data.jsonl \
  --out_csv analysis_v2/data/models_results_real.csv \
  --out_dir analysis_v2/data

Suposiciones:
- Estructura por modelo:
    <model_dir>/
      inference_results_run_1.jsonl
      inference_results_run_2.jsonl
      inference_results_run_3.jsonl
- Cada línea JSONL tiene:
    original_index, question, selected_level (1..5), parsed_output, inference_run (opcional)
- validated_data.jsonl tiene:
    split, index, question, domain, dimension_code ∈ {IDV, IVR, LTO, MAS, PDI, UAI}
- score_raw = selected_level; scale_min=1; scale_max=5.
- El pipeline escalará a 0–100 automáticamente.

Edita los diccionarios VENDOR_PATTERNS / VENDOR_COUNTRY / SIZE_BUCKETS si necesitas ajustes.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import math


# ============================
# Configuraciones editables
# ============================

# Dimensiones Hofstede → nombres de columna esperados por analysis_v2
DIM_MAP = {
    "IDV": "Individualism",
    "IVR": "Indulgence",
    "LTO": "Long_Term_Orientation",
    "MAS": "Masculinity",
    "PDI": "Power_Distance",
    "UAI": "Uncertainty_Avoidance",
}

# Nombres y valores de referencia (0–100)
# Nota: el pipeline espera el país europeo como "Europe-France"
HOFSTEDE_REFERENCE = [
    {"country":"China",          "Power_Distance":80, "Individualism":43, "Masculinity":66, "Uncertainty_Avoidance":30, "Long_Term_Orientation":77, "Indulgence":24},
    {"country":"USA",            "Power_Distance":40, "Individualism":60, "Masculinity":62, "Uncertainty_Avoidance":46, "Long_Term_Orientation":50, "Indulgence":68},
    {"country":"Europe-France",  "Power_Distance":68, "Individualism":74, "Masculinity":43, "Uncertainty_Avoidance":86, "Long_Term_Orientation":60, "Indulgence":48},
]

# Detección de vendor por nombre de carpeta
VENDOR_PATTERNS = [
    (r"^deepseek-ai", "deepseek"),
    (r"^Qwen", "qwen"),
    (r"^google", "google"),
    (r"^ibm-granite", "ibm"),
    (r"^meta-llama", "meta"),
    (r"^microsoft[_-]?phi", "microsoft-phi"),
    (r"^mistral|^mistralai", "mistral"),
]

# País de referencia por vendor
VENDOR_COUNTRY = {
    "deepseek": "China",
    "qwen": "China",
    "google": "USA",
    "ibm": "USA",
    "meta": "USA",
    "microsoft-phi": "USA",
    "mistral": "Europe-France",
}

# Buckets de tamaño (en Billions) → etiqueta
SIZE_BUCKETS = [
    (1.0,  "XS"),
    (2.0,  "S"),
    (4.0,  "S"),
    (7.0,  "M"),
    (8.0,  "M"),
    (12.0, "L"),
    (14.0, "L"),
    (27.0, "XL"),
    (32.0, "XL"),
    (70.0, "XXL"),
    (405.0,"XXXL"),
]


# ============================
# Utilidades
# ============================

def detect_vendor(dirname: str):
    for pat, vendor in VENDOR_PATTERNS:
        if re.search(pat, dirname, re.IGNORECASE):
            return vendor
    return None

def parse_params_and_size(model_dirname: str):
    """
    Extrae numero de parámetros (Billions) de tokens tipo '0.6B', '1b', '7B', '32b', '70B', '405B'
    Devuelve (params_b, params_log10, size_label).
    """
    m = re.search(r'(\d+(\.\d+)?)\s*[bB]\b', model_dirname)
    if not m:
        return (None, None, "NA")
    b = float(m.group(1))
    plog = math.log10(b*1e9) if b > 0 else None
    label = "NA"
    for cutoff, lab in SIZE_BUCKETS:
        if b <= cutoff:
            label = lab
            break
    return (b, plog, label)

def infer_run_from_filename(name: str):
    m = re.search(r'run_(\d+)\.jsonl$', name)
    return int(m.group(1)) if m else None

def load_validated(path: Path):
    """
    Carga validated_data.jsonl en un dict: key=original_index (o index) -> objeto
    """
    gold = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            idx = obj.get("original_index", obj.get("index"))
            if idx is not None:
                gold[idx] = obj
    return gold


# ============================
# Acciones principales
# ============================

def write_hofstede(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(HOFSTEDE_REFERENCE)
    out_path = out_dir / "hofstede_reference.csv"
    df.to_csv(out_path, index=False)
    print(f"✓ Escrito: {out_path}")
    return out_path

def convert_jsonl_to_models_csv(models_root: Path, validated_jsonl: Path, out_csv: Path):
    validated = load_validated(validated_jsonl)
    if not validated:
        print("ERROR: validated_data.jsonl vacío o ilegible.", file=sys.stderr)
        sys.exit(2)

    rows = []
    now_iso = datetime.now().isoformat(timespec="seconds")

    model_dirs = [p for p in models_root.iterdir() if p.is_dir()]
    if not model_dirs:
        print(f"ERROR: No se encontraron subdirectorios en {models_root}", file=sys.stderr)
        sys.exit(2)

    for model_dir in sorted(model_dirs):
        vendor = detect_vendor(model_dir.name)
        if not vendor:
            # Si quieres incluir vendors desconocidos, reemplaza continue por asignaciones por defecto
            # p.ej., country_ref="Unknown"
            # Aquí omitimos para evitar contaminar análisis
            continue
        country_ref = VENDOR_COUNTRY.get(vendor, "Unknown")
        params_b, params_log10, size_label = parse_params_and_size(model_dir.name)

        jsonl_files = sorted(model_dir.glob("inference_results_run_*.jsonl"))
        if not jsonl_files:
            # Si tu naming es distinto, ajusta el patrón
            continue

        for jsonl in jsonl_files:
            run_id = infer_run_from_filename(jsonl.name) or 1
            with open(jsonl, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)

                    # Campos base
                    oi = rec.get("original_index")
                    selected = rec.get("selected_level")
                    if oi is None or selected is None:
                        # registros incompletos → saltar
                        continue

                    gold = validated.get(oi, {})
                    dim_code = gold.get("dimension_code")
                    dimension = DIM_MAP.get(dim_code)
                    if dimension is None:
                        # Si falta el code o es desconocido, lo saltamos
                        continue

                    rows.append({
                        "model_id": model_dir.name,
                        "size": size_label,
                        "params_log": params_log10,
                        "country_ref": country_ref,
                        "dimension": dimension,
                        "item_id": oi,
                        "score_raw": int(selected),
                        "scale_min": 1,
                        "scale_max": 5,
                        "run_id": int(run_id),
                        "prompt_id": 1,
                        "time_stamp": now_iso,
                    })

    if not rows:
        print("ERROR: No se generaron filas. Revisa rutas y patrones de archivo.", file=sys.stderr)
        sys.exit(2)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✓ Escrito: {out_csv} ({len(rows)} filas)")
    return out_csv


# ============================
# CLI
# ============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write_hofstede", action="store_true",
                    help="Escribe data/hofstede_reference.csv con China, USA, Europe-France")
    ap.add_argument("--convert_jsonl", action="store_true",
                    help="Convierte tus JSONL de inferencia a data/models_results_*.csv")
    ap.add_argument("--models_root", type=Path,
                    help="Directorio que contiene una carpeta por modelo")
    ap.add_argument("--validated_jsonl", type=Path,
                    help="Ruta a validated_data.jsonl")
    ap.add_argument("--out_csv", type=Path,
                    help="Ruta de salida para models_results CSV")
    ap.add_argument("--out_dir", type=Path, default=Path("analysis_v2/data"),
                    help="Directorio de salida para hofstede_reference.csv (por defecto analysis_v2/data)")
    args = ap.parse_args()

    if not args.write_hofstede and not args.convert_jsonl:
        print("Nada que hacer. Pasa --write_hofstede y/o --convert_jsonl", file=sys.stderr)
        sys.exit(1)

    if args.write_hofstede:
        write_hofstede(args.out_dir)

    if args.convert_jsonl:
        if not args.models_root or not args.validated_jsonl or not args.out_csv:
            print("Faltan argumentos: --models_root --validated_jsonl --out_csv", file=sys.stderr)
            sys.exit(1)
        convert_jsonl_to_models_csv(args.models_root, args.validated_jsonl, args.out_csv)


if __name__ == "__main__":
    main()
