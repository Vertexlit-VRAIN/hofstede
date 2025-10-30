#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusiona varios CSV, elimina 'time_stamp' (si existe) y añade una columna 'temperature'
específica para cada fichero.

Cómo usar:
1) Edita la lista FILES con tus rutas y temperaturas.
2) Edita OUTPUT con la ruta de salida deseada.
3) Ejecuta:  python merge_by_file_temperature.py

Requisitos: pandas, numpy
    pip install pandas numpy
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Tuple

FILES: List[Tuple[str, float]] = [
    ("./models_results_0_5.csv", 0.5),
    ("./models_results_1_0.csv", 1),
    ("./models_results_0.csv", 0)
]

# Ruta del CSV de salida
OUTPUT = "merged.csv"

# Columnas esperadas (sin 'time_stamp', que se elimina)
EXPECTED_COLUMNS = [
    "model_id","size","params_log","country_ref","dimension",
    "item_id","score_raw","scale_min","scale_max","run_id","prompt_id"
]

def main():
    if not FILES:
        sys.stderr.write("Configura la lista FILES con (path, temperature).\n")
        sys.exit(1)

    dfs = []
    for path, temp in FILES:
        if not isinstance(temp, (int, float)):
            sys.stderr.write(f"[ERROR] Temperatura no numérica para '{path}': {temp}\n")
            sys.exit(1)

        if not os.path.exists(path):
            sys.stderr.write(f"[ERROR] No existe el fichero: {path}\n")
            sys.exit(1)

        try:
            df = pd.read_csv(path)
        except Exception as e:
            sys.stderr.write(f"[ERROR] Al leer '{path}': {e}\n")
            sys.exit(1)

        # Eliminar 'time_stamp' si está presente
        if "time_stamp" in df.columns:
            df = df.drop(columns=["time_stamp"])

        # Añadir temperatura del fichero
        df["temperature"] = float(temp)

        dfs.append(df)

    # Concatenar todos
    if not dfs:
        sys.stderr.write("[ERROR] No se cargó ningún DataFrame.\n")
        sys.exit(1)

    merged = pd.concat(dfs, ignore_index=True, sort=False)

    # Preparar orden de columnas: (las esperadas que existan) + temperature
    ordered_cols = [c for c in EXPECTED_COLUMNS if c in merged.columns] + ["temperature"]

    # Si hay otras columnas extra, las colocamos después
    other_cols = [c for c in merged.columns if c not in ordered_cols]
    final_cols = ordered_cols + other_cols

    merged = merged[final_cols]

    # Guardar
    try:
        merged.to_csv(OUTPUT, index=False)
    except Exception as e:
        sys.stderr.write(f"[ERROR] No se pudo escribir '{OUTPUT}': {e}\n")
        sys.exit(1)

    # Resumen rápido
    n_rows = len(merged)
    temps = merged["temperature"].unique()
    print(f"✅ Guardado '{OUTPUT}' con {n_rows} filas.")
    print(f"   Temperaturas presentes: {np.array2string(temps, precision=3)}")
    print(f"   Columnas: {final_cols}")

if __name__ == "__main__":
    main()
