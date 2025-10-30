#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analiza si cambiar la temperatura de inferencia del LLM produce diferencias
ESTADÍSTICAMENTE SIGNIFICATIVAS en las dimensiones de Hofstede.

Se trabaja por **modelo** y **dimensión**, usando los 150 ítems como medidas
repetidas (bloques pareados) a través de las temperaturas.

Entrada (CSV con columnas):
    model_id,size,params_log,country_ref,dimension,item_id,score_raw,
    scale_min,scale_max,run_id,prompt_id,temperature

Ejemplo:
    Qwen_Qwen3-32B,XL,10.505149978319906,China,Individualism,0,2,1,5,1,1,0.5

Qué calcula:
  1) Escala score_raw a [0,100] usando scale_min/max si están presentes.
  2) Descriptivos por (modelo, dimensión, temperatura): media, sd, se, IC95%.
  3) Test global de medidas repetidas por faceta (modelo×dimensión):
        • FRIEDMAN (no paramétrico; k ≥ 3 temperaturas)
        • Tamaño de efecto: Kendall’s W
  4) Post-hoc pareados entre temperaturas con:
        • Wilcoxon (signed-rank, two-sided)
        • Ajuste de p-valores por Holm
        • Tamaño de efecto: rank-biserial r
  5) Exporta CSVs:
        - summary_descriptives.csv
        - friedman_results.csv
        - pairwise_wilcoxon.csv
        - agg_item_level.csv    (bloques pareados por ítem para trazado/QA)

Uso:
    python analyze_temperature_significance.py \
        --input models_results.csv \
        --outdir ./salidas \
        --temps 0 0.5 1

Notas importantes:
- Este script **NO** cambia el nombre de los modelos. Analiza EXACTAMENTE estos:
    Qwen_Qwen3-32B
    mistralai_Mistral-7B-Instruct-v0.3
    google_gemma-3-27b-it
    deepseek-ai_DeepSeek-R1-Distill-Qwen-32B
- Si faltan observaciones para alguna temperatura en ciertos ítems, esos ítems
  se excluyen del test inferencial (pero cuentan en descriptivos).
- Requisitos: pandas, numpy, scipy
"""

import argparse
import itertools
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats


TARGET_MODELS = {
    "Qwen_Qwen3-32B",
    "mistralai_Mistral-7B-Instruct-v0.3",
    "google_gemma-3-27b-it",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B",
}


# -----------------------
# Utilidades estadísticas
# -----------------------

def scale_to_0_100(df: pd.DataFrame) -> pd.Series:
    """
    Escala score_raw a [0,100] cuando hay scale_min/max; si no, asume ya 0-100.
    """
    if "score_raw" not in df.columns:
        raise ValueError("Falta la columna 'score_raw' en el CSV.")
    if {"scale_min", "scale_max"}.issubset(df.columns):
        rng = (df["scale_max"] - df["scale_min"]).replace(0, np.nan)
        scaled = 100 * (df["score_raw"] - df["scale_min"]) / rng
        scaled = scaled.fillna(df["score_raw"]).clip(0, 100)
        return scaled
    return df["score_raw"].clip(0, 100)


def ci95_halfwidth_from_se(se: float) -> float:
    # IC95% aproximado: ±1.96 * SE (válido con n≥30)
    return 1.96 * se if se is not None and np.isfinite(se) else np.nan


def kendalls_w(chi2: float, n_blocks: int, k_conditions: int) -> float:
    # W = chi2 / (n * (k - 1))
    if n_blocks <= 0 or k_conditions <= 1 or not np.isfinite(chi2):
        return np.nan
    return chi2 / (n_blocks * (k_conditions - 1))


def rank_biserial_from_wilcoxon(T_stat: float, n_eff: int) -> Optional[float]:
    """
    Aproxima el rank-biserial r para Wilcoxon:
      r_rb ≈ 1 - (2*T / (n*(n+1)/2))
    n_eff = número de pares con diferencia != 0
    """
    if n_eff is None or n_eff <= 0 or not np.isfinite(T_stat):
        return np.nan
    denom = n_eff * (n_eff + 1) / 2.0
    if denom == 0:
        return np.nan
    return 1.0 - (2.0 * T_stat / denom)


def holm_adjust(pvals: List[float]) -> List[float]:
    """
    Ajuste Holm-Bonferroni para un conjunto de p-valores (misma familia).
    Devuelve los p ajustados en el orden original.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    p_sorted = np.array([pvals[i] for i in order])
    adj_sorted = np.minimum.accumulate(((m - np.arange(m)) * p_sorted)[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.clip(adj_sorted, 0, 1)
    return adj.tolist()


# --------------
# Script principal
# --------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Ruta al CSV (models_results.csv)")
    p.add_argument("--outdir", required=True, help="Directorio de salida")
    p.add_argument("--temps", type=float, nargs="*", default=None,
                   help="Lista de temperaturas a considerar (por ej. --temps 0 0.5 1). "
                        "Si se omite, se infiere del CSV.")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = args.outdir

    df = pd.read_csv(args.input)

    # Compatibilidad: 'temperature' o 'Temperature'
    if "temperature" in df.columns:
        df["temperature"] = df["temperature"].astype(float)
    elif "Temperature" in df.columns:
        df = df.rename(columns={"Temperature": "temperature"})
        df["temperature"] = df["temperature"].astype(float)
    else:
        raise ValueError("No se encontró la columna 'temperature' ni 'Temperature'.")

    # Filtra a los 4 modelos indicados (exact match)
    if "model_id" not in df.columns:
        raise ValueError("Falta la columna 'model_id'.")
    df = df[df["model_id"].isin(TARGET_MODELS)].copy()
    if df.empty:
        raise ValueError("Tras filtrar por los 4 modelos, no quedan filas. "
                         "Revisa 'model_id' en tu CSV.")

    # Tipos mínimos
    for col in ["dimension", "item_id"]:
        if col not in df.columns:
            raise ValueError(f"Falta la columna '{col}'.")
    df["item_id"] = df["item_id"].astype(str)

    # Escalar a [0,100]
    df["score"] = scale_to_0_100(df)

    # Temperaturas a considerar
    temps = sorted(df["temperature"].unique().tolist()) if args.temps is None else sorted(args.temps)
    if len(temps) < 2:
        raise ValueError("Se requieren al menos 2 temperaturas para comparar.")
    temp_pairs = list(itertools.combinations(temps, 2))

    # -------------------------------
    # 1) DESCRIPTIVOS por temperatura
    # -------------------------------
    desc = (df.groupby(["model_id", "dimension", "temperature"])
              .agg(n=("score", "count"),
                   mean=("score", "mean"),
                   sd=("score", "std"))
              .reset_index())
    desc["se"] = desc["sd"] / np.sqrt(desc["n"].replace(0, np.nan))
    desc["ci95_halfwidth"] = desc["se"].apply(ci95_halfwidth_from_se)
    desc = desc.sort_values(["model_id", "dimension", "temperature"])
    desc.to_csv(f"{outdir}/summary_descriptives.csv", index=False)

    # ------------------------------------------------------
    # 2) MATRIZ PAREADA por bloque (modelo, dimensión, ítem)
    # ------------------------------------------------------
    keys = ["model_id", "dimension", "item_id"]
    extra_keys = [c for c in ["prompt_id", "run_id"] if c in df.columns]
    agg_keys = keys + extra_keys + ["temperature"]

    agg = (df.groupby(agg_keys, dropna=False)
             .agg(score=("score", "mean"))
             .reset_index())

    wide = agg.pivot_table(index=keys + extra_keys, columns="temperature", values="score")
    agg_out = wide.reset_index()
    agg_out.to_csv(f"{outdir}/agg_item_level.csv", index=False)

    # Bloques completos: tienen todas las temperaturas presentes
    has_all_cols = all(t in wide.columns for t in temps)
    if not has_all_cols:
        # Si falta alguna temperatura por completo, no se puede testear esa faceta
        missing_ts = [t for t in temps if t not in wide.columns]
        print(f"[AVISO] No hay datos para estas temperaturas: {missing_ts}. "
              f"Se continuará con las disponibles para descriptivos, "
              f"pero las facetas sin k≥2 no podrán testearse.")
    mask_complete = wide[temps].notna().all(axis=1) if has_all_cols else pd.Series(False, index=wide.index)
    wide_complete = wide.loc[mask_complete].copy()

    # ------------------------------------------------------
    # 3) FRIEDMAN (global) y WILCOXON (post-hoc) por faceta
    # ------------------------------------------------------
    friedman_rows = []
    pairwise_rows = []

    # iterar por (modelo, dimensión)
    # (usamos el MultiIndex de wide_complete para seleccionar submatrices)
    if wide_complete.empty:
        print("[AVISO] No hay bloques completos con todas las temperaturas. "
              "No se podrán realizar tests inferenciales. "
              "Revisa que cada ítem tenga puntajes a TODAS las temperaturas.")
    else:
        # Ubicaciones de niveles para 'model_id' y 'dimension'
        mi = wide_complete.index
        level_names = mi.names
        try:
            idx_model = level_names.index("model_id")
            idx_dim = level_names.index("dimension")
        except ValueError:
            raise RuntimeError("No se encontró 'model_id' o 'dimension' en el índice de bloques.")

        # Recopila todos los pares (model_id, dimension) presentes
        facets = sorted(set((ix[idx_model], ix[idx_dim]) for ix in mi))

        for model, dim in facets:
            sub_idx = [ix for ix in mi if ix[idx_model] == model and ix[idx_dim] == dim]
            blocks = wide_complete.loc[sub_idx, temps]  # shape: n_blocks × k
            n_blocks = blocks.shape[0]
            k = len(temps)

            # Friedman (solo si k >= 3 y n_blocks >= 2)
            if k >= 3 and n_blocks >= 2:
                try:
                    fried = stats.friedmanchisquare(*[blocks[t].values for t in temps])
                    chi2, pval = float(fried.statistic), float(fried.pvalue)
                except Exception:
                    chi2, pval = np.nan, np.nan
            else:
                chi2, pval = np.nan, np.nan

            W = kendalls_w(chi2, n_blocks, k) if np.isfinite(chi2) else np.nan

            friedman_rows.append({
                "model_id": model,
                "dimension": dim,
                "k_temps": k,
                "n_blocks": int(n_blocks),
                "friedman_chi2": chi2,
                "friedman_p": pval,
                "kendalls_W": W,
            })

            # Pairwise Wilcoxon (si hay ≥ 2 temperaturas y ≥ 1 bloque)
            if n_blocks >= 1 and k >= 2:
                pvals_raw = []
                tmp_rows = []
                for (ta, tb) in temp_pairs:
                    if ta not in blocks.columns or tb not in blocks.columns:
                        T, p, r_rb, n_eff = np.nan, np.nan, np.nan, 0
                    else:
                        a = blocks[ta].values
                        b = blocks[tb].values
                        # Wilcoxon ignora diferencias exactamente 0; contamos pares efectivos
                        diffs = a - b
                        n_eff = int(np.sum(np.abs(diffs) > 0))
                        if n_eff == 0:
                            T, p, r_rb = np.nan, np.nan, np.nan
                        else:
                            try:
                                w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", correction=False)
                                T, p = float(w.statistic), float(w.pvalue)
                                r_rb = rank_biserial_from_wilcoxon(T, n_eff)
                            except Exception:
                                T, p, r_rb = np.nan, np.nan, np.nan
                    tmp_rows.append((ta, tb, T, p, r_rb, n_eff))
                    pvals_raw.append(p if np.isfinite(p) else 1.0)

                # Ajuste Holm dentro de la faceta
                pvals_holm = holm_adjust(pvals_raw) if len(pvals_raw) else []

                for i, (ta, tb, T, p, r_rb, n_eff) in enumerate(tmp_rows):
                    pairwise_rows.append({
                        "model_id": model,
                        "dimension": dim,
                        "temp_a": ta,
                        "temp_b": tb,
                        "wilcoxon_T": T,
                        "p_raw": p,
                        "p_holm": pvals_holm[i] if pvals_holm else np.nan,
                        "rank_biserial_r": r_rb,
                        "n_pairs": int(n_blocks),
                    })

    # Guardar resultados
    pd.DataFrame(friedman_rows).to_csv(f"{outdir}/friedman_results.csv", index=False)
    pd.DataFrame(pairwise_rows).to_csv(f"{outdir}/pairwise_wilcoxon.csv", index=False)

    # Mensaje final en consola
    print("Listo ✅")
    print(f"- Descriptivos: {outdir}/summary_descriptives.csv")
    print(f"- Friedman:     {outdir}/friedman_results.csv")
    print(f"- Wilcoxon:     {outdir}/pairwise_wilcoxon.csv")
    print(f"- Bloques agg:  {outdir}/agg_item_level.csv")
    print("\nCómo justificar 'significativo':")
    print("  • Si en friedman_results.csv el p<0.05 → hay efecto de temperatura (global).")
    print("  • Revisa pairwise_wilcoxon.csv: p_holm<0.05 indica qué pares de temperaturas difieren.")
    print("  • Reporta también tamaños de efecto: Kendall’s W (global) y r (pareado).")


if __name__ == "__main__":
    main()
