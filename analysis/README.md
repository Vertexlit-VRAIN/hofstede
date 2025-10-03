# Hofstede LLM Proximity – Pipeline

Este proyecto calcula **métricas de proximidad cultural** entre perfiles de **LLMs** y **referencias Hofstede** (0–100), y genera las **figuras** para el paper.

## 📁 Estructura
```
hofstede_llm_proximity/
├─ data/
│  ├─ hofstede_reference.csv         # China, USA, Europe-France (según lo aportado)
│  ├─ models_results_template.csv    # Plantilla de entradas por ítem
│  └─ example_models_results.csv     # Ejemplo reproducible
├─ src/
│  ├─ compute_metrics.py             # Cálculo de métricas y deltas
│  ├─ plot_figures.py                # Gráficas (radar, heatmap, PCA, scatter)
│  └─ run_all.py                     # Orquestador
├─ outputs/                          # CSVs con resultados
└─ figures/                          # PNGs con figuras
```

## 🧮 Entrada esperada (`models_results_*.csv`)
Formato **tidy** por respuesta/ítem:
- `model_id` (p.ej. LLM-A)
- `size` (S, M, L, XL)
- `params_log` (log10 de parámetros; opcional pero útil)
- `country_ref` (China, USA, Europe-France, …)
- `dimension` (Power_Distance, Individualism, Masculinity, Uncertainty_Avoidance, Long_Term_Orientation, Indulgence)
- `item_id`
- `score_raw` (escala original p.ej. 1–7)
- `scale_min`, `scale_max`
- `run_id`, `prompt_id`, `time_stamp`

> El script **escala automáticamente** a 0–100 con: `100*(raw-min)/(max-min)`. Si ya tienes `score_scaled`, déjalo en el CSV y se utilizará tal cual.

## 🔧 Instalación rápida
Requiere Python 3.10+.
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## ▶️ Ejecución
Con los datos de ejemplo:
```bash
python src/run_all.py
```
o de forma manual:
```bash
# 1) Métricas
python src/compute_metrics.py --models_csv data/example_models_results.csv --hofstede_csv data/hofstede_reference.csv --out_dir outputs

# 2) Figuras
python src/plot_figures.py   --model_profiles_csv outputs/model_profiles_6D.csv   --metrics_csv outputs/metrics_per_model_country.csv   --deltas_csv outputs/per_dimension_deltas.csv   --hofstede_csv data/hofstede_reference.csv   --out_dir figures
```

## 📈 Qué genera

**CSVs (en `outputs/`):**
- `model_profiles_6D.csv` → perfiles promedio por modelo×tamaño×país (0–100, 6D).
- `metrics_per_model_country.csv` → MAE, RMSE, L2, Cosine, Pearson_r, Bias por modelo×país.
- `per_dimension_deltas.csv` → Δ y |Δ| por dimensión.
- `same_model_different_sizes.csv` → comparación **mismo modelo, tamaños distintos**.
- `all_models_rank_by_country.csv` → ranking de **todos los modelos** por país.
- `country_group_comparisons.csv` → comparaciones específicas (Europa, media China, media USA).
- `usa_models_comparison.csv` → **comparación entre modelos de USA**.

**Figuras (en `figures/`):**
- `radar_<country>.png` → Perfil país vs modelos que referencian ese país.
- `heatmap_abs_deltas.png` → Heatmap de errores absolutos por dimensión.
- `pca_map.png` → Mapa de proximidad 2D (PCA) con países + modelos.
- `size_vs_mae.png` → Dispersión tamaño (categoría) vs MAE.

## 🔍 Responde a tus comparaciones
- **¿Se compara el mismo modelo de diferente tamaño?** Sí → `same_model_different_sizes.csv` + figuras (scatter).
- **Comparación de todos los modelos** → `all_models_rank_by_country.csv` y `pca_map.png`.
- **Comparación específica de países (Europa, media China, media USA)** → `country_group_comparisons.csv`.
  - Con los datos actuales: Europa ≡ Europe-France; media China ≡ China; media USA ≡ USA. Si añades más países por grupo, el script calculará la media automáticamente.
- **Comparación entre modelos para USA** → `usa_models_comparison.csv`.

## 🧪 Notas metodológicas
- Métricas: MAE, RMSE, L2, Cosine, Pearson r, Bias; Δ por dimensión.
- Perfiles basados en media por dimensión; añade `run_id/prompt_id` para estratificar si extiendes a IC/bootstraps.
- PCA es exploratorio; usa con cautela en el paper y reporta varianza explicada (puedes extender `plot_figures.py`).

## 🔁 Extensiones (opcional)
- Añadir ANOVA/regresión (statsmodels) para efecto del tamaño sobre MAE/|Δ|.
- Bootstraps para IC95%.
- Agrupar por `region` usando `hofstede_reference.csv` (columna `region`).

## ✅ Validación rápida
Con `data/example_models_results.csv` podrás generar todos los outputs para revisar el **pipeline** antes de conectar tus datos reales.

---

## 🧪 Métricas incluidas (por par modelo×país)
- **Δ por dimensión** (signed) y **|Δ|**; además **Zerr** (Δ/100).
- **MAE** (ponderado opcionalmente con `data/dimension_weights.json`).
- **RMSE** (ponderado).  
- **L2 (Euclidiana)**, **L1 (Manhattan)**, **L∞ (Chebyshev)**.
- **Cosine similarity** y **Cosine distance (1−cos)**.
- **Pearson r**, **Spearman ρ**, **R² (de Pearson)**.
- **Canberra** y **Bray–Curtis**.
- **Bias** (media de Δ).

Para activar pesos por dimensión, pasa `--weights_json data/dimension_weights.json` al `compute_metrics.py`.

### Ejecución con pesos
```bash
python src/compute_metrics.py   --models_csv data/example_models_results.csv   --hofstede_csv data/hofstede_reference.csv   --out_dir outputs   --weights_json data/dimension_weights.json
```

## 🗺️ Mapas y figuras adicionales
- `figures/bland_altman/` – Histogramas de Δ por dimensión (simplificación BA).
- `figures/correlation_matrix_models.png` – Matriz de correlación entre perfiles de modelos.
