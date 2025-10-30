# 1) `compute_metrics.py`  → métricas “core”

**Nivel:** por *(model_id, size, params_log, country_ref)* salvo que se indique lo contrario.
**Salida principal:** `metrics_per_model_country.csv`, `per_dimension_deltas.csv`, y varios agregados.

**Vector/diferencias por dimensión (6D Hofstede):**

* `Delta_<DIM>`: diferencia (modelo − país) por dimensión.
* `AbsDelta_<DIM>`: valor absoluto de la diferencia.
* `Zerr_<DIM>`: error estandarizado = Δ/100 (escala-libre).

**Métricas de error/distancia entre el vector 6D del modelo y el país:**

* `MAE` (media ponderada de |Δ|; pesos opcionales normalizados a 1 si se pasa `--weights_json`).
* `RMSE` (raíz del error cuadrático medio ponderado).
* `L2` (norma euclídea de Δ, sin ponderar).
* `L1` (suma de |Δ|, sin ponderar).
* `L_inf` (máximo |Δ|).
* `Cosine` (similitud coseno).
* `CosineDist` (= 1 − Cosine).
* `Pearson_r` (correlación Pearson entre vectores).
* `Spearman_rho` (correlación Spearman entre vectores).
* `R2_from_Pearson` (= Pearson_r²).
* `Canberra` (distancia Canberra).
* `BrayCurtis` (distancia Bray–Curtis).
* `Bias` (media de Δ en las 6 dimensiones).

**Perfiles y agregados que también se guardan:**

* `model_profiles_6D.csv`: los **perfiles 6D** (medias por dimensión) por *(model_id, size, params_log, country_ref)*.
* `per_dimension_deltas.csv`: todas las `Delta_*/AbsDelta_*/Zerr_*`.
* `metrics_per_model_country.csv`: todas las métricas anteriores (MAE, RMSE, etc.).
* `same_model_different_sizes.csv`: para comparar tamaños, re-agrega **MAE, L2, Cosine** por *(model_id, country_ref, size)*.
* `all_models_rank_by_country.csv`: ranking por país ordenando por *(MAE, L2, CosineDist)*.
* `country_group_comparisons.csv`: por grupos (Europa-France, China, USA) re-agrega **MAE, L2, L1, L_inf, Cosine, CosineDist** por *(model_id, size)*.
* `usa_models_comparison.csv`: subconjunto para `USA`.

> **Dimensiones usadas (constante `DIMENSIONS`):** `Power_Distance`, `Individualism`, `Masculinity`, `Uncertainty_Avoidance`, `Long_Term_Orientation`, `Indulgence`.

---

# 2) `stats_descriptives.py`  → descriptivos y fiabilidad

**Nivel:** por *(model_id, size, country_ref, dimension)*.
**Salida:** `table1_descriptives.csv`.

**Métricas por dimensión (sobre `score_scaled`):**

* `mean`, `sd`, `N`.
* `CI95_low`, `CI95_high` (IC 95% con t de Student).
* **Fiabilidad intra-dimensión:**

  * `alpha` = **α de Cronbach** (items como columnas; si hay ≥2 items y ≥2 observaciones).
  * `ICC_2k` = **ICC(2,k)**, dos vías aleatorio, **absoluto** (si hay ≥2 items y ≥2 runs).

---

# 3) `anova_regression.py`  → inferencia (tamaño del modelo y tamaño de parámetros)

**Entradas:** `metrics_per_model_country.csv` y `per_dimension_deltas.csv`.
**Salidas:**

* `table3_anova_size_by_dimension.csv`
* `table3_kruskal_size_by_dimension.csv`
* `table3_regression_mae_paramslog_countryFE.csv`
* `table3_spearman_mae_paramslog.csv`

**ANOVA por dimensión (sobre `AbsDelta_<DIM>`):**

* `F` (efecto de `C(size)`), `p`, `eta_sq` (η² parcial respecto a `C(size)`), `N`.

**Kruskal–Wallis por dimensión (robusto a no-normalidad):**

* `H`, `p`, `N` (grupos por `size`, requiere ≥2 grupos con n>1).

**Regresión OLS (errores agrupados por `model_id`):**

* Modelo: `MAE ~ params_log + C(country_ref)`
  → **coeficiente de `params_log`**: `beta`, `ci_low`, `ci_high`, `p`; además `R2`.

**Correlación Spearman (robusta a no linealidad):**

* `spearman_rho` y `p` entre `params_log` y `MAE`.

---

# 4) `plot_figures.py` / `make_maps.py`

Solo **visualizan** a partir de las métricas anteriores (no añaden métricas nuevas), p. ej.:

* Radar por país (perfiles 6D).
* Heatmap de `AbsDelta_*`.
* PCA/MDS de proximidad modelos-países.
* Dispersión `size` vs `MAE` (con mediana y tendencia).
* Histos tipo Bland–Altman de `Delta_*`.
* Matriz de correlaciones entre **perfiles 6D** de modelos.
* Barras Gemma: medias por dimensión y **IC95%** por tamaño.
