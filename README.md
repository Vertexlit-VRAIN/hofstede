# 🌍 Hofstede Cultural Dimensions Model Analysis

This repository contains a comprehensive framework for **generating, evaluating, and analyzing LLM behaviors** through the lens of **Hofstede's Cultural Dimensions**. It provides an end-to-end pipeline: from synthetic dataset generation to multi-model inference and manual human evaluation.

---

## 🚀 Overview

The project aims to quantify cultural biases and tendencies in Large Language Models (LLMs) by testing them against the six dimensions of national culture defined by Geert Hofstede:

1.  **Power Distance (PDI)**
2.  **Individualism vs. Collectivism (IDV)**
3.  **Masculinity vs. Femininity (MAS)**
4.  **Uncertainty Avoidance (UAI)**
5.  **Long-term vs. Short-term Orientation (LTO)**
6.  **Indulgence vs. Restraint (IVR)**

### Key Features
-   🤖 **Automated Dataset Generation**: Create high-quality, culturally-aligned synthetic datasets.
-   ⚡ **Scalable Inference**: Multi-GPU/Multi-Run support for Hugging Face and Ollama models.
-   🖥️ **Interactive Evaluation UI**: Gradio-based interface for human verification and grounding.
-   📊 **Statistical Analysis**: Tools for ANOVA, regression, and visualization (radars, maps).

---

## 🛠️ Project Structure

```text
.
├── generate_dataset/     # Scripts to create synthetic prompts (Hugging Face)
├── inference/            # Model execution (SLURM/Python scripts)
│   └── output/           # RAW Inference results (.jsonl)
├── evaluation_ui/        # Gradio app for human evaluation
│   ├── inputs/           # Data to be evaluated (JSON)
│   └── outputs/          # Ground-truth evaluated data (JSONL)
├── analysis/             # Statistical analysis and plotting
│   ├── src/              # Core calculation and plotting scripts
│   ├── data/             # Reference CSVs (Hofstede indices)
│   └── metrics/          # Calculated metrics and figures
├── data/                 # Common data storage
├── pyproject.toml        # Dependency definitions
└── README.md             # This file
```

---

## ⚙️ Setup and Installation

### Prerequisites
-   **Python 3.10+**
-   **NVIDIA GPUs** (recommended for inference phase)
-   `uv` (recommended) or `conda`

### Dependency Installation

Initialize the environment using **uv**:
```bash
uv sync
# This installs all dependencies from pyproject.toml and uv.lock
```

Using **conda**:
```bash
conda create -n hofstede python=3.10
conda activate hofstede
pip install -r generate_dataset/requirements_gen_ds.txt
# Additional dependencies can be installed as needed
```

---

## 🏃 Workflow Execution

### 1. Dataset Generation
Generates synthetic data points based on cultural dimensions.
```bash
cd generate_dataset/
./run_job.sh  # Requires HF_TOKEN
```

### 2. Model Inference
Executes inference across multiple models.
```bash
# Example: Multi-GPU inference with quantization
python inference/multi_gpu_inference.py \
  --model_name "google/gemma-3-4b-it" \
  --input_file "evaluation_ui/inputs/part_1.json" \
  --output_dir "inference/output/" \
  --num_inferences 3
```

### 3. Human Evaluation (UI)
Manual review of model outputs to ensure quality and correctness.
```bash
cd evaluation_ui/
python main.py 1 # 1-5 depends on the part you are evaluating
```
The UI allows marking outputs as **Good**, **Bad**, or **Correction required**, and saves results to `evaluation_ui/outputs/`.

### 4. Statistical Analysis
Generate metrics, significance tests, and visualizations.
```bash
cd analysis/src/
python run_all.py --models_csv ../data/results.csv --out_dir ../metrics/
```
This script runs the full pipeline:
1.  `compute_metrics.py`: Basic statistics.
2.  `stats_descriptives.py`: Summary statistics.
3.  `plot_figures.py`: Radar charts and performance plots.
4.  `anova_regression.py`: Significance testing.
5.  `make_maps.py`: World maps visualization.

---

## 📊 Data Schema (Evaluated Format)

The evaluated data is stored as `.jsonl`. Each line contains:
```json
{
  "index": 12,
  "domain": "Education",
  "dimension_code": "PDI",
  "question": "How should a teacher...",
  "model_output_raw": "...",
  "model_output_parsed": [
    {"statement": "Low PDI approach...", "level": 1},
    {"statement": "High PDI approach...", "level": 5}
  ],
  "evaluation": "good"
}
```

---

## 🔬 Tested Models

We benchmarked a wide range of architectures on **NVIDIA A40 GPUs** (limited to 24GB active VRAM per GPU).

| Family | Variants Tested |
| :--- | :--- |
| **Gemma-3** | 1B, 4B, 12B, 27B |
| **DeepSeek-R1** | 1.5B, 7B, 8B, 14B, 32B |
| **Qwen-3** | 0.6B, 1.7B, 4B, 8B, 14B, 32B |
| **Mistral** | 7B |
| **Llama-3.1** | 8B |

> [!NOTE]
> 32B models required 4 GPUs with 4-bit quantization, while Gemma-3 27B required 6 GPUs due to memory constraints.
