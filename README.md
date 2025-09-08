# Hofstede Cultural Dimensions Model Analysis

This project focuses on generating datasets, performing inference, and evaluating models related to Hofstede's cultural dimensions. It provides tools for automated data generation, multi-model inference, and a user interface for evaluating outputs.  

## Table of Contents
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Dependency Installation](#dependency-installation)
- [Usage](#usage)
  - [Dataset Generation](#dataset-generation)
  - [Model Inference](#model-inference)
  - [Evaluation UI](#evaluation-ui)
- [Tested Models](#tested-models)
- [Execution Environment](#execution-environment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Automated Dataset Generation**: Scripts for creating synthetic datasets aligned with Hofstede's cultural dimensions.
- **Model Inference**: Supports inference across multiple large language models (LLMs), with multi-GPU and quantization options.
- **Inference Outputs**: Each tested model has its own folder with **three independent runs**, stored as `.jsonl` files for reproducibility and variance analysis.
- **Interactive Evaluation UI**: A Gradio-based web interface for human evaluation, correction, and annotation of model outputs.

---

## Setup and Installation

### Prerequisites
- Python 3.10 or higher.
- `uv` for dependency management (recommended).  
  Alternatively, `conda` can be used (as shown in the SLURM job scripts).

### Dependency Installation

Using **uv**:
```bash
pip install uv
uv sync
````

Using **conda**:

```bash
conda create -n hofstede python=3.10
conda activate hofstede
pip install -r generate_dataset/requirements_gen_ds.txt
# Then install additional dependencies from pyproject.toml if needed
```

---

## Usage

### Dataset Generation

Located in `generate_dataset/`.

Example (with SLURM):

```bash
cd generate_dataset/
./run_job.sh
```

This generates `hofstede_generated.json` inside the `outputs/` folder.
Make sure to set your Hugging Face token (`HF_TOKEN`) in `run_job.sh`.

---

### Model Inference

Located in `inference/`.

Scripts:

* `inference.py`: Single-model inference using Ollama API.
* `multi_inference.py`: Runs **multiple models** across multiple runs (default: 3 runs per model).
* `multi_gpu_inference.py`: Multi-GPU Hugging Face inference, supporting quantization for large models.

Example:

```bash
python inference/multi_gpu_inference.py \
  --model_name "google/gemma-3-4b-it" \
  --input_file "evaluation_ui/inputs/part_1.json" \
  --output_dir "inference/output/" \
  --num_inferences 3
```

**Inference Outputs:**
Each model has its own folder under `inference/output/`, e.g.:

```
inference/output/
├── google_gemma-3-4b-it/
│   ├── inference_results_run_1.jsonl
│   ├── inference_results_run_2.jsonl
│   └── inference_results_run_3.jsonl
├── Qwen_Qwen3-14B/
│   ├── inference_results_run_1.jsonl
│   ├── inference_results_run_2.jsonl
│   └── inference_results_run_3.jsonl
└── ...
```

Each `.jsonl` file contains one JSON object per evaluated item.

---

### Evaluation UI

Located in `evaluation_ui/`.

Run with:

```bash
cd evaluation_ui/
python main.py 1
```

This launches a local Gradio app (default: [http://127.0.0.1:7860](http://127.0.0.1:7860)) for manual review.

* Inputs: `inputs/part_X.json`
* Outputs: Saved as `outputs/evaluated_data_part_X.jsonl`
* Features: Progress tracking, reset buttons, manual fixes for invalid JSON.

---

## Tested Models

The following models have been tested (see `inference/model_to_analyze.md`):

| Family                  | Variants Tested              |
| ----------------------- | ---------------------------- |
| **Gemma-3**             | 1B, 4B, 12B, 27B             |
| **DeepSeek-R1 Distill** | 1.5B, 7B, 8B, 14B, 32B       |
| **Qwen-3**              | 0.6B, 1.7B, 4B, 8B, 14B, 32B |
| **Mistral**             | 7B                           |
| **Llama-3.1**           | 8B                           |
| **Granite-3.3**         | 2B, 8B                       |
| **Phi-4**               | 14B                          |

**Special settings:**

* Gemma-3 27B → requires 6 GPUs.
* DeepSeek-R1 32B → 4 GPUs, 4-bit quantization.
* Qwen-3 32B → 4 GPUs, 4-bit quantization.
* All other models → 4 GPUs (full precision).

---

## Execution Environment

All experiments were executed on **NVIDIA A40 GPUs** (48GB each), but each GPU was **limited to 24GB of usable memory**.

This GPU constraint was the main factor influencing which models and configurations could be run (e.g., Gemma-3 27B required 6 GPUs; 32B models required 4 GPUs in 4-bit precision).

---

## Project Structure

* `generate_dataset/` → dataset generation scripts.
* `inference/` → inference scripts + outputs per model.
* `evaluation_ui/` → Gradio interface for manual evaluation.
* `pyproject.toml` → dependency management.
* `run_job.sh` / `run_slurm_inference.sh` → SLURM job scripts.
