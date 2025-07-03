# Hofstede Cultural Dimensions Model Analysis

This project focuses on generating datasets, performing inference, and evaluating models related to Hofstede's cultural dimensions. It provides tools for automated data generation, model inference on this data, and a user interface for evaluating the model outputs.

## Table of Contents
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Dependency Installation](#dependency-installation)
- [Usage](#usage)
  - [Dataset Generation](#dataset-generation)
  - [Model Inference](#model-inference)
  - [Evaluation UI](#evaluation-ui)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Automated Dataset Generation**: Tools to create synthetic datasets based on Hofstede's cultural dimensions.
- **Model Inference**: Scripts to run various language models (LLMs) on the generated datasets, supporting multi-GPU setups.
- **Interactive Evaluation UI**: A Gradio-based web interface for human evaluation and correction of model outputs.

## Setup and Installation

### Prerequisites
- Python 3.10 or higher.
- `uv` for dependency management (recommended, as indicated by `uv.lock` and `pyproject.toml`). Alternatively, `conda` can be used as seen in `run_job.sh` scripts.

### Dependency Installation

It is recommended to use `uv` for managing dependencies.

1.  **Install `uv`**:
    ```bash
    pip install uv
    ```

2.  **Install Project Dependencies**:
    Navigate to the root of the project and run:
    ```bash
    uv sync
    ```
    This will install all dependencies listed in `pyproject.toml`.

    If you prefer `conda`, you can create an environment and install dependencies manually:
    ```bash
    conda create -n hofstede python=3.10
    conda activate hofstede
    pip install -r requirements_gen_ds.txt # for dataset generation
    # Install other dependencies from pyproject.toml manually or using pip install <package_name>
    ```

## Usage

### Dataset Generation
The `generate_dataset/` directory contains scripts for generating synthetic datasets.

To run the dataset generation job (e.g., using SLURM as indicated by `run_job.sh`):
```bash
cd generate_dataset/
./run_job.sh
```
**Note**: Ensure you set your Hugging Face token in the `run_job.sh` script (e.g., `export HF_TOKEN="hf_YOUR_TOKEN_HERE"`).
This script typically runs `dataset_multigpu.py` or `dataset_ollama.py` to create `generated_dataset.json` in the `data/` directory.

### Model Inference
The `inference/` directory contains scripts for running inference with various models.

- `inference.py`: General inference script.
- `multi_gpu_inference.py`: For inference across multiple GPUs.
- `multi_inference.py`: Another multi-inference script.

Example of running inference (adjust parameters as needed):
```bash
python inference/inference.py --model_name "google/gemma-3-4b-it" --input_file "data/generated_dataset.json" --output_dir "inference_output/"
```
Inference results will be saved in the `inference_output/` directory, typically in a subdirectory named after the model.

### Evaluation UI
The `evaluation_ui/` provides a web-based interface for evaluating and correcting model outputs.

#### Setup
You only need to install the `gradio` library if not already installed with `uv sync`:
```bash
pip install gradio
```

#### Directory Structure
Ensure your input data for evaluation is organized as follows:
```
evaluation_ui/
├── main.py
└── inputs/
    ├── part_1.json   <-- Your input data file
    ├── part_2.json
    └── ...
```
The `outputs/` directory will be created automatically by the script when you save your first item.

#### How to Run the Application
1.  Open your terminal or command prompt.
2.  Navigate to the `evaluation_ui/` directory.
3.  Run the script, providing the `part_id` of the data file you want to evaluate as a command-line argument. The `part_id` must be a number from 1 to 5 (or as per your input file naming convention).

    **Example:** To evaluate the `part_1.json` file:
    ```bash
    python main.py 1
    ```

4.  The script will start a local web server and print a URL, usually `http://127.0.0.1:7860`. Open this URL in your web browser to access the UI.

#### Using the Interface
The interface is designed for a straightforward evaluation workflow:

1.  **Information Panel:** The top text boxes (`Domain`, `Dimension`, `Question`) show the context for the current item. These are for display only.

2.  **Statements Panel:**
    -   The five `Level X Statement` boxes contain the model-generated output. **You can edit the text in these boxes directly.**
    -   The **Reset Button (🔄)** next to each statement will restore its text to the original value from the input file.

3.  **Action Buttons:** After reviewing (and editing, if necessary), click one of the three buttons to save and move to the next item.
    -   `Accept (Good)`: Click this if the five statements are accurate and well-formed as they are. No changes are needed.
    -   `Save Changes (Bad)`: Click this if the statements were semantically incorrect or poorly phrased and **you have edited them to be correct**.
    -   `Save Manual Fix (Format Error)`: This button **only appears** if the original model output could not be parsed (e.g., it was not valid JSON). In this case, you must manually type in all five statements before clicking this button to save your work.

4.  **Progress Bar:** The label at the bottom shows your overall progress for the current data part.

#### Key Features
-   **Automatic Progress Saving:** The application saves your work after every item is evaluated. You can safely stop the script (Ctrl+C in the terminal) and relaunch it later. It will automatically resume from the last completed item.
-   **Output File:** Your evaluated data is saved in the `outputs/` directory in a file named `evaluated_data_part_<PART_ID>.jsonl`. Each line in this file is a complete JSON object representing one evaluated item.

## Project Structure
- `.gitignore`: Specifies intentionally untracked files to ignore.
- `pyproject.toml`: Project metadata and dependencies.
- `run_job.sh`: Example SLURM job script for main tasks.
- `uv.lock`: Lock file for `uv` dependency management.
- `data/`: Contains generated datasets.
- `evaluation_ui/`: Contains the Gradio-based evaluation interface.
- `generate_dataset/`: Scripts for generating synthetic datasets.
- `inference/`: Scripts for running model inference.
