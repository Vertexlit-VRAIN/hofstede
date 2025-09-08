#!/bin/bash
#
# Usage: sbatch run_slurm_inference.sh
#

# --- SLURM Directives ---
#SBATCH --job-name=multi_model_inference
#SBATCH --output=slurm_logs/inference_%A_%a.out
#SBATCH --error=slurm_logs/inference_%A_%a.err
#SBATCH --gpus=6

# ----- 32b models with 4 gpus and 4 bit, gemma 27b with 6 gpus, all others default

# --- Environment Setup ---

export HF_TOKEN="<HF_token>"

# --- Logging ---
echo "----------------------------------------------------"
echo "Timestamp: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "----------------------------------------------------"

# --- Run Inference Script ---
cd ~/hofstede/inference
conda run -n hofstede python multi_gpu_inference.py

EXIT_CODE=$?

echo "----------------------------------------------------"
echo "Finished SLURM job for model: $MODEL_NAME"
echo "Exit Code: $EXIT_CODE"
echo "Timestamp: $(date)"
echo "----------------------------------------------------"

exit $EXIT_CODE
