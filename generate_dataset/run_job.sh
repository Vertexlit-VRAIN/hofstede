#!/bin/bash
#
# Nombre del job
#SBATCH --job-name=vertexlit-gemma3-27b
#
# Partición y recursos GPU
#SBATCH --gpus=4                  # número de GPUs
#
# Archivos de salida y error
#SBATCH --output=logs/gemma3-27b_%j.out
#SBATCH --error=logs/gemma3-27b_%j.err

echo "Iniciando job $SLURM_JOB_NAME ($SLURM_JOB_ID) en $(hostname) a $(date)"

export HF_TOKEN="YOUR_TOKEN_HERE"

cd ~/hofstede

conda run -n hofstede python dataset_multigpu.py

echo "Job finalizado a $(date)"
