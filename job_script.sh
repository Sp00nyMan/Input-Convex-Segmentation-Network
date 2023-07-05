#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1

module load miniconda3
source ~/.bashrc
conda deactivate
conda activate torch

echo "-----------------------------------------------------------------------------------------"

echo "Job ID: " $SLURM_JOB_ID
echo "Job Name: " $SLURM_JOB_NAME

python main.py --image-name=bf --model=$1