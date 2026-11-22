#!/bin/bash
#SBATCH --job-name=unlearning-wildcard
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --time=00:30:00
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
srun python code.py "$@"
