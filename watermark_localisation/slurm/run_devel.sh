#!/bin/bash
# DEVEL (no reservation, dc-gpu-devel). Usage:  sbatch slurm/run_devel.sh <python module> [args...]
#   e.g. sbatch slurm/run.sh scripts.step1_probe --method pixelseal --list-layers
# Log: the task's output/ folder (../output/<jobid>.out). The folder must exist before
# sbatch, because Slurm opens the log before this script starts.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=../output/%j.out

cd "$SLURM_SUBMIT_DIR"                           # the wmloc/ folder you ran sbatch from
export TASK_DIR="${TASK_DIR:-$(dirname "$PWD")}"
source "$TASK_DIR/.venv/bin/activate"
srun --ntasks=1 python -m "$@"
