#!/bin/bash
# GENERAL QUEUE (no cispahack reservation): does NOT count toward the one-node team limit, may wait longer.
# Usage (from the wmloc/ folder):  sbatch slurm/run3.sh <python module> [args...]
#   e.g. sbatch slurm/run3.sh scripts.step3_stage_a
# Runs the module once per method, in parallel, one GPU each (GPUs 0, 1, 2), adding
# --method <m>. One job gets a whole node (4 GPUs), so this uses it instead of wasting 3.
# Logs: ../output/<jobid>.out (this script) and ../output/<jobid>_<method>.out (each method).
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --output=../output/%j.out

cd "$SLURM_SUBMIT_DIR"                           # the wmloc/ folder you ran sbatch from
export TASK_DIR="${TASK_DIR:-$(dirname "$PWD")}"
source "$TASK_DIR/.venv/bin/activate"
export OMP_NUM_THREADS=32                        # the three runs share 128 cores
mod="$1"; shift
gpu=0
for m in pixelseal trustmark stegastamp; do
  # GPUs are not fenced off per process on this cluster: pin each run to its own GPU,
  # otherwise all three land on GPU 0.
  CUDA_VISIBLE_DEVICES=$gpu python -m "$mod" --method "$m" "$@" \
      > "../output/${SLURM_JOB_ID}_${m}.out" 2>&1 &
  gpu=$((gpu + 1))
done
wait
echo "done: see ../output/${SLURM_JOB_ID}_<method>.out"
