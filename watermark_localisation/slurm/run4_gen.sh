#!/bin/bash
# GENERAL QUEUE (no cispahack reservation): does NOT count toward the one-node team limit, may wait longer.
# Usage (from the wmloc folder):  sbatch slurm/run4.sh "<command for GPU 0>" ["<GPU 1>" ["<GPU 2>" ["<GPU 3>"]]]
#   Each quoted string is a full shell command (use ';' to chain several); each runs on its own GPU.
#   ONE node in total (team limit). Logs: ../output/<jobid>.out and ../output/<jobid>_g<k>.out
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --output=../output/%j.out

cd "$SLURM_SUBMIT_DIR"                           # the wmloc folder you ran sbatch from
export TASK_DIR="${TASK_DIR:-$(dirname "$PWD")}"
source "$TASK_DIR/.venv/bin/activate"
export OMP_NUM_THREADS=32                        # four runs share 128 cores
gpu=0
for cmd in "$@"; do
  # GPUs are not fenced off per process on this cluster: pin each command to its own GPU.
  CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" > "../output/${SLURM_JOB_ID}_g${gpu}.out" 2>&1 &
  gpu=$((gpu + 1))
done
wait
echo "done: see ../output/${SLURM_JOB_ID}_g<k>.out"
