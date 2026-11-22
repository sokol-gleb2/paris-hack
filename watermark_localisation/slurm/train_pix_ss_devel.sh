#!/bin/bash
# DEVEL job (NO reservation; dc-gpu-devel nodes are not in cispahack). Training only, no compare:
# 4 PixNet variants for StegaStamp (ensemble members for Step 4; tuned on fold 3 like `pix`).
#   GPU0 pixS1 width 32, 15k steps        GPU1 pixS2 width 48
#   GPU2 pixS3 crop 384 (bs 12)           GPU3 pixS4 + fake data v2 (fake-weight 0.3)
# Usage (~/wmloc_src; WORK_DIR, WMLOC_MODELS, SYNTH_DIR=~/wmloc_synth2 exported): sbatch slurm/train_pix_ss_devel.sh
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH -J wmloc
#SBATCH --output=../output/%j.out

cd "$SLURM_SUBMIT_DIR"
export TASK_DIR="${TASK_DIR:-$(dirname "$PWD")}"
source "$TASK_DIR/.venv/bin/activate"
export OMP_NUM_THREADS=16
J=$SLURM_JOB_ID
P='python -m scripts.step8_pixnet --method stegastamp --steps 15000 --eval-every 500 --hours 0.8'
run() { CUDA_VISIBLE_DEVICES=$1 bash -c "$2" > "../output/${J}_train_g$1.out" 2>&1 & }
run 0 "$P --tag pixS1"
run 1 "$P --width 48 --tag pixS2"
run 2 "$P --crop 384 --bs 12 --tag pixS3"
run 3 "$P --fake-weight 0.3 --tag pixS4"
wait
echo "$(date +%H:%M) PixNet StegaStamp variants done"
