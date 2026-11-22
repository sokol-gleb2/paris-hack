#!/bin/bash
# ONE reservation job (team limit: one job per team in cispahack, running or pending).
# Runs three phases one after another on the same node (4 GPUs):
#   1) fake data v2 (bash ~/gen.sh: 3 methods x 2 shards per GPU) into $SYNTH_DIR
#   2) dense decoder (Pixel Seal, TrustMark) + big pixel detector (StegaStamp, TrustMark)
#   3) Step 4 compare (no submit)
# Usage (from the wmloc folder, with WORK_DIR, WMLOC_MODELS, SYNTH_DIR exported):
#   sbatch slurm/chain_dense.sh            [SKIP_GEN=1 sbatch ... to skip phase 1]
# Logs: ../output/<jobid>.out and ../output/<jobid>_<phase>_g<k>.out
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=02:30:00
#SBATCH -J wmloc
#SBATCH --output=../output/%j.out

cd "$SLURM_SUBMIT_DIR"
export TASK_DIR="${TASK_DIR:-$(dirname "$PWD")}"
source "$TASK_DIR/.venv/bin/activate"
export OMP_NUM_THREADS=16
J=$SLURM_JOB_ID

run_phase() {        # run_phase <name> "<cmd gpu0>" "<cmd gpu1>" ... ; waits for all
  local name=$1; shift; local g=0
  for cmd in "$@"; do
    CUDA_VISIBLE_DEVICES=$g bash -c "$cmd" > "../output/${J}_${name}_g${g}.out" 2>&1 &
    g=$((g + 1))
  done
  wait
  echo "$(date +%H:%M) phase $name done"
}

if [ -z "$SKIP_GEN" ]; then
  run_phase gen 'bash ~/gen.sh 20 21' 'bash ~/gen.sh 22 23' 'bash ~/gen.sh 24 25' 'bash ~/gen.sh 26 27'
fi

run_phase train \
  'python -m scripts.step6_stage_b --method pixelseal --unfreeze convnext.stages.2,convnext.stages.3,pixel_decoder --dec-lr 3e-5 --bit-loss 1 --clean-loss 0.01 --real-weight 0.2 --steps 20000 --eval-every 1000 --hours 1.3 --tag dense' \
  'python -m scripts.step6_stage_b --method trustmark --unfreeze layer3,layer4,fc --dec-lr 3e-5 --bit-loss 1 --clean-loss 0.01 --real-weight 0.2 --steps 20000 --eval-every 1000 --hours 1.3 --tag dense' \
  'python -m scripts.step8_pixnet --method stegastamp --fake-weight 0.5 --width 64 --tag pixf64 --steps 20000 --hours 1.3 --eval-every 1000' \
  'python -m scripts.step8_pixnet --method trustmark --fake-weight 0.5 --width 64 --tag pixf64 --steps 20000 --hours 1.3 --eval-every 1000'

run_phase compare 'python -m scripts.step4_tune_and_predict --tags prior,stageA,heat,stageAh,box,tmbox,tmboxAh,pix,dense,pixf64'
echo "chain done: see ../output/${J}_compare_g0.out"
