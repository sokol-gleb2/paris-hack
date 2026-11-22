#!/bin/bash
# DEVEL copy: NO reservation, dc-gpu-devel (nodes not in cispahack, which is MAGNETIC on dc-gpu).
# Dense decoder v2: more fake data (shards 28-35 on top of 20-27), longer training, 4 decoders:
#   GPU0 Pixel Seal  (stages 1-3 + pixel decoder)   tag dense2
#   GPU1 TrustMark   (layer2-4 + fc)                 tag dense2
#   GPU2 Pixel Seal  (all stages)                    tag dense3
#   GPU3 TrustMark   (all layers)                    tag dense3
# then Step 4 compare (no submit). Submit ONLY when no other team job is in the reservation.
# Usage (wmloc folder; WORK_DIR, WMLOC_MODELS, SYNTH_DIR=~/wmloc_synth2 exported): sbatch slurm/chain_dense2.sh
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH -J wmloc
#SBATCH --output=../output/%j.out

cd "$SLURM_SUBMIT_DIR"
export TASK_DIR="${TASK_DIR:-$(dirname "$PWD")}"
source "$TASK_DIR/.venv/bin/activate"
export OMP_NUM_THREADS=16
J=$SLURM_JOB_ID

run_phase() {
  local name=$1; shift; local g=0
  for cmd in "$@"; do
    CUDA_VISIBLE_DEVICES=$g bash -c "$cmd" > "../output/${J}_${name}_g${g}.out" 2>&1 &
    g=$((g + 1))
  done
  wait
  echo "$(date +%H:%M) phase $name done"
}

if false; then   # devel: 2 h cap, reuse existing fake data
  run_phase gen 'bash ~/gen.sh 28 29' 'bash ~/gen.sh 30 31' 'bash ~/gen.sh 32 33' 'bash ~/gen.sh 34 35'
fi

COMMON='--bit-loss 1 --clean-loss 0.01 --real-weight 0.2 --steps 60000 --eval-every 2000 --hours 1.4'
run_phase train \
  "python -m scripts.step6_stage_b --method pixelseal --unfreeze convnext.stages.1,convnext.stages.2,convnext.stages.3,pixel_decoder --dec-lr 3e-5 $COMMON --tag dense2" \
  "python -m scripts.step6_stage_b --method trustmark --unfreeze layer2,layer3,layer4,fc --dec-lr 3e-5 $COMMON --tag dense2" \
  "python -m scripts.step6_stage_b --method pixelseal --unfreeze convnext,pixel_decoder --dec-lr 2e-5 $COMMON --tag dense3" \
  "python -m scripts.step6_stage_b --method trustmark --unfreeze conv1,bn1,layer1,layer2,layer3,layer4,fc --dec-lr 2e-5 $COMMON --tag dense3"

run_phase compare 'python -m scripts.step4_tune_and_predict --tags prior,stageA,heat,stageAh,box,tmbox,tmboxAh,pix,dense,dense_g,dense2,dense3,dense2p'
echo "chain done: see ../output/${J}_compare_g0.out"
