#!/bin/bash
# DEVEL job (NO reservation; dc-gpu-devel nodes are not in cispahack). Training only, no compare:
# 4 more dense-decoder variants for ensembling (Step 4 averages top-k tags sharing tuning fold 0).
#   GPU0 Pixel Seal  dense6: stages 2-3 + pixel decoder (as dense), real-weight 0.35, dec-lr 5e-5
#   GPU1 TrustMark   dense6: layer3-4 + fc (as dense),          real-weight 0.35, dec-lr 5e-5
#   GPU2 Pixel Seal  dense7: stages 1-3 + pixel decoder, bit-loss 2, clean-loss 0.03
#   GPU3 TrustMark   dense7: layer2-4 + fc,              bit-loss 2, clean-loss 0.03
# Usage (~/wmloc_src; WORK_DIR, WMLOC_MODELS, SYNTH_DIR=~/wmloc_synth2 exported): sbatch slurm/train_dense67_devel.sh
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
S='--steps 60000 --eval-every 2000 --hours 1.6'
run() { CUDA_VISIBLE_DEVICES=$1 bash -c "$2" > "../output/${J}_train_g$1.out" 2>&1 & }
run 0 "python -m scripts.step6_stage_b --method pixelseal --unfreeze convnext.stages.2,convnext.stages.3,pixel_decoder --dec-lr 5e-5 --bit-loss 1 --clean-loss 0.01 --real-weight 0.35 $S --tag dense6"
run 1 "python -m scripts.step6_stage_b --method trustmark --unfreeze layer3,layer4,fc --dec-lr 5e-5 --bit-loss 1 --clean-loss 0.01 --real-weight 0.35 $S --tag dense6"
run 2 "python -m scripts.step6_stage_b --method pixelseal --unfreeze convnext.stages.1,convnext.stages.2,convnext.stages.3,pixel_decoder --dec-lr 3e-5 --bit-loss 2 --clean-loss 0.03 --real-weight 0.2 $S --tag dense7"
run 3 "python -m scripts.step6_stage_b --method trustmark --unfreeze layer2,layer3,layer4,fc --dec-lr 3e-5 --bit-loss 2 --clean-loss 0.03 --real-weight 0.2 $S --tag dense7"
wait
echo "$(date +%H:%M) dense6/dense7 training done"
