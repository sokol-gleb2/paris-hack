#!/bin/bash
# ONE reservation job (team limit: one job, running OR pending; no other job of ours anywhere).
# dense5 recipe (whole decoder unfrozen, 4,000 fake/method) won after only ~10 min (job killed): train it longer.
#   GPU0 PS dense8 (= dense5)   GPU1 TM dense8 (= dense5)   GPU2 PS dense9 (dec-lr 3e-5)   GPU3 TM dense9 (dec-lr 3e-5)
#   + 4 StegaStamp PixNet variants pixS1-4 on GPU1/GPU3 + StegaStamp RISE (4 shards, tag rise)
# then Step 4 compare over all tags (no submit). ~54 min train + ~15 min compare.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=01:45:00
#SBATCH -J wmloc
#SBATCH --output=../output/%j.out

cd "$SLURM_SUBMIT_DIR"
export TASK_DIR="${TASK_DIR:-$(dirname "$PWD")}"
source "$TASK_DIR/.venv/bin/activate"
export OMP_NUM_THREADS=16
J=$SLURM_JOB_ID
C='--bit-loss 1 --clean-loss 0.01 --real-weight 0.2 --steps 60000 --eval-every 2000 --hours 0.9'
PS='python -m scripts.step6_stage_b --method pixelseal --unfreeze convnext,pixel_decoder'
TM='python -m scripts.step6_stage_b --method trustmark --unfreeze conv1,bn1,layer1,layer2,layer3,layer4,fc'
run() { CUDA_VISIBLE_DEVICES=$1 bash -c "$2" > "../output/${J}_train_g$1.out" 2>&1 & }
run 0 "$PS --dec-lr 2e-5 $C --tag dense8"
run 1 "$TM --dec-lr 2e-5 $C --tag dense8"
run 2 "$PS --dec-lr 3e-5 $C --tag dense9"
run 3 "$TM --dec-lr 3e-5 $C --tag dense9"
# StegaStamp PixNet variants (small, 3.5M params) share the TrustMark GPUs
P='python -m scripts.step8_pixnet --method stegastamp --steps 15000 --eval-every 500 --hours 0.9'
pix() { CUDA_VISIBLE_DEVICES=$1 bash -c "$2" > "../output/${J}_pix_$3.out" 2>&1 & }
pix 1 "$P --tag pixS1" pixS1
pix 1 "$P --width 48 --tag pixS2" pixS2
pix 3 "$P --crop 384 --bs 12 --tag pixS3" pixS3
pix 3 "$P --fake-weight 0.3 --tag pixS4" pixS4
# StegaStamp RISE (black-box saliency with the known message): 4 shards, one per GPU, then merge
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k OMP_NUM_THREADS=6 timeout 50m python -m scripts.step3h_rise --method stegastamp --n 160 --shard $k --nshards 4 \
    > "../output/${J}_rise_$k.out" 2>&1 &
done
wait
echo "$(date +%H:%M) train + RISE shards done"
python -m scripts.step3h_rise --method stegastamp --merge --nshards 4 > "../output/${J}_rise_merge.out" 2>&1
echo "$(date +%H:%M) RISE merge done"
python -m scripts.step4_tune_and_predict --tags prior,stageA,heat,stageAh,box,tmbox,tmboxAh,pix,rise,pixS1,pixS2,pixS3,pixS4,dense,dense_g,dense2,dense3,dense4,dense5,dense6,dense7,dense8,dense9 > "../output/${J}_compare.out" 2>&1
echo "$(date +%H:%M) compare done: ../output/${J}_compare.out"
