CL="--clean-dir $HOME/coco/val2017 --coco-ann $HOME/coco/annotations/instances_val2017.json --n 250"
for m in pixelseal trustmark stegastamp; do for s in "$@"; do python -m scripts.step5_synth --method $m $CL --shard $s & done; done; wait
