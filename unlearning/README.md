# Task 3 · Adaptive unlearning

**Team checkpoint for final testing: submission 3905**. The public scorer reported **0.17655230543752248**. The model is a 100-class ViT-B/16 with a classifier trained on an image-specific mixture of clean and occluded teacher predictions. Read the [architecture and reproduction report](docs/architecture_3905.pdf) for the training pipeline, equations, controls and measurements. The [verification record](docs/verification_3905.json) includes the original acceptance receipt, public leaderboard check and fresh A100 replay.

## Recover the exact scored weights on CPU

From this directory, supply the official original model, Python 3.11+ and the Zstandard CLI. This path has no Python package dependencies.

```sh
python3 code.py --restore --model /path/to/model.pt
python3 code.py --verify
```

The resulting `output/submission.pt` is a 343,628,063-byte plain PyTorch state dictionary. The SHA-256 is `cb6ecbdf2cc27d6485850483fb47176ee52255dbbb36a004a31c935c98dc068b`. The compact patch in `assets/` recovers the complete file byte for byte. An existing output from our previous release, 3768, upgrades atomically after the replacement passes verification.

## Reproduce training on JURECA

In the task directory, activate the shared environment by making its `.venv/` available here. Place or link the organizer's `model.pt` and `forget_images.npz` here. For a fresh Linux environment, use Python 3.12 and the CUDA package versions in `requirements.txt`.

```sh
mkdir -p output
sbatch main.sh
```

The job runs the adaptive fit on one A100. To use inputs outside the directory, pass `--model /path/to/model.pt --images /path/to/forget_images.npz` after `main.sh`. The program checks the SHA-256 before publishing the checkpoint. Slurm writes logs to `output/<job-id>.out` and `output/<job-id>.err`.

## Code map

| File | Role |
| --- | --- |
| `adaptive.py` | Final training entry point, targeted and random masks, cross-view reliability, teacher mixing and state export |
| `phase9/destinations_dual.py` | Conditional probability constraints and classifier optimizer |
| `phase9/destinations_recipe.py` | Dual-prior parent (3552) |
| `phase9/recipe.py` | Sparse MLP and class-dependent corrections (3321 and 3468) |
| `phase7/solution.py` | CPU classifier fitting (3058) |
| `task_template.py` | Official architecture, image preprocessing and inference |
| `code.py` | Command line for training, exact recovery and SHA-256 verification |

The training CLI uses `assets/cpu_readout.npz`, a 100 × 769 float32 classifier derived from the original model and forget set. Calling `adaptive.unlearn(model, forget_images)` refits this stage directly. The archived raw-only GPU fit produced 122 differing top predictions across 17,500 image/view evaluations compared with 3905. The supplied intermediate reproduces the scored fit.

| Component | SHA-256 |
| --- | --- |
| Original model | `485881a03ad064b5e6defc245ae3f45fd96714bf065ff96543a5eeca9f642571` |
| Forget images | `5b1d9deaf475db0905973aa028511ab35aebc33196412f5ef9ebbef09f1541c7` |
| CPU readout | `6dd838db325ec761d880d21794fb99a3b0e5239d3f320f7d4c5b736af90f49ae` |
