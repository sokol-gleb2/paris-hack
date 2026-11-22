# Task 3 · Wildcard unlearning

The wildcard is a 100-class ViT-B/16 for the Paris AI Championship. The organizers scored this checkpoint as public submission **3768**, with a public score of **0.1762**. The held-out score is unknown.

## Recover the scored checkpoint

Place the organizer-supplied `model.pt` in this directory, or pass its path with `--model`. From this directory, run the following with Python 3.11+ and the Zstandard CLI. This path runs on CPU and needs no Python packages.

```sh
python3 code.py --restore
python3 code.py --verify
```

This writes the complete 343,628,063-byte PyTorch state dictionary to `output/submission.pt`. Git tracks a lossless Zstandard patch against the original model. The program verifies the base model, patch and resulting file. The scored checkpoint SHA-256 is `2be9bdf5e38b6c00be867f941bd51ad1462b2a73c96e8d4e7f56405c8b8bcacb`.

## Reproduce the fit on JURECA

Start in this directory with the task's shared `.venv/`, `model.pt` and `forget_images.npz`. Symlinks to the shared copies work. The reference environment used Python 3.12.15 and an NVIDIA A100. The dependency versions are in `requirements.txt`. For a fresh Linux CUDA environment, create a local virtual environment and install that file. Each machine supplies its own environment.

```sh
mkdir -p output
sbatch main.sh
```

The job runs `code.py` in fitting mode and writes the same checkpoint. Logs are in `output/<job-id>.out` and `output/<job-id>.err`. To use inputs elsewhere, pass `--model /path/to/model.pt --images /path/to/forget_images.npz` after `main.sh`. The code checks the pinned output digest before installing the candidate.

## What is fitted

1. The original classifier supplies pseudo-labels for the 3,500 unlabeled forget images. The parent fit combines a classifier readout, a sparse update to the last block’s MLP projection and classifier corrections informed by class counts and alternative predictions.
2. For original and cropped views, the original teacher differentiates its pseudo-label margin with respect to pixels. The algorithm ranks 49 tiles and masks the 12 most supportive tiles with the image’s channel means. Occluded teacher predictions supply alternative-class targets.
3. The student fits those targets on intact image features. The split uses 2,100 fit images, 700 calibration images and 700 held images. Calibration selected half of the fitted classifier update. Inference uses the standard ViT forward pass. The changed tensors are `blocks.11.mlp.fc2.weight`, `head.weight` and `head.bias`.

The fitting entry point is `phase9/wildcard_occlusion.py::unlearn`. Exact replay uses `assets/cpu_readout.npz`, a 100 × 769 intermediate fitted on three CPU threads from original-model features and pseudo-labels. Its fitting code is in `phase7/solution.py`. An A100 replay with this intermediate matched every tensor and every logit for all 17,500 image/view pairs. Calling `unlearn(model, images)` also refits the intermediate from raw inputs. That run matched 17,318 of 17,500 top predictions across backends. The checkpoint patch is used only by `--restore`.

| Organizer input | SHA-256 |
| --- | --- |
| `model.pt` | `485881a03ad064b5e6defc245ae3f45fd96714bf065ff96543a5eeca9f642571` |
| `forget_images.npz` | `5b1d9deaf475db0905973aa028511ab35aebc33196412f5ef9ebbef09f1541c7` |
