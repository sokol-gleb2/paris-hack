# Watermark localisation (Paris hackathon, team "J'ai Deux Passion")

Find **which pixels of a photo carry an invisible watermark**, for three watermarking methods:
**Pixel Seal**, **TrustMark** and **StegaStamp**. Code package: `wmloc`; pipeline scripts: `scripts/`;
JURECA batch scripts: `slurm/`.

Final result: **mean holdout mIoU 0.619** (from 0.29 for the "average mask" baseline). Public leaderboard
history is in [Results](#results).

---

## Contents

1. [The task](#the-task)
2. [Results](#results)
3. [Data facts we found](#data-facts-we-found)
4. [How we validate (folds and holdout)](#how-we-validate-folds-and-holdout)
5. [Repository layout](#repository-layout)
6. [Pipeline, step by step](#pipeline-step-by-step)
7. [Per method: what works](#per-method-what-works)
8. [Everything we tried, with numbers](#everything-we-tried-with-numbers)
9. [Running it on JURECA](#running-it-on-jureca)
10. [Lessons and next steps](#lessons-and-next-steps)

---

## The task

- Each photo is 512 x 512. Part of it (sometimes almost all of it) carries a watermark from **one** method.
- For every photo we are given: the **method** and the **hidden message** (bits) that was embedded.
  Pixel Seal messages are 256 bits; TrustMark and StegaStamp 100 bits.
- We predict a **mask on a 128 x 128 grid** (each cell = 4 x 4 pixels): 1 = watermarked, 0 = clean.
- **Score = mean IoU** (overlap / union, per image, averaged). Example: we mark 100 cells, the truth is
  100 cells, 50 overlap -> IoU = 50 / 150 = 0.33.
- Train: 170 photos per method (510) with masks. Test: 340 per method (1,020), no masks.
- Submission: a torch `.pt` file `{"image_names": [...], "masks": tensor[N,128,128] of 0/1}`, every test
  image exactly once, <= 20 MB (`wmloc/submission.py` builds, validates, saves and submits it).

## Results

Public leaderboard = 30% of the test set (the final ranking uses the hidden rest). Our "holdout" is our
own 34-photos-per-method fold that no model ever trained on (see next sections).

| Submission | What changed | Mean holdout | Public |
| --- | --- | --- | --- |
| #1 | `prior`: average train mask per method | ~0.30 | 0.2934 |
| #1903 | Stage A head on frozen decoders (+ crop heatmaps) | 0.4454 | 0.4457 |
| #2412 | + StegaStamp **box search** | 0.4969 | **0.4943** |
| #2728 | + PixNet (`pix`) in the StegaStamp ensemble | 0.5012 | 0.4935 (noise) |
| #3275 | + **dense decoder** (`dense`, `dense_g`) for Pixel Seal | (PS 0.553) | **0.5122** |
| next | + `dense2..7` + top-k ensembles (Step 4 fix) | **0.5483** | ~0.54 |
| final | + `dense8`, `dense9`, StegaStamp PixNet variants `pixS1..4` | **0.6190** | (see board) |

Final choices per method (job 15905914): Pixel Seal `dense9`; TrustMark ensemble
`top3:dense5+dense9+dense4`; StegaStamp `pixS2` (PixNet, width 48).
The holdout also picks among ~25 candidates, so it is slightly optimistic; expect the public score a little lower.

## Data facts we found

- **0 empty true masks** in train, so predictions are never empty (`postprocess.ALLOW_EMPTY = False`;
  an early submission had 73 empty masks because the tuner turned "never empty" off).
- **No repeat photos** (max grey-thumbnail correlation 0.86 < 0.95), so folds need no grouping
  (`groups.json` is empty).
- **Two regimes, exactly 85/170 per method each** (train metadata `full_watermark`):
  - *full*: the whole photo is watermarked, then 1-5 random smooth blobs (often touching the border,
    unrelated to the content) are restored to the clean original; mask area ~0.90.
  - *partial*: an outpainting **rectangle** (often ~9% of the image, down to 1%) or a **transplanted object**.
  - Test metadata has no flag. A whole-image message match detects the regime 97-98% correctly
    (`step4c_regime.py`), but separate cutoffs per regime gained nothing.
- Photos are AI-generated; edits are seamless (`scripts/viz_train.py` draws photo / true mask / prediction grids).
- The **decoders' weak spot**: small transplanted objects are read at near chance (bit accuracy 0.55-0.63).

## How we validate (folds and holdout)

- 5 folds per method (34 photos each), fixed seed (`data.make_folds`).
- **Fold 4 = holdout** (`config.HOLDOUT_FOLD`): never trained on, never used to pick a checkpoint.
  It is only used by Step 4 to choose between candidates and report the score.
- Cheap models (Stage A heads, logistic fits for heat/box/rise) are **cross-trained** on folds 0-3:
  train on three folds, predict the fourth -> out-of-fold (OOF) predictions for all 136 photos.
- Expensive models (dense decoders, PixNet) are trained **once**: real photos of three folds + fake data,
  checkpoint picked on one fold (`--val-fold`: 0 for dense, 3 for PixNet). Their OOF files record
  `tune_names` (the picking fold), and Step 4 tunes their cutoff only on those 34 photos.
- Step 4 ranks candidates by the mean of tune and holdout score, and keeps the baseline unless the winner
  beats it by `--margin` (0.01) **and** does not lose on the holdout.
- The holdout predicted the public board well (e.g. 0.4454 vs 0.4457).

## Repository layout

```
wmloc/                 shared package
  config.py            paths (TASK_DIR, WORK_DIR, SYNTH_DIR via env vars), methods, folds, per-method decoder settings
  data.py              Record (image, bits, 128x128 mask, metadata), loading the starter kit split, folds, caching
  decoders.py          loads the starter kit's wrappers/embedders by path; DecoderAdapter = decoder as feature
                       extractor (+ per-position bit scores "agreement map"); to_scores / decode_bits helpers
  head.py              LocHead (decoder features + extra maps + image branch -> 128x128 logits); BCE + symmetric Lovasz loss
  heat.py              known-message match (match_gap), crop heatmaps, tile agreement, per-cell logistic fit
  metric.py            exact task metric (mIoU), 512 -> 128 downsampling, breakdown by edit type and size
  postprocess.py       probability -> mask: cutoff, never-empty, optional cleanup (min blob, hole fill, rect fill), tuning
  submission.py        build / validate / save / submit (needs WMLOC_API_KEY and optionally WMLOC_BASE_URL in the env)
scripts/               pipeline steps (python -m scripts.<name>), see below
slurm/                 JURECA batch scripts (see "Running it on JURECA"); gen.sh = fake-data generator wrapper
```

Work files (not in git) go to `$WORK_DIR`: `cache/` (OOF / test predictions per tag, maps, records),
`checkpoints/`, `submissions/`. A **tag** is one model's predictions: `cache/oof_<tag>_<method>.pt`
(all train photos) + `cache/test_<tag>_<method>.pt` (test photos, or computed by Step 4 from the checkpoint).

## Pipeline, step by step

| Step | Script | What it does |
| --- | --- | --- |
| 0 | `step0_prior_submission.py` | proves the submission pipeline: checks our 512->128 downsampling against the given masks, trivial baselines, submits the average mask |
| 0b | `step0b_checks.py` | empty masks, repeat photos (-> groups.json), our metric vs the kit's scorer |
| 1 | `step1_probe.py` | decoder probe per method: which network, output format, preprocessing, bit accuracy on whole image / true crop / flips / sizes, agreement-map check, layer list |
| 3 | `step3_stage_a.py` | Stage A: frozen decoder, cached features, train `LocHead`, 5-fold CV (`--heat` adds heatmaps -> tag `stageAh`) |
| 3b | `step3b_heatmaps.py` | known-message heatmaps from overlapping crops (256/128 px) + shuffle check; tag `heat`; `--tile-agree` for Pixel Seal tiles |
| 3c | `step3c_prior.py` | tag `prior` = average train mask (Step 4 baseline) |
| 3d | `step3d_boxes.py` | **box search**: decode ~700-800 rectangles (64-px grid, sides 128-512, aspect limit), top-5 by message match -> map; tag `box` (logistic on [map, average mask]); shardable |
| 3e | `step3e_tm_boxes.py` | Adobe TrustMark box finder (`localize`, newer trustmark in `~/pylib_tm`) -> tags `tmbox`, `tmboxAh` |
| 3f | `step3f_seeded_boxes.py` | rectangles from a seed map, edges refined 32->16->8 px by message match; tags `srect`, `sbox` |
| 3g | `step3g_matched.py` | matched filter: re-embed the known message, correlate fine detail with the expected pattern; tag `mf` |
| 3h | `step3h_rise.py` | RISE black-box saliency with the known message (StegaStamp); tag `rise` |
| 4 | `step4_tune_and_predict.py` | per method: load all listed tags, candidates = each tag, all-tag average, **top-2..5 ensembles**; tune post-processing; choose on the holdout; predict test; write `submissions/sub_<tags>.pt` (+ `.json` with the choices) |
| 4c | `step4c_regime.py` | two-regime post-processing (full vs partial) |
| 5 | `step5_synth.py` | fake training data following the dataset recipe (packs of 250 PNGs in `SYNTH_DIR`) |
| 6 | `step6_stage_b.py` | Stage B0 / B and the **dense decoder** (`--unfreeze`, `--bit-loss`, `--clean-loss`, `--real-weight`) |
| 8 | `step8_pixnet.py` | **PixNet**: small U-Net on raw pixels + high-pass, real photos (+ optional fake) |
| - | `viz_train.py` | look at the data and predictions |

### Key components in detail

**Decoder adapter (`decoders.py`).** Loads the starter kit's own wrappers by file path (the kit's package is
also called `wmloc`, so we import its files as `kit_watermarks` / `kit_utils`). Uses the exact kit
preprocessing (Pixel Seal 256 px; TrustMark 245 px, `x*2-1`; StegaStamp: the wrapper resizes to 400 px).
Forward hooks tap intermediate feature maps (`config.LAYERS`) and an **agreement map**: the decoder's
per-position bit scores before its global average (Pixel Seal `pixel_decoder.output_upscaling` ->
`pixel_decoder.linear`; TrustMark `layer4` -> `fc`), checked exact against the full decoder (diff ~1e-6).
`pos_logits` keeps those per-position scores for the dense losses.

**Known-message match (`heat.match_gap`).** For decoder scores s and the known bits: match = mean of
sign(bit) x tanh(s/4), minus the average match over all shuffles of the message (computed exactly, no
sampling). About 0 = "no better than a reordered message"; high = "this region reads our message".
Used by heatmaps, box search, seeded boxes, regime detector and RISE.

**Head (`head.py`).** Lateral 1x1 blocks per feature level, upsampled to 128x128, plus extra maps
(heatmaps / agreement) and an optional image branch (512 -> 128 by two stride-2 convs, keeps fine edges);
fuse block -> 1 logit per cell. Loss = BCE + symmetric Lovasz hinge (an IoU-shaped loss).

**Post-processing (`postprocess.py`).** Cutoff (tuned), never-empty (fallback: largest blob of the top 5%
cells), optional cleanup (minimum blob size, hole filling, rectangle fill) only if it beats the plain cutoff
by a margin on the tuning photos.

**Fake data (`step5_synth.py`, v2).** Follows the dataset recipe, with real-photo statistics (holdout fold
excluded): clean COCO val2017 photos (random crop/resize to 512), watermarked with the **kit's own embedders**;
50% *full* (whole image watermarked, 1-5 random smooth blobs restored, hole sizes from real full photos),
partial = *rect* (outpainting: watermark a rectangle sized like real ones, paste back) or *object*
(transplant an object from a watermarked photo onto another photo; COCO outline or random blob, size
0.7-1.4x real objects). Packs of 250 images as PNG bytes (`pack_<shard>_<k>.pt`; few files because the
scratch file-count cap; written as `tmp_*` then renamed, so readers never see half-written files).
Checks against real data: holes area 0.78-0.86 vs real 0.82-0.88; bit accuracy 0.81-0.86 vs 0.93-0.97.
Final set: 4,000 fake photos per method (shards 20-35, `slurm/gen.sh`).

**Dense decoder (`step6_stage_b.py`)** - the method that moved us most. Instead of a head on a frozen
decoder, we **retrain the decoder itself** together with the head, on fake + real photos
(`--real-weight 0.2` = 20% of batches are real photos), with three losses:
1. mask loss (BCE + Lovasz) on the head output;
2. `--bit-loss`: every decoder position **inside** the watermark must read the known bits
   (BCE of the per-position bit logits, weighted by the watermarked share of that position);
3. `--clean-loss`: positions in clean areas must read ~nothing (their logits pushed towards 0).
The decoder becomes a "where is my message" detector. Checkpoint picked on the real photos of fold 0
(`saved best`); bit accuracy stays healthy (0.70-0.87), so the decoder is not destroyed.

**PixNet (`step8_pixnet.py`).** U-Net (3.5M params at width 32) on the image + its high-pass
(x - 5x5 average), 4 levels, GroupNorm + GELU, output 128x128. Trained on random crops of real photos
(folds 0-2), checkpoint on fold 3; optional `--fake-weight` mixes fake packs. Writes its own test predictions.

**Step 4 ensembles.** Old behaviour: the "average of all tags" candidate was useless once tags were tuned on
different folds (`pix` on fold 3, `dense` on fold 0 -> 0 shared tuning photos, score `nan`). Now:
candidates with < 20 shared tuning photos are skipped, and `topK:` candidates average the best 2-5 single
tags (ranked by tune + holdout) that share a tuning set. This is what produced the TrustMark
`top3:dense5+dense9+dense4` and StegaStamp `top2:box+pix` choices.

## Per method: what works

### Pixel Seal

- Decoder `det`: ConvNeXt-V2 + pixel decoder, 33M params, input 256 px, 256-bit message, raw scores
  (257 outputs, column 0 = detection). Not flip-safe (vflip/rot90 break it).
- Probe bit accuracy on the true region: rectangle 0.86 / holes 0.93 / object 0.63.
- From the paper (`2512.16874`): the watermark is computed at 256 px, stretched to the image size and
  multiplied by a JND map (strong on edges/texture, ~0 in flat areas) -> flat regions carry almost no signal.
- Path: prior 0.29 -> Stage A 0.52 (holdout 0.521) -> crop heatmaps help slightly (only method where they
  work: 256-px crops, AUC 0.88) -> **dense decoder** `dense` holdout 0.557 / `dense_g` 0.553 ->
  whole-decoder variants on 4,000 fake photos (`dense5`, `dense9`).
- **Final: `dense9`** (whole ConvNeXt + pixel decoder unfrozen, dec-lr 3e-5, 4,000 fake/method, 54 min).
  Selection-fold mIoU during training reached 0.62.
- Weak spots (holdout of `dense_g`): small edits 0.44, rectangles 0.49; holes 0.62, objects 0.57.

### TrustMark

- Decoder `net`: ResNet-50, 24M params, input 245 px (`x*2-1`), 100 bits. Not flip-safe.
- Probe bit accuracy: rectangle 0.97 / holes 0.92 / object 0.55. Reads the exact watermark box very well,
  half-size crops not at all (crop heatmaps AUC ~0.5).
- Path: Stage A + heatmaps `stageAh` 0.52-0.53 -> Adobe box finder, box search, seeded boxes, matched
  filter: all no gain -> **dense decoder** (selection fold 0.558 -> 0.57).
- **Final: ensemble `top3:dense5+dense9+dense4`** (whole-decoder and layer2-4 variants, 4,000 fake/method).

### StegaStamp

- Decoder is an **ONNX file only** (no PyTorch network): no feature maps, no gradients, so no Stage A
  features and no dense decoder. Input 400 px (wrapper resizes), 100 bits. Not flip-safe (all flips break it).
- Probe bit accuracy: rectangle 0.89 / holes 0.94 / object 0.57.
- Path: Stage A / heatmaps 0.25-0.28 (below the prior 0.29!) -> **box search** 0.442 holdout (big jump) ->
  + PixNet in the ensemble 0.456 -> matched filter (AUC 0.65, no gain) -> RISE (no signal) ->
  **PixNet variants**.
- **Final: `pixS2`** (PixNet width 48, 15,000 steps, real photos). The weakest method and the main gap
  to the top teams; part of this last gain may be holdout luck (one model picked on 34 photos).

## Everything we tried, with numbers

"OOF" = cross-validated train score; "holdout" = fold 4; "sel" = checkpoint-picking fold (optimistic).

| Idea | Script / tag | Result | Kept? |
| --- | --- | --- | --- |
| Average mask | `prior` | ~0.29-0.30 | baseline |
| Stage A head on frozen decoder | `stageA` | OOF PS 0.520, TM 0.518; holdout PS 0.522 | yes (early) |
| + crop heatmaps as input | `stageAh` | OOF PS 0.526, TM 0.532, SS 0.275; holdout TM 0.524 | yes (early) |
| Crop heatmaps alone | `heat` | PS only has signal (256 px AUC 0.88); SS holdout 0.254 | in ensembles |
| Pixel Seal tile agreement (32-px grid) | `--tile-agree` | stageAh OOF 0.523 (was 0.526) | no |
| Memorise test (pipeline check) | Stage A `--memorise` | train mIoU 0.9986 | check passed |
| Box search | `box` | SS OOF 0.447, holdout 0.442 (prior 0.29); TM OOF 0.479 | SS yes |
| Adobe TrustMark box finder | `tmbox`, `tmboxAh` | box found on 100% but scores 0.35 vs stageAh 0.536; worst on objects 0.18 | no |
| Fake data v1 + frozen decoder (B0) | `stageB0` | PS 0.497, TM 0.496, SS 0.27-0.29; mean holdout 0.4455 | no |
| PixNet, real photos only | `pix` | PS 0.25, TM 0.27, **SS 0.37 (holdout 0.40)**; SS ensemble 0.443 -> 0.456 | SS yes |
| Seeded boxes + edge refinement | `srect`, `sbox` | TM/PS rect alone 0.43 vs seed 0.536/0.518; SS dev 0.451, holdout 0.445; compare 0.4970 < 0.5012 | no |
| Matched filter (8-bit templates) | `mf` | AUC SS 0.641 (15-px filter, 64-px windows), TM 0.530, PS 0.515 | no |
| Matched filter, full-precision templates | `step3g` `template()` | no change (SS 0.641, PS 0.508, TM 0.519): pattern buried in texture | no |
| Two-regime cutoffs | `step4c_regime` | detector 97-98% accurate; gain PS +0.001, TM 0.000 | no |
| Big PixNet + fake data | `pixf64` (width 64) | collapsed (val 0.25, loss flat 1.69) | no |
| **Dense decoder**, last stages, 2,000 fake | `dense`, `dense_g` | sel PS 0.590, TM 0.558; holdout PS 0.557 / 0.553 | yes |
| dense2: one more stage unfrozen | `dense2` | sel PS 0.607, TM 0.564 (overfits after ~20 min on 2,000 fake) | candidate |
| dense3: whole decoder, 2,000 fake | `dense3` | sel PS 0.591, TM 0.575 | candidate |
| dense4/5: as dense2/3 with 4,000 fake | `dense4`, `dense5` | sel PS 0.590 / 0.613, TM 0.570 / 0.570 after ~10 min; **chosen (mean holdout 0.548)** | yes |
| dense6/7: lr 5e-5 + more real / stronger bit+clean loss | `dense6`, `dense7` | sel TM 0.544 / 0.572 (killed after ~10 min) | candidate |
| dense8/9: dense5 recipe, 54 min (lr 2e-5 / 3e-5) | `dense8`, `dense9` | sel PS 0.607 / **0.621**, TM 0.572 / 0.566 (after ~10 min); **chosen** | yes |
| StegaStamp PixNet variants | `pixS1..4` (width 32 / 48 / crop 384 / +fake 0.3) | `pixS2` chosen for SS | yes |
| RISE saliency with known message | `rise` | raw cell AUC **0.514** (no signal); rise 0.448 = box 0.447 | no |
| Top-k ensembles in Step 4 | Step 4 fix | TM `top3`, SS `top2` chosen | yes |

## Running it on JURECA

Environment (every new shell; the team folder name has a space and an apostrophe, always quote it):

```bash
jutil env activate -p training2666
export TASK_DIR="/p/scratch/training2666/basit2/J'ai Deux Passion/Hackathon_Watermark_Localisation_Task"
export WORK_DIR=~/wmloc_work WMLOC_MODELS=~/wmloc_models SYNTH_DIR=~/wmloc_synth2
source "$TASK_DIR/.venv/bin/activate"        # Python 3.11, torch 2.5.1; starter kit in $TASK_DIR/starter_kit
cd ~/wmloc_src                                # this folder
export WMLOC_API_KEY=...                      # only in the shell, only for submitting; never in a file
```

- Code was run from `$HOME` (`~/wmloc_src`, logs in `~/output`, work data in `~/wmloc_work`) because the
  project's scratch file-count quota (4.4M inodes, shared by all teams) was full ("Disk quota exceeded").
  Home quota is 20 GB soft / 22 GB hard: keep an eye on `du -sh ~`.
- `slurm/gen.sh` must be copied to `~/gen.sh` (the chain scripts call it); it runs `step5_synth` for the 3
  methods and the given shards (COCO in `~/coco`).
- Submit a saved file: `python -c "import os; from wmloc import submission; submission.submit(os.environ['WORK_DIR'] + '/submissions/<file>.pt')"`
  (do not put `$TASK_DIR` inside the Python quotes: the apostrophe breaks it).

Slurm scripts (all `--account=training2666`, `--gres=gpu:N`):

| Script | Use |
| --- | --- |
| `run.sh` / `run3.sh` / `run4.sh` | one step on 1 / 3 / 4 GPUs of one node, reservation `cispahack` |
| `chain_dense.sh` | gen fake data + `dense` + `pixf64` + compare (one job) |
| `chain_dense2.sh` | gen +2,000/method + `dense4`/`dense5` + compare (one job) |
| **`chain_final.sh`** | the final job: `dense8`/`dense9` + StegaStamp PixNets + RISE, then Step 4 over all tags |
| `*_gen.sh`, `*_devel.sh`, `train_*_devel.sh`, `run_devel.sh` | general-queue / devel copies. **Do not run in parallel with a reservation job** (see below) |

**Cluster rules we learned the hard way:**
- **One job of the team at a time, period** (running or pending, any partition). We were banned twice on
  Nov 22: once for several reservation jobs, once for running `dc-gpu-devel` jobs next to a reservation job.
  Multi-step work = **one** job script that runs the steps in sequence or on its 4 GPUs (like `chain_final.sh`).
- The `cispahack` reservation is **MAGNETIC**: plain `dc-gpu` jobs without a reservation get pulled into it.
- Before every submit, both must print nothing:
  `squeue -r --reservation=cispahack -h -o '%u|%T|%Z' | grep -F "J'ai Deux Passion"` and
  `squeue -r --reservation=cispahack -h -u <team logins>` (the first misses jobs submitted from `$HOME`).
- Some login nodes miss `/p/scratch` or `$HOME`: check `ls -d ~/wmloc_src /p/scratch/training2666`, reconnect
  if missing. After the unban only `jureca08.fz-juelich.de` (jrlogin08) had both scratch and Slurm access.
- No heavy work on login nodes; Slurm 25.11 needs `--gres=gpu:X`.

## Lessons and next steps

- The decoders already "know" the message; teaching them **where** they read it (bit loss inside, clean
  loss outside) beat every hand-made search (heatmaps, boxes, refinement, matched filter, RISE).
- More fake data that follows the dataset recipe helped (2,000 -> 4,000 per method), and the dense models
  overfit fake data after ~20 min at 2,000 images, so more data / more variety is the next lever.
- Next with more time: cross-train the best dense recipe on folds 0-3 and average the 4 models;
  final retrain on all 170 photos with the chosen settings; convert the StegaStamp ONNX decoder to PyTorch
  (e.g. `onnx2torch`) so it can get the dense treatment, since StegaStamp is the weakest method.
