"""Central settings. Edit the ADAPT lines after Step 1 (step1_probe.py)."""
import os
from pathlib import Path

# ---- paths -----------------------------------------------------------------
# export TASK_DIR="/p/scratch/training2666/basit2/J'ai Deux Passion/Hackathon_Watermark_Localisation_Task"
TASK_DIR = Path(os.environ.get("TASK_DIR", "."))
DATA_DIR = Path(os.environ.get("DATA_DIR", TASK_DIR / "release"))  # unzipped train/ and test/
WORK_DIR = Path(os.environ.get("WORK_DIR", TASK_DIR / "output" / "wmloc_work"))
CACHE_DIR = WORK_DIR / "cache"        # cached features, heatmaps, OOF predictions, groups.json
CKPT_DIR = WORK_DIR / "checkpoints"   # head / decoder weights
SYNTH_DIR = Path(os.environ.get("SYNTH_DIR", WORK_DIR / "synth"))  # fake data (put it in $HOME: scratch is full)
SUB_DIR = WORK_DIR / "submissions"
for _d in (CACHE_DIR, CKPT_DIR, SYNTH_DIR, SUB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- task constants (from the task description) -----------------------------
IMG = 512          # image side in pixels
GRID = 128         # mask side in cells
CELL = IMG // GRID # 4 pixels per cell side
N_FOLDS = 5
SEED = 0

# Exact method strings as they appear in the dataset metadata (checked in Step 0)
METHODS = ["pixelseal", "trustmark", "stegastamp"]

# Max aspect ratio of the watermarked rectangle, before the edit (task description)
MAX_ASPECT = {"pixelseal": 16.0, "trustmark": 2.0, "stegastamp": 16.0}

# ---- validation -------------------------------------------------------------
# Fold kept untouched until the final choice (Step 4/7): no training, checkpoint picking,
# tuning or fake-data design uses it (34 images per method). None = off.
HOLDOUT_FOLD = 4

# ---- decoder wiring: fill these in after Step 1 -----------------------------
# Attribute path of the decoder network inside the starter-kit wrapper, e.g. "decoder".
# None = guess the largest network inside the wrapper, which can be the EMBEDDER.
DECODER_NET = {"pixelseal": "det", "trustmark": "net", "stegastamp": "none"}  # kit: "none" = ONNX, no torch net
# What the outputs are (step1_probe prints them; no guessing from value ranges):
#   "logits" = raw score, > 0 means bit 1 (public Pixel Seal and TrustMark code; StegaStamp's original)
#   "probs"  = probability, > 0.5 means bit 1        "bits" = already 0/1
WRAPPER_OUTPUT = {"pixelseal": "bits", "trustmark": "bits", "stegastamp": "bits"}   # ADAPT: wrapper.decode()
NET_OUTPUT = {"pixelseal": "logits", "trustmark": "logits", "stegastamp": "logits"} # ADAPT: the network itself
# Names of decoder submodules whose outputs feed the head (2-4 layers, from
# fine to coarse). Get candidates from step1_probe.py --list-layers.
LAYERS = {
    "pixelseal": ["convnext.stages.0", "convnext.stages.1", "convnext.stages.2", "convnext.stages.3"],  # GUESS: confirm with --list-layers
    "trustmark": ["layer1", "layer2", "layer3", "layer4"],   # kit's TrustMark.features()
    "stegastamp": [],                                         # ONNX: no feature maps
}
# Side length the decoder network expects (the wrapper usually resizes to it). Take it from
# the probe's "wrapper feeds net" line. Public code: Pixel Seal 256; TrustMark 245 for
# variants Q/B/C and 224 for P (Adobe's trustmark.py); StegaStamp 400.
DECODER_INPUT_SIZE = {"pixelseal": 256, "trustmark": 245, "stegastamp": 400}  # ADAPT
# True only if step1_probe.py shows bit accuracy survives flips/rot90.
FLIP_SAFE = {"pixelseal": False, "trustmark": False, "stegastamp": False}     # ADAPT
# Agreement channel: (module whose OUTPUT is the spatial map that gets averaged, the Linear
# layer applied after the average). Only exact if nothing sits between the average and the
# Linear; step1_probe checks it ("agreement check").
#   pixelseal (public code): ("<prefix>.pixel_decoder.output_upscaling", "<prefix>.pixel_decoder.linear")
#   trustmark: a ResNet-50 in Adobe's code; if --list-layers shows avgpool -> fc, try ("<prefix>.layer4", "<prefix>.fc")
#   stegastamp: None (flatten + dense layers, no average)
AGREEMENT = {"pixelseal": ("pixel_decoder.output_upscaling", "pixel_decoder.linear"),
             "trustmark": ("layer4", "fc"), "stegastamp": None}   # both exact per the kit
# Stage B: name prefixes of decoder modules to unfreeze (the last stage).
UNFREEZE = {"pixelseal": ["convnext.stages.3"], "trustmark": ["layer4"], "stegastamp": []}  # pixelseal name: GUESS

# ---- crop heatmaps (Step 3b) -----------------------------------------------
HEAT_SIZES = (256, 128)   # crop sides in pixels (multiples of 4); crops overlap by half
