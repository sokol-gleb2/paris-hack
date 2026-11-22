"""Fit or recover the Task 3 wildcard checkpoint (public entry 3768)."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

ASSETS = Path(__file__).resolve().parent / "assets"
MODEL_SHA = "485881a03ad064b5e6defc245ae3f45fd96714bf065ff96543a5eeca9f642571"
IMAGES_SHA = "5b1d9deaf475db0905973aa028511ab35aebc33196412f5ef9ebbef09f1541c7"
READOUT_SHA = "6dd838db325ec761d880d21794fb99a3b0e5239d3f320f7d4c5b736af90f49ae"
PATCH_SHA = "23621ff3129efef330ababad3e6ba3af5a4b76e583fc8cc9412718713665d3aa"
SCORED_SHA = "2be9bdf5e38b6c00be867f941bd51ad1462b2a73c96e8d4e7f56405c8b8bcacb"


def verify(path, expected):
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"{path}: expected SHA-256 {expected}, got {actual}")


def fit(model_path, images_path, stream):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    import task_template
    from phase9.wildcard_occlusion import unlearn

    if not torch.cuda.is_available():
        raise RuntimeError("Fitting requires CUDA. Use --restore to recover the scored file on CPU.")
    verify(images_path, IMAGES_SHA)
    readout_path = ASSETS / "cpu_readout.npz"
    verify(readout_path, READOUT_SHA)
    torch.set_num_threads(4)
    model = task_template.load_model(model_path)
    images = task_template.load_forget_images(images_path)
    with np.load(readout_path, allow_pickle=False) as cache:
        readout = torch.from_numpy(cache["head"])
    unlearn(model, images, cpu_readout=readout)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("model.pt"))
    parser.add_argument("--images", type=Path, default=Path("forget_images.npz"))
    parser.add_argument("--output", type=Path, default=Path("output/submission.pt"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--restore", action="store_true", help="Apply the scored checkpoint patch using zstd.")
    mode.add_argument("--verify", action="store_true", help="Verify an existing output file.")
    args = parser.parse_args()
    if args.verify:
        verify(args.output, SCORED_SHA)
    else:
        verify(args.model, MODEL_SHA)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            verify(args.output, SCORED_SHA)
        with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with temporary.open("wb") as stream:
                if args.restore:
                    patch = ASSETS / "submission.patch.zst"
                    verify(patch, PATCH_SHA)
                    subprocess.run(["zstd", "-q", "-dc", "--long=29",
                                    f"--patch-from={args.model}", str(patch)], stdout=stream, check=True)
                else:
                    fit(args.model, args.images, stream)
            verify(temporary, SCORED_SHA)
            os.replace(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Verified {args.output}  sha256={SCORED_SHA}", flush=True)


if __name__ == "__main__":
    main()
