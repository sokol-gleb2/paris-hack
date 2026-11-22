"""Step 3c: the "prior" model per method: the average train mask, no image input.
Use it as the FIRST (baseline) tag in Step 4: a method keeps the prior unless a model beats it.
CPU, seconds. Run: python -m scripts.step3c_prior
"""
import numpy as np
import torch

from wmloc import config as C
from wmloc import data


def main():
    train, test = data.cached_split("train"), data.cached_split("test")
    folds = data.make_folds(train)
    for m in C.METHODS:
        recs = data.by_method(train, m)
        names = [r.name for r in recs]
        gts = np.stack([r.mask128 for r in recs]).astype(np.float32)
        oof = np.zeros_like(gts)
        for k in range(C.N_FOLDS):          # fit on other folds, never on the holdout
            fit = [i for i, n in enumerate(names) if folds[n] != k and folds[n] != C.HOLDOUT_FOLD]
            for i, n in enumerate(names):
                if folds[n] == k:
                    oof[i] = gts[fit].mean(0)
        te = data.by_method(test, m)
        prior = np.repeat(gts.mean(0)[None], len(te), axis=0)
        torch.save({"names": names, "probs": torch.from_numpy(oof).half(),
                    "gts": torch.from_numpy(gts.astype(np.uint8)), "folds": [folds[n] for n in names]},
                   C.CACHE_DIR / f"oof_prior_{m}.pt")
        torch.save({"names": [r.name for r in te], "probs": torch.from_numpy(prior).half()},
                   C.CACHE_DIR / f"test_prior_{m}.pt")
        print(f"{m}: prior saved")


if __name__ == "__main__":
    main()
