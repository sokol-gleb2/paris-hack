# Final model-equality submission

The final submitted file is **`RESULTS/final_predictions.csv`**. It contains 33 endpoints, five predicted families, and seven predicted model groups. The portal accepted it with a **public-split score of 0.05**. The receipt is `RESULTS/submission_attempt.json`.

## Retained files

| Path | Purpose |
| --- | --- |
| `DATA/` | Frozen shared five-prompt panel: 450 labeled reference replies and 1,650 unlabeled endpoint replies, manifests and validation |
| `METHOD/` | Frozen Bigram scorer, copied mathematical dependencies, calibration artifacts and baseline predictions |
| `RESULTS/run.py` | Offline final prediction pipeline |
| `RESULTS/final_predictions.csv` | The exact submitted endpoint/family/model labels |
| `RESULTS/configuration.json` | Bigram cutoff, K-means settings, centroids, provenance and predicted groups |
| `RESULTS/report.md` | Final method explanation |
| `submission.py` | Portal upload script, already configured for the final CSV |
| `requirements.txt` | Minimal runtime packages |

This publication contains the final DATA, METHOD and RESULTS package. Local credentials, Git history, agent instructions and historical experiment archives are not included. Archived source paths in provenance records identify the original local evidence and are not runtime dependencies.

## Reproduce predictions offline

From `model_equality`, with Python 3.10 or newer and NumPy installed:

```powershell
python -B RESULTS/run.py --output-dir RESULTS/reproduced
```

Choose a fresh output directory under `RESULTS/`. This uses the retained DATA snapshot and frozen Bigram parameters, generates the final CSV, and does not submit or query endpoints. It refuses to overwrite previous predictions.

The method uses signed Bigram MMD² on the first 200 Unicode characters, equally weighted across five prompts. K-means with K=5 clusters each endpoint's score profile into families. Complete-linkage model clustering runs within those families with the frozen cutoff `0.02375075405834256`. Family names are arbitrary labels; the references contain only family P, so the family rule is unvalidated. Full details and seeds are in `RESULTS/configuration.json`.

## Upload the existing final CSV

Only when another upload is intended:

```powershell
python -u -B submission.py
```

This sends `RESULTS/final_predictions.csv` to the portal using the PARIS_API_KEY environment variable. Cleanup and offline verification do not run this command. No new upload was made during cleanup.



## API credential configuration

Offline prediction requires no credential. Before an explicitly intended portal upload, set the credential in your shell without committing it:

```powershell
$env:PARIS_API_KEY = "your-team-api-key"
python -u -B submission.py
```

The literal placeholder above is not a credential. Keep the real value private. The repository contains no API key.
