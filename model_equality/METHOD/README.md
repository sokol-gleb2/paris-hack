# Frozen final Bigram method

The active default `results/frozen_method.json` is an exact copy of `results/bigram/frozen_method.json`: Bigram cutoff `0.02375075405834256`, raw Unicode prefix 200, signed unbiased MMD², equal prompt weighting and complete-linkage clustering.

`vendor/` contains all copied mathematical dependencies needed by the scorer. It makes no imports from archived experiments. `source_manifest.json` records historical source paths and hashes; those original source files now reside in `../../Extra/model_equality_archive_20261122_104141/`. Frozen source checks verify the retained copies here.

The labeled calibration data, training-half configurations, reference validation and baseline endpoint predictions remain available. Unused method configurations and the broad benchmark runner were archived.

For Bigram model-only predictions, from `model_equality` run:

```powershell
python -B METHOD/score_endpoints.py --responses DATA/endpoint_responses.jsonl --output-dir METHOD/bigram_reproduced
```

For the final family-and-model pipeline, use `RESULTS/run.py` as documented in the parent README. The model-only scorer does not add K-means family labels. The final submitted CSV is `../RESULTS/final_predictions.csv`.
