# Bigram models and K-means families

Final CSV: `final_predictions.csv`,33 endpoint rows. Frozen Bigram cutoff: `0.02375075405834256`. No network, collection or submission.

Family inference uses K-means with exactly5 clusters on33-dimensional endpoint score profiles: signed Bigram MMD² to every endpoint, diagonal zero. No scaling or clipping. Fixed seed0,20 K-means++ restarts, Lloyd updates, lowest inertia. Family letters A–E are arbitrary cluster labels; this does not identify known familyP or validate five true families.

Model inference uses the same frozen Bigram complete-linkage cutoff inside each predicted family. This enforces same-model implies same-family. No cutoff is fitted to unknown endpoints. Original global Bigram decisions and any changes caused by the hierarchy are retained in configuration.json and pair_scores.csv.

Model pair decisions changed by the family constraint: **17 of528**. Total predicted models: **7**.

| Family | Endpoints | Models |
| --- | --- | --- |
| A | ep-01, ep-02, ep-03, ep-04, ep-06, ep-07, ep-08, ep-12, ep-14, ep-17, ep-18, ep-20, ep-21, ep-22, ep-23, ep-24, ep-25, ep-26, ep-28, ep-30, ep-32, ep-33 | A1 |
| B | ep-05, ep-09, ep-11, ep-13, ep-15, ep-16 | B1, B2, B3 |
| C | ep-10, ep-19, ep-31 | C1 |
| D | ep-27 | D1 |
| E | ep-29 | E1 |

No real-endpoint MCC is available; there are no ground-truth labels. The labeled references all belong toP, so this family rule cannot be validated with them. K-means forces five families because requested, not because five was learned from validated data.

Reproduce from model_equality:

```powershell
python -B RESULTS/run.py --output-dir RESULTS/reproduced
```

The runner refuses to overwrite final_predictions.csv. Use a fresh --output-dir inside RESULTS; running it never submits. configuration.json stores source hashes, centroid coordinates, endpoint feature order, every restart inertia and the full Bigram configuration.
