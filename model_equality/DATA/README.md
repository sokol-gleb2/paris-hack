# Frozen data used by the final submission

| File | Endpoints | Replies | Labels |
| --- | ---: | ---: | --- |
| `reference_responses.jsonl` | 9 | 450 | Organizer family/model/kind |
| `endpoint_responses.jsonl` | 33 | 1,650 | No identity labels |

Both files contain the exact shared prompts `L01_P01` through `L05_P01`, with ten draws per prompt and endpoint. Exact prompt text and per-cell counts are in `selected_prompts.jsonl` and `manifest.json`. The snapshot was frozen on 2026-11-22 at 09:24:14 UTC. Other prompts from the 900-reference source were incomplete across all scored endpoints at that time and are excluded.

All source fields are preserved except endpoint identity-label fields. Nested `response_body.model` is API routing metadata, not ground truth. Identical answers from distinct request records remain separate.

These immutable files are the inputs to the final retained pipeline. The original collections and snapshot exporter are archived in `../../Extra/model_equality_archive_20261122_104141/`; manifest source paths are relative to that archived original tree. No archive lookup is needed for prediction. Do not add later replies to this final dataset.
