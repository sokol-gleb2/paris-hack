import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'METHOD'))
import inference
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def kmeans(x, k=5, seed=0, restarts=20):
    rng = np.random.default_rng(seed)
    if len(np.unique(x, axis=0)) < k:
        raise ValueError('Fewer distinct score profiles than requested clusters')
    best = None
    runs = []
    for restart in range(restarts):
        chosen = [int(rng.integers(len(x)))]
        while len(chosen) < k:
            distances = ((x[:, None, :] - x[chosen][None, :, :])**2).sum(axis=2).min(axis=1)
            chosen.append(int(rng.choice(len(x), p=distances/distances.sum())))
        centers = x[chosen].copy()
        previous = None
        for iteration in range(300):
            squared = ((x[:, None, :] - centers[None, :, :])**2).sum(axis=2)
            labels = squared.argmin(axis=1)
            counts = np.bincount(labels, minlength=k)
            for empty in np.flatnonzero(counts == 0):
                candidates = [i for i in range(len(x)) if counts[labels[i]] > 1]
                farthest = max(candidates, key=lambda i: (squared[i, labels[i]], -i))
                counts[labels[farthest]] -= 1
                labels[farthest] = empty
                counts[empty] += 1
            centers = np.array([x[labels == i].mean(axis=0) for i in range(k)])
            if previous is not None and np.array_equal(previous, labels):
                break
            previous = labels.copy()
        else:
            raise ValueError('K-means did not converge')
        groups = sorted(tuple(np.flatnonzero(labels == i).tolist()) for i in range(k))
        canonical = np.empty(len(x), dtype=int)
        for i, members in enumerate(groups):
            canonical[list(members)] = i
        centers = np.array([x[canonical == i].mean(axis=0) for i in range(k)])
        inertia = float(((x-centers[canonical])**2).sum())
        runs.append({'restart': restart, 'inertia': inertia, 'iterations': iteration+1})
        rank = inertia, groups
        if best is None or rank < best[0]:
            best = rank, canonical.copy(), centers.copy()
    _, labels, centers = best
    squared = ((x[:, None, :] - centers[None, :, :])**2).sum(axis=2)
    if not np.allclose(squared[np.arange(len(x)), labels], squared.min(axis=1), atol=1e-12, rtol=0):
        raise AssertionError('K-means assignments are not nearest-centroid assignments')
    return labels, centers, runs


def main():
    global HERE
    parser = argparse.ArgumentParser(description='Offline Bigram and fixed-K family prediction pipeline; never submits.')
    parser.add_argument('--output-dir', type=Path, default=HERE)
    parser.add_argument('--k', type=int, default=5, choices=range(2, 27))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT/'RESULTS'):
        raise ValueError('Final pipeline outputs must remain inside RESULTS')
    HERE = output
    HERE.mkdir(parents=True, exist_ok=True)
    final = HERE/'final_predictions.csv'
    if final.exists():
        raise FileExistsError('Final predictions already exist; preserve previous output')
    source = ROOT/'DATA/endpoint_responses.jsonl'
    config_path = ROOT/'METHOD/results/bigram/frozen_method.json'
    protected = [source, config_path, ROOT/'DATA/reference_responses.jsonl']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['method'] != 'Bigram' or config['recipe']['cutoff'] != 0.02375075405834256:
        raise ValueError('Expected the frozen shared-panel Bigram configuration')
    raw = inference.read_rows(source)
    result = inference.infer(config, raw)
    endpoints = sorted(result['groups'])
    if endpoints != [f'ep-{i:02d}' for i in range(1,34)] or len(result['pairs']) != 528:
        raise ValueError('Expected all33 endpoints and528 pairs')
    indices = {e: i for i,e in enumerate(endpoints)}
    matrix = np.zeros((len(endpoints), len(endpoints)))
    for (a,b), score in zip(result['pairs'], result['scores'], strict=True):
        matrix[indices[a], indices[b]] = matrix[indices[b], indices[a]] = score
    labels, centers, runs = kmeans(matrix, k=args.k)
    families = {e: chr(65+int(labels[i])) for i,e in enumerate(endpoints)}
    model_groups = {}
    family_groups = {}
    for family in sorted(set(families.values())):
        members = [e for e in endpoints if families[e] == family]
        family_groups[family] = members
        pairs = [(a,b) for i,a in enumerate(members) for b in members[i+1:]]
        if len(members) == 1:
            groups = {members[0]: 'M01'}
        else:
            scores = [matrix[indices[a], indices[b]] for a,b in pairs]
            groups = inference.core.cluster(scores, pairs, members, config['recipe']['cutoff'])
        model_groups.update({e: family+str(int(label[1:])) for e,label in groups.items()})
    changed = []
    for a,b in result['pairs']:
        original = result['groups'][a] == result['groups'][b]
        hierarchical = model_groups[a] == model_groups[b]
        if original != hierarchical:
            changed.append({'endpoint_a': a, 'endpoint_b': b, 'original_same_model': original, 'final_same_model': hierarchical})
    baseline = json.loads((ROOT/'METHOD/results/bigram/endpoint_inference/inference_summary.json').read_text())
    difference = max(abs(a-b) for a,b in zip(result['scores'],baseline['scores'],strict=True))
    if difference > 1e-12 or result['groups'] != baseline['groups']:
        raise AssertionError('Frozen baseline reproduction failed')
    again = inference.infer(config, list(reversed(raw)))
    if again['scores'] != result['scores'] or again['groups'] != result['groups']:
        raise AssertionError('Input row order affected predictions')
    repeated_labels, repeated_centers, _ = kmeans(matrix, k=args.k)
    if not np.array_equal(labels, repeated_labels) or not np.array_equal(centers, repeated_centers):
        raise AssertionError('Fixed seed K-means is not reproducible')
    with final.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['endpoint', 'family', 'model'])
        writer.writerows((e, families[e], model_groups[e]) for e in endpoints)
    with (HERE/'pair_scores.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['endpoint_a','endpoint_b','bigram_mmd2','global_bigram_same_model','same_family','final_same_model'])
        writer.writerows((a,b,score,int(result['groups'][a]==result['groups'][b]),int(families[a]==families[b]),int(model_groups[a]==model_groups[b])) for (a,b),score in zip(result['pairs'],result['scores'],strict=True))
    with (HERE/'score_profiles.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['endpoint', *endpoints])
        writer.writerows([e,*matrix[i].tolist()] for i,e in enumerate(endpoints))
    if any(digest(ROOT/p) != value for p,value in hashes.items()):
        raise AssertionError('Protected inputs changed')
    if len(family_groups) != args.k or len(model_groups) != 33 or any(not model_groups[e].startswith(families[e]) for e in endpoints):
        raise AssertionError('Final hierarchy or coverage is invalid')
    settings = {'created_utc': datetime.now(timezone.utc).isoformat(), 'method': 'Frozen Bigram with complete linkage within K-means predicted families', 'model_cutoff': config['recipe']['cutoff'], 'family_algorithm': 'NumPy Lloyd K-means, K-means++ initialization, unscaled Euclidean score profiles', 'k': args.k, 'seed': 0, 'restarts': 20, 'max_iterations': 300, 'feature_columns': endpoints, 'feature_definition': 'Each endpoint is represented by its33 signed Bigram MMD2 scores to all endpoints; symmetric matrix, diagonal0; no clipping/scaling/MDS.', 'family_labels': 'Arbitrary alphabetical IDs ordered by smallest member endpoint, not known family names.', 'centers': centers.tolist(), 'restart_results': runs, 'family_groups': family_groups, 'model_groups': model_groups, 'original_global_bigram_groups': result['groups'], 'changed_model_pairs': changed, 'source_hashes': hashes, 'bigram_configuration': config, 'family_validation': f'Unvalidated: k={args.k} is user prescribed; references provide only familyP; no real ground truth.', 'requests': 0}
    save(HERE/'configuration.json', settings)
    save(HERE/'validation.json', {'records': len(raw), 'endpoints': len(endpoints), 'pairs': len(result['pairs']), 'families': len(family_groups), 'models': len(set(model_groups.values())), 'baseline_max_score_difference': difference, 'row_order_invariant': True, 'seed_reproducible': True, 'centroid_means_and_nearest_assignments_checked': True, 'protected_inputs_unchanged': True, 'hierarchy_valid': True, 'final_csv_sha256': digest(final), 'requests': 0})
    lines = ['# Bigram models and K-means families', '', f'Final CSV: `final_predictions.csv`,33 endpoint rows. Frozen Bigram cutoff: `{config["recipe"]["cutoff"]!r}`. No network, collection or submission.', '', f'Family inference uses K-means with exactly{args.k} clusters on33-dimensional endpoint score profiles: signed Bigram MMD² to every endpoint, diagonal zero. No scaling or clipping. Fixed seed0,20 K-means++ restarts, Lloyd updates, lowest inertia. Family letters A–E are arbitrary cluster labels; this does not identify known familyP or validate five true families.', '', 'Model inference uses the same frozen Bigram complete-linkage cutoff inside each predicted family. This enforces same-model implies same-family. No cutoff is fitted to unknown endpoints. Original global Bigram decisions and any changes caused by the hierarchy are retained in configuration.json and pair_scores.csv.', '', f'Model pair decisions changed by the family constraint: **{len(changed)} of528**. Total predicted models: **{len(set(model_groups.values()))}**.', '', '| Family | Endpoints | Models |', '| --- | --- | --- |']
    for family,members in family_groups.items():
        lines.append('| '+family+' | '+', '.join(members)+' | '+', '.join(sorted({model_groups[e] for e in members}))+' |')
    lines += ['', 'No real-endpoint MCC is available; there are no ground-truth labels. The labeled references all belong toP, so this family rule cannot be validated with them. K-means forces the requested number of families; this was not learned from validated data.', '', 'Reproduce from model_equality:', '', '```powershell', f'python -B RESULTS/run.py --k {args.k} --output-dir RESULTS/reproduced_k{args.k}', '```', '', 'Use a fresh output directory inside RESULTS; the runner refuses to overwrite an existing final_predictions.csv. configuration.json stores source hashes, centroid coordinates, endpoint feature order, every restart inertia and the full Bigram configuration. Running this script never submits.', '']
    (HERE/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'csv': str(final), 'families': {f: len(es) for f,es in family_groups.items()}, 'models': len(set(model_groups.values())), 'changed_model_pairs': len(changed), 'baseline_max_score_difference': difference}, indent=2))


if __name__ == '__main__':
    main()
