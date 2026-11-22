import itertools
import math
import os
import string
import sys
from collections import Counter
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np


from vendor.identity_rule_comparison.engine import MAX_DEPTH, MIN_LEAF, _grow, _tree_score
from vendor.kernel_comparison.engine import auroc, bigram_similarity, normalized_bigrams, signed_mmd
from vendor.looop.hamming import clusters, fit_cutoff

WIDTH = 200
BASE_FEATURES = ['Hamming', 'Bigram', 'Trigram', 'Structural']
METHODS = ['Hamming', 'Bigram', 'Trigram', 'Hamming + bigram', 'Bigram + trigram', 'Structural', 'Logistic regression', 'Hamming + decision tree']
TREE_FEATURES = ['mean', 'std', 'min', 'median', 'max']


def normalized_ngrams(text, n):
    text = text[:WIDTH]
    if n == 2:
        return normalized_bigrams(text)
    counts = Counter(text[i:i + n] for i in range(max(0, len(text) - n + 1)))
    norm = math.sqrt(sum(value * value for value in counts.values()))
    return {key: value / norm for key, value in counts.items()} if norm else {}


def structural_vector(text):
    prefix = text[:WIDTH]
    return np.array([len(prefix), prefix.count('\n'), sum(c.isspace() for c in prefix), sum(c in string.punctuation for c in prefix)], dtype=float) / WIDTH


def gram_matrix(texts, name):
    if name == 'Hamming':
        encoded = np.full((len(texts), WIDTH), -1, dtype=np.int64)
        for index, text in enumerate(texts):
            prefix = text[:WIDTH]
            encoded[index, :len(prefix)] = list(map(ord, prefix))
        return np.count_nonzero(encoded[:, None, :] == encoded[None, :, :], axis=2).astype(float) / WIDTH
    if name in ('Bigram', 'Trigram'):
        vectors = [normalized_ngrams(text, 2 if name == 'Bigram' else 3) for text in texts]
        result = np.empty((len(texts), len(texts)), dtype=float)
        for i, first in enumerate(vectors):
            for j in range(i, len(vectors)):
                value = bigram_similarity(first, vectors[j])
                result[i, j] = result[j, i] = value
        return result
    if name == 'Structural':
        vectors = np.array([structural_vector(text) for text in texts])
        return vectors @ vectors.T / 4.0
    raise ValueError('Unknown base kernel: ' + name)


def _aggregate(per_prompt):
    return {name: np.array([math.fsum(float(value) for value in matrix[:, column]) / len(matrix) for column in range(matrix.shape[1])]) for name, matrix in per_prompt.items()}


def compute_features(rows, prompt_ids=None):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('At least one saved response is required')
    seen = set()
    prompts = {}
    endpoints = set()
    cells = {}
    for row in rows:
        endpoint, identifier, repeat = row.get('endpoint'), row.get('id'), row.get('repeat')
        if not isinstance(endpoint, str) or not endpoint or not isinstance(identifier, str) or not identifier:
            raise ValueError('Endpoint and prompt IDs must be nonempty strings')
        if type(repeat) is not int or repeat < 0:
            raise ValueError('Each saved draw needs a nonnegative integer repeat')
        if not isinstance(row.get('prompt'), str) or not row['prompt'] or not isinstance(row.get('response'), str):
            raise ValueError('Each draw needs exact prompt and response strings')
        key = endpoint, identifier, repeat
        if key in seen:
            raise ValueError('Duplicate endpoint/prompt/repeat collection record')
        seen.add(key)
        if identifier in prompts and prompts[identifier] != row['prompt']:
            raise ValueError('Prompt text differs across saved draws or endpoints')
        prompts[identifier] = row['prompt']
        endpoints.add(endpoint)
        cells.setdefault((endpoint, identifier), []).append((repeat, row['response']))
    endpoints = sorted(endpoints)
    if len(endpoints) < 2:
        raise ValueError('At least two endpoints are required')
    if prompt_ids is None:
        prompt_ids = sorted(prompts)
    else:
        if not isinstance(prompt_ids, (list, tuple)) or not prompt_ids or len(set(prompt_ids)) != len(prompt_ids) or not set(prompt_ids) <= set(prompts):
            raise ValueError('Requested prompt IDs must be unique and present')
        prompt_ids = sorted(prompt_ids)
    pairs = list(itertools.combinations(endpoints, 2))
    per_prompt = {name: np.empty((len(prompt_ids), len(pairs)), dtype=float) for name in BASE_FEATURES}
    counts = {}
    for row_index, identifier in enumerate(prompt_ids):
        texts, indices = [], {}
        counts[identifier] = {}
        for endpoint in endpoints:
            cell = sorted(cells.get((endpoint, identifier), []))
            if len(cell) < 2:
                raise ValueError('Each endpoint/shared prompt requires at least two distinct draws')
            counts[identifier][endpoint] = len(cell)
            indices[endpoint] = list(range(len(texts), len(texts) + len(cell)))
            texts.extend(text for _, text in cell)
        for name in BASE_FEATURES:
            gram = gram_matrix(texts, name)
            for column, (left, right) in enumerate(pairs):
                per_prompt[name][row_index, column] = signed_mmd(gram, indices[left], indices[right])
    return {'endpoints': endpoints, 'prompt_ids': prompt_ids, 'pairs': pairs, 'per_prompt': per_prompt, 'aggregates': _aggregate(per_prompt), 'counts': counts, 'prompts': {p: prompts[p] for p in prompt_ids}, 'response_count': sum(sum(v.values()) for v in counts.values())}


def subset_bundle(bundle, prompt_ids):
    if not prompt_ids or len(set(prompt_ids)) != len(prompt_ids) or not set(prompt_ids) <= set(bundle['prompt_ids']):
        raise ValueError('Subset must contain unique known prompt IDs')
    prompt_ids = sorted(prompt_ids)
    index = [bundle['prompt_ids'].index(p) for p in prompt_ids]
    per_prompt = {name: values[index, :].copy() for name, values in bundle['per_prompt'].items()}
    result = {'endpoints': list(bundle['endpoints']), 'prompt_ids': prompt_ids, 'pairs': list(bundle['pairs']), 'per_prompt': per_prompt, 'aggregates': _aggregate(per_prompt)}
    if 'counts' in bundle:
        result['counts'] = {p: dict(bundle['counts'][p]) for p in prompt_ids}
        result['response_count'] = sum(sum(v.values()) for v in result['counts'].values())
    if 'prompts' in bundle:
        result['prompts'] = {p: bundle['prompts'][p] for p in prompt_ids}
    return result


def logistic_matrix(bundle):
    return np.column_stack([bundle['aggregates'][name] for name in BASE_FEATURES])


def tree_matrix(bundle):
    values = bundle['per_prompt']['Hamming']
    return np.column_stack([bundle['aggregates']['Hamming'], values.std(axis=0), values.min(axis=0), np.median(values, axis=0), values.max(axis=0)])


def sigmoid(values):
    return np.exp(-np.logaddexp(0.0, -np.asarray(values, dtype=float)))


def fit_logistic(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if set(y.tolist()) != {0.0, 1.0}:
        raise ValueError('Logistic fitting requires both identity classes')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    z = (x - mean) / scale
    sample_weight = np.array([len(y) / (2.0 * np.count_nonzero(y == label)) for label in y])
    total = sample_weight.sum()
    design = np.column_stack([np.ones(len(y)), z])
    penalty = np.diag([0.0] + [1.0 / total] * x.shape[1])
    backend_error = None
    try:
        from sklearn.linear_model import LogisticRegression
    except (ImportError, ValueError) as error:
        LogisticRegression = None
        backend_error = f'{type(error).__name__}: {error}'
    def objective(theta):
        logits = design @ theta
        return float(np.dot(sample_weight, np.logaddexp(0, logits) - y * logits) / total + 0.5 * theta @ penalty @ theta)
    if LogisticRegression is not None:
        model = LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000, tol=1e-10, random_state=0)
        model.fit(z, y)
        theta = np.r_[model.intercept_, model.coef_[0]]
        backend, iteration = 'scikit-learn LogisticRegression', int(model.n_iter_[0])
    else:
        theta = np.zeros(design.shape[1])
        backend = 'NumPy Newton solver for the same fixed C=1 balanced L2 objective'
        for iteration in range(100):
            probability = sigmoid(design @ theta)
            gradient = design.T @ (sample_weight * (probability - y)) / total + penalty @ theta
            if np.max(np.abs(gradient)) < 1e-10:
                break
            hessian = design.T @ ((sample_weight * probability * (1.0 - probability))[:, None] * design) / total + penalty
            step = np.linalg.solve(hessian, gradient)
            alpha = 1.0
            while alpha > 2**-30 and objective(theta - alpha * step) > objective(theta) + 1e-14:
                alpha *= 0.5
            theta -= alpha * step
        else:
            raise ValueError('Fixed logistic regression failed to converge')
    gradient = design.T @ (sample_weight * (sigmoid(design @ theta) - y)) / total + penalty @ theta
    if np.max(np.abs(gradient)) > 1e-6:
        raise ValueError('Logistic optimum check failed')
    return {'feature_names': list(BASE_FEATURES), 'mean': mean.tolist(), 'scale': scale.tolist(), 'intercept': float(theta[0]), 'coefficients': theta[1:].tolist(), 'C': 1.0, 'class_weight': {'same_0': float(len(y) / (2 * np.count_nonzero(y == 0))), 'different_1': float(len(y) / (2 * np.count_nonzero(y == 1)))}, 'target': 'different identity = 1', 'backend': backend, 'backend_import_error': backend_error, 'iterations': iteration, 'objective': objective(theta), 'max_absolute_gradient': float(np.max(np.abs(gradient)))}


def recipe_scores(recipe, bundle):
    name, parameters = recipe['method'], recipe.get('parameters', {})
    if name in BASE_FEATURES:
        return bundle['aggregates'][name].copy()
    if name == 'Hamming + bigram':
        return 0.5 * bundle['aggregates']['Hamming'] + 0.5 * bundle['aggregates']['Bigram']
    if name == 'Bigram + trigram':
        return 0.5 * bundle['aggregates']['Bigram'] + 0.5 * bundle['aggregates']['Trigram']
    if name == 'Logistic regression':
        if parameters['feature_names'] != BASE_FEATURES:
            raise ValueError('Logistic feature names differ from the frozen definition')
        x = logistic_matrix(bundle)
        return sigmoid(parameters['intercept'] + ((x - np.array(parameters['mean'])) / np.array(parameters['scale'])) @ np.array(parameters['coefficients']))
    if name == 'Hamming + decision tree':
        if parameters['feature_names'] != TREE_FEATURES:
            raise ValueError('Tree feature names differ from the frozen definition')
        return np.array([_tree_score(parameters['tree'], values) for values in tree_matrix(bundle)])
    raise ValueError('Unknown recipe: ' + name)


def fit_recipe(name, bundle, truth):
    if name not in METHODS:
        raise ValueError('Unknown recipe: ' + name)
    if set(bundle['endpoints']) != set(truth):
        raise ValueError('Training truth must cover exactly the saved endpoints')
    if len({tuple(value)[0] for value in truth.values()}) != 1:
        raise ValueError('This benchmark fits exact-model identity within one labeled family')
    same = np.array([tuple(truth[a]) == tuple(truth[b]) for a, b in bundle['pairs']])
    parameters = {}
    if name == 'Logistic regression':
        parameters = fit_logistic(logistic_matrix(bundle), (~same).astype(float))
    elif name == 'Hamming + decision tree':
        if (MAX_DEPTH, MIN_LEAF) != (2, 4):
            raise ValueError('Source tree constants changed')
        x = tree_matrix(bundle)
        records = [(tuple(map(float, values)), bool(label)) for values, label in zip(x, same, strict=True)]
        parameters = {'feature_names': list(TREE_FEATURES), 'tree': _grow(records, TREE_FEATURES), 'max_depth': 2, 'min_samples_leaf': 4, 'criterion': 'gini', 'smoothing': '(same_count+1)/(leaf_count+2)', 'score': '-p_same', 'split_tie_break': 'lexical feature name, then smallest threshold'}
    elif name == 'Hamming + bigram':
        parameters = {'weights': {'Hamming': 0.5, 'Bigram': 0.5}}
    elif name == 'Bigram + trigram':
        parameters = {'weights': {'Bigram': 0.5, 'Trigram': 0.5}}
    recipe = {'method': name, 'parameters': parameters}
    scores = recipe_scores(recipe, bundle)
    score_map = dict(zip(bundle['pairs'], map(float, scores), strict=True))
    cutoff, _, candidates = fit_cutoff(score_map, bundle['endpoints'], {e: tuple(truth[e]) for e in bundle['endpoints']})
    return {**recipe, 'cutoff': float(cutoff), 'candidate_cutoffs': list(map(float, candidates))}


def cluster(scores, pairs, endpoints, cutoff):
    values = np.asarray(scores, dtype=float)
    if values.shape != (len(pairs),) or not np.all(np.isfinite(values)):
        raise ValueError('One finite distance is required for every pair')
    return clusters(dict(zip(pairs, map(float, values), strict=True)), float(cutoff), endpoints)


def evaluate(scores, bundle, truth, cutoff):
    groups = cluster(scores, bundle['pairs'], bundle['endpoints'], cutoff)
    actual = np.array([tuple(truth[a]) == tuple(truth[b]) for a, b in bundle['pairs']])
    predicted = np.array([groups[a] == groups[b] for a, b in bundle['pairs']])
    tp, fp, tn, fn = (int(np.count_nonzero(mask)) for mask in (actual & predicted, ~actual & predicted, ~actual & ~predicted, actual & ~predicted))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    raw_mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    values = np.asarray(scores, dtype=float)
    both = bool(np.any(actual) and np.any(~actual))
    recovered = [list(pair) for pair, a, p in zip(bundle['pairs'], actual, predicted, strict=True) if a and p]
    false_merges = [list(pair) for pair, a, p in zip(bundle['pairs'], actual, predicted, strict=True) if not a and p]
    missed = [list(pair) for pair, a, p in zip(bundle['pairs'], actual, predicted, strict=True) if a and not p]
    s_endpoints = [e for e in bundle['endpoints'] if truth[e][1] == 'S']
    t_endpoints = [e for e in bundle['endpoints'] if truth[e][1] != 'S']
    checks = {'all_S_together': len({groups[e] for e in s_endpoints}) == 1 if s_endpoints else None}
    for model in ('T1', 'T2', 'T3', 'T4'):
        selected = [e for e in t_endpoints if truth[e][1] == model]
        checks['S_separate_' + model] = all(groups[s] != groups[t] for s in s_endpoints for t in selected) if s_endpoints and selected else None
    checks['all_T_mutually_separate'] = all(groups[a] != groups[b] for a, b in itertools.combinations(t_endpoints, 2) if truth[a] != truth[b])
    checks['exact_truth_groups'] = bool(np.array_equal(actual, predicted))
    checks['exact_five_groups'] = bool(checks['exact_truth_groups'] and len(set(groups.values())) == 5)
    return {'mcc': max(0.0, raw_mcc), 'raw_mcc': raw_mcc, 'accuracy': (tp + tn) / len(actual), 'auroc': auroc(actual.tolist(), (-values).tolist()) if both else None, 'margin': float(values[~actual].min() - values[actual].max()) if both else None, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'recall': tp / (tp + fn) if tp + fn else None, 'groups': groups, 'number_of_groups': len(set(groups.values())), 'true_matches_recovered': recovered, 'false_merges': false_merges, 'missed_matches': missed, 'identity_checks': checks}
