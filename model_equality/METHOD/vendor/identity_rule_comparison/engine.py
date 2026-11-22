import itertools
import math
from fractions import Fraction

from vendor.looop.hamming import fit_cutoff


MAX_DEPTH = 2
MIN_LEAF = 4


def _prompt_ids(prompt_ids):
    if not isinstance(prompt_ids, (list, tuple)) or not prompt_ids:
        raise ValueError("At least one prompt ID is required")
    result = list(prompt_ids)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError("Prompt IDs must be nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError("Prompt IDs must be unique")
    return result


def _vector(values, width):
    if not isinstance(values, (list, tuple)) or len(values) != width:
        raise ValueError("Each pair needs one feature per prompt")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError("Every feature must be finite and numeric")
    return tuple(values)


def _impurity(positives, count):
    return Fraction(2 * positives * (count - positives), count * count)


def _midpoint(lower, upper):
    value = lower / 2 + upper / 2
    return lower if value >= upper else value


def _grow(records, prompt_ids, depth=0):
    count = len(records)
    positives = sum(label for _, label in records)
    node = {
        "type": "leaf", "depth": depth, "samples": count,
        "positives": positives, "p_same": (positives + 1) / (count + 2),
    }
    if depth >= MAX_DEPTH or count < 2 * MIN_LEAF or positives in (0, count):
        return node
    impurity = _impurity(positives, count)
    best = None
    for index in sorted(range(len(prompt_ids)), key=lambda item: prompt_ids[item]):
        unique = sorted({values[index] for values, _ in records})
        for lower, upper in itertools.pairwise(unique):
            threshold = _midpoint(lower, upper)
            left = [record for record in records if record[0][index] <= threshold]
            right = [record for record in records if record[0][index] > threshold]
            if min(len(left), len(right)) < MIN_LEAF:
                continue
            gain = impurity - Fraction(len(left), count) * _impurity(sum(label for _, label in left), len(left)) - Fraction(len(right), count) * _impurity(sum(label for _, label in right), len(right))
            if gain > 0 and (best is None or gain > best[0]):
                best = gain, index, threshold, left, right
    if best is None:
        return node
    gain, index, threshold, left, right = best
    node.update({
        "type": "split", "feature_id": prompt_ids[index], "feature_index": index,
        "threshold": threshold, "gain": float(gain),
        "left": _grow(left, prompt_ids, depth + 1),
        "right": _grow(right, prompt_ids, depth + 1),
    })
    return node


def _tree_score(tree, values):
    node = tree
    while node["type"] == "split":
        node = node["left"] if values[node["feature_index"]] <= node["threshold"] else node["right"]
    return -node["p_same"]


def predict_scores(fitted, features):
    prompt_ids = _prompt_ids(fitted["prompt_ids"])
    method = fitted["method"]
    if method not in ("mean", "tree"):
        raise ValueError("Method must be mean or tree")
    result = {}
    for pair, values in features.items():
        if not isinstance(pair, tuple) or len(pair) != 2 or any(not isinstance(endpoint, str) for endpoint in pair) or pair[0] >= pair[1]:
            raise ValueError("Feature keys must be ordered endpoint pairs")
        values = _vector(values, len(prompt_ids))
        result[pair] = math.fsum(values) / len(values) if method == "mean" else _tree_score(fitted["tree"], values)
    return result


def fit_model(features, prompt_ids, truth, endpoints, method):
    prompt_ids = _prompt_ids(prompt_ids)
    if method not in ("mean", "tree"):
        raise ValueError("Method must be mean or tree")
    if not isinstance(endpoints, (list, tuple)) or len(endpoints) < 2:
        raise ValueError("At least two training endpoints are required")
    if any(not isinstance(endpoint, str) or not endpoint.strip() for endpoint in endpoints) or len(set(endpoints)) != len(endpoints):
        raise ValueError("Training endpoints must be unique nonempty strings")
    endpoints = sorted(endpoints)
    pairs = list(itertools.combinations(endpoints, 2))
    subset_truth = {endpoint: tuple(truth[endpoint]) for endpoint in endpoints}
    training = {pair: _vector(features[pair], len(prompt_ids)) for pair in pairs}
    records = [(training[pair], subset_truth[pair[0]] == subset_truth[pair[1]]) for pair in pairs]
    fitted = {
        "method": method, "prompt_ids": prompt_ids,
        "tree": _grow(records, prompt_ids) if method == "tree" else None,
        "training_endpoints": endpoints, "training_pairs": [list(pair) for pair in pairs],
    }
    scores = predict_scores(fitted, training)
    threshold, calibration, candidates = fit_cutoff(scores, endpoints, subset_truth)
    fitted.update({
        "threshold": threshold, "calibration": calibration,
        "candidate_cutoffs": candidates, "training_scores": scores,
    })
    return fitted
