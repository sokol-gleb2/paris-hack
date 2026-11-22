import hashlib
import itertools
import math
from pathlib import Path
import sys



from vendor.identity_features.scoring import score_partition


WIDTH = 200
AUTHORIZED_ENDPOINTS = ("ref-01", "ref-02", "ref-03")
SPLIT_REPEATS = {"calibration": (0, 1, 2, 3, 4), "heldout": (5, 6, 7, 8, 9)}


def encode_response(text):
    if not isinstance(text, str):
        raise ValueError("Response must be a string")
    characters = tuple(map(ord, text[:WIDTH]))
    return characters + (-1,) * (WIDTH - len(characters))


def _similarity(left, right):
    return sum(a == b for a, b in zip(left, right)) / WIDTH


def hamming_similarity(left, right):
    return _similarity(encode_response(left), encode_response(right))


def hamming_components(left, right):
    left = [encode_response(text) for text in left]
    right = [encode_response(text) for text in right]
    if len(left) < 2 or len(right) < 2:
        raise ValueError("The signed estimator requires at least two draws per endpoint")
    within_a = math.fsum(_similarity(a, b) for a, b in itertools.combinations(left, 2)) / math.comb(len(left), 2)
    within_b = math.fsum(_similarity(a, b) for a, b in itertools.combinations(right, 2)) / math.comb(len(right), 2)
    across = math.fsum(_similarity(a, b) for a, b in itertools.product(left, right)) / (len(left) * len(right))
    return {
        "within_a": within_a,
        "within_b": within_b,
        "across": across,
        "statistic": within_a + within_b - 2 * across,
        "samples_a": len(left),
        "samples_b": len(right),
        "unordered_within_pairs_a": math.comb(len(left), 2),
        "unordered_within_pairs_b": math.comb(len(right), 2),
        "cross_pairs": len(left) * len(right),
    }


def signed_hamming_mmd(left, right):
    return hamming_components(left, right)["statistic"]


def _checked_scores(scores, endpoints):
    expected = set(itertools.combinations(sorted(endpoints), 2))
    if set(scores) != expected:
        raise ValueError("Pair scores must cover exactly every unordered endpoint pair")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in scores.values()):
        raise ValueError("Every pair score must be finite")
    return scores


def clusters(scores, threshold, endpoints):
    endpoints = tuple(sorted(endpoints))
    _checked_scores(scores, endpoints)
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    groups = [(endpoint,) for endpoint in endpoints]
    while len(groups) > 1:
        distance, left, right = min(
            (max(scores[tuple(sorted((a, b)))] for a, b in itertools.product(first, second)), first, second)
            for first, second in itertools.combinations(groups, 2)
        )
        if distance > threshold:
            break
        groups.remove(left)
        groups.remove(right)
        groups.append(tuple(sorted(left + right)))
        groups.sort()
    return {endpoint: f"M{index:02d}" for index, group in enumerate(groups, 1) for endpoint in group}


def evaluate_scores(scores, threshold, endpoints, truth):
    groups = clusters(scores, threshold, endpoints)
    predictions = {endpoint: ("P", groups[endpoint]) for endpoint in sorted(endpoints)}
    metrics = score_partition(truth, predictions)
    counts = metrics["model"]["confusion_matrix"]
    metrics["model"]["accuracy"] = (counts["tp"] + counts["tn"]) / metrics["model"]["pair_count"]
    return {"metrics": metrics, "groups": groups, "predictions": predictions}


def fit_cutoff(calibration_scores, endpoints, truth):
    _checked_scores(calibration_scores, endpoints)
    unique = sorted(set(calibration_scores.values()))
    candidates = [math.nextafter(unique[0], -math.inf), *((left + right) / 2 for left, right in itertools.pairwise(unique)), unique[-1]]
    best = None
    for threshold in candidates:
        evaluation = evaluate_scores(calibration_scores, threshold, endpoints, truth)
        model = evaluation["metrics"]["model"]
        rank = model["mcc"], model["accuracy"], -threshold
        if best is None or rank > best[0]:
            best = rank, threshold, evaluation
    return best[1], best[2], candidates
