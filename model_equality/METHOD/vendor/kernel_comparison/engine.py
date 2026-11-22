import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TOLERANCE = 1e-12
WIDTH = 200


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_bigrams(text):
    counts = Counter(zip(text[:WIDTH], text[:WIDTH][1:]))
    norm = math.sqrt(sum(count * count for count in counts.values()))
    return {key: value / norm for key, value in counts.items()} if norm else {}


def bigram_similarity(left, right):
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def signed_mmd(kernel, left, right):
    import numpy as np

    within_left = [kernel[a, b] for a, b in itertools.combinations(left, 2)]
    within_right = [kernel[a, b] for a, b in itertools.combinations(right, 2)]
    across = kernel[np.ix_(left, right)]
    return float(np.mean(within_left) + np.mean(within_right) - 2 * np.mean(across))


def auroc(labels, similarities):
    positive = [score for label, score in zip(labels, similarities, strict=True) if label]
    negative = [score for label, score in zip(labels, similarities, strict=True) if not label]
    if not positive or not negative:
        raise ValueError("AUROC requires both positive and negative labels")
    return sum((a > b) + 0.5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative))


def sanity_checks():
    import numpy as np

    result = bigram_similarity(normalized_bigrams("ABAB"), normalized_bigrams("BABA"))
    if not math.isclose(result, 0.8, abs_tol=TOLERANCE):
        raise ValueError("Bigram sanity example failed")
    if normalized_bigrams("") or normalized_bigrams("A"):
        raise ValueError("Short responses must yield zero vectors")
    left, right = normalized_bigrams("Ab a!\n"), normalized_bigrams("a!\nAb ")
    if not math.isclose(bigram_similarity(left, right), bigram_similarity(right, left), abs_tol=TOLERANCE):
        raise ValueError("Bigram symmetry failed")
    if auroc([True, False], [1, 0]) != 1 or auroc([True, False], [0, 1]) != 0 or auroc([True, False], [0, 0]) != 0.5:
        raise ValueError("AUROC sanity check failed")
    sample = np.array([[1.0, 0.2, 0.8, 0.7], [0.2, 1.0, 0.9, 0.6], [0.8, 0.9, 1.0, 0.1], [0.7, 0.6, 0.1, 1.0]])
    estimate = signed_mmd(sample, [0, 1], [2, 3])
    if estimate >= 0:
        raise ValueError("Signed estimator must preserve negative estimates")
    return result
