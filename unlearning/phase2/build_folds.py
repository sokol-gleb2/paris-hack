import numpy as np
from phase2.common import validate_folds


def make_folds(labels, seed=2026):
    rng = np.random.default_rng(seed)
    groups = [[] for _ in range(5)]
    cursor = 0
    for label in range(100):
        members = np.flatnonzero(labels == label)
        rng.shuffle(members)
        for index in members:
            groups[cursor % 5].append(int(index))
            cursor += 1
    folds = {f'fold_{i}': np.sort(np.asarray(group, dtype=np.int64)) for i, group in enumerate(groups)}
    validate_folds(folds)
    return folds
