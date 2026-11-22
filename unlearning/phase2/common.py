import numpy as np


def validate_indices(indices, n=3500):
    if indices.ndim != 1 or indices.dtype.kind not in 'iu' or len(np.unique(indices)) != len(indices):
        raise ValueError('Indices must be a unique integer vector.')
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError('Index outside dataset.')


def validate_folds(folds):
    if set(folds) != {f'fold_{i}' for i in range(5)}:
        raise ValueError('Expected five named folds.')
    for indices in folds.values():
        validate_indices(indices)
        if len(indices) != 700:
            raise ValueError('Every fold must have 700 indices.')
    if not np.array_equal(np.sort(np.concatenate(list(folds.values()))), np.arange(3500)):
        raise ValueError('Folds must be disjoint and cover all 3500 indices.')
