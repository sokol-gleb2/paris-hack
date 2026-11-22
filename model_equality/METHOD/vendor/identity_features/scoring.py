from collections.abc import Mapping
from itertools import combinations
from math import sqrt


def _partition(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must map endpoints to (family, model) labels")
    result = {}
    for endpoint, labels in value.items():
        if not isinstance(endpoint, str) or not endpoint or endpoint != endpoint.strip():
            raise ValueError(f"{name} has an invalid endpoint identifier")
        if not isinstance(labels, (tuple, list)) or len(labels) != 2:
            raise ValueError(f"{name}[{endpoint!r}] must contain exactly two labels")
        for label in labels:
            if not isinstance(label, str) or not label.strip() or label != label.strip():
                raise ValueError(f"{name}[{endpoint!r}] has an empty or malformed label")
            if any(character in label for character in (",", "\r", "\n")):
                raise ValueError(f"{name}[{endpoint!r}] has a label incompatible with the submission format")
        result[endpoint] = tuple(labels)
    return result


def _pairs(endpoints, pairs):
    if pairs is None:
        return list(combinations(sorted(endpoints), 2))
    if isinstance(pairs, (str, bytes)):
        raise ValueError("pairs must be an iterable of endpoint pairs")
    try:
        iterator = iter(pairs)
    except TypeError as error:
        raise ValueError("pairs must be an iterable of endpoint pairs") from error
    selected = set()
    for pair in iterator:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Each selected pair must contain exactly two endpoint identifiers")
        a, b = pair
        if not isinstance(a, str) or not isinstance(b, str) or a not in endpoints or b not in endpoints:
            raise ValueError("A selected pair contains an unknown endpoint")
        if a >= b:
            raise ValueError("Selected pairs must use distinct endpoints in ascending order")
        if (a, b) in selected:
            raise ValueError("A selected pair occurs more than once")
        selected.add((a, b))
    if not selected:
        raise ValueError("At least one pair must be selected")
    return sorted(selected)


def _component(decisions):
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for truth, prediction in decisions:
        key = ("tp" if prediction else "fn") if truth else ("fp" if prediction else "tn")
        counts[key] += 1
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    raw_mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {
        "pair_count": sum(counts.values()),
        "confusion_matrix": counts,
        "raw_mcc": raw_mcc,
        "mcc": max(0.0, raw_mcc),
        "mcc_defined": bool(denominator),
        "denominator_zero": not bool(denominator),
        "truth_positive_count": tp + fn,
        "truth_negative_count": tn + fp,
        "predicted_positive_count": tp + fp,
        "predicted_negative_count": tn + fn,
        "truth_has_both_classes": bool(tp + fn) and bool(tn + fp),
    }


def score_partition(truth, predicted, pairs=None):
    truth = _partition(truth, "truth")
    predicted = _partition(predicted, "predicted")
    if len(truth) < 2:
        raise ValueError("At least two ground-truth endpoints are required")
    if set(truth) != set(predicted):
        missing = sorted(set(truth) - set(predicted))
        extra = sorted(set(predicted) - set(truth))
        raise ValueError(f"Endpoint coverage differs: missing={missing}, extra={extra}")
    selected = _pairs(set(truth), pairs)
    pair_results = []
    for a, b in selected:
        true_family = truth[a][0] == truth[b][0]
        pair_results.append({
            "endpoints": [a, b],
            "true_same_family": true_family,
            "predicted_same_family": predicted[a][0] == predicted[b][0],
            "true_same_model": truth[a] == truth[b],
            "predicted_same_model": predicted[a] == predicted[b],
            "included_in_model": true_family,
        })
    family = _component((row["true_same_family"], row["predicted_same_family"]) for row in pair_results)
    model = _component((row["true_same_model"], row["predicted_same_model"]) for row in pair_results if row["included_in_model"])
    limitations = []
    for name, component in (("family", family), ("model", model)):
        if not component["truth_has_both_classes"]:
            limitations.append(f"{name}: selected ground truth lacks both positive and negative pairs")
        if component["denominator_zero"]:
            limitations.append(f"{name}: MCC denominator is zero; reported value uses the organizer-compatible zero convention")
    return {
        "endpoint_count": len(truth),
        "selected_pair_count": len(selected),
        "possible_pair_count": len(truth) * (len(truth) - 1) // 2,
        "family": family,
        "model": model,
        "weights": {"family": 0.1, "model": 0.9},
        "weighted_score": 0.1 * family["mcc"] + 0.9 * model["mcc"],
        "full_score_informative": family["truth_has_both_classes"] and model["truth_has_both_classes"],
        "limitations": limitations,
        "pair_results": pair_results,
    }
