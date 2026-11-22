import argparse
import csv
import hashlib
import importlib
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
UPSTREAM = ROOT / "upstream-model-equality-testing" / "model_equality_testing" / "src"
ENDPOINTS = tuple(f"ref-{number:02d}" for number in range(1, 10))
PAIRS = tuple(itertools.combinations(ENDPOINTS, 2))
SPLITS = ("calibration", "heldout")
REPEATS = range(5)


def workspace_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise argparse.ArgumentTypeError("Path must stay inside model_equality")
    return path


def jsonl(path):
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_inputs(prompt_path, answer_path):
    prompts = {}
    for row in jsonl(prompt_path):
        identifier = row["id"]
        if not isinstance(identifier, str) or not identifier or identifier in prompts:
            raise ValueError(f"Invalid or duplicate prompt ID {identifier!r}")
        tokens = row["input_tokens"]
        if type(tokens) is not int or not 0 < tokens < 1024:
            raise ValueError(f"Invalid input token count or estimate for {identifier}")
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
            raise ValueError(f"Prompt text hash mismatch for {identifier}")
        prompts[identifier] = row
    if Counter(row["split"] for row in prompts.values()) != dict.fromkeys(SPLITS, 10):
        raise ValueError("Expected ten calibration and ten heldout prompts")

    answers = {}
    for row in jsonl(answer_path):
        endpoint, identifier, repeat = row["endpoint"], row["id"], row["repeat"]
        key = (endpoint, identifier, repeat)
        if endpoint not in ENDPOINTS or identifier not in prompts or type(repeat) is not int or repeat not in REPEATS:
            raise ValueError(f"Unexpected response key {key!r}")
        if key in answers:
            raise ValueError(f"Duplicate response key {key!r}")
        prompt = prompts[identifier]
        for field in ("prompt", "prompt_sha256", "split", "input_tokens"):
            if row[field] != prompt[field]:
                raise ValueError(f"Response {key!r} differs from its manifest in {field}")
        if row["max_tokens"] != 256 or not isinstance(row["response"], str) or not row["response"].strip():
            raise ValueError(f"Invalid response for {key!r}")
        usage = row.get("usage") or {}
        for field, cap in (("prompt_tokens", 1023), ("completion_tokens", 256)):
            count = usage.get(field)
            if count is not None and (type(count) is not int or not 0 <= count <= cap):
                raise ValueError(f"Response {key!r} has invalid server-reported {field}")
        answers[key] = row["response"]
    expected = set(itertools.product(ENDPOINTS, prompts, REPEATS))
    if missing := expected - answers.keys():
        raise ValueError(f"Missing {len(missing)} reference responses, first missing key {min(missing)!r}")
    return prompts, answers


def load_upstream():
    name = "model_equality_testing"
    if name in sys.modules:
        module = sys.modules[name]
        if str(UPSTREAM) not in module.__path__:
            raise ImportError("An alternate model_equality_testing package is already loaded")
        return module
    module = ModuleType(name)
    module.__path__ = [str(UPSTREAM)]
    sys.modules[name] = module
    try:
        for submodule in ("algorithm", "distribution", "pvalue", "utils"):
            importlib.import_module(f"{name}.{submodule}")
    except ImportError:
        sys.modules.pop(name)
        raise
    return module


def permutation_scores(rows, group_size, rng, permutations):
    import numpy as np

    width = rows.shape[1]
    kernel = np.count_nonzero(rows[:, None, :] == rows[None, :, :], axis=2) / width
    np.fill_diagonal(kernel, 0)
    groups = np.asarray(list(itertools.combinations(range(len(rows)), group_size)))
    membership = np.zeros((len(groups), len(rows)), dtype=bool)
    membership[np.arange(len(groups))[:, None], groups] = True
    other_groups = np.nonzero(~membership)[1].reshape(groups.shape)
    within_a = kernel[groups[:, :, None], groups[:, None, :]].sum(axis=(1, 2))
    within_b = kernel[other_groups[:, :, None], other_groups[:, None, :]].sum(axis=(1, 2))
    between = kernel[groups[:, :, None], other_groups[:, None, :]].sum(axis=(1, 2))
    within_count = group_size * (group_size - 1)
    null = (within_a / within_count - 2 * between / group_size**2 + within_b / within_count)
    return null[rng.integers(len(groups), size=permutations)]


def upstream_scores(prompts, answers, permutations, seed, width):
    import numpy as np

    method = load_upstream()
    rng = np.random.default_rng(seed)
    width = width or max(map(len, answers.values()))
    samples = {}
    for endpoint, identifier in itertools.product(ENDPOINTS, sorted(prompts)):
        responses = [answers[endpoint, identifier, repeat][:width] for repeat in REPEATS]
        rows = method.utils.pad_to_length(method.utils.tokenize_unicode(responses), L=width)
        samples[endpoint, identifier] = method.distribution.CompletionSample(
            prompts=np.zeros(len(REPEATS), dtype=np.int64), completions=rows, m=1
        )

    scores = []
    for first, second in PAIRS:
        observed_by_prompt, null_by_prompt = {}, {}
        for identifier, row in sorted(prompts.items()):
            sample_a = samples[first, identifier]
            sample_b = samples[second, identifier]
            pooled = np.concatenate((sample_a.sequences.numpy()[:, 1:], sample_b.sequences.numpy()[:, 1:]))
            null = permutation_scores(pooled, sample_a.N, rng, permutations)
            _, statistic = method.algorithm.run_two_sample_test(
                sample_a, sample_b, pvalue_type="dummy", stat_type="mmd_hamming"
            )
            pvalue = np.mean(null >= statistic - 1e-12)
            observed_by_prompt[identifier] = statistic
            null_by_prompt[identifier] = null
            scores.append({
                "endpoints": [first, second],
                "split": row["split"],
                "prompt_id": identifier,
                "response_keys": [[endpoint, identifier, repeat] for endpoint in (first, second) for repeat in REPEATS],
                "samples_per_endpoint": len(REPEATS),
                "statistic": float(statistic),
                "pvalue": float(pvalue),
            })
        for split in (*SPLITS, "all"):
            identifiers = [identifier for identifier in sorted(prompts) if split == "all" or prompts[identifier]["split"] == split]
            statistic = float(np.mean([observed_by_prompt[identifier] for identifier in identifiers]))
            null = np.mean([null_by_prompt[identifier] for identifier in identifiers], axis=0)
            scores.append({
                "endpoints": [first, second],
                "split": split,
                "prompt_ids": identifiers,
                "samples_per_endpoint": len(REPEATS) * len(identifiers),
                "statistic": statistic,
                "pvalue": float(np.mean(null >= statistic - 1e-12)),
            })
    return scores, width


def clusters(scores, threshold):
    groups = [(endpoint,) for endpoint in ENDPOINTS]
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


def performance(truth, groups):
    counts = Counter()
    predictions = []
    for first, second in PAIRS:
        same_model = truth[first][:2] == truth[second][:2]
        predicted = groups[first] == groups[second]
        counts[same_model, predicted] += 1
        predictions.append({
            "endpoints": [first, second],
            "true_same_model": same_model,
            "predicted_same_model": predicted,
            "kinds": [truth[first][2], truth[second][2]],
        })
    tp, tn = counts[True, True], counts[False, False]
    fp, fn = counts[False, True], counts[True, False]
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {
        "mcc": max(0.0, mcc),
        "raw_mcc": mcc,
        "accuracy": (tp + tn) / len(PAIRS),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "wrong_pairs": [row for row in predictions if row["true_same_model"] != row["predicted_same_model"]],
        "pair_predictions": predictions,
    }


def calibrate(scores, truth):
    distances = {
        split: {
            tuple(row["endpoints"]): row["statistic"]
            for row in scores if row["split"] == split and "prompt_ids" in row
        }
        for split in SPLITS
    }
    unique = sorted(set(distances["calibration"].values()))
    thresholds = [
        math.nextafter(unique[0], -math.inf),
        *((left + right) / 2 for left, right in itertools.pairwise(unique)),
        unique[-1],
    ]
    best = None
    for threshold in thresholds:
        result = performance(truth, clusters(distances["calibration"], threshold))
        rank = (result["mcc"], result["accuracy"], -threshold)
        if best is None or rank > best[0]:
            best = rank, threshold, result
    _, threshold, calibration = best
    heldout_groups = clusters(distances["heldout"], threshold)
    heldout = performance(truth, heldout_groups)
    return threshold, calibration, heldout, heldout_groups


def main():
    parser = argparse.ArgumentParser(description="Evaluate all 900 practice responses locally with the vendored MMD Hamming test.")
    parser.add_argument("--prompts", type=workspace_path, default=OUTPUT / "reference_prompts.jsonl")
    parser.add_argument("--answers", type=workspace_path, default=OUTPUT / "reference_answers.jsonl")
    parser.add_argument("--output-dir", type=workspace_path, default=OUTPUT)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--completion-characters", type=int, default=200, help="Use the first N Unicode characters or 0 for all characters.")
    args = parser.parse_args()
    if args.permutations < 1 or args.completion_characters < 0:
        parser.error("Permutations must be positive and completion characters cannot be negative")
    try:
        from score_practice import read

        prompts, answers = load_inputs(args.prompts, args.answers)
        truth = read(ROOT / "practice_answers.csv", with_kind=True)
        if set(truth) != set(ENDPOINTS):
            raise ValueError("Organizer practice labels must cover the nine reference endpoints")
        scores, width = upstream_scores(prompts, answers, args.permutations, args.seed, args.completion_characters)
    except (OSError, ImportError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Evaluation blocked: {exc}\n")
    threshold, calibration, heldout, groups = calibrate(scores, truth)
    report = {
        "method": "Upstream MMD Hamming on fixed prompt strata",
        "upstream_source": "https://github.com/irenasaracay/model-equality-testing/tree/fd2ee24d75c9fef87debff8caefa0c04d4a5d374/model_equality_testing/src",
        "aggregation": "Equal mean across prompt statistics and within-prompt permutation draws",
        "permutations": args.permutations,
        "permutation_sampling": "Uniform balanced split sampling with replacement from cached Hamming MMD scores, NumPy RNG, 1e-12 tie tolerance.",
        "seed": args.seed,
        "completion_characters": width,
        "input_artifacts": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.prompts, args.answers)
        },
        "clustering": "Complete linkage on aggregate MMD. Threshold selected by calibration MCC, then accuracy, then smallest threshold.",
        "threshold": threshold,
        "caveat": "Heldout prompts are disjoint from calibration prompts. The labeled endpoint pairs repeat across both splits, so this estimates transfer to new prompts on these endpoints.",
        "calibration": calibration,
        "heldout": heldout,
        "heldout_model_groups": groups,
        "passes_target": heldout["mcc"] > 0.90,
        "scores": scores,
    }
    args.output_dir.mkdir(exist_ok=True, parents=True)
    score_path = args.output_dir / "reference_scores.json"
    score_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    prediction_path = args.output_dir / "reference_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("endpoint", "family", "model"))
        writer.writerows((endpoint, "P", groups[endpoint]) for endpoint in ENDPOINTS)
    print(f"Heldout MCC {heldout['mcc']:.3f}, accuracy {heldout['accuracy']:.3f}")
    print(score_path.relative_to(ROOT))
    print(prediction_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
