import argparse
from collections import Counter
import copy
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

HERE = Path(__file__).resolve().parent
ROOT = HERE
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vendor.final_900_benchmark import core
from vendor.identity_features.scoring import score_partition

FIELDS = ('endpoint', 'id', 'prompt', 'repeat', 'response')
OUTPUTS = ('model_predictions.csv', 'pair_scores.csv', 'inference_summary.json')
KERNELS = {
    'Hamming': 'positional agreement divided by200 with -1 padding',
    'Bigram': 'overlapping2-character counts, L2 normalized, dot product, zero when short',
    'Trigram': 'overlapping3-character counts, L2 normalized, dot product, zero when short',
    'Structural': 'phi=[length,newlines,whitespace,ASCII punctuation]/200 on prefix; kernel dot/4',
}
MIXTURES = {'Hamming + bigram': {'Hamming': 0.5, 'Bigram': 0.5}, 'Bigram + trigram': {'Bigram': 0.5, 'Trigram': 0.5}}
AGGREGATION = 'equal prompt weights, actual per-cell sample counts, signed unbiased squared MMD'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def expected_counts(prompt):
    for key in ('reference_draw_counts', 'reference_counts'):
        if key in prompt:
            counts = prompt[key]
            if not isinstance(counts, dict) or not counts or any(type(v) is not int or v < 2 for v in counts.values()):
                raise ValueError('Invalid frozen reference draw counts')
            return counts
    for key in ('responses_per_endpoint', 'reference_draws_per_endpoint'):
        if key in prompt:
            value = prompt[key]
            if type(value) is not int or value < 2:
                raise ValueError('Invalid frozen reference draw count')
            return {'*': value}
    return {}


def validate_tree(node, depth=0):
    if not isinstance(node, dict) or depth > 2 or node.get('type') not in ('leaf', 'split'):
        raise ValueError('Invalid frozen shallow tree')
    if not finite(node.get('p_same')) or not 0 <= node['p_same'] <= 1:
        raise ValueError('Invalid frozen same-model leaf proportion')
    if node['type'] == 'split':
        index = node.get('feature_index')
        if type(index) is not int or not 0 <= index < 5 or node.get('feature_id') != core.TREE_FEATURES[index] or not finite(node.get('threshold')):
            raise ValueError('Invalid frozen tree split feature or threshold')
        validate_tree(node.get('left'), depth + 1)
        validate_tree(node.get('right'), depth + 1)


def validate_config(config):
    if not isinstance(config, dict) or config.get('schema_version') != 2:
        raise ValueError('Expected frozen panel schema_version 2')
    fixed = {'unicode_prefix_length': 200, 'normalization': 'none', 'hamming_padding': -1}
    if config.get('preprocessing') != fixed:
        raise ValueError('Only raw text, Unicode prefix200 and Hamming padding -1 are supported')
    if config.get('minimum_draws_per_cell') != 2:
        raise ValueError('Signed unbiased MMD requires the frozen minimum of two draws')
    recipe = config.get('recipe')
    if not isinstance(recipe, dict) or recipe.get('method') != config.get('method') or recipe.get('method') not in core.METHODS:
        raise ValueError('Unknown or inconsistent frozen method')
    cutoff, parameters = recipe.get('cutoff'), recipe.get('parameters')
    if not finite(cutoff) or not isinstance(parameters, dict):
        raise ValueError('A finite cutoff and frozen recipe parameters are required')
    name = recipe['method']
    if name in KERNELS and parameters:
        raise ValueError('Basic kernels do not accept learned recipe parameters')
    if name in MIXTURES and parameters != {'weights': MIXTURES[name]}:
        raise ValueError('Only the specified equal mixture weights are supported')
    if name == 'Logistic regression':
        if parameters.get('feature_names') != core.BASE_FEATURES:
            raise ValueError('Logistic features differ from the fixed kernel feature order')
        for field in ('mean', 'scale', 'coefficients'):
            if not isinstance(parameters.get(field), list) or len(parameters[field]) != len(core.BASE_FEATURES) or any(not finite(v) for v in parameters[field]):
                raise ValueError('Invalid frozen logistic ' + field)
        if any(v <= 0 for v in parameters['scale']) or not finite(parameters.get('intercept')):
            raise ValueError('Invalid frozen logistic scaling or intercept')
    if name == 'Hamming + decision tree':
        if parameters.get('feature_names') != core.TREE_FEATURES or parameters.get('max_depth') != 2 or parameters.get('min_samples_leaf') != 4:
            raise ValueError('Unsupported frozen tree features or settings')
        validate_tree(parameters.get('tree'))
    for field, expected in [('kernels', KERNELS), ('mixture_weights', MIXTURES), ('aggregation', AGGREGATION)]:
        if field in config and config[field] != expected:
            raise ValueError('Unsupported frozen ' + field)
    required_clustering = {'method': 'complete', 'merge_rule': 'maximum cross-pair score <= cutoff', 'tie_break': 'lexical sorted endpoint cluster tuples', 'cutoff': cutoff, 'number_of_groups': 'unconstrained'}
    if config.get('clustering') != required_clustering:
        raise ValueError('Clustering must match the fixed complete-linkage procedure and cutoff')
    prompts = config.get('prompts')
    if not isinstance(prompts, list) or not prompts:
        raise ValueError('The exact frozen prompt panel is required')
    index, texts = {}, set()
    for prompt in prompts:
        if not isinstance(prompt, dict):
            raise ValueError('Every frozen prompt must be a record')
        identifier, text = prompt.get('id'), prompt.get('prompt')
        if not isinstance(identifier, str) or not identifier.strip() or identifier in index or not isinstance(text, str) or not text or text in texts:
            raise ValueError('Frozen prompt IDs and texts must be nonempty and unique')
        if hashlib.sha256(text.encode('utf-8')).hexdigest() != prompt.get('prompt_sha256'):
            raise ValueError('Frozen prompt hash mismatch')
        expected_counts(prompt)
        index[identifier] = prompt
        texts.add(text)
    hashes = config.get('source_code_sha256', {})
    if not isinstance(hashes, dict):
        raise ValueError('Frozen source hashes must be a mapping')
    for relative, expected in hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError('Invalid source hash record')
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or digest(path) != expected:
            raise ValueError('Frozen source hash mismatch or unsafe path: ' + relative)
    return index


def validate_rows(rows, config):
    prompts = validate_config(config)
    by_text = {p['prompt']: identifier for identifier, p in prompts.items()}
    seen, input_ids, clean, aliases = set(), {}, [], set()
    counts = Counter()
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict) or any(field not in row for field in FIELDS):
            raise ValueError(f'Record {number} must contain {FIELDS}')
        endpoint, original_id, text, repeat = row['endpoint'], row['id'], row['prompt'], row['repeat']
        if not isinstance(endpoint, str) or not endpoint or endpoint != endpoint.strip() or any(c in endpoint for c in ('\r', '\n', ',')):
            raise ValueError(f'Record {number} endpoint is invalid')
        if not isinstance(original_id, str) or not original_id or not isinstance(text, str) or text not in by_text:
            raise ValueError(f'Record {number} has an unknown exact prompt text or invalid ID')
        if original_id in input_ids and input_ids[original_id] != text:
            raise ValueError('Reused input prompt ID has different text: ' + original_id)
        if original_id in prompts and prompts[original_id]['prompt'] != text:
            raise ValueError('Frozen prompt ID is reused with different text: ' + original_id)
        input_ids[original_id] = text
        identifier = by_text[text]
        if row.get('prompt_sha256', hashlib.sha256(text.encode('utf-8')).hexdigest()) != hashlib.sha256(text.encode('utf-8')).hexdigest():
            raise ValueError('Input prompt hash mismatch')
        if type(repeat) is not int or repeat < 0 or not isinstance(row['response'], str):
            raise ValueError('Repeat must be a nonnegative integer and response must be a string')
        if row.get('status', 'success') not in ('success', 'ok', 'completed'):
            raise ValueError('Input contains an explicitly unsuccessful response record')
        key = endpoint, identifier, repeat
        if key in seen:
            raise ValueError('Duplicate canonical endpoint/prompt/repeat draw: ' + repr(key))
        seen.add(key)
        counts[endpoint, identifier] += 1
        clean.append({'endpoint': endpoint, 'id': identifier, 'prompt': text, 'repeat': repeat, 'response': row['response']})
        if original_id != identifier:
            aliases.add((original_id, identifier))
    endpoints = sorted({r['endpoint'] for r in clean})
    if len(endpoints) < 2:
        raise ValueError('At least two endpoints are required')
    warnings = []
    for endpoint, identifier in itertools.product(endpoints, sorted(prompts)):
        actual = counts[endpoint, identifier]
        if actual < 2:
            raise ValueError(f'Missing or unusable cell {endpoint}/{identifier}: {actual} draws, minimum two')
        reference = expected_counts(prompts[identifier])
        supported = {reference[endpoint]} if endpoint in reference else set(reference.values())
        if supported and actual not in supported:
            warnings.append({'endpoint': endpoint, 'prompt_id': identifier, 'actual_draws': actual, 'reference_draw_counts': sorted(supported), 'note': 'All actual draws used; frozen cutoff transfer across draw counts is unvalidated'})
    return sorted(clean, key=lambda r: (r['endpoint'], r['id'], r['repeat'])), [{'input_id': a, 'canonical_id': b} for a, b in sorted(aliases)], warnings


def read_rows(path):
    rows = []
    with Path(path).open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f'Invalid JSON on line {number}') from error
    return rows


def infer(config, rows):
    clean, aliases, warnings = validate_rows(rows, config)
    bundle = core.compute_features(clean)
    scores = list(map(float, core.recipe_scores(config['recipe'], bundle)))
    groups = core.cluster(scores, bundle['pairs'], bundle['endpoints'], config['recipe']['cutoff'])
    records = [{'endpoint_a': a, 'endpoint_b': b, 'score': score, 'raw_cutoff_same': int(score <= config['recipe']['cutoff']), 'final_same_model': int(groups[a] == groups[b]), 'group_a': groups[a], 'group_b': groups[b]} for (a, b), score in zip(bundle['pairs'], scores, strict=True)]
    return {'method': config['method'], 'cutoff': config['recipe']['cutoff'], 'scores': scores, 'pairs': list(bundle['pairs']), 'pair_scores': records, 'groups': groups, 'number_of_groups': len(set(groups.values())), 'counts': bundle['counts'], 'record_count': len(clean), 'endpoint_count': len(bundle['endpoints']), 'prompt_count': len(bundle['prompt_ids']), 'responses_per_endpoint': dict(sorted(Counter(r['endpoint'] for r in clean).items())), 'aliases': aliases, 'count_warnings': warnings, 'family_inference': 'not performed; family labels pending; model-only output is not a submission', 'training': 'none; frozen parameters and cutoff only'}


def run(responses_path, config_path, output_dir):
    responses_path, config_path, output_dir = Path(responses_path), Path(config_path), Path(output_dir).resolve()
    if not output_dir.is_relative_to(ROOT):
        raise ValueError('Inference outputs must remain inside this METHOD package')
    if any((output_dir / name).exists() for name in OUTPUTS):
        raise FileExistsError('Refusing to overwrite existing inference outputs; choose a new output directory')
    source_hash, configuration_hash = digest(responses_path), digest(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    result = infer(config, read_rows(responses_path))
    if digest(responses_path) != source_hash or digest(config_path) != configuration_hash:
        raise ValueError('Inference source or frozen configuration changed while scoring')
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / OUTPUTS[0]).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['endpoint', 'model'])
        writer.writerows(sorted(result['groups'].items()))
    with (output_dir / OUTPUTS[1]).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['endpoint_a', 'endpoint_b', 'score', 'raw_cutoff_same', 'final_same_model', 'group_a', 'group_b'])
        writer.writeheader()
        writer.writerows(result['pair_scores'])
    result.update({'config_sha256': configuration_hash, 'responses_sha256': source_hash})
    write_json(output_dir / OUTPUTS[2], result)
    return result


def rejected(call):
    try:
        call()
    except (ValueError, FileExistsError):
        return True
    return False


def validate(config, rows, expected_scores, expected_groups, truth):
    source = json.dumps(rows, sort_keys=True, ensure_ascii=False, allow_nan=False)
    expected_scores = list(map(float, expected_scores))
    clean, _, _ = validate_rows(rows, config)
    shuffled = copy.deepcopy(clean)
    random.Random(20261122).shuffle(shuffled)
    poisoned = [dict(row, family='FAKE_FAMILY', model='FAKE_MODEL_' + str(i % 3), kind='FAKE_KIND', true_same_model=False) for i, row in enumerate(clean)]
    renamed = {e: f'unknown-{i:04d}' for i, e in enumerate(sorted(expected_groups), 1)}
    aliased = [dict(row, id='alias-' + row['id']) for row in clean]
    cases = [('original', rows), ('shuffled_unlabeled', shuffled), ('poisoned_labels', poisoned), ('prompt_aliases', aliased), ('renamed_endpoints', [dict(row, endpoint=renamed[row['endpoint']]) for row in clean])]
    maximum = 0.0
    for name, case in cases:
        result = infer(config, case)
        groups = {renamed[e]: g for e, g in expected_groups.items()} if name == 'renamed_endpoints' else expected_groups
        if result['groups'] != groups or len(result['scores']) != len(expected_scores):
            raise AssertionError('Frozen inference partition or pair coverage differs: ' + name)
        difference = max((abs(a-b) for a,b in zip(result['scores'], expected_scores, strict=True)), default=0.0)
        maximum = max(maximum, difference)
        if difference > 1e-12:
            raise AssertionError('Frozen inference scores differ: ' + name)
    guards = {}
    guards['duplicate_draw'] = rejected(lambda: validate_rows(clean + [clean[0]], config))
    removed_cell = [r for r in clean if (r['endpoint'], r['id']) != (clean[0]['endpoint'], clean[0]['id'])]
    guards['missing_cell'] = rejected(lambda: validate_rows(removed_cell, config))
    single = [r for r in clean if (r['endpoint'], r['id']) != (clean[0]['endpoint'], clean[0]['id'])] + [clean[0]]
    guards['one_draw_cell'] = rejected(lambda: validate_rows(single, config))
    changed = copy.deepcopy(clean)
    changed[0]['prompt'] += ' '
    guards['prompt_mismatch'] = rejected(lambda: validate_rows(changed, config))
    changed = copy.deepcopy(clean)
    changed[0]['repeat'] = True
    guards['boolean_repeat'] = rejected(lambda: validate_rows(changed, config))
    altered = copy.deepcopy(config)
    altered['clustering']['cutoff'] += 1.0
    guards['inconsistent_cutoff'] = rejected(lambda: validate_config(altered))
    altered = copy.deepcopy(config)
    altered['preprocessing']['unicode_prefix_length'] = 100
    guards['unsupported_preprocessing'] = rejected(lambda: validate_config(altered))
    altered = copy.deepcopy(config)
    altered['source_code_sha256'] = {'inference.py': '0' * 64}
    guards['source_hash_mismatch'] = rejected(lambda: validate_config(altered))
    if len(config['prompts']) > 1:
        changed = copy.deepcopy(clean)
        first, second = config['prompts'][:2]
        changed = [dict(r, id='reused-id') if r['id'] in (first['id'], second['id']) else r for r in changed]
        guards['id_reused_with_different_text'] = rejected(lambda: validate_rows(changed, config))
    if not all(guards.values()):
        raise AssertionError('Inference validation guard failed: ' + repr(guards))
    metrics = score_partition(truth, {e: (truth[e][0], g) for e, g in expected_groups.items()})['model']
    if json.dumps(rows, sort_keys=True, ensure_ascii=False, allow_nan=False) != source:
        raise AssertionError('Validation mutated source records')
    return {'status': 'passed', 'inference_variants': len(cases), 'maximum_pair_score_difference': maximum, 'pair_count': len(expected_scores), 'same_partition': True, 'row_order_invariant': True, 'ground_truth_fields_unnecessary': True, 'poisoned_labels_ignored': True, 'exact_text_aliases_supported': True, 'new_endpoint_ids_supported': True, 'reference_mcc': metrics['mcc'], 'confusion_matrix': metrics['confusion_matrix'], 'guards': guards, 'requests': 0}


def main(default_config):
    parser = argparse.ArgumentParser(description='Offline model-only inference for a frozen reference prompt panel; no fitting, API calls or submissions.')
    parser.add_argument('--responses', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=Path(default_config))
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    result = run(args.responses, args.config, args.output_dir)
    print(json.dumps({key: result[key] for key in ('method', 'record_count', 'endpoint_count', 'number_of_groups', 'groups', 'count_warnings', 'family_inference')}, indent=2))
