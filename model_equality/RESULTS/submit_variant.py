import argparse
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import submission


def main():
    parser = argparse.ArgumentParser(description='Submit one explicitly requested K2 or K3 variant once via submission.py.')
    parser.add_argument('--k', type=int, choices=[2, 3], required=True)
    parser.add_argument('--retry', action='store_true', help='Run one explicitly authorized retry, preserving previous receipts.')
    args = parser.parse_args()
    folder = HERE/f'k{args.k}'
    prediction = folder/'final_predictions.csv'
    receipt = folder/'submission_attempt.json'
    previous = None
    if args.retry:
        previous = json.loads(receipt.read_text(encoding='utf-8'))
        receipt = folder/('submission_retry_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    if receipt.exists():
        raise FileExistsError('An upload has already been attempted; no automatic retries.')
    rows = list(csv.DictReader(prediction.open(encoding='utf-8', newline='')))
    if len(rows) != 33 or {r['endpoint'] for r in rows} != {f'ep-{i:02d}' for i in range(1,34)}:
        raise ValueError('Expected exactly33 unique endpoints')
    if len({r['family'] for r in rows}) != args.k or not all(set(r)=={'endpoint','family','model'} and r['model'].startswith(r['family']) for r in rows):
        raise ValueError('CSV family/model hierarchy does not match requested variant')
    frozen = json.loads((folder/'configuration.json').read_text(encoding='utf-8'))
    if frozen['k'] != args.k or frozen['model_cutoff'] != 0.02375075405834256:
        raise ValueError('Variant configuration differs')
    before = hashlib.sha256(prediction.read_bytes()).hexdigest()
    if previous is not None and previous['csv_sha256'] != before:
        raise ValueError('Retry CSV differs from the previously authorized artifact')
    record = {'started_utc': datetime.now(timezone.utc).isoformat(), 'k': args.k, 'csv': str(prediction.relative_to(ROOT)), 'csv_sha256': before, 'rows': 33, 'families': args.k, 'models': len({r['model'] for r in rows}), 'submission_script': 'submission.py (SUBMISSION_FILE overridden for this invocation only)', 'attempts': 1, 'status': 'started; delivery unknown until response recorded'}
    receipt.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    submission.SUBMISSION_FILE = str(prediction)
    captured = io.StringIO()
    print(f'Uploading K={args.k} once: {prediction.name}', flush=True)
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        code = submission.main()
    output = captured.getvalue()
    print(output, end='', flush=True)
    match = re.search(r'Score \(public split\):\s*([-+0-9.eE]+)', output)
    record.update({'completed_utc': datetime.now(timezone.utc).isoformat(), 'status': 'success' if code == 0 else 'failed_or_unconfirmed', 'process_exit_code': code, 'public_split_score': float(match.group(1)) if match else None, 'portal_output': output, 'csv_unchanged': hashlib.sha256(prediction.read_bytes()).hexdigest() == before})
    receipt.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    return code


if __name__ == '__main__':
    sys.exit(main())
