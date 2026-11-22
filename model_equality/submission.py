import os
import sys
from pathlib import Path

import requests

API_KEY = os.environ.get('PARIS_API_KEY', '')
SUBMISSION_FILE = str(Path(__file__).resolve().parent / 'RESULTS/final_predictions.csv')

TASK_ID = "33-model-equality"                           
SERVER_URL = "http://34.147.229.55"

ALLOWED_SUFFIXES = (".csv",)

def main() -> int:
    file = Path(SUBMISSION_FILE)

    if API_KEY == "YOUR_API_KEY_HERE" or not API_KEY.strip():
        print("Set the PARIS_API_KEY environment variable to your team's API key.", file=sys.stderr)
        return 1
    if not file.is_file():
        print(f"File not found: {file}. Set SUBMISSION_FILE at the top of submission.py.", file=sys.stderr)
        return 1
    if file.suffix.lower() not in ALLOWED_SUFFIXES:
        print(f"{TASK_ID} expects a {' or '.join(ALLOWED_SUFFIXES)} file, got {file.name}", file=sys.stderr)
        return 1

    print(f"Submitting {file.name} to {TASK_ID}. Scoring can take a few minutes...")
    with file.open("rb") as f:
        try:
            reply = requests.post(
                f"{SERVER_URL.rstrip('/')}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files={"file": (file.name, f)},
                timeout=(600, 600),                                              
            )
        except requests.RequestException as exc:
            print(f"Could not reach the server: {exc}", file=sys.stderr)
            return 1

    try:
        body = reply.json()
    except ValueError:
        print(f"HTTP {reply.status_code}: {reply.text[:300]}", file=sys.stderr)
        return 1

    if reply.status_code != 200:
        print(f"Rejected (HTTP {reply.status_code}): {body.get('detail', body)}", file=sys.stderr)
        return 1
    status = body.get("status")
    if status == "success":
        print(f"Score (public split): {body.get('score')}")
        if "previous_best" in body:
            print(f"Previous best: {body['previous_best']}  (improved: {body.get('improved')})")
        print(body.get("message", ""))
        return 0
    print(f"{status}: {body.get('message')}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
