import os
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


API_KEY = os.environ.get('PARIS_API_KEY', '')
SERVER_URL = "http://34.147.229.55"
TASK_ID = "33-model-equality"
PROMPTS_FILE = Path("reference_prompts.txt")
ANSWERS_FILE = Path("reference_answers.jsonl")

NUM_ENDPOINTS = 9
ENDPOINTS = [f"ref-{i:02d}" for i in range(1, NUM_ENDPOINTS + 1)]
REPEATS = 1
MAX_TOKENS = 256
PARALLEL = 4

BASE_URL = f"{SERVER_URL.rstrip('/')}/{TASK_ID}/v1"


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers["X-API-Key"] = API_KEY
    return session


def get_budget(session: requests.Session) -> dict:

    reply = session.get(f"{BASE_URL}/models", timeout=60)
    if reply.status_code != 200:
        raise RuntimeError(f"Error (HTTP {reply.status_code}): {reply.text[:300]}")
    return {m["id"]: m["budget"]["limit"] - m["budget"]["used"] for m in reply.json()["data"]}


def read_prompts(path: Path) -> list:

    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            return [(row["id"], row["prompt"]) for row in csv.DictReader(f)]
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [(str(r["id"]), r["prompt"]) for r in (json.loads(line) for line in f if line.strip())]
    with path.open(encoding="utf-8") as f:
        return [(str(i), line.rstrip("\n")) for i, line in enumerate(f)]


def query(prompts: list, output: Path = ANSWERS_FILE, endpoints: list = ENDPOINTS, repeats: int = REPEATS,max_tokens: int = MAX_TOKENS) -> None:




    session = make_session()
    left = get_budget(session)
    unknown = [e for e in endpoints if e not in left]
    if unknown:
        raise ValueError(f"unknown endpoints {unknown}; valid: {', '.join(left)}")

    done = set()
    if output.exists():
        with output.open(encoding="utf-8") as f:
            done = {(r["endpoint"], r["id"], r["repeat"]) for r in (json.loads(line) for line in f if line.strip())}

    jobs, planned, over_budget = [], {ep: 0 for ep in endpoints}, 0
    for repeat in range(repeats):
        for pid, prompt in prompts:
            for ep in endpoints:
                if (ep, pid, repeat) in done:
                    continue
                if planned[ep] < left[ep]:
                    planned[ep] += 1
                    jobs.append((ep, pid, repeat, prompt))
                else:
                    over_budget += 1
    print(f"{len(prompts)} prompts x {len(endpoints)} endpoints x {repeats} repeats: {len(done)} answers already "
          f"in {output}, {len(jobs)} to query" + (f", {over_budget} left out (over budget)" if over_budget else ""))

    lock, exhausted, counter = threading.Lock(), set(), [0]
    out = output.open("a", encoding="utf-8")

    def one(job):
        ep, pid, repeat, prompt = job
        while ep not in exhausted:
            try:
                r = session.post(f"{BASE_URL}/chat/completions", timeout=180,
                                 json={"model": ep, "max_tokens": max_tokens,
                                       "messages": [{"role": "user", "content": prompt}]})
            except requests.RequestException as exc:
                print(f"Server unreachable ({type(exc).__name__}); retrying in 30 s.")
                time.sleep(30)
                continue
            if r.status_code == 200:
                body = r.json()
                line = {"endpoint": ep, "id": pid, "repeat": repeat, "prompt": prompt,
                        "response": body["choices"][0]["message"]["content"]}
                with lock:
                    out.write(json.dumps(line, ensure_ascii=False) + "\n")
                    out.flush()
                    counter[0] += 1
                    if counter[0] % 20 == 0 or counter[0] == len(jobs):
                        b = body["budget"]
                        print(f"{counter[0]}/{len(jobs)} answers ({ep}: {b['used']}/{b['limit']} used)")
                return
            detail = r.json().get("detail", r.text[:300]) if r.headers.get("content-type", "").startswith(
                "application/json") else r.text[:300]
            if r.status_code == 429 and "Retry-After" in r.headers:
                time.sleep(int(r.headers["Retry-After"]))
            elif r.status_code == 429:
                with lock:
                    if ep not in exhausted:
                        exhausted.add(ep)
                        print(detail)
            elif r.status_code in (502, 503, 504):
                print(f"{detail} Retrying in 15 s.")
                time.sleep(15)
            else:
                print(f"{ep} id {pid}: HTTP {r.status_code}: {detail}", file=sys.stderr)
                return

    try:
        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            list(pool.map(one, jobs))
    finally:
        out.close()
    print(f"Finished: {output}")


def main() -> None:
    if API_KEY == "YOUR_API_KEY_HERE" or not API_KEY.strip():
        print("Set API_KEY at the top of query_reference.py to your team's API key.", file=sys.stderr)
        raise SystemExit(1)
    if not PROMPTS_FILE.exists():
        print(f"Error: {PROMPTS_FILE} not found. Write your prompts there (one per line).", file=sys.stderr)
        raise SystemExit(1)

    budget = get_budget(make_session())
    endpoints = [ep for ep in budget if ep.startswith("ref-")]
    print("Reference endpoints:", ", ".join(endpoints))
    query(read_prompts(PROMPTS_FILE), endpoints=endpoints)


if __name__ == "__main__":
    main()
