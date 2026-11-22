import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import dotenv_values

import query_reference as organizer

ROOT = Path(__file__).resolve().parent
ENDPOINTS = tuple(f"ref-{index:02d}" for index in range(1, 10))
REPEATS = 5
INPUT_LIMIT = 1024
OUTPUT_LIMIT = 256
SPLITS = {"calibration": 10, "heldout": 10}
EXPECTED_COUNT = sum(SPLITS.values()) * len(ENDPOINTS) * REPEATS
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
ATTEMPTS = 3


def local_path(path):
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("All collection files must stay inside model_equality.")
    return resolved


def digest(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def json_records(path):
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError()
            except ValueError as exc:
                raise ValueError(f"{path.name}, line {line_number}: expected a JSON object.") from exc
            yield row


def load_prompts(path):
    prompts = {}
    seen_text = set()
    for row in json_records(path):
        prompt_id = row.get("id")
        prompt = row.get("prompt")
        if not isinstance(prompt_id, str) or not prompt_id.strip() or prompt_id in prompts:
            raise ValueError("Prompt IDs must be unique nonempty strings.")
        if not isinstance(prompt, str) or not prompt.strip() or prompt in seen_text:
            raise ValueError(f"Prompt {prompt_id}: expected a unique nonempty prompt.")
        count = row.get("input_tokens")
        if type(count) is not int or not 0 < count < INPUT_LIMIT:
            raise ValueError(f"Prompt {prompt_id}: input_tokens must be 1–1023.")
        endpoint_counts = row.get("token_counts")
        if endpoint_counts is not None and (
            not isinstance(endpoint_counts, dict)
            or set(endpoint_counts) != set(ENDPOINTS)
            or any(type(value) is not int or not 0 < value <= count for value in endpoint_counts.values())
            or max(endpoint_counts.values()) != count
        ):
            raise ValueError(f"Prompt {prompt_id}: invalid endpoint token counts.")
        if not isinstance(row.get("source"), (str, dict)) or not row["source"]:
            raise ValueError(f"Prompt {prompt_id}: missing source provenance.")
        if row.get("split") not in SPLITS:
            raise ValueError(f"Prompt {prompt_id}: invalid evaluation split.")
        prompt_hash = digest(prompt)
        if row.get("prompt_sha256", prompt_hash) != prompt_hash:
            raise ValueError(f"Prompt {prompt_id}: prompt hash mismatch.")
        prompts[prompt_id] = {**row, "prompt_sha256": prompt_hash}
        seen_text.add(prompt)
    if Counter(row["split"] for row in prompts.values()) != SPLITS:
        raise ValueError("Expected 10 calibration prompts and 10 held-out prompts.")
    return prompts


def check_reference_labels():
    with (ROOT / "practice_answers.csv").open(newline="", encoding="utf-8-sig") as stream:
        endpoints = [row["endpoint"] for row in csv.DictReader(stream)]
    if sorted(endpoints) != list(ENDPOINTS):
        raise ValueError("The organizer's practice list must contain ref-01 through ref-09.")


def usage_flags(usage, reference_prompt_tokens):
    if usage is not None and not isinstance(usage, dict):
        raise ValueError("API usage must be an object or null.")
    usage = usage or {}
    flags = []
    for field, limit in (("prompt_tokens", INPUT_LIMIT - 1), ("completion_tokens", OUTPUT_LIMIT)):
        count = usage.get(field)
        if count is None:
            flags.append(f"missing_{field}")
        elif type(count) is not int or not 0 <= count <= limit:
            raise ValueError(f"Reported {field} exceeds its allowed range.")
    if reference_prompt_tokens is not None and usage.get("prompt_tokens") not in (None, reference_prompt_tokens):
        flags.append("prompt_token_count_differs")
    return flags


def response_flags(row, prompt):
    endpoint_counts = prompt.get("token_counts") or {}
    expected = None if prompt.get("count_method") == "estimated" else endpoint_counts.get(row["endpoint"], prompt["input_tokens"])
    flags = usage_flags(row.get("usage"), expected)
    if not row["response"].strip():
        flags.append("empty_response")
    if row.get("finish_reason") == "length":
        flags.append("output_truncated")
    return flags


def read_answers(path, prompts):
    done = {}
    if not path.exists():
        return done
    for row in json_records(path):
        endpoint = row.get("endpoint")
        prompt_id = row.get("id")
        repeat = row.get("repeat")
        if endpoint not in ENDPOINTS or not isinstance(prompt_id, str) or prompt_id not in prompts:
            raise ValueError("Answer file contains an unexpected endpoint or prompt ID.")
        if type(repeat) is not int or not 0 <= repeat < REPEATS:
            raise ValueError("Answer file contains an unexpected request.")
        key = endpoint, prompt_id, repeat
        if key in done:
            raise ValueError(f"Answer file contains a duplicate request: {key}.")
        prompt = prompts[prompt_id]
        for field in ("prompt", "prompt_sha256", "split", "source", "input_tokens"):
            if row.get(field) != prompt[field]:
                raise ValueError(f"Answer metadata differs from the prompt manifest for {key}: {field}.")
        if row.get("max_tokens") != OUTPUT_LIMIT or not isinstance(row.get("response"), str):
            raise ValueError(f"Invalid saved response or output limit for {key}.")
        done[key] = response_flags(row, prompt)
    return done


def append_record(path, row):
    payload = json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n"
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() > 0:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                stream.write(b"\n")
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def recover_partial_record(path):
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    end = data.rfind(b"\n") + 1
    try:
        json.loads(data[end:])
    except (ValueError, UnicodeError):
        with path.open("r+b") as stream:
            stream.truncate(end)
            stream.flush()
            os.fsync(stream.fileno())
        print("Recovered an incomplete final response record. Its request will be retried.", file=sys.stderr)


def retry_delay(headers, attempt):
    raw = headers.get("Retry-After")
    if raw and raw.isascii() and raw.isdecimal() and len(raw) <= 6:
        return int(raw) if int(raw) <= 60 else None
    return 2 ** attempt


def request_json(method, url, **kwargs):
    for attempt in range(ATTEMPTS):
        try:
            reply = method(url, allow_redirects=False, **kwargs)
        except (requests.ConnectionError, requests.Timeout) as exc:
            error = type(exc).__name__
            delay = retry_delay({}, attempt)
        else:
            if reply.status_code == 200:
                try:
                    return reply.json()
                except ValueError as exc:
                    raise RuntimeError("Organizer returned invalid JSON.") from exc
            error = f"HTTP {reply.status_code}"
            retryable = reply.status_code in RETRY_STATUS
            if reply.status_code == 429 and "Retry-After" not in reply.headers:
                retryable = False
            delay = retry_delay(reply.headers, attempt) if retryable else None
        if delay is None or attempt == ATTEMPTS - 1:
            raise RuntimeError(f"Organizer request failed ({error}).")
        time.sleep(delay)


def open_session():
    config = dotenv_values(ROOT / ".env", encoding="utf-8-sig", interpolate=False)
    key = os.environ.get("MODEL_EQUALITY_API_KEY") or config.get("MODEL_EQUALITY_API_KEY") or config.get("token")
    session = organizer.make_session()
    key = (key or session.headers.get("X-API-Key", "")).strip()
    if not key or key == "YOUR_API_KEY_HERE":
        session.close()
        raise RuntimeError("Set MODEL_EQUALITY_API_KEY or token in the local .env file.")
    session.headers["X-API-Key"] = key
    return session


def check_discovery():
    with open_session() as session:
        body = request_json(session.get, f"{organizer.BASE_URL}/models", timeout=15)
    listed = {row["id"] for row in body["data"]}
    missing = sorted(set(ENDPOINTS) - listed)
    if missing:
        raise RuntimeError("Organizer did not list reference endpoints " + ", ".join(missing) + ".")
    print("Organizer lists all nine reference endpoints.")


def collect_one(endpoint, prompt, repeat):
    if endpoint not in ENDPOINTS:
        raise ValueError("Only reference models are allowed.")
    message = {"role": "user", "content": prompt["prompt"]}
    payload = {"model": endpoint, "messages": [message], "max_tokens": OUTPUT_LIMIT}
    with open_session() as session:
        body = request_json(
            session.post, f"{organizer.BASE_URL}/chat/completions", json=payload, timeout=180
        )
    choice = body["choices"][0]
    response = choice["message"]["content"]
    if not isinstance(response, str):
        raise ValueError("Organizer response content must be a string.")
    row = {
        **{key: prompt[key] for key in ("id", "prompt", "prompt_sha256", "split", "source", "input_tokens")},
        "endpoint": endpoint,
        "repeat": repeat,
        "response": response,
        "max_tokens": OUTPUT_LIMIT,
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    row["flags"] = response_flags(row, prompt)
    return row


def list_jobs(prompts, done):
    return [
        (endpoint, prompt_id, repeat)
        for repeat in range(REPEATS)
        for prompt_id in prompts
        for endpoint in ENDPOINTS
        if (endpoint, prompt_id, repeat) not in done
    ]


def run_collection(prompts, done, jobs, output, failures):
    stopped = threading.Event()
    save_lock = threading.Lock()

    def worker(key):
        if stopped.is_set():
            return
        endpoint, prompt_id, repeat = key
        try:
            row = collect_one(endpoint, prompts[prompt_id], repeat)
        except (RuntimeError, ValueError, KeyError, IndexError, TypeError, requests.RequestException) as exc:
            stopped.set()
            details = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
            with save_lock:
                append_record(failures, {
                    "endpoint": endpoint,
                    "id": prompt_id,
                    "repeat": repeat,
                    "error": details,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                print(f"Collection stopped at {endpoint}/{prompt_id}/{repeat} ({details}).", file=sys.stderr)
            return
        with save_lock:
            append_record(output, row)
            done[key] = row["flags"]
            if len(done) % 25 == 0:
                print(f"Saved {len(done)}/{EXPECTED_COUNT} responses.", flush=True)

    with ThreadPoolExecutor(max_workers=min(4, organizer.PARALLEL)) as pool:
        for _ in pool.map(worker, jobs):
            pass


def summarize(done):
    endpoint_counts = Counter(key[0] for key in done)
    flags = Counter(flag for row_flags in done.values() for flag in row_flags)
    return {
        "expected": EXPECTED_COUNT,
        "saved": len(done),
        "missing": EXPECTED_COUNT - len(done),
        "saved_by_endpoint": {endpoint: endpoint_counts[endpoint] for endpoint in ENDPOINTS},
        "flags": dict(flags),
        "valid": True,
        "complete": len(done) == EXPECTED_COUNT,
    }


def save_report(path, staging, report):
    serialized = json.dumps(report, indent=2) + "\n"
    staging.write_text(serialized, encoding="utf-8")
    staging.replace(path)
    print(serialized, end="")


def main():
    parser = argparse.ArgumentParser(description="Collect and verify the 900 reference-only responses.")
    parser.add_argument("--prompts", type=Path, default=Path("output/reference_prompts.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("output/reference_answers.jsonl"))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verify-only", action="store_true", help="Validate saved responses offline.")
    action.add_argument("--check-service", action="store_true", help="Check model discovery without generating answers.")
    args = parser.parse_args()
    try:
        check_reference_labels()
        if args.check_service:
            check_discovery()
            return 0
        prompt_path = local_path(args.prompts)
        output = local_path(args.output)
        failures = local_path(output.with_suffix(".failures.jsonl"))
        lock_path = local_path(output.with_suffix(".lock.sqlite3"))
        report_path = local_path(output.with_suffix(".validation.json"))
        report_staging = local_path(report_path.with_suffix(".tmp"))
        if len({prompt_path, output, failures, lock_path, report_path, report_staging}) != 6:
            raise ValueError("Collection file paths must be distinct.")
        prompts = load_prompts(prompt_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(lock_path, timeout=0)) as lock:
            try:
                lock.execute("BEGIN EXCLUSIVE")
            except sqlite3.OperationalError as exc:
                raise RuntimeError("Could not acquire the collection lock.") from exc
            if not args.verify_only:
                recover_partial_record(output)
            try:
                done = read_answers(output, prompts)
            except ValueError as exc:
                save_report(report_path, report_staging, {
                    "expected": EXPECTED_COUNT, "valid": False, "complete": False, "error": str(exc)
                })
                raise
            jobs = list_jobs(prompts, done)
            if jobs and not args.verify_only:
                try:
                    check_discovery()
                    run_collection(prompts, done, jobs, output, failures)
                except RuntimeError as exc:
                    save_report(report_path, report_staging, {**summarize(done), "error": str(exc)})
                    raise
            report = summarize(done)
            save_report(report_path, report_staging, report)
            return 0 if report["complete"] else 1
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        print(f"Collection error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
