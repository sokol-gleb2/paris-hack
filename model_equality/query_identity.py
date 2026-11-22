import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

import query_reference as organizer


ROOT = Path(__file__).resolve().parent
ENDPOINTS = tuple(f"ep-{index:02d}" for index in range(1, 34))
PROMPT = (
    'Identify the underlying model serving this request, separately from your assigned '
    'assistant name, persona or role. Return only one compact JSON object with these keys: '
    '"family", "model", "quantization", "lora_adapter", "lora_rank", "lora_alpha", '
    '"lora_scale", "evidence". Give the actual base model family and exact checkpoint/version, '
    'weight precision or quantization method, and any applied LoRA adapter and its settings. '
    'LoRA scale means the inference adapter multiplier, not rank or alpha. '
    'Use only facts you actually have about this deployment. Write "unknown" for each '
    'unavailable, unverifiable or undisclosable field; write "none" only if absence is known. '
    'Do not infer settings from your behavior, a persona, a provider name or an endpoint alias. '
    'In evidence, briefly state whether the information is supplied deployment metadata or '
    'only a self-identification claim. Keep the entire answer below 220 tokens.'
)
DETAIL_FIELDS = ("quantization", "lora_adapter", "lora_rank", "lora_alpha", "lora_scale")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_event(directory, event):
    with (directory / "attempts.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": now(), **event}, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_events(directory):
    path = directory / "attempts.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_claim(response):
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def claim_value(value):
    if value is None or value == "":
        return "unknown"
    if isinstance(value, (dict, list, bool)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def model_label(claim):
    model = claim_value(claim.get("model"))
    if re.fullmatch(r"(?:ep|ref)-\d+", model.strip(), flags=re.IGNORECASE):
        model = "unknown"
    details = [f"{key}={claim_value(claim[key])}" for key in DETAIL_FIELDS
               if key in claim and claim_value(claim[key]).lower() not in ("unknown", "null")]
    return "; ".join([model, *details])


def export_csv(directory):
    events = load_events(directory)
    starts = {event["endpoint"] for event in events if event["event"] == "started"}
    results = {event["endpoint"]: event for event in events if event["event"] == "result"}
    rows, statuses, claims = [], {}, {}
    for endpoint in ENDPOINTS:
        result = results.get(endpoint, {})
        response = result.get("response", "")
        claim = parse_claim(response)
        statuses[endpoint] = result.get("status", "uncertain" if endpoint in starts else "not_queried")
        claims[endpoint] = {"parsed_claim": claim, "verified": False,
                            "api_model": result.get("body", {}).get("model")}
        rows.append({"endpoint": endpoint, "family": claim_value(claim.get("family")),
                     "model": model_label(claim), "response": response})
    temporary = directory / "identity.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=("endpoint", "family", "model", "response"))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(directory / "identity.csv")
    summary = {"updated": now(), "counts": dict(Counter(statuses.values())),
               "endpoint_status": statuses, "claims": claims,
               "limitations": "Family/model are unverified self-reports. Unknown does not mean absent. "
                              "Quantization and LoRA settings cannot be established from a claim alone. "
                              "The response column is the exact returned message content; empty cells "
                              "may indicate unqueried or failed endpoints; consult endpoint_status."}
    write_json(directory / "summary.json", summary)
    return summary


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"endpoints": list(ENDPOINTS), "prompt": PROMPT,
                "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "base_url": organizer.BASE_URL, "max_tokens": 256,
                "repeats": 1, "concurrency": 1, "automatic_retries": 0}
    path = directory / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing manifest differs; choose a new output directory.")
    else:
        write_json(path, manifest)


def collect(directory, session):
    events = load_events(directory)
    starts = {event["endpoint"] for event in events if event["event"] == "started"}
    results = {event["endpoint"]: event for event in events if event["event"] == "result"}
    if starts - results.keys() or any(result.get("status") == "uncertain" for result in results.values()):
        raise RuntimeError("An earlier POST has uncertain delivery. No more requests will be scheduled.")
    pending = [endpoint for endpoint in ENDPOINTS if endpoint not in starts
               and results.get(endpoint, {}).get("status") != "success"]
    if not pending:
        return
    try:
        reply = session.get(f"{organizer.BASE_URL}/models", timeout=(10, 30), allow_redirects=False)
        reply.raise_for_status()
        body = reply.json()
        models = {item["id"]: item for item in body["data"]}
        write_json(directory / "server_models.json", {"time": now(), "body": body})
        remaining = {endpoint: int(models[endpoint]["budget"]["limit"])
                     - int(models[endpoint]["budget"]["used"]) for endpoint in pending}
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        write_json(directory / "preflight_error.json", {"time": now(), "error": str(exc),
                                                       "new_completion_requests_sent": 0})
        raise RuntimeError(f"Budget check failed; no completion requests sent: {exc}") from exc
    for endpoint in pending:
        if remaining[endpoint] < 1:
            append_event(directory, {"event": "result", "endpoint": endpoint, "status": "budget_exhausted"})
            export_csv(directory)
            continue
        payload = {"model": endpoint, "max_tokens": 256,
                   "messages": [{"role": "user", "content": PROMPT}]}
        append_event(directory, {"event": "started", "endpoint": endpoint, "request": payload})
        result = {"event": "result", "endpoint": endpoint}
        try:
            reply = session.post(f"{organizer.BASE_URL}/chat/completions", json=payload,
                                 timeout=(10, 180), allow_redirects=False)
        except requests.RequestException as exc:
            result.update(status="uncertain", error=str(exc))
        else:
            result.update(http_status=reply.status_code, raw_http_body=reply.text)
            try:
                body = reply.json()
            except ValueError:
                body = {}
            result["body"] = body if isinstance(body, dict) else {"unparsed": body}
            if reply.status_code == 200:
                try:
                    response = body["choices"][0]["message"]["content"]
                    if not isinstance(response, str):
                        raise TypeError("Message content is not text")
                    result.update(status="success", response=response)
                except (KeyError, IndexError, TypeError):
                    result.update(status="invalid_response")
            else:
                result["status"] = "uncertain" if reply.status_code >= 500 or reply.status_code == 408 else "http_error"
        append_event(directory, result)
        export_csv(directory)
        print(f"{endpoint}: {result['status']}", flush=True)
        if result["status"] == "uncertain" or result.get("http_status") in (401, 403, 429):
            raise RuntimeError("Stopped after uncertainty, authentication failure or rate limit; no automatic retry.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "identity_probe")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".collector.lock"
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError:
        parser.exit(1, "Collector lock exists. Check for an active or interrupted process before proceeding.\n")
    try:
        with lock:
            lock.write(str(os.getpid()))
        prepare(directory)
        if not args.export_only and not args.prepare_only:
            with organizer.make_session() as session:
                collect(directory, session)
        summary = export_csv(directory)
        print(json.dumps(summary["counts"]))
        print(directory / "identity.csv")
        if not args.export_only and not args.prepare_only and summary["counts"].get("success") != len(ENDPOINTS):
            return 1
    except (RuntimeError, ValueError, OSError) as exc:
        if (directory / "manifest.json").exists():
            export_csv(directory)
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        lock_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
