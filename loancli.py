"""Small local runner for the supplied 12-page loan assignment."""

import argparse
import hashlib
from collections import defaultdict
from contextlib import ExitStack, nullcontext
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

from dotenv import dotenv_values
from google import genai
from google.genai import errors, types
from jsonschema import Draft202012Validator
from PIL import Image


ROOT = Path(__file__).resolve().parent
MODEL = "gemma-4-26b-a4b-it"
EXTRACTION_MODEL = "gemini-3.1-flash-lite"
DEFAULT_TIMEOUT = 300
RETRY_DELAYS = (5, 10, 20)
FIELDS = ("borrower_name", "property_address", "loan_number")


class CLIError(Exception):
    """An actionable input, model, or reconciliation failure."""


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def publish_result(source: Path, target: Path):
    staged = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".result-", suffix=".tmp", delete=False
        ) as temporary:
            staged = Path(temporary.name)
            temporary.write(source.read_bytes())
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(staged, target)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def extraction_schema(pages):
    """Constrain a request to its physical pages without changing the merged schema."""
    schema = read_json(ROOT / "schemas/extraction.schema.json")
    entries = schema["properties"]["page_extractions"]
    entries["description"] = f"One entry for each physical PDF page in {pages}, in ascending order."
    entries["minItems"] = entries["maxItems"] = len(pages)
    entries["items"]["properties"]["page_number"]["enum"] = list(pages)
    return schema


def validate(data: dict, stage: str, expected_pages=None):
    schema = (extraction_schema(expected_pages) if stage == "extraction" and expected_pages is not None
              else read_json(ROOT / "schemas" / f"{stage}.schema.json"))
    error = next(Draft202012Validator(schema).iter_errors(data), None)
    if error:
        location = ".".join(map(str, error.absolute_path)) or "root"
        raise CLIError(f"{stage}: invalid {location}: {error.message}")
    key = "page_labels" if stage == "classification" else "page_extractions"
    expected = list(expected_pages) if expected_pages is not None else list(range(1, 13))
    if [p["page_number"] for p in data[key]] != expected:
        raise CLIError(f"{stage}: expected physical pages {expected} exactly once in order")
    if stage == "classification":
        grouped = []
        labels = {p["page_number"]: p["label"] for p in data[key]}
        for doc in data["documents"]:
            pages = doc["pages"]
            if not pages or pages != sorted(set(pages)):
                raise CLIError("classification: groups must be nonempty and sorted")
            if any(labels.get(page) != doc["label"] for page in pages):
                raise CLIError("classification: document and page labels disagree")
            grouped.extend(pages)
        if sorted(grouped) != list(range(1, 13)):
            raise CLIError("classification: every page must belong to exactly one group")
        return
    for page in data[key]:
        for field in FIELDS:
            item = page[field]
            readable = [o for o in item["observations"] if o["raw_value"] is not None]
            where = f"extraction: page {page['page_number']} {field}"
            if item["status"] == "present" and not readable:
                raise CLIError(f"{where}: present requires a readable value")
            if item["status"] == "not present" and item["observations"]:
                raise CLIError(f"{where}: not present requires no observations")
            if item["status"] == "unreadable" and (
                readable or not (item["notes"].strip() or any(
                    o["notes"].strip() for o in item["observations"]
                ))
            ):
                raise CLIError(f"{where}: unreadable requires uncertainty notes and no readable values")
            for observation in readable:
                if not observation["raw_value"].strip() or not (
                    observation["evidence_text"] or ""
                ).strip():
                    raise CLIError(f"{where}: readable values require nonempty evidence")


def sort_classification(data):
    """Order physical page references without mutating raw input or removing duplicates."""
    try:
        return {
            **data,
            "page_labels": sorted(data["page_labels"], key=lambda page: page["page_number"]),
            "documents": [{**document, "pages": sorted(document["pages"])}
                          for document in data["documents"]],
        }
    except (KeyError, TypeError):
        # Leave malformed structures for the existing schema validation.
        return data


def read_classification(path: Path):
    data = read_json(path)
    if isinstance(data, dict) and "structured_output" in data:
        if data.get("status") != "SUCCESS":
            raise CLIError("Supplied classification run was not successful")
        data = data["structured_output"]
    data = sort_classification(data)
    validate(data, "classification")
    return data


def run_fingerprint(command, classification=None):
    """Bind resume to source inputs and the code that prepares/configures requests."""
    stages = ["extraction"] if command == "extract" else ["classification"]
    if command == "full":
        stages.append("extraction")
    paths = ["AREAL_LOAN.pdf"]
    for stage in stages:
        paths.extend([f"prompts/{stage}.prompt.txt", f"schemas/{stage}.schema.json"])
    if "classification" in stages:
        paths.append("candidate_classification_labels.txt")
    return {
        "command": command,
        "files": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths},
        "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "models": {stage: {"model": EXTRACTION_MODEL if stage == "extraction" else MODEL,
                           "thinking_level": "high" if stage == "extraction" else "minimal"}
                   for stage in stages},
        "classification": classification if command == "extract" else None,
    }


def completed_stage(stage, output, pages=None):
    """Revalidate a completed artifact and check it against its saved raw response."""
    path = output / f"{stage}.json"
    if not path.exists():
        return None
    data = read_json(path)
    validate(data, stage, expected_pages=pages)
    raw = read_json(output / f"{stage}-run.json")
    expected = raw.get("structured_output")
    if stage == "classification":
        expected = sort_classification(expected)
    if raw.get("status") != "SUCCESS" or data != expected:
        raise CLIError(f"Cannot resume: {path} does not match its successful raw response")
    print(f"[{stage}] reusing validated {path}", file=sys.stderr, flush=True)
    return data


def compact(value: str) -> str:
    return " ".join(value.split())


def normalize(field: str, value: str) -> str:
    value = compact(value)
    if field == "borrower_name" and value.count(",") == 1:
        surname, given = value.split(",")
        value = compact(given + " " + surname)
    if field == "property_address":
        value = compact(value.replace(",", " "))
    return value.casefold()


def reconcile(classification: dict, extraction: dict) -> dict:
    documents = [
        {"document_id": f"document_{i}", **doc}
        for i, doc in enumerate(classification["documents"], 1)
    ]
    result = {
        "normalization_applied": [
            "Collapse repeated whitespace and line breaks; compare without capitalization differences.",
            "Reorder unambiguous surname-comma-given-name occurrences; preserve spelling and internal single spaces.",
            "Ignore address separator commas when comparing otherwise identical address tokens.",
            "Preserve identifier digits, name spellings, hyphens, unit numbers, and internal word-spacing differences.",
        ],
        "voting_rule": "Each distinct document contributes at most one vote per candidate. A document containing competing candidates can support each candidate once.",
        "documents": documents,
        "loan_level_fields": {},
        "reconciliation": {},
    }
    ties = {}
    for field in FIELDS:
        grouped = defaultdict(list)
        unreadable = []
        for page in extraction["page_extractions"]:
            for observation in page[field]["observations"]:
                observed = {"page_number": page["page_number"], **observation}
                if observation["raw_value"] is None:
                    unreadable.append(observed)
                else:
                    grouped[normalize(field, observation["raw_value"])].append(observed)
        candidates = []
        for value, observations in grouped.items():
            pages = sorted({o["page_number"] for o in observations})
            supported = [d["document_id"] for d in documents if set(d["pages"]) & set(pages)]
            candidates.append({
                "normalized_value": value,
                "document_count": len(supported),
                "supporting_document_ids": supported,
                "source_pages": pages,
                "observations": observations,
            })
        candidates.sort(key=lambda x: (
            -x["document_count"], x["source_pages"][0], x["normalized_value"]
        ))
        winners = [x for x in candidates if x["document_count"] == candidates[0]["document_count"]]
        winner = winners[0] if len(winners) == 1 else None
        if len(winners) > 1:
            ties[field] = [x["normalized_value"] for x in winners]
            rationale = "Exact document-vote tie; no selection rule has been defined."
        elif winner:
            rationale = f"Selected the unique candidate supported by the most distinct documents: {winner['document_count']}. Repeated occurrences within a document add no votes."
        else:
            rationale = "No readable candidate was observed; no value selected."
        alternatives = [
            {"value": compact(x["observations"][0]["raw_value"]),
             "document_count": x["document_count"], "source_pages": x["source_pages"]}
            for x in candidates if x is not winner
        ]
        uncertainty = (
            "Competing observations remain separate: "
            + json.dumps(alternatives, ensure_ascii=False)
            + ". Their cause and equivalence are not inferred from voting."
            if alternatives else "No competing readable values were observed under the stated normalization."
        )
        if unreadable:
            uncertainty += " Unreadable observations are preserved and do not receive votes."
        if not candidates:
            uncertainty = "No readable value; consult page-level status and notes in extraction.json."
        result["loan_level_fields"][field] = {
            "value": compact(winner["observations"][0]["raw_value"]) if winner else "",
            "source_pages": winner["source_pages"] if winner else [],
        }
        result["reconciliation"][field] = {
            "selected_normalized_value": winner["normalized_value"] if winner else None,
            "rationale": rationale,
            "remaining_uncertainty": uncertainty,
            "candidates": candidates,
            "unreadable_observations": unreadable,
        }
    if ties:
        result["ties"] = ties
    return result


def sdk_worker(request_path: Path) -> int:
    """One request with bounded 503 retries; never log the API key."""
    import logging

    logging.disable(logging.CRITICAL)
    api_key = None
    try:
        request = read_json(request_path)
        deadline = time.monotonic() + request["timeout"]
        api_key = dotenv_values(request["env_path"], interpolate=False).get("GEMINI_API_KEY")
        if not api_key or not api_key.strip():
            raise CLIError("Set GEMINI_API_KEY in the project .env file")
        model = request["model"]
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=request["schema"],
            thinking_config=types.ThinkingConfig(thinking_level=request["thinking_level"]),
        )
        with ExitStack() as stack:
            contents = []
            for item in request["images"]:
                image = stack.enter_context(Image.open(item["path"]))
                rgb = image.convert("RGB")
                stack.callback(rgb.close)
                contents.extend([item["caption"], rgb])
            contents.append(request["prompt"])
            client = stack.enter_context(genai.Client(
                api_key=api_key, vertexai=False,
                http_options=types.HttpOptions(
                    timeout=max(1, int(request["timeout"] * 1000)),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            ))
            for attempt in range(1, len(RETRY_DELAYS) + 2):
                if time.monotonic() >= deadline:
                    raise CLIError("Hard timeout before model attempt")
                print(json.dumps({"event": "request", "model": model,
                                  "image_count": len(request["images"]), "attempt": attempt}), flush=True)
                try:
                    response = client.models.generate_content(model=model, contents=contents, config=config)
                    break
                except errors.APIError as error:
                    detail = str(error).replace(api_key, "[REDACTED]")
                    print(json.dumps({"event": "attempt_error", "attempt": attempt,
                                      "http_status": error.code, "error": detail}), flush=True)
                    if error.code != 503 or attempt > len(RETRY_DELAYS):
                        raise
                    delay = RETRY_DELAYS[attempt - 1]
                    if time.monotonic() + delay >= deadline:
                        raise CLIError("Hard timeout leaves insufficient time for 503 retry delay") from error
                    print(json.dumps({"event": "retry", "attempt": attempt,
                                      "next_attempt": attempt + 1, "http_status": 503,
                                      "delay_seconds": delay}), flush=True)
                    time.sleep(delay)
        print(json.dumps({"event": "response", "attempt": attempt, "text": response.text}, ensure_ascii=False), flush=True)
        if not response.text:
            raise CLIError("Model returned no response text")
        data = json.loads(response.text)
        if not isinstance(data, dict):
            raise CLIError("Model response must be a JSON object")
        usage = response.usage_metadata
        envelope = {"status": "SUCCESS", "model": model,
                    "structured_output": data,
                    "usage_metadata": usage.model_dump(mode="json") if usage else None}
        print(json.dumps({"event": "result", "attempt": attempt, "result": envelope}, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        # SDK failures may contain request details. Redact before any output.
        detail = str(error)
        if api_key:
            detail = detail.replace(api_key, "[REDACTED]")
        print(f"{type(error).__name__}: {detail}", file=sys.stderr, flush=True)
        return 1


def stop_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # Children may survive after the group leader exits.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def render_command(command, log, deadline):
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        stop_process(process)


def prepare_views(images: Path, views: Path, log, deadline, pages=range(1, 13)):
    """Three full-width 40% bands, with 10% overlap and no resampling."""
    views.mkdir()
    for page in pages:
        with Image.open(images / f"page-{page:02}.png") as image:
            width, height = image.size
            for name, start, end in (("top", 0, 4), ("middle", 3, 7), ("bottom", 6, 10)):
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired("image preparation", 0)
                with image.crop((0, height * start // 10, width, height * end // 10)) as view:
                    view.save(views / f"page-{page:02}-{name}.png", dpi=(300, 300))
    if time.monotonic() >= deadline:
        raise subprocess.TimeoutExpired("image preparation", 0)


def run_stage(stage: str, output: Path, classification=None,
              timeout: float = DEFAULT_TIMEOUT, prepared_images: Path | None = None,
              pages=None, resume=False) -> dict:
    started = time.monotonic()
    deadline = started + timeout
    pages = list(pages) if pages is not None else list(range(1, 13))
    if resume:
        saved = completed_stage(stage, output, pages if stage == "extraction" else None)
        if saved is not None:
            return saved
    model = EXTRACTION_MODEL if stage == "extraction" else MODEL
    prompt = (ROOT / "prompts" / f"{stage}.prompt.txt").read_text(encoding="utf-8")
    if stage == "classification":
        prompt += "\nALLOWED LABELS\n" + (ROOT / "candidate_classification_labels.txt").read_text(encoding="utf-8")
    else:
        prompt = prompt.replace("{{PAGE_NUMBERS}}", json.dumps(pages))
        context = {
            "page_labels": [{"page_number": p["page_number"], "label": p["label"]}
                            for p in classification["page_labels"] if p["page_number"] in pages],
            "documents": [d for d in classification["documents"] if set(d["pages"]) <= set(pages)],
        }
        prompt += "\nCLASSIFICATION CONTEXT\n" + json.dumps(context, ensure_ascii=False)
    (output / f"{stage}.prompt.txt").write_text(prompt, encoding="utf-8")
    if stage == "extraction":
        write_json(output / f"{stage}.schema.json", extraction_schema(pages))
    else:
        shutil.copyfile(ROOT / "schemas" / f"{stage}.schema.json", output / f"{stage}.schema.json")
    stream_path = output / f"{stage}.stream.jsonl"
    stderr_path = output / f"{stage}.stderr.log"
    print(f"[{stage}] starting {model}; logs: {stream_path}", file=sys.stderr, flush=True)
    temp_root = ROOT / ".temp"
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root, prefix="") as directory:
        run_directory = Path(directory)
        workspace = run_directory / stage
        workspace.mkdir()
        rendered = prepared_images if prepared_images is not None else run_directory / "rendered"
        rendered.mkdir(exist_ok=True)
        render_log_path = output / f"{stage}.render.log"
        with render_log_path.open("a", encoding="utf-8") as render_log:
            try:
                if not (rendered / "page-01.png").exists():
                    print(f"[{stage}] rendering 12 source pages at 300 DPI", file=sys.stderr, flush=True)
                    render_command(
                        ["pdftoppm", "-png", "-r", "300", str(ROOT / "AREAL_LOAN.pdf"), str(rendered / "page")],
                        render_log, deadline,
                    )
                else:
                    print(f"[{stage}] reusing shared 300 DPI page images", file=sys.stderr, flush=True)
                    render_log.write("Reused shared 300 DPI page images; PDF was not rendered again.\n")
            except subprocess.TimeoutExpired as error:
                raise CLIError(f"{stage}: hard timeout after {timeout:g}s during page preparation. Logs: {render_log_path}") from error
            except (OSError, subprocess.CalledProcessError) as error:
                raise CLIError(f"{stage}: page rendering failed: {error}. Logs: {render_log_path}") from error
        images = workspace / "page_images"
        images.mkdir()
        for page in pages:
            source = rendered / f"page-{page:02}.png"
            if not source.is_file():
                raise CLIError(f"{stage}: rendered physical page {page} is missing")
            shutil.copyfile(source, images / source.name)
        if stage == "extraction":
            print(f"[{stage}] preparing three overlapping views per page", file=sys.stderr, flush=True)
            with render_log_path.open("a", encoding="utf-8") as render_log:
                try:
                    prepare_views(rendered, workspace / "page_views", render_log, deadline, pages)
                except subprocess.TimeoutExpired as error:
                    raise CLIError(f"{stage}: hard timeout after {timeout:g}s during page preparation. Logs: {render_log_path}") from error
                except (OSError, subprocess.CalledProcessError) as error:
                    raise CLIError(f"{stage}: page view preparation failed: {error}. Logs: {render_log_path}") from error
        attachments = [{"caption": f"Physical PDF page {page}",
                        "path": str(images / f"page-{page:02}.png")}
                       for page in pages]
        if stage == "extraction":
            attachments.extend(
                {"caption": f"Physical PDF page {page}, {view} view",
                 "path": str(workspace / "page_views" / f"page-{page:02}-{view}.png")}
                for page in pages for view in ("top", "middle", "bottom")
            )
        request_path = workspace / "request.json"
        write_json(request_path, {
            "model": model, "thinking_level": "high" if stage == "extraction" else "minimal",
            "prompt": prompt, "schema": read_json(output / f"{stage}.schema.json"),
            "images": attachments, "env_path": str(ROOT / ".env"),
            "timeout": max(0.001, deadline - time.monotonic()),
        })
        command = [sys.executable, str(Path(__file__).resolve()), "--sdk-worker", str(request_path)]
        with stream_path.open("a", encoding="utf-8") as log, stderr_path.open("a", encoding="utf-8") as stderr:
            log.write(json.dumps({"event": "invocation", "run_id": run_directory.name}) + "\n")
            log.flush()
            process = subprocess.Popen(command, cwd=workspace, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=stderr, text=True,
                                       encoding="utf-8", start_new_session=True)
            lines = queue.Queue()

            def read_lines():
                try:
                    for line in process.stdout:
                        lines.put(line)
                finally:
                    lines.put(None)

            reader = threading.Thread(target=read_lines, daemon=True)
            reader.start()
            envelope = None
            last_progress = time.monotonic()
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise CLIError(f"hard timeout after {timeout:g}s")
                    try:
                        line = lines.get(timeout=min(15, remaining))
                    except queue.Empty:
                        line = ""
                    if line is None:
                        break
                    event = {}
                    if line:
                        log.write(line)
                        log.flush()
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise CLIError("SDK worker emitted invalid JSON") from error
                        if event.get("event") == "result":
                            if envelope is not None:
                                raise CLIError("SDK worker emitted duplicate results")
                            envelope = event["result"]
                        elif event.get("event") == "request":
                            print(f"[{stage}] model attempt {event['attempt']}/4", file=sys.stderr, flush=True)
                        elif event.get("event") == "retry":
                            print(f"[{stage}] HTTP 503 on attempt {event['attempt']}; retrying in "
                                  f"{event['delay_seconds']}s (attempt {event['next_attempt']}/4)",
                                  file=sys.stderr, flush=True)
                    now = time.monotonic()
                    if now - last_progress >= 15:
                        print(f"[{stage}] waiting for model ({int(now - started)}s)", file=sys.stderr, flush=True)
                        last_progress = now
                try:
                    code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired as error:
                    raise CLIError(f"hard timeout after {timeout:g}s") from error
                if code:
                    detail = stderr_path.read_text(encoding="utf-8").strip()
                    raise CLIError(f"SDK worker exited with code {code}: {detail}")
                if not isinstance(envelope, dict) or envelope.get("status") != "SUCCESS":
                    raise CLIError("SDK worker ended without a successful result")
                write_json(output / f"{stage}-run.json", envelope)
                data = envelope["structured_output"]
                if stage == "classification":
                    data = sort_classification(data)
                validate(data, stage, expected_pages=pages if stage == "extraction" else None)
                write_json(output / f"{stage}.json", data)
                print(f"[{stage}] validated ({int(time.monotonic() - started)}s)", file=sys.stderr, flush=True)
                return data
            except CLIError as error:
                raise CLIError(f"{stage}: {error}. Logs: {stream_path}, {stderr_path}") from error
            finally:
                stop_process(process)
                reader.join(timeout=5)
                while not lines.empty():
                    pending = lines.get_nowait()
                    if pending:
                        log.write(pending)
                process.stdout.close()


def run_extraction(output: Path, classification: dict, timeout=DEFAULT_TIMEOUT,
                   prepared_images: Path | None = None, resume=False) -> dict:
    """Extract each document once; publish a merged artifact only after all succeed."""
    started = time.monotonic()
    deadline = started + timeout
    (ROOT / ".temp").mkdir(exist_ok=True)
    preparation = (nullcontext(prepared_images) if prepared_images is not None
                   else tempfile.TemporaryDirectory(dir=ROOT / ".temp", prefix="shared-"))
    merged = {"page_extractions": []}
    artifacts = []
    with preparation as directory:
        for index, document in enumerate(classification["documents"], 1):
            name = f"document-{index:02}"
            group_output = output / "groups" / name
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CLIError(f"extraction: hard timeout after {timeout:g}s; completed groups saved in {output / 'groups'}")
            group_output.mkdir(parents=True, exist_ok=resume)
            print(f"[extraction] {name} ({index}/{len(classification['documents'])}), physical pages {document['pages']}",
                  file=sys.stderr, flush=True)
            data = run_stage("extraction", group_output, classification,
                             timeout=remaining, prepared_images=Path(directory), pages=document["pages"],
                             resume=resume)
            merged["page_extractions"].extend(data["page_extractions"])
            artifacts.append({"pages": document["pages"],
                              "raw_response": f"groups/{name}/extraction-run.json"})
    merged["page_extractions"].sort(key=lambda page: page["page_number"])
    validate(merged, "extraction")
    write_json(output / "extraction-run.json", {
        "status": "SUCCESS", "model": EXTRACTION_MODEL,
        "groups": artifacts, "structured_output": merged,
    })
    write_json(output / "extraction.json", merged)
    print(f"[extraction] merged and validated ({time.monotonic() - started:.1f}s)", file=sys.stderr, flush=True)
    return merged


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("classify", "extract", "full"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True, help="new or empty output directory")
        command.add_argument("--resume", action="store_true",
                             help="resume this output directory only when its saved inputs/configuration match")
        command.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                             help="hard timeout in seconds per stage, including image preparation (default: 300)")
        if name == "extract":
            command.add_argument("--classification", type=Path, required=True, help="validated classification JSON or successful saved run envelope")
    args = parser.parse_args(argv)
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be a finite positive number")
    try:
        classification = read_classification(args.classification) if args.command == "extract" else None
        output = args.output.resolve()
        manifest = output / "run-manifest.json"
        if args.resume:
            if not manifest.is_file():
                raise CLIError(f"Cannot resume: saved run manifest is missing: {manifest}")
            if read_json(manifest) != run_fingerprint(args.command, classification):
                raise CLIError("Cannot resume: inputs, prompt, schema, command, and model/runner configuration must be unchanged")
        else:
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise CLIError(f"Output must be a new or empty directory: {output}")
            output.mkdir(parents=True, exist_ok=True)
            write_json(manifest, run_fingerprint(args.command, classification))
        preparation = nullcontext(None)
        if args.command == "full":
            (ROOT / ".temp").mkdir(exist_ok=True)
            preparation = tempfile.TemporaryDirectory(dir=ROOT / ".temp", prefix="shared-")
        with preparation as directory:
            shared_images = Path(directory) if directory else None
            if args.command in ("classify", "full"):
                classification = run_stage("classification", output, timeout=args.timeout,
                                           prepared_images=shared_images, resume=args.resume)
            else:
                write_json(output / "classification.json", classification)
            if args.command in ("extract", "full"):
                extraction = run_extraction(output, classification, timeout=args.timeout,
                                            prepared_images=shared_images, resume=args.resume)
        if args.command == "full":
            reconciliation_started = time.monotonic()
            reconciliation = reconcile(classification, extraction)
            reconciliation["source_artifacts"] = {
                "extraction_run": "extraction-run.json",
                "extraction": "extraction.json",
                "classification": "classification.json",
            }
            write_json(output / "reconciliation.json", reconciliation)
            if reconciliation.get("ties"):
                raise CLIError("Concrete document-vote tie: " + json.dumps(reconciliation["ties"]) + "; candidates saved in reconciliation.json; result.json was not generated")
            write_json(output / "result.json", {
                "page_labels": classification["page_labels"],
                "documents": classification["documents"],
                "loan_level_fields": reconciliation["loan_level_fields"],
            })
            publish_result(output / "result.json", ROOT / "result.json")
            print(f"[reconciliation] complete ({time.monotonic() - reconciliation_started:.2f}s)", file=sys.stderr)
        print(f"Saved {args.command} output to {output}")
        return 0
    except (CLIError, OSError, ValueError) as error:
        print(f"loancli: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("loancli: interrupted; partial logs are preserved", file=sys.stderr)
        return 130


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--sdk-worker":
        sys.exit(sdk_worker(Path(sys.argv[2])))
    sys.exit(main())
