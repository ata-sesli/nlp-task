"""Small local runner for the supplied 12-page loan assignment."""

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zlib

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parent
MODEL = "gemini-3.8-flash-low"
DEFAULT_TIMEOUT = 300
FIELDS = ("borrower_name", "property_address", "loan_number")
CONTEXT_PATH = Path(
    "classification-runs/classification-flash-low-ordering-run.json"
)


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


def validate(data: dict, stage: str):
    schema = read_json(ROOT / "schemas" / f"{stage}.schema.json")
    error = next(Draft202012Validator(schema).iter_errors(data), None)
    if error:
        location = ".".join(map(str, error.absolute_path)) or "root"
        raise CLIError(f"{stage}: invalid {location}: {error.message}")
    key = "page_labels" if stage == "classification" else "page_extractions"
    if [p["page_number"] for p in data[key]] != list(range(1, 13)):
        raise CLIError(f"{stage}: expected physical pages 1–12 exactly once in order")
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


def read_classification(path: Path):
    data = read_json(path)
    if isinstance(data, dict) and "structured_output" in data:
        if data.get("status") != "SUCCESS":
            raise CLIError("Supplied classification run was not successful")
        data = data["structured_output"]
    validate(data, "classification")
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


class StreamParser:
    def __init__(self, workspace: Path | None = None):
        self.result = None
        self.workspace = workspace.resolve() if workspace else None
        self.cwd_verified = False

    def feed(self, line: str):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise CLIError("agy emitted invalid streaming JSON") from error
        if not isinstance(event, dict):
            raise CLIError("agy stream event must be an object")
        if event.get("event") == "init" and self.workspace is not None:
            init = event.get("init")
            cwd = init.get("cwd") if isinstance(init, dict) else None
            if not isinstance(cwd, str) or not Path(cwd).is_absolute() or Path(cwd).resolve() != self.workspace:
                raise CLIError(f"agy streamed working directory {cwd!r}; expected {self.workspace}")
            self.cwd_verified = True
        if event.get("event") == "result":
            if self.result is not None or not isinstance(event.get("result"), dict):
                raise CLIError("agy emitted an invalid or duplicate result event")
            self.result = event["result"]
        return event

    def finish(self, exit_code: int) -> dict:
        if exit_code:
            detail = self.result.get("error", "") if self.result else ""
            raise CLIError(f"agy exited with code {exit_code}: {detail}")
        if self.result is None:
            raise CLIError("agy stream ended without a result event")
        if self.workspace is not None and not self.cwd_verified:
            raise CLIError("agy did not confirm the stage working directory in its init event")
        if self.result.get("status") != "SUCCESS":
            raise CLIError(f"agy status {self.result.get('status')}: {self.result.get('error', '')}")
        if not isinstance(self.result.get("structured_output"), dict):
            raise CLIError("agy result has no structured_output object")
        return self.result


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


def read_rendered_pixels(path: Path):
    """Read Poppler's raw RGB PPM output; no resampling or image library."""
    with path.open("rb") as image:
        if image.readline().strip() != b"P6":
            raise CLIError(f"Invalid rendered image: {path}")
        width, height = map(int, image.readline().split())
        if image.readline().strip() != b"255":
            raise CLIError(f"Unsupported rendered image depth: {path}")
        pixels = image.read()
    if width <= 0 or height <= 0 or len(pixels) != width * height * 3:
        raise CLIError(f"Incomplete rendered image: {path}")
    return width, height, pixels


def write_png(path: Path, width, height, pixels, deadline):
    if time.monotonic() >= deadline:
        raise subprocess.TimeoutExpired("image preparation", 0)

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))

    stride = width * 3
    rows = b"".join(b"\0" + pixels[y * stride:(y + 1) * stride] for y in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"pHYs", struct.pack(">IIB", 11811, 11811, 1))  # 300 DPI
        + chunk(b"IDAT", zlib.compress(rows, level=1)) + chunk(b"IEND", b""))
    if time.monotonic() >= deadline:
        raise subprocess.TimeoutExpired("image preparation", 0)


def prepare_views(images: Path, views: Path, log, deadline):
    """Three full-width 40% bands, with 10% overlap and no resampling."""
    views.mkdir()
    for page in range(1, 13):
        width, height, pixels = read_rendered_pixels(images / f"page-{page:02}.ppm")
        for name, start, end in (("top", 0, 4), ("middle", 3, 7), ("bottom", 6, 10)):
            y, bottom = height * start // 10, height * end // 10
            write_png(views / f"page-{page:02}-{name}.png", width, bottom - y,
                      pixels[y * width * 3:bottom * width * 3], deadline)


def isolate_agy(command):
    """Exclude global coding skills for this process without changing setup."""
    sandbox = Path("/usr/bin/sandbox-exec")
    if sys.platform != "darwin" or not sandbox.is_file():
        raise CLIError("Per-run skill isolation currently requires macOS sandbox-exec")
    home = Path.home().resolve()
    profile = (
        "(version 1)(allow default)(deny file-read* "
        f"(subpath {json.dumps(str(home / '.gemini/config/skills'))}) "
        f"(subpath {json.dumps(str(home / '.agents/skills'))}) "
        f"(literal {json.dumps(str(home / '.gemini/GEMINI.md'))}))"
    )
    return [str(sandbox), "-p", profile, *command]


def run_stage(stage: str, output: Path, classification=None,
              timeout: float = DEFAULT_TIMEOUT, prepared_images: Path | None = None) -> dict:
    started = time.monotonic()
    deadline = started + timeout
    executable = shutil.which("agy")
    if executable is None:
        raise CLIError("agy is not on PATH; install and authenticate Antigravity CLI first")
    prompt = (
        "STAGE INPUTS\n"
        "The source is 12 freshly rendered images, page_images/page-01.png\n"
        "through page_images/page-12.png. Their filenames give physical PDF\n"
        "page positions. The original PDF is not provided in this workspace.\n"
        "These input instructions replace all PDF-reading, text-extraction,\n"
        "and rendering instructions in the task below. Inspect these images\n"
        "directly; do not search for the PDF or previous outputs elsewhere.\n\n"
        + (ROOT / "prompts" / f"{stage}.prompt.txt").read_text(encoding="utf-8")
    )
    (output / f"{stage}.prompt.txt").write_text(prompt, encoding="utf-8")
    shutil.copyfile(ROOT / "schemas" / f"{stage}.schema.json", output / f"{stage}.schema.json")
    stream_path = output / f"{stage}.stream.jsonl"
    stderr_path = output / f"{stage}.stderr.log"
    print(f"[{stage}] starting {MODEL}; logs: {stream_path}", file=sys.stderr, flush=True)
    temp_root = ROOT / ".temp"
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root, prefix="") as directory:
        run_directory = Path(directory)
        workspace = run_directory / stage
        workspace.mkdir()
        rendered = prepared_images if prepared_images is not None else run_directory / "rendered"
        rendered.mkdir(exist_ok=True)
        render_log_path = output / f"{stage}.render.log"
        with render_log_path.open("w", encoding="utf-8") as render_log:
            try:
                if not (rendered / "page-01.png").exists():
                    print(f"[{stage}] rendering 12 source pages at 300 DPI", file=sys.stderr, flush=True)
                    render_command(
                        ["pdftoppm", "-r", "300", str(ROOT / "AREAL_LOAN.pdf"), str(rendered / "page")],
                        render_log, deadline,
                    )
                    for page in range(1, 13):
                        width, height, pixels = read_rendered_pixels(rendered / f"page-{page:02}.ppm")
                        write_png(rendered / f"page-{page:02}.png", width, height, pixels, deadline)
                else:
                    print(f"[{stage}] reusing shared 300 DPI page images", file=sys.stderr, flush=True)
                    render_log.write("Reused shared 300 DPI page images; PDF was not rendered again.\n")
            except subprocess.TimeoutExpired as error:
                raise CLIError(f"{stage}: hard timeout after {timeout:g}s during page preparation. Logs: {render_log_path}") from error
            except (OSError, subprocess.CalledProcessError) as error:
                raise CLIError(f"{stage}: page rendering failed: {error}. Logs: {render_log_path}") from error
        images = workspace / "page_images"
        images.mkdir()
        for page in range(1, 13):
            source = rendered / f"page-{page:02}.png"
            if not source.is_file():
                raise CLIError(f"{stage}: rendered physical page {page} is missing")
            shutil.copyfile(source, images / source.name)
        if stage == "extraction":
            print(f"[{stage}] preparing three overlapping views per page", file=sys.stderr, flush=True)
            with render_log_path.open("a", encoding="utf-8") as render_log:
                try:
                    prepare_views(rendered, workspace / "page_views", render_log, deadline)
                except subprocess.TimeoutExpired as error:
                    raise CLIError(f"{stage}: hard timeout after {timeout:g}s during page preparation. Logs: {render_log_path}") from error
                except (OSError, subprocess.CalledProcessError) as error:
                    raise CLIError(f"{stage}: page view preparation failed: {error}. Logs: {render_log_path}") from error
        for suffix in ("prompt.txt", "schema.json"):
            shutil.copyfile(output / f"{stage}.{suffix}", workspace / f"{stage}.{suffix}")
        if stage == "classification":
            shutil.copyfile(ROOT / "candidate_classification_labels.txt", workspace / "candidate_classification_labels.txt")
        if classification is not None:
            context = workspace / CONTEXT_PATH
            context.parent.mkdir()
            write_json(context, {"structured_output": {
                "page_labels": [{"page_number": p["page_number"], "label": p["label"]} for p in classification["page_labels"]],
                "documents": classification["documents"],
            }})
        command = [executable, "-p", (workspace / f"{stage}.prompt.txt").read_text(),
                   "--new-project", "--sandbox", "--disable-slash-commands",
                   "--model", MODEL, "--output-format", "stream-json",
                   "--json-schema", str(workspace / f"{stage}.schema.json")]
        command = isolate_agy(command)
        with stream_path.open("w", encoding="utf-8") as log, stderr_path.open("w", encoding="utf-8") as stderr:
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
            parser = StreamParser(workspace)
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
                        event = parser.feed(line)
                    now = time.monotonic()
                    step = event.get("step_update", {})
                    if now - last_progress >= 15:
                        tool = step.get("tool_name") if isinstance(step, dict) else None
                        detail = f"tool {tool}" if tool else "working"
                        print(f"[{stage}] {detail} ({int(now - started)}s)", file=sys.stderr, flush=True)
                        last_progress = now
                try:
                    code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired as error:
                    raise CLIError(f"hard timeout after {timeout:g}s") from error
                envelope = parser.finish(code)
                write_json(output / f"{stage}-run.json", envelope)
                validate(envelope["structured_output"], stage)
                write_json(output / f"{stage}.json", envelope["structured_output"])
                print(f"[{stage}] validated ({int(time.monotonic() - started)}s)", file=sys.stderr, flush=True)
                return envelope["structured_output"]
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("classify", "extract", "full"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True, help="new or empty output directory")
        command.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                             help="hard timeout in seconds per stage, including image preparation (default: 300)")
        if name == "extract":
            command.add_argument("--classification", type=Path, required=True, help="validated classification JSON or successful agy JSON envelope")
    args = parser.parse_args(argv)
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be a finite positive number")
    try:
        classification = read_classification(args.classification) if args.command == "extract" else None
        output = args.output.resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise CLIError(f"Output must be a new or empty directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        preparation = nullcontext(None)
        if args.command == "full":
            (ROOT / ".temp").mkdir(exist_ok=True)
            preparation = tempfile.TemporaryDirectory(dir=ROOT / ".temp", prefix="shared-")
        with preparation as directory:
            shared_images = Path(directory) if directory else None
            if args.command in ("classify", "full"):
                classification = run_stage("classification", output, timeout=args.timeout,
                                           prepared_images=shared_images)
            else:
                write_json(output / "classification.json", classification)
            if args.command in ("extract", "full"):
                extraction = run_stage("extraction", output, classification, timeout=args.timeout,
                                       prepared_images=shared_images)
        if args.command == "full":
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
            print("[reconciliation] complete", file=sys.stderr)
        print(f"Saved {args.command} output to {output}")
        return 0
    except (CLIError, OSError, ValueError) as error:
        print(f"loancli: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("loancli: interrupted; partial logs are preserved", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
