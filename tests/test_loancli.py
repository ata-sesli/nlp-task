import copy
import json
import io
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

import httpx

from PIL import Image

import loancli


ROOT = Path(__file__).resolve().parents[1]


class ViewTests(unittest.TestCase):
    def test_views_use_rendered_pixels_without_renderer(self):
        with tempfile.TemporaryDirectory() as directory:
            images = Path(directory) / "images"
            images.mkdir()
            pixels = b"".join(bytes([y, x, 127]) for y in range(100) for x in range(10))
            for page in range(1, 13):
                Image.frombytes("RGB", (10, 100), pixels).save(images / f"page-{page:02}.png")
            views = Path(directory) / "views"
            with patch.object(loancli, "render_command", side_effect=AssertionError("must not render regions")):
                loancli.prepare_views(images, views, None, time.monotonic() + 30)
            for name, y in [("top", 0), ("middle", 30), ("bottom", 60)]:
                with Image.open(views / f"page-01-{name}.png") as image:
                    self.assertEqual(image.size, (10, 40))
                    self.assertEqual(image.tobytes(), pixels[y * 30:(y + 40) * 30])


def load(name):
    return json.loads((ROOT / name).read_text())


def accepted_classification():
    result = load("result.json")
    return {key: result[key] for key in ["page_labels", "documents"]}


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.classification = accepted_classification()
        self.extraction = load("extraction.json")

    def test_accepted_outputs_and_wrapped_classification(self):
        loancli.validate(self.classification, "classification")
        loancli.validate(self.extraction, "extraction")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classification-run.json"
            path.write_text(json.dumps({"status": "SUCCESS", "structured_output": self.classification}))
            self.assertEqual(loancli.read_classification(path), self.classification)

    def test_supplied_classification_sorts_without_changing_source_or_duplicates(self):
        unsorted = copy.deepcopy(self.classification)
        unsorted["page_labels"].reverse()
        for document in unsorted["documents"]:
            document["pages"].reverse()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classification.json"
            for data in [unsorted, {"status": "SUCCESS", "structured_output": unsorted}]:
                path.write_text(json.dumps(data))
                original = path.read_bytes()
                self.assertEqual(loancli.read_classification(path), self.classification)
                self.assertEqual(path.read_bytes(), original)
            for kind in ["page_labels", "documents"]:
                duplicate = copy.deepcopy(unsorted)
                if kind == "page_labels":
                    duplicate[kind].append(copy.deepcopy(duplicate[kind][0]))
                else:
                    duplicate[kind][0]["pages"].append(duplicate[kind][0]["pages"][0])
                path.write_text(json.dumps(duplicate))
                with self.assertRaises(loancli.CLIError):
                    loancli.read_classification(path)

    def test_invalid_page_coverage_and_grouping_are_rejected(self):
        bad = copy.deepcopy(self.classification)
        bad["page_labels"][-1]["page_number"] = 11
        with self.assertRaises(loancli.CLIError):
            loancli.validate(bad, "classification")
        bad = copy.deepcopy(self.classification)
        bad["documents"][0]["pages"].append(3)
        with self.assertRaises(loancli.CLIError):
            loancli.validate(bad, "classification")
        bad = copy.deepcopy(self.classification)
        bad["documents"][0]["label"] = bad["documents"][1]["label"]
        with self.assertRaises(loancli.CLIError):
            loancli.validate(bad, "classification")

    def test_group_extraction_rejects_missing_extra_foreign_and_duplicate_pages(self):
        group = {"page_extractions": copy.deepcopy(self.extraction["page_extractions"][:2])}
        loancli.validate(group, "extraction", expected_pages=[1, 2])
        bad_cases = [group["page_extractions"][:1], group["page_extractions"] + [copy.deepcopy(group["page_extractions"][0])],
                     [group["page_extractions"][0], group["page_extractions"][0]],
                     [group["page_extractions"][0], self.extraction["page_extractions"][2]]]
        for entries in bad_cases:
            with self.assertRaises(loancli.CLIError):
                loancli.validate({"page_extractions": entries}, "extraction", expected_pages=[1, 2])

    def test_missing_evidence_and_inconsistent_status_are_rejected(self):
        bad = copy.deepcopy(self.extraction)
        bad["page_extractions"][0]["borrower_name"]["observations"][0]["evidence_text"] = None
        with self.assertRaises(loancli.CLIError):
            loancli.validate(bad, "extraction")

    def test_unreadable_evidence_is_valid_and_retained_without_votes(self):
        extraction = copy.deepcopy(self.extraction)
        item = extraction["page_extractions"][2]["borrower_name"]
        item.update(status="unreadable", notes="Name remains unclear in supplied views.",
                    observations=[{"raw_value": None, "evidence_text": None,
                                   "field_label": None, "notes": "Unreadable name."}])
        loancli.validate(extraction, "extraction")
        result = loancli.reconcile(self.classification, extraction)
        self.assertEqual(len(result["reconciliation"]["borrower_name"]["unreadable_observations"]), 1)
        self.assertNotIn(3, result["loan_level_fields"]["borrower_name"]["source_pages"])
        bad = copy.deepcopy(self.extraction)
        bad["page_extractions"][0]["borrower_name"]["status"] = "not present"
        with self.assertRaises(loancli.CLIError):
            loancli.validate(bad, "extraction")


class ReconciliationTests(unittest.TestCase):
    def test_saved_results_and_alternatives_are_reproduced(self):
        c = accepted_classification()
        e = load("extraction.json")
        expected = load("reconciliation.json")
        actual = loancli.reconcile(c, e)
        self.assertEqual(actual["loan_level_fields"], expected["loan_level_fields"])
        for field in loancli.FIELDS:
            self.assertEqual(
                actual["reconciliation"][field]["candidates"],
                expected["reconciliation"][field]["candidates"],
            )

    def test_normalization_preserves_real_differences(self):
        self.assertEqual(loancli.normalize("borrower_name", "Doe, JANE"), "jane doe")
        self.assertNotEqual(
            loancli.normalize("borrower_name", "Al ya Doe"),
            loancli.normalize("borrower_name", "Alya Doe"),
        )
        self.assertEqual(loancli.normalize("loan_number", "0012"), "0012")
        self.assertEqual(
            loancli.normalize("property_address", "10 A St,\nUnit 2"), "10 a st unit 2"
        )

    def test_repeated_pages_do_not_outvote_distinct_documents(self):
        c = {"documents": [
            {"label": "a", "pages": [1, 2, 3]},
            {"label": "b", "pages": [4]},
            {"label": "c", "pages": [5]},
        ]}
        pages = []
        for page in range(1, 6):
            value = "A" if page <= 3 else "B"
            pages.append({"page_number": page, **{
                field: {"observations": [{"raw_value": value,
                    "evidence_text": value, "field_label": None, "notes": ""}]}
                for field in loancli.FIELDS
            }})
        result = loancli.reconcile(c, {"page_extractions": pages})
        self.assertEqual(result["loan_level_fields"]["loan_number"]["value"], "B")
        self.assertEqual(result["loan_level_fields"]["loan_number"]["source_pages"], [4, 5])

    def test_exact_tie_is_preserved_without_selecting_a_winner(self):
        c = {"documents": [{"label": "a", "pages": [1]}]}
        e = {"page_extractions": [{"page_number": 1, **{
            field: {"observations": [
                {"raw_value": value, "evidence_text": value,
                 "field_label": None, "notes": ""} for value in ["A", "B"]
            ]} for field in loancli.FIELDS
        }}]}
        result = loancli.reconcile(c, e)
        self.assertIn("loan_number", result["ties"])
        self.assertIsNone(result["reconciliation"]["loan_number"]["selected_normalized_value"])


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.project = self.work / "project"
        self.project.mkdir()
        for name in ["AREAL_LOAN.pdf", "prompts/classification.prompt.txt", "schemas/classification.schema.json", "prompts/extraction.prompt.txt", "schemas/extraction.schema.json", "candidate_classification_labels.txt", "result.json"]:
            (self.project / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, self.project / name)
        self.secret = "offline-secret-never-log"
        (self.project / ".env").write_text("GEMINI_API_KEY=" + self.secret + "\n")
        # Patch the genuine SDK boundary in every test subprocess. Never use the network.
        (self.work / "sitecustomize.py").write_text(r'''
import json, os, pathlib, subprocess, sys, time
from google import genai
from PIL import Image
class FakeClient:
    def __init__(self, *, api_key, http_options, vertexai):
        assert vertexai is False
        assert api_key == "offline-secret-never-log"
        assert http_options.retry_options.attempts == 1
        assert http_options.timeout > 0
        self.models = self
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def generate_content(self, *, model, contents, config):
        assert config.response_mime_type == "application/json"
        assert not config.tools
        assert config.response_json_schema
        prompt = contents[-1]
        kind = "extraction" if "page_extractions" in config.response_json_schema["properties"] else "classification"
        pages = list(config.response_json_schema["properties"]["page_extractions"]["items"]["properties"]["page_number"]["enum"]) if kind == "extraction" else list(range(1,13))
        assert model == ("gemini-3.1-flash-lite" if kind == "extraction" else "gemma-4-26b-a4b-it")
        assert config.thinking_config.thinking_level == ("HIGH" if kind == "extraction" else "MINIMAL")
        images = [p for p in contents if isinstance(p, Image.Image)]
        assert len(images) == (4*len(pages) if kind == "extraction" else 12)
        assert all(p.mode == "RGB" for p in images)
        assert [p for p in contents if isinstance(p,str) and p.startswith("Physical PDF page ") and "," not in p] == [f"Physical PDF page {i}" for i in pages]
        assert "page_images/" not in prompt and "classification-runs/" not in prompt
        if kind == "extraction":
            context = json.loads(prompt.split("CLASSIFICATION CONTEXT\n")[-1])
            assert all(set(p) == {"page_number", "label"} for p in context["page_labels"])
            assert set(context) == {"page_labels", "documents"}
            assert [p["page_number"] for p in context["page_labels"]] == pages
            assert len(context["documents"]) == 1
            assert context["documents"][0]["pages"] == pages
            assert config.response_json_schema["properties"]["page_extractions"]["minItems"] == len(pages)
            assert config.response_json_schema["properties"]["page_extractions"]["maxItems"] == len(pages)
        if os.environ.get("REAL_RENDER"):
            if 3 in pages:
                assert images[pages.index(3)].size == (2550, 3300)
                if kind == "extraction": assert all(p.size == (2550, 1320) for p in images[len(pages):])
        with open(os.environ["SDK_CALLS"], "a") as log:
            log.write(json.dumps({"stage": kind, "image_count": len(images), "pages": pages, "page_pixels": [p.tobytes()[:20].hex() for p in images[:len(pages)]]}) + "\n")
        if os.environ.get("FAKE_HANG"):
            child = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
            pathlib.Path(os.environ["FAKE_CHILD_PID"]).write_text(str(child.pid))
            time.sleep(60)
        if os.environ.get("FAKE_FAILURE") == kind or (kind == "extraction" and os.environ.get("FAKE_FAIL_PAGE") == str(pages[0])):
            raise RuntimeError("fixture failure offline-secret-never-log")
        root = pathlib.Path(os.environ["FIXTURE_ROOT"])
        data = json.loads((root / ("result.json" if kind == "classification" else "extraction.json")).read_text())
        if kind == "classification": data = {k:data[k] for k in ["page_labels", "documents"]}
        else:
            data["page_extractions"] = [p for p in data["page_extractions"] if p["page_number"] in pages]
            if os.environ.get("FAKE_WRONG_COVERAGE"): data["page_extractions"][0]["page_number"] = 12
        if os.environ.get("FAKE_INVALID"): data = {}
        class Response:
            text = json.dumps(data)
            usage_metadata = None
        if os.environ.get("FAKE_NONJSON"): Response.text = "not json"
        return Response()
genai.Client = FakeClient
''')
        guard = self.work / "agy"
        guard.write_text("#!/bin/sh\necho legacy-agy-forbidden >&2\nexit 1\n")
        guard.chmod(0o755)
        renderer = self.work / "pdftoppm"
        renderer.write_text(
            "#!" + sys.executable + "\n"
            "import json,os,pathlib,sys\nfrom PIL import Image\n"
            "a=sys.argv[1:]; assert a[a.index('-r')+1]=='300'\n"
            "with open(os.environ['RENDER_CALLS'],'a') as log: log.write(json.dumps(a)+'\\n')\n"
            "for i in range(1,13): Image.new('RGB',(100,100),(i,80,120)).save(a[-1]+f'-{i:02}'+('.png' if '-png' in a else '.ppm'))\n"
        )
        renderer.chmod(0o755)
        self.env = {**os.environ, "PATH": str(self.work) + os.pathsep + os.environ["PATH"],
                    "PYTHONPATH": str(self.work) + os.pathsep + str(ROOT),
                    "FIXTURE_ROOT": str(ROOT), "SDK_CALLS": str(self.work / "sdk-calls.jsonl"),
                    "RENDER_CALLS": str(self.work / "render-calls.jsonl")}

    def run_cli(self, *args, test_timeout=30):
        return subprocess.run(
            [sys.executable, "-c", "import sys; from pathlib import Path; import loancli; loancli.ROOT=Path(sys.argv[1]); sys.exit(loancli.main(sys.argv[2:]))", str(self.project), *args],
            env=self.env, text=True, capture_output=True, timeout=test_timeout)

    def test_full_makes_one_extraction_request_per_group_and_renders_once(self):
        output = self.work / "full"
        run = self.run_cli("full", "--output", str(output))
        self.assertEqual(run.returncode, 0, run.stderr)
        expected = {**accepted_classification(), "loan_level_fields": loancli.reconcile(accepted_classification(), load("extraction.json"))["loan_level_fields"]}
        self.assertEqual(json.loads((output / "result.json").read_text()), expected)
        self.assertEqual((self.project / "result.json").read_bytes(), (output / "result.json").read_bytes())
        calls = [json.loads(line) for line in (self.work / "sdk-calls.jsonl").read_text().splitlines()]
        self.assertEqual([c["stage"] for c in calls], ["classification"] + ["extraction"] * 5)
        self.assertEqual([c["pages"] for c in calls[1:]], [d["pages"] for d in accepted_classification()["documents"]])
        self.assertEqual(calls[0]["page_pixels"], [pixels for c in calls[1:] for pixels in c["page_pixels"]])
        self.assertEqual(len((self.work / "render-calls.jsonl").read_text().splitlines()), 1)
        self.assertFalse(any((self.project / ".temp").iterdir()))
        for folder, stage in [(output, "classification")] + [(output / "groups" / f"document-{i:02}", "extraction") for i in range(1,6)]:
            for suffix in [".stream.jsonl", ".stderr.log", ".render.log", "-run.json", ".schema.json", ".prompt.txt"]:
                self.assertTrue((folder / (stage + suffix)).is_file())
            self.assertIn(f"[{stage}] validated", run.stderr)
        self.assertIn("[reconciliation] complete (", run.stderr)
        self.assert_secret_absent(output, run)

    def test_generated_classification_sorts_but_keeps_raw_response(self):
        fixtures = self.work / "unordered-fixtures"
        fixtures.mkdir()
        raw = copy.deepcopy(accepted_classification())
        raw["page_labels"].reverse()
        for document in raw["documents"]:
            document["pages"].reverse()
        (fixtures / "result.json").write_text(json.dumps(raw))
        self.env["FIXTURE_ROOT"] = str(fixtures)
        output = self.work / "ordered"
        run = self.run_cli("classify", "--output", str(output))
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads((output / "classification.json").read_text()), accepted_classification())
        self.assertEqual(json.loads((output / "classification-run.json").read_text())["structured_output"], raw)
        events = [json.loads(line) for line in (output / "classification.stream.jsonl").read_text().splitlines()]
        self.assertEqual(json.loads(next(e["text"] for e in events if e["event"] == "response")), raw)

    def assert_secret_absent(self, output, run):
        self.assertNotIn(self.secret, run.stdout + run.stderr)
        for path in output.rglob("*"):
            if path.is_file(): self.assertNotIn(self.secret, path.read_text())

    def test_classify_and_extract_keep_root_unchanged(self):
        original = (self.project / "result.json").read_bytes()
        cdir, edir = self.work / "classify", self.work / "extract"
        run = self.run_cli("classify", "--output", str(cdir))
        self.assertEqual(run.returncode, 0, run.stderr)
        supplied = json.loads((cdir / "classification.json").read_text())
        supplied["documents"].reverse()
        (cdir / "classification.json").write_text(json.dumps(supplied))
        run = self.run_cli("extract", "--classification", str(cdir / "classification.json"), "--output", str(edir))
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads((edir / "extraction.json").read_text()), load("extraction.json"))
        self.assertFalse((edir / "result.json").exists())
        self.assertEqual((self.project / "result.json").read_bytes(), original)

    def test_failure_preserves_logs_and_redacts_key(self):
        original = (self.project / "result.json").read_bytes()
        self.env["FAKE_FAIL_PAGE"] = "3"
        output = self.work / "failed"
        run = self.run_cli("full", "--output", str(output))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Logs:", run.stderr)
        self.assertIn("fixture failure", (output / "groups/document-02/extraction.stderr.log").read_text())
        self.assertTrue((output / "classification.json").is_file())
        self.assertTrue((output / "groups/document-01/extraction.json").is_file())
        self.assertFalse((output / "groups/document-03").exists())
        self.assertFalse((output / "extraction.json").exists())
        self.assertFalse((output / "result.json").exists())
        self.assertFalse(any((self.project / ".temp").iterdir()))
        self.assertEqual((self.project / "result.json").read_bytes(), original)
        self.assert_secret_absent(output, run)

    def test_resume_reuses_completed_groups_and_preserves_failed_attempt_logs(self):
        original = (self.project / "result.json").read_bytes()
        output = self.work / "resume"
        self.env["FAKE_FAIL_PAGE"] = "3"
        failed = self.run_cli("full", "--output", str(output))
        self.assertNotEqual(failed.returncode, 0)
        completed = output / "groups/document-01/extraction-run.json"
        completed_bytes = completed.read_bytes()
        log = output / "groups/document-02/extraction.stream.jsonl"
        previous_log = log.read_bytes()
        previous_error = (log.parent / "extraction.stderr.log").read_bytes()
        self.assertEqual((self.project / "result.json").read_bytes(), original)
        del self.env["FAKE_FAIL_PAGE"]
        run = self.run_cli("full", "--output", str(output), "--resume")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("reusing validated", run.stderr)
        self.assertEqual(completed.read_bytes(), completed_bytes)
        self.assertTrue(log.read_bytes().startswith(previous_log))
        self.assertGreater(len(log.read_bytes()), len(previous_log))
        self.assertTrue((log.parent / "extraction.stderr.log").read_bytes().startswith(previous_error))
        calls = [json.loads(line) for line in (self.work / "sdk-calls.jsonl").read_text().splitlines()]
        self.assertEqual([c["pages"] for c in calls],
                         [list(range(1, 13)), [1, 2], [3], [3], [4, 5, 6], [7, 8, 9, 10], [11, 12]])
        self.assertEqual((self.project / "result.json").read_bytes(), (output / "result.json").read_bytes())
        self.assert_secret_absent(output, run)

    def test_resume_rejects_changed_inputs_and_untracked_runs_before_requests(self):
        output = self.work / "resume-mismatch"
        self.env["FAKE_FAIL_PAGE"] = "3"
        self.assertNotEqual(self.run_cli("full", "--output", str(output)).returncode, 0)
        before = (self.work / "sdk-calls.jsonl").read_bytes()
        for name in ["AREAL_LOAN.pdf", "prompts/extraction.prompt.txt", "schemas/extraction.schema.json"]:
            path = self.project / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            run = self.run_cli("full", "--output", str(output), "--resume")
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("unchanged", run.stderr)
            path.write_bytes(original)
        manifest = output / "run-manifest.json"
        original = manifest.read_bytes()
        changed = json.loads(original)
        changed["models"]["extraction"]["thinking_level"] = "minimal"
        manifest.write_text(json.dumps(changed))
        run = self.run_cli("full", "--output", str(output), "--resume")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("unchanged", run.stderr)
        manifest.write_bytes(original)
        legacy = self.work / "legacy"
        legacy.mkdir()
        (legacy / "classification.json").write_text(json.dumps(accepted_classification()))
        run = self.run_cli("full", "--output", str(legacy), "--resume")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("manifest", run.stderr)
        self.assertEqual((self.work / "sdk-calls.jsonl").read_bytes(), before)

    def test_resume_revalidates_completed_groups_and_supplied_classification(self):
        classification = self.work / "classification.json"
        classification.write_text(json.dumps(accepted_classification()))
        output = self.work / "resume-extract"
        self.env["FAKE_FAIL_PAGE"] = "3"
        self.assertNotEqual(self.run_cli("extract", "--classification", str(classification), "--output", str(output)).returncode, 0)
        before = (self.work / "sdk-calls.jsonl").read_bytes()
        original = classification.read_bytes()
        changed = json.loads(original)
        changed["page_labels"][0]["notes"] += "changed"
        classification.write_text(json.dumps(changed))
        run = self.run_cli("extract", "--classification", str(classification), "--output", str(output), "--resume")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("unchanged", run.stderr)
        classification.write_bytes(original)
        path = output / "groups/document-01/extraction.json"
        valid = json.loads(path.read_text())
        invalid = copy.deepcopy(valid)
        invalid["page_extractions"].pop()
        for data in [invalid, {**valid, "page_extractions": copy.deepcopy(valid["page_extractions"])}]:
            if len(data["page_extractions"]) == 2:
                data["page_extractions"][0]["loan_number"]["observations"][0]["raw_value"] = "edited"
            path.write_text(json.dumps(data))
            run = self.run_cli("extract", "--classification", str(classification), "--output", str(output), "--resume")
            self.assertNotEqual(run.returncode, 0)
        self.assertEqual((self.work / "sdk-calls.jsonl").read_bytes(), before)
        self.assertFalse((output / "extraction.json").exists())

    def test_invalid_and_nonjson_responses_fail_without_publication(self):
        for variable in ["FAKE_INVALID", "FAKE_NONJSON"]:
            self.env[variable] = "1"
            output = self.work / variable
            run = self.run_cli("full", "--output", str(output))
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("Logs:", run.stderr)
            self.assertFalse((output / "result.json").exists())
            self.assertFalse((output / "classification.json").exists())
            self.env.pop(variable)

    def test_group_coverage_mismatch_fails_without_merge(self):
        self.env["FAKE_WRONG_COVERAGE"] = "1"
        original = (self.project / "result.json").read_bytes()
        output = self.work / "wrong-group"
        run = self.run_cli("full", "--output", str(output))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Logs:", run.stderr)
        self.assertTrue((output / "groups/document-01/extraction-run.json").is_file())
        self.assertFalse((output / "extraction.json").exists())
        self.assertEqual((self.project / "result.json").read_bytes(), original)

    def test_missing_key_fails_without_sdk_call(self):
        (self.project / ".env").unlink()
        self.env["GEMINI_API_KEY"] = "ambient-key-must-not-be-used"
        run = self.run_cli("classify", "--output", str(self.work / "no-key"))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("GEMINI_API_KEY", run.stderr)
        self.assertFalse((self.work / "sdk-calls.jsonl").exists())

    def test_timeout_kills_group_and_preserves_root(self):
        original = (self.project / "result.json").read_bytes()
        child_pid = self.work / "child.pid"
        self.env.update(FAKE_HANG="1", FAKE_CHILD_PID=str(child_pid))
        output = self.work / "timeout"
        started = time.monotonic()
        run = self.run_cli("full", "--output", str(output), "--timeout", "5")
        self.assertLess(time.monotonic() - started, 14)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("timeout", run.stderr.lower())
        self.assertTrue((output / "classification.stream.jsonl").is_file())
        self.assertFalse(any((self.project / ".temp").iterdir()))
        self.assertEqual((self.project / "result.json").read_bytes(), original)
        pid = int(child_pid.read_text())
        for _ in range(20):
            check = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True)
            if check.returncode or check.stdout.strip().startswith("Z"): break
            time.sleep(0.05)
        else: self.fail("SDK worker child survived cleanup")

    def test_real_poppler_prepares_300_dpi_images_and_fixed_views(self):
        (self.work / "pdftoppm").unlink()
        self.env["REAL_RENDER"] = "1"
        classification = self.work / "classification.json"
        classification.write_text(json.dumps(accepted_classification()))
        run = self.run_cli("extract", "--classification", str(classification),
                           "--output", str(self.work / "real-render"), test_timeout=90)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_render_failure_cleans_temp_and_preserves_log(self):
        renderer = self.work / "pdftoppm"
        renderer.write_text("#!" + sys.executable + "\nimport sys\nprint('fixture render failure', file=sys.stderr)\nsys.exit(2)\n")
        output = self.work / "render-failed"
        run = self.run_cli("classify", "--output", str(output))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("fixture render failure", (output / "classification.render.log").read_text())
        self.assertFalse(any((self.project / ".temp").iterdir()))

    def test_full_preserves_concrete_tie_without_publication(self):
        fixtures = self.work / "fixtures"
        fixtures.mkdir()
        (fixtures / "result.json").write_text(json.dumps(load("result.json")))
        extraction = load("extraction.json")
        for page in extraction["page_extractions"]:
            for observation in page["loan_number"]["observations"]:
                observation["raw_value"] = "A" if page["page_number"] in [1, 2, 4, 5, 6] else "B"
        (fixtures / "extraction.json").write_text(json.dumps(extraction))
        self.env["FIXTURE_ROOT"] = str(fixtures)
        original = (self.project / "result.json").read_bytes()
        output = self.work / "tie"
        run = self.run_cli("full", "--output", str(output))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Concrete document-vote tie", run.stderr)
        self.assertEqual(set(json.loads((output / "reconciliation.json").read_text())["ties"]["loan_number"]), {"a", "b"})
        self.assertFalse((output / "result.json").exists())
        self.assertEqual((self.project / "result.json").read_bytes(), original)

    def test_required_arguments_timeout_and_existing_output(self):
        run = self.run_cli("extract", "--output", str(self.work / "missing"))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("--classification", run.stderr)
        for value in ["0", "-1", "nan", "inf"]:
            run = self.run_cli("classify", "--output", str(self.work / "invalid"), "--timeout", value)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("finite positive", run.stderr)
        output = self.work / "existing"
        output.mkdir()
        sentinel = output / "result.json"
        sentinel.write_text("keep me")
        run = self.run_cli("full", "--output", str(output))
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(sentinel.read_text(), "keep me")


class SDKSerializationTests(unittest.TestCase):
    def test_503_retries_are_bounded_visible_and_preserve_request_inputs(self):
        real_client = loancli.genai.Client
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("GEMINI_API_KEY=offline-key")
            image = root / "page.png"
            Image.new("RGB", (10, 10)).save(image)
            path = root / "request.json"
            loancli.write_json(path, {"model": loancli.EXTRACTION_MODEL, "thinking_level": "high",
                "env_path": str(root / ".env"), "schema": loancli.extraction_schema([3]),
                "timeout": 90, "prompt": "test", "images": [{"caption": "Physical PDF page 3", "path": str(image)}]})
            for codes, status, delays, timeout in [([503, 503, 200], 0, [5, 10], 90),
                    ([503] * 4, 1, [5, 10, 20], 90), ([429], 1, [], 90),
                    ([500], 1, [], 90), ([503], 1, [], 1)]:
                request_data = loancli.read_json(path)
                request_data["timeout"] = timeout
                loancli.write_json(path, request_data)
                bodies = []
                def respond(request):
                    bodies.append(request.content)
                    code = codes[len(bodies) - 1]
                    if code != 200:
                        return httpx.Response(code, json={"error": {"code": code,
                            "message": "fixture unavailable offline-key", "status": "UNAVAILABLE"}})
                    return httpx.Response(200, json={"candidates": [{"content": {
                        "role": "model", "parts": [{"text": "{}"}]}, "finishReason": "STOP"}]})
                transport = httpx.Client(transport=httpx.MockTransport(respond))
                client = real_client(api_key="offline-key", http_options=loancli.types.HttpOptions(
                    httpx_client=transport, retry_options=loancli.types.HttpRetryOptions(attempts=1)))
                output, errors = io.StringIO(), io.StringIO()
                with patch.object(loancli.genai, "Client", return_value=client), patch.object(loancli.time, "sleep") as sleep, redirect_stdout(output), redirect_stderr(errors):
                    self.assertEqual(loancli.sdk_worker(path), status, errors.getvalue())
                self.assertEqual(len(bodies), len(codes))
                self.assertTrue(all(body == bodies[0] for body in bodies))
                self.assertEqual([c.args[0] for c in sleep.call_args_list], delays)
                events = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual([e["attempt"] for e in events if e["event"] == "request"], list(range(1, len(codes)+1)))
                self.assertEqual([e["delay_seconds"] for e in events if e["event"] == "retry"], delays)
                self.assertEqual(len([e for e in events if e["event"] == "attempt_error"]), len(codes) - codes.count(200))
                self.assertNotIn("offline-key", output.getvalue() + errors.getvalue())
                transport.close()

    def test_real_sdk_serializes_pillow_and_existing_schemas_offline(self):
        real_client = loancli.genai.Client
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("GEMINI_API_KEY=offline-key")
            image = root / "page.png"
            Image.new("RGB", (10, 10)).save(image)
            for stage in ["classification", "extraction"]:
                schema = loancli.extraction_schema([3]) if stage == "extraction" else load(f"schemas/{stage}.schema.json")
                level = "high" if stage == "extraction" else "minimal"
                model = loancli.EXTRACTION_MODEL if stage == "extraction" else loancli.MODEL
                requests = []

                def respond(request):
                    body = json.loads(request.content)
                    requests.append(body)
                    self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")
                    self.assertEqual(body["generationConfig"]["responseJsonSchema"], schema)
                    self.assertIn(level.upper(), body["generationConfig"]["thinkingConfig"].values(), body["generationConfig"]["thinkingConfig"])
                    self.assertNotIn("tools", body)
                    self.assertIn("inlineData", body["contents"][0]["parts"][1])
                    return httpx.Response(200, json={"candidates": [{"content": {
                        "role": "model", "parts": [{"text": "{}"}]}, "finishReason": "STOP"}]})

                transport = httpx.Client(transport=httpx.MockTransport(respond))
                client = real_client(api_key="offline-key", http_options=loancli.types.HttpOptions(
                    httpx_client=transport, retry_options=loancli.types.HttpRetryOptions(attempts=1)))
                request_path = root / "request.json"
                loancli.write_json(request_path, {"model": model, "thinking_level": level, "env_path": str(root / ".env"), "schema": schema,
                    "timeout": 5, "prompt": "test", "images": [{"caption": "Physical PDF page 1", "path": str(image)}]})
                output, errors = io.StringIO(), io.StringIO()
                with patch.object(loancli.genai, "Client", return_value=client), redirect_stdout(output), redirect_stderr(errors):
                    self.assertEqual(loancli.sdk_worker(request_path), 0, errors.getvalue())
                self.assertEqual(len(requests), 1)
                self.assertNotIn("offline-key", output.getvalue() + errors.getvalue())
                transport.close()


class PublicationTests(unittest.TestCase):
    def test_full_overwrites_root_only_after_success(self):
        classification = accepted_classification()
        extraction = load("extraction.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "result.json"
            target.write_text("previous result")
            for name in ["AREAL_LOAN.pdf", "prompts/classification.prompt.txt", "prompts/extraction.prompt.txt",
                         "schemas/classification.schema.json", "schemas/extraction.schema.json", "candidate_classification_labels.txt"]:
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, root / name)
            with patch.object(loancli, "ROOT", root), patch.object(loancli, "run_stage", return_value=classification), patch.object(loancli, "run_extraction", return_value=extraction):
                self.assertEqual(loancli.main(["full", "--output", str(root / "output")]), 0)
            self.assertEqual(target.read_bytes(), (root / "output/result.json").read_bytes())
            self.assertNotEqual(target.read_text(), "previous result")

    def test_replace_failure_leaves_previous_root_and_cleans_staging_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "new.json", root / "result.json"
            source.write_text('{"new":true}\n')
            target.write_text("previous result")

            def failed_replace(staged, destination):
                self.assertEqual(Path(staged).parent, target.parent)
                self.assertEqual(Path(destination), target)
                self.assertEqual(Path(staged).read_bytes(), source.read_bytes())
                self.assertEqual(target.read_text(), "previous result")
                raise OSError("fixture replace failure")

            with patch.object(loancli.os, "replace", side_effect=failed_replace):
                with self.assertRaises(OSError):
                    loancli.publish_result(source, target)
            self.assertEqual(target.read_text(), "previous result")
            self.assertEqual(set(root.iterdir()), {source, target})

if __name__ == "__main__":
    unittest.main()
