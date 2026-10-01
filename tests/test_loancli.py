import copy
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import loancli


ROOT = Path(__file__).resolve().parents[1]


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

    def test_missing_evidence_and_inconsistent_status_are_rejected(self):
        bad = copy.deepcopy(self.extraction)
        bad["page_extractions"][0]["borrower_name"]["observations"][0]["evidence_text"] = None
        with self.assertRaises(loancli.CLIError):
            loancli.validate(bad, "extraction")
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


class StreamTests(unittest.TestCase):
    def test_working_directory_must_match_the_streamed_init(self):
        workspace = ROOT / ".temp" / "test-run" / "classification"
        parser = loancli.StreamParser(workspace)
        parser.feed(json.dumps({"event": "init", "init": {"cwd": str(workspace)}}))
        parser.feed(json.dumps({"event": "result", "result": {
            "status": "SUCCESS", "structured_output": {}
        }}))
        self.assertEqual(parser.finish(0)["status"], "SUCCESS")
        for cwd in [str(ROOT), None]:
            with self.assertRaises(loancli.CLIError):
                loancli.StreamParser(workspace).feed(json.dumps({
                    "event": "init", "init": {"cwd": cwd}
                }))
        parser = loancli.StreamParser(workspace)
        parser.feed(json.dumps({"event": "result", "result": {
            "status": "SUCCESS", "structured_output": {}
        }}))
        with self.assertRaises(loancli.CLIError):
            parser.finish(0)

    def test_terminal_result_is_used_instead_of_response_deltas(self):
        parser = loancli.StreamParser()
        parser.feed(json.dumps({"event": "step_update", "step_update": {"text_delta": "partial"}}))
        result = {"status": "SUCCESS", "structured_output": {"answer": 1}}
        parser.feed(json.dumps({"event": "result", "result": result}))
        self.assertEqual(parser.finish(0), result)

    def test_truncated_invalid_failed_and_duplicate_streams_are_rejected(self):
        with self.assertRaises(loancli.CLIError):
            loancli.StreamParser().finish(0)
        with self.assertRaises(loancli.CLIError):
            loancli.StreamParser().feed("not json")
        for status, code in [("ERROR", 0), ("SUCCESS", 2)]:
            parser = loancli.StreamParser()
            parser.feed(json.dumps({"event": "result", "result": {
                "status": status, "structured_output": {}, "error": "failed"
            }}))
            with self.assertRaises(loancli.CLIError):
                parser.finish(code)
        parser = loancli.StreamParser()
        line = json.dumps({"event": "result", "result": {"status": "SUCCESS", "structured_output": {}}})
        parser.feed(line)
        with self.assertRaises(loancli.CLIError):
            parser.feed(line)


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.project = self.work / "project"
        self.project.mkdir()
        for name in ["AREAL_LOAN.pdf", "classification.prompt.txt", "classification.schema.json", "extraction.prompt.txt", "extraction.schema.json", "candidate_classification_labels.txt", "result.json"]:
            shutil.copyfile(ROOT / name, self.project / name)
        fake = self.work / "agy"
        fake.write_text(
            "#!" + sys.executable + "\n"
            "import json, os, pathlib, sys\n"
            "args=sys.argv[1:]\n"
            "assert args[args.index('--model')+1]=='gemini-3.8-flash-low'\n"
            "assert args[args.index('--output-format')+1]=='stream-json'\n"
            "assert all(f in args for f in ['--new-project','--sandbox','--disable-slash-commands'])\n"
            "assert not any(f in args for f in ['--continue','-c','--conversation','--add-dir','--project'])\n"
            "kind=pathlib.Path(args[args.index('--json-schema')+1]).name.split('.')[0]\n"
            "cwd=pathlib.Path.cwd()\n"
            "assert cwd.name==kind and cwd.parent.parent.name=='.temp'\n"
            "files={str(p.relative_to(cwd)) for p in cwd.rglob('*') if p.is_file()}\n"
            "assert not any(p.is_symlink() for p in cwd.rglob('*'))\n"
            "images={f'page_images/page-{i:02}.png' for i in range(1,13)}\n"
            "required=images|{f'{kind}.prompt.txt',f'{kind}.schema.json'}\n"
            "required.add('candidate_classification_labels.txt' if kind=='classification' else 'classification-runs/classification-flash-low-ordering-run.json')\n"
            "assert files==required,files\n"
            "assert all((cwd/p).read_bytes().startswith(b'\\x89PNG\\r\\n\\x1a\\n') for p in images)\n"
            "assert 'These input instructions replace' in args[args.index('-p')+1]\n"
            "assert not pathlib.Path('extraction.json').exists()\n"
            "assert not pathlib.Path('reconciliation.json').exists()\n"
            "if kind=='extraction':\n"
            " c=json.loads(pathlib.Path('classification-runs/classification-flash-low-ordering-run.json').read_text())['structured_output']\n"
            " assert all(set(p)=={'label','page_number'} for p in c['page_labels'])\n"
            "stream_cwd=os.environ.get('FAKE_CWD',os.getcwd())\n"
            "print(json.dumps({'event':'init','init':{'cwd':stream_cwd}}),flush=True)\n"
            "print(json.dumps({'event':'step_update','step_update':{'step_type':'tool','tool_name':'view_file','state':'ACTIVE','step_index':1}}),flush=True)\n"
            "if os.environ.get('FAKE_FAILURE'):\n"
            " print('fixture authentication failure',file=sys.stderr)\n"
            " sys.exit(2)\n"
            "root=pathlib.Path(os.environ['FIXTURE_ROOT'])\n"
            "data=json.loads((root/('result.json' if kind=='classification' else 'extraction.json')).read_text())\n"
            "if kind=='classification': data={k:data[k] for k in ['page_labels','documents']}\n"
            "result={'conversation_id':'fixture','status':'SUCCESS','response':json.dumps(data),'duration_seconds':0,'num_turns':1,'structured_output':data}\n"
            "print(json.dumps({'event':'result','result':result}),flush=True)\n"
        )
        fake.chmod(0o755)
        self.env = {**os.environ, "PATH": str(self.work) + os.pathsep + os.environ["PATH"], "FIXTURE_ROOT": str(ROOT)}

    def run_cli(self, *args, failure=False):
        env = {**self.env, **({"FAKE_FAILURE": "1"} if failure else {})}
        return subprocess.run(
            [sys.executable, "-c", "import sys; from pathlib import Path; import loancli; loancli.ROOT=Path(sys.argv[1]); sys.exit(loancli.main(sys.argv[2:]))", str(self.project), *args],
            env=env, text=True, capture_output=True, timeout=15,
        )

    def test_full_produces_validated_artifacts_and_stream_logs(self):
        output = self.work / "full"
        run = self.run_cli("full", "--output", str(output))
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads((output / "result.json").read_text()), load("result.json"))
        self.assertEqual((self.project / "result.json").read_bytes(), (output / "result.json").read_bytes())
        for stage in ["classification", "extraction"]:
            self.assertTrue((output / f"{stage}.stream.jsonl").is_file())
            self.assertTrue((output / f"{stage}.stderr.log").is_file())
            self.assertTrue((output / f"{stage}-run.json").is_file())
            self.assert_workspace_cleaned(output, stage)
        directories = [json.loads((output / f"{stage}.stream.jsonl").read_text().splitlines()[0])["init"]["cwd"] for stage in ["classification", "extraction"]]
        self.assertNotEqual(Path(directories[0]).parent, Path(directories[1]).parent)

    def assert_workspace_cleaned(self, output, stage):
        init = json.loads((output / f"{stage}.stream.jsonl").read_text().splitlines()[0])
        cwd = Path(init["init"]["cwd"])
        self.assertEqual(cwd.name, stage)
        self.assertEqual(cwd.parent.parent.resolve(), (self.project / ".temp").resolve())
        self.assertFalse(cwd.parent.exists(), "run directory must be removed")

    def test_classify_and_extract_use_explicit_classification(self):
        original = (self.project / "result.json").read_bytes()
        cdir, edir = self.work / "classify", self.work / "extract"
        run = self.run_cli("classify", "--output", str(cdir))
        self.assertEqual(run.returncode, 0, run.stderr)
        run = self.run_cli("extract", "--classification", str(cdir / "classification.json"), "--output", str(edir))
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads((edir / "extraction.json").read_text()), load("extraction.json"))
        self.assertFalse((edir / "result.json").exists())
        self.assertEqual((self.project / "result.json").read_bytes(), original)

    def test_failure_keeps_logs_and_does_not_publish_result(self):
        original = (self.project / "result.json").read_bytes()
        output = self.work / "failed"
        run = self.run_cli("full", "--output", str(output), failure=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("classification", run.stderr)
        self.assertIn("logs", run.stderr.lower())
        self.assertIn("fixture authentication failure", (output / "classification.stderr.log").read_text())
        self.assertTrue((output / "classification.stream.jsonl").is_file())
        self.assertFalse((output / "result.json").exists())
        self.assert_workspace_cleaned(output, "classification")
        self.assertEqual((self.project / "result.json").read_bytes(), original)

    def test_wrong_streamed_cwd_fails_and_cleans_temporary_files(self):
        self.env["FAKE_CWD"] = str(ROOT)
        output = self.work / "wrong-cwd"
        before = set((self.project / ".temp").glob("*"))
        run = self.run_cli("classify", "--output", str(output))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("working directory", run.stderr)
        self.assertEqual(set((self.project / ".temp").glob("*")), before)
        self.assertTrue((output / "classification.stream.jsonl").is_file())
        self.assertFalse((output / "classification.json").exists())

    def test_render_failure_cleans_the_run_and_preserves_render_log(self):
        renderer = self.work / "pdftoppm"
        renderer.write_text("#!" + sys.executable + "\nimport sys\nprint('fixture render failure', file=sys.stderr)\nsys.exit(2)\n")
        renderer.chmod(0o755)
        output = self.work / "render-failed"
        before = set((self.project / ".temp").glob("*"))
        run = self.run_cli("classify", "--output", str(output))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("page rendering failed", run.stderr)
        self.assertIn("fixture render failure", (output / "classification.render.log").read_text())
        self.assertEqual(set((self.project / ".temp").glob("*")), before)
        self.assertFalse((output / "classification.json").exists())

    def test_full_preserves_a_concrete_tie_without_publishing_result(self):
        original = (self.project / "result.json").read_bytes()
        fixtures = self.work / "fixtures"
        fixtures.mkdir()
        (fixtures / "result.json").write_text(json.dumps(load("result.json")))
        extraction = load("extraction.json")
        for page in extraction["page_extractions"]:
            for observation in page["loan_number"]["observations"]:
                observation["raw_value"] = "A" if page["page_number"] in [1, 2, 4, 5, 6] else "B"
        (fixtures / "extraction.json").write_text(json.dumps(extraction))
        self.env["FIXTURE_ROOT"] = str(fixtures)
        output = self.work / "tie"
        run = self.run_cli("full", "--output", str(output))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Concrete document-vote tie", run.stderr)
        reconciliation = json.loads((output / "reconciliation.json").read_text())
        self.assertEqual(set(reconciliation["ties"]["loan_number"]), {"a", "b"})
        self.assertFalse((output / "result.json").exists())
        self.assertEqual((self.project / "result.json").read_bytes(), original)


    def test_extract_requires_classification_and_output_is_not_overwritten(self):
        run = self.run_cli("extract", "--output", str(self.work / "missing"))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("--classification", run.stderr)
        output = self.work / "existing"
        output.mkdir()
        sentinel = output / "result.json"
        sentinel.write_text("keep me")
        run = self.run_cli("full", "--output", str(output))
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(sentinel.read_text(), "keep me")


class PublicationTests(unittest.TestCase):
    def test_full_overwrites_root_only_after_success(self):
        classification = accepted_classification()
        extraction = load("extraction.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "result.json"
            target.write_text("previous result")
            with patch.object(loancli, "ROOT", root), patch.object(loancli, "run_stage", side_effect=[classification, extraction]):
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
