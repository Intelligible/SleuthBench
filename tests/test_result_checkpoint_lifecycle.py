from __future__ import annotations

import io
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import eval_pipeline  # noqa: E402
import grade_pipeline  # noqa: E402
from io_utils import (  # noqa: E402
    _atomic_temp_candidate,
    _truncate_to_filename_units,
    partial_output_path,
    save_json_atomic,
)
from shared.path_utils import windows_utf16_units  # noqa: E402


class _FatalEval(BaseException):
    pass


def _write_eval_instance(
    root: Path,
    qa_count: int = 2,
    *,
    instance_name: str = "fake",
) -> Path:
    instance_dir = (
        root / "instances" / "dataset_100" / "seed_42" / instance_name
    )
    instance_dir.mkdir(parents=True)
    (instance_dir / "table.csv").write_text(
        "feature,target\n1,2\n",
        encoding="utf-8",
    )
    manifest_path = instance_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_name": "dataset_100",
                "seed": 42,
                "target": "target",
                "phenomenon": {
                    "injector_type": instance_name,
                    "effects": {},
                },
                "validation": {"passed": True},
                "qa_pairs": [
                    {
                        "template_id": f"{instance_name}_v{i}",
                        "question": f"Question {i}?",
                        "answer": i,
                        "answer_format": "value",
                    }
                    for i in range(qa_count)
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


class EvalCheckpointTests(unittest.TestCase):
    def test_injection_delay_sleeps_only_between_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_manifest = _write_eval_instance(
                root,
                qa_count=2,
                instance_name="first",
            )
            second_manifest = _write_eval_instance(
                root,
                qa_count=1,
                instance_name="second",
            )
            output_path = root / "results" / "eval_results.json"
            events: list[object] = []

            def query(*_args):
                events.append("query")
                return "answer", None, {}

            sleep = Mock(
                side_effect=lambda seconds: events.append(("sleep", seconds))
            )
            fake_client = SimpleNamespace(close=Mock())
            fake_openai = SimpleNamespace(
                OpenAI=Mock(return_value=fake_client)
            )

            with (
                patch.dict(
                    os.environ,
                    {"OPENAI_API_KEY": "test-key"},
                    clear=False,
                ),
                patch.dict(sys.modules, {"openai": fake_openai}),
                patch("dotenv.load_dotenv", return_value=False),
                patch.dict(eval_pipeline.QUERY_FN, {"openai": query}),
                patch.object(eval_pipeline.time, "sleep", sleep),
                redirect_stdout(io.StringIO()),
            ):
                results = eval_pipeline.run_eval(
                    ["gpt-test"],
                    root / "instances",
                    output_path,
                    manifest_paths=[first_manifest, second_manifest],
                    injection_delay_seconds=10,
                )

            self.assertEqual(len(results), 3)
            self.assertEqual(
                events,
                ["query", "query", ("sleep", 10.0), "query"],
            )
            sleep.assert_called_once_with(10.0)

    def test_duplicate_models_fail_before_output_or_model_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "eval_results.json"

            with self.assertRaisesRegex(
                ValueError,
                r"duplicate model\(s\): gpt-test",
            ):
                eval_pipeline.run_eval(
                    ["gpt-test", "gpt-test"],
                    root / "instances",
                    output_path,
                )

            self.assertFalse(output_path.exists())
            self.assertFalse(partial_output_path(output_path).exists())

    @unittest.skipUnless(os.name == "nt", "Windows read-only attributes only")
    def test_read_only_formal_target_fails_before_model_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "eval_results.json"
            output_path.write_text('[{"old": true}]', encoding="utf-8")
            output_path.chmod(stat.S_IREAD)
            query = Mock()
            try:
                with (
                    patch.dict(
                        eval_pipeline.QUERY_FN,
                        {"openai": query},
                    ),
                    self.assertRaisesRegex(
                        PermissionError,
                        "read-only",
                    ),
                ):
                    eval_pipeline.run_eval(
                        ["gpt-test"],
                        root / "instances",
                        output_path,
                    )
            finally:
                output_path.chmod(stat.S_IWRITE)

            query.assert_not_called()
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                [{"old": True}],
            )
            self.assertFalse(partial_output_path(output_path).exists())

    def test_non_file_formal_target_fails_before_model_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "eval_results.json"
            output_path.mkdir()
            query = Mock()

            with (
                patch.dict(
                    eval_pipeline.QUERY_FN,
                    {"openai": query},
                ),
                self.assertRaisesRegex(
                    IsADirectoryError,
                    "not a replaceable file",
                ),
            ):
                eval_pipeline.run_eval(
                    ["gpt-test"],
                    root / "instances",
                    output_path,
                )

            query.assert_not_called()
            self.assertTrue(output_path.is_dir())
            self.assertFalse(partial_output_path(output_path).exists())

    def test_fatal_rerun_preserves_formal_output_and_marks_partial_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_eval_instance(root)
            output_path = root / "results" / "eval_results.json"
            output_path.parent.mkdir()
            old_results = [{"old": True}]
            output_path.write_text(
                json.dumps(old_results),
                encoding="utf-8",
            )

            query = Mock(
                side_effect=[
                    ("first answer", None, {"latency_s": 0.1}),
                    _FatalEval("interrupted"),
                ]
            )
            fake_client = SimpleNamespace(close=Mock())
            fake_openai = SimpleNamespace(
                OpenAI=Mock(return_value=fake_client)
            )

            with (
                patch.dict(
                    os.environ,
                    {"OPENAI_API_KEY": "test-key"},
                    clear=False,
                ),
                patch.dict(sys.modules, {"openai": fake_openai}),
                patch("dotenv.load_dotenv", return_value=False),
                patch.dict(
                    eval_pipeline.QUERY_FN,
                    {"openai": query},
                ),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(_FatalEval, "interrupted"):
                    eval_pipeline.run_eval(
                        ["gpt-test"],
                        root / "instances",
                        output_path,
                        manifest_paths=[manifest_path],
                    )

            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                old_results,
            )
            partial_path = partial_output_path(output_path)
            self.assertEqual(partial_path.name, "eval_results.json.partial")
            self.assertNotIn(
                partial_path,
                list(output_path.parent.glob("*.json")),
            )
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            self.assertEqual(partial["status"], "partial")
            self.assertEqual(partial["kind"], "eval")
            self.assertEqual(partial["completed"], 1)
            self.assertEqual(len(partial["results"]), 1)

    def test_success_atomically_publishes_and_removes_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_eval_instance(root, qa_count=1)
            output_path = root / "results" / "eval_results.json"
            fake_client = SimpleNamespace(close=Mock())
            fake_openai = SimpleNamespace(
                OpenAI=Mock(return_value=fake_client)
            )

            with (
                patch.dict(
                    os.environ,
                    {"OPENAI_API_KEY": "test-key"},
                    clear=False,
                ),
                patch.dict(sys.modules, {"openai": fake_openai}),
                patch("dotenv.load_dotenv", return_value=False),
                patch.dict(
                    eval_pipeline.QUERY_FN,
                    {"openai": Mock(return_value=("answer", None, {}))},
                ),
                redirect_stdout(io.StringIO()),
            ):
                results = eval_pipeline.run_eval(
                    ["gpt-test"],
                    root / "instances",
                    output_path,
                    manifest_paths=[manifest_path],
                )

            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                results,
            )
            self.assertFalse(
                partial_output_path(output_path).exists()
            )


class GradeCheckpointTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows read-only attributes only")
    def test_read_only_formal_target_fails_before_judge_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "eval.json"
            input_path.write_text("[]", encoding="utf-8")
            output_path = root / "graded.json"
            output_path.write_text('[{"old": true}]', encoding="utf-8")
            output_path.chmod(stat.S_IREAD)
            fake_openai = SimpleNamespace(OpenAI=Mock())
            try:
                with (
                    patch.dict(sys.modules, {"openai": fake_openai}),
                    patch.object(
                        grade_pipeline,
                        "grade_entry",
                    ) as grade_entry,
                    self.assertRaisesRegex(
                        PermissionError,
                        "read-only",
                    ),
                ):
                    grade_pipeline.run(input_path, output_path)
            finally:
                output_path.chmod(stat.S_IWRITE)

            fake_openai.OpenAI.assert_not_called()
            grade_entry.assert_not_called()
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                [{"old": True}],
            )
            self.assertFalse(partial_output_path(output_path).exists())

    def test_non_file_formal_target_fails_before_judge_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "eval.json"
            input_path.write_text("[]", encoding="utf-8")
            output_path = root / "graded.json"
            output_path.mkdir()
            fake_openai = SimpleNamespace(OpenAI=Mock())

            with (
                patch.dict(sys.modules, {"openai": fake_openai}),
                patch.object(grade_pipeline, "grade_entry") as grade_entry,
                self.assertRaisesRegex(
                    IsADirectoryError,
                    "not a replaceable file",
                ),
            ):
                grade_pipeline.run(input_path, output_path)

            fake_openai.OpenAI.assert_not_called()
            grade_entry.assert_not_called()
            self.assertTrue(output_path.is_dir())
            self.assertFalse(partial_output_path(output_path).exists())

    def test_failed_rerun_preserves_formal_output_and_marks_partial_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = [
                {"model": "gpt-test", "dataset": "d", "injector": "i"},
                {"model": "gpt-test", "dataset": "d", "injector": "i"},
            ]
            input_path = root / "eval.json"
            input_path.write_text(json.dumps(entries), encoding="utf-8")
            output_path = root / "graded.json"
            old_results = [{"old": True}]
            output_path.write_text(
                json.dumps(old_results),
                encoding="utf-8",
            )
            fake_openai = SimpleNamespace(
                OpenAI=Mock(return_value=object())
            )

            with (
                patch.dict(
                    os.environ,
                    {"OPENAI_API_KEY": "test-key"},
                    clear=False,
                ),
                patch.dict(sys.modules, {"openai": fake_openai}),
                patch.object(
                    grade_pipeline,
                    "load_dotenv",
                    return_value=False,
                ),
                patch.object(
                    grade_pipeline,
                    "grade_entry",
                    side_effect=[
                        ("CORRECT", "first complete"),
                        RuntimeError("judge failed"),
                    ],
                ),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "judge failed"):
                    grade_pipeline.run(input_path, output_path)

            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                old_results,
            )
            partial_path = partial_output_path(output_path)
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            self.assertEqual(partial["status"], "partial")
            self.assertEqual(partial["kind"], "grade")
            self.assertEqual(partial["completed"], 1)
            self.assertEqual(partial["graded"][0]["grade"], "CORRECT")

    def test_empty_success_replaces_formal_output_and_removes_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "eval.json"
            input_path.write_text("[]", encoding="utf-8")
            output_path = root / "graded.json"
            output_path.write_text('[{"old": true}]', encoding="utf-8")
            fake_openai = SimpleNamespace(
                OpenAI=Mock(return_value=object())
            )

            with (
                patch.dict(
                    os.environ,
                    {"OPENAI_API_KEY": "test-key"},
                    clear=False,
                ),
                patch.dict(sys.modules, {"openai": fake_openai}),
                patch.object(
                    grade_pipeline,
                    "load_dotenv",
                    return_value=False,
                ),
                patch.object(grade_pipeline, "grade_entry") as grade_entry,
                redirect_stdout(io.StringIO()),
            ):
                grade_pipeline.run(input_path, output_path)

            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                [],
            )
            self.assertFalse(
                partial_output_path(output_path).exists()
            )
            grade_entry.assert_not_called()


class AtomicJsonPathLengthTests(unittest.TestCase):
    def test_posix_component_budget_counts_encoded_bytes(self):
        text = "\N{LATIN SMALL LETTER E WITH ACUTE}" * 120
        bounded = _truncate_to_filename_units(
            text,
            max_units=201,
            use_windows_units=False,
        )

        self.assertLessEqual(len(os.fsencode(bounded)), 201)
        self.assertGreater(len(os.fsencode(f"{bounded}é")), 201)

    def test_temp_candidate_uses_utf16_units_and_preserves_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / ("\N{GRINNING FACE}" * 20) / "nested"
            target = parent / ".artifact_generation_complete.json"
            token = "0123456789abcdef"
            absolute_parent_units = windows_utf16_units(
                Path(os.path.abspath(parent))
            )
            max_path_units = absolute_parent_units + 1 + 40

            candidate = _atomic_temp_candidate(
                target,
                token,
                max_path_units=max_path_units,
            )

            self.assertEqual(candidate.parent, target.parent)
            self.assertTrue(candidate.name.endswith(f".{token}.tmp"))
            self.assertLessEqual(
                windows_utf16_units(Path(os.path.abspath(candidate))),
                max_path_units,
            )
            self.assertNotEqual(
                candidate.name,
                f".{target.name}.{token}.tmp",
            )

    @unittest.skipUnless(os.name == "nt", "Windows MAX_PATH regression")
    def test_atomic_write_succeeds_when_old_sidecar_would_exceed_max_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            while windows_utf16_units(Path(os.path.abspath(parent))) < 210:
                parent /= "abcdefghij"
            parent.mkdir(parents=True)
            target = parent / ".artifact_generation_complete.json"
            old_sidecar = parent / (
                f".{target.name}.0123456789abcdef.tmp"
            )

            self.assertLessEqual(
                windows_utf16_units(Path(os.path.abspath(target))),
                259,
            )
            self.assertGreater(
                windows_utf16_units(Path(os.path.abspath(old_sidecar))),
                259,
            )

            save_json_atomic({"version": 1}, target)

            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"version": 1},
            )

    @unittest.skipUnless(os.name == "nt", "Windows MAX_PATH regression")
    def test_atomic_write_handles_non_bmp_parent_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / ("\N{GRINNING FACE}" * 40)
            while windows_utf16_units(Path(os.path.abspath(parent))) < 210:
                parent /= "abcdefghij"
            parent.mkdir(parents=True)
            target = parent / ".artifact_generation_complete.json"

            candidate = _atomic_temp_candidate(
                target,
                "0123456789abcdef",
            )
            self.assertLessEqual(
                windows_utf16_units(Path(os.path.abspath(candidate))),
                259,
            )

            save_json_atomic({"version": 1}, target)

            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"version": 1},
            )

    @unittest.skipUnless(os.name == "nt", "Windows extended paths only")
    def test_atomic_write_preserves_an_explicit_extended_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            extended_tmp = Path("\\\\?\\" + str(Path(tmp).absolute()))
            cleanup_root = extended_tmp / "extended"
            parent = cleanup_root
            while windows_utf16_units(parent) < 240:
                parent /= "abcdefghij"
            parent.mkdir(parents=True)
            target = parent / ".artifact_generation_complete.json"
            token = "0123456789abcdef"

            try:
                candidate = _atomic_temp_candidate(target, token)
                self.assertEqual(
                    candidate.name,
                    f".{target.name}.{token}.tmp",
                )
                self.assertGreater(
                    windows_utf16_units(Path(os.path.abspath(candidate))),
                    259,
                )

                save_json_atomic({"version": 1}, target)

                self.assertEqual(
                    json.loads(target.read_text(encoding="utf-8")),
                    {"version": 1},
                )
            finally:
                if cleanup_root.exists():
                    shutil.rmtree(cleanup_root)

    @unittest.skipIf(os.name == "nt", "POSIX filename byte limit")
    def test_atomic_write_bounds_a_multibyte_posix_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            target = parent / (
                ("\N{LATIN SMALL LETTER E WITH ACUTE}" * 115) + ".json"
            )
            token = "0123456789abcdef"
            old_sidecar = parent / f".{target.name}.{token}.tmp"

            self.assertLessEqual(len(os.fsencode(target.name)), 255)
            self.assertGreater(len(os.fsencode(old_sidecar.name)), 255)
            candidate = _atomic_temp_candidate(target, token)
            self.assertLessEqual(len(os.fsencode(candidate.name)), 255)

            save_json_atomic({"version": 1}, target)

            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"version": 1},
            )


@unittest.skipIf(os.name == "nt", "POSIX permission modes only")
class AtomicJsonPermissionTests(unittest.TestCase):
    def test_new_file_uses_umask_and_replacement_preserves_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "result.json"
            previous_umask = os.umask(0o027)
            try:
                save_json_atomic({"version": 1}, output_path)
            finally:
                os.umask(previous_umask)

            self.assertEqual(output_path.stat().st_mode & 0o777, 0o640)
            output_path.chmod(0o664)
            save_json_atomic({"version": 2}, output_path)
            self.assertEqual(output_path.stat().st_mode & 0o777, 0o664)


if __name__ == "__main__":
    unittest.main()
