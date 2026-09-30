from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import answer_pipeline  # noqa: E402
import phenomena_pipeline  # noqa: E402
import run_experiment  # noqa: E402


class PhenomenaFailurePropagationTests(unittest.TestCase):
    @staticmethod
    def _run_with_injectors(injectors, specs):
        summary = {
            "dataset": "dataset.csv",
            "target": "target",
            "by_kind": {},
        }
        dataframe = Mock()
        dataframe.copy.return_value = dataframe
        with tempfile.TemporaryDirectory() as tmp:
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        phenomena_pipeline,
                        "load_json",
                        return_value=summary,
                    )
                )
                stack.enter_context(
                    patch.object(
                        phenomena_pipeline,
                        "load_csv",
                        return_value=dataframe,
                    )
                )
                stack.enter_context(
                    patch.object(
                        phenomena_pipeline,
                        "get_template_matches",
                        return_value=[],
                    )
                )
                stack.enter_context(
                    patch.object(
                        phenomena_pipeline,
                        "collect_required_phenomena",
                        return_value=specs,
                    )
                )
                stack.enter_context(
                    patch.dict(
                        phenomena_pipeline.PHENOMENA,
                        injectors,
                        clear=True,
                    )
                )
                stdout = stack.enter_context(
                    redirect_stdout(io.StringIO())
                )
                result = phenomena_pipeline.build_instance(
                    Path(tmp)
                    / "standardized"
                    / "summaries"
                    / "dataset.json",
                    42,
                    output_root=Path(tmp) / "instances",
                )
        return result, stdout.getvalue()

    def test_injection_rejection_is_an_expected_skip(self):
        reject = SimpleNamespace(
            inject=Mock(
                side_effect=phenomena_pipeline.InjectionRejected(
                    "no suitable candidate"
                )
            ),
            summary_fields=(),
        )

        result, output = self._run_with_injectors(
            {"reject": reject},
            [{"type": "reject", "params": {}}],
        )

        self.assertEqual(result, [])
        self.assertIn("SKIP (no suitable candidate)", output)

    def test_plain_value_error_is_an_unexpected_failure(self):
        crash = SimpleNamespace(
            inject=Mock(side_effect=ValueError("bad array shape")),
            summary_fields=(),
        )

        with self.assertRaises(
            phenomena_pipeline.UnexpectedInjectionErrors
        ) as raised:
            self._run_with_injectors(
                {"crash": crash},
                [{"type": "crash", "params": {}}],
            )

        self.assertIsInstance(raised.exception.__cause__, ValueError)

    def test_malformed_injector_return_is_an_unexpected_failure(self):
        malformed = SimpleNamespace(
            inject=Mock(
                return_value=(
                    phenomena_pipeline.pd.DataFrame({"target": [1]}),
                    {},
                )
            ),
            summary_fields=(),
        )

        with self.assertRaises(
            phenomena_pipeline.UnexpectedInjectionErrors
        ) as raised:
            self._run_with_injectors(
                {"malformed": malformed},
                [{"type": "malformed", "params": {}}],
            )

        self.assertIsInstance(raised.exception.__cause__, KeyError)

    def test_wrong_injector_return_arity_is_an_unexpected_failure(self):
        malformed = SimpleNamespace(
            inject=Mock(
                return_value=(
                    phenomena_pipeline.pd.DataFrame({"target": [1]}),
                )
            ),
            summary_fields=(),
        )

        with self.assertRaises(
            phenomena_pipeline.UnexpectedInjectionErrors
        ) as raised:
            self._run_with_injectors(
                {"malformed": malformed},
                [{"type": "malformed", "params": {}}],
            )

        self.assertIsInstance(raised.exception.__cause__, TypeError)

    def test_mismatched_injector_type_is_an_unexpected_failure(self):
        mismatched = SimpleNamespace(
            inject=Mock(
                return_value=(
                    phenomena_pipeline.pd.DataFrame({"target": [1]}),
                    {
                        "type": "other",
                        "params": {},
                        "effects": {},
                    },
                )
            ),
            summary_fields=(),
        )

        with self.assertRaises(
            phenomena_pipeline.UnexpectedInjectionErrors
        ) as raised:
            self._run_with_injectors(
                {"expected": mismatched},
                [{"type": "expected", "params": {}}],
            )

        self.assertIsInstance(raised.exception.__cause__, ValueError)

    def test_unexpected_error_finishes_injector_batch_then_raises(self):
        crash = SimpleNamespace(
            inject=Mock(side_effect=RuntimeError("broken injector")),
            summary_fields=(),
        )
        reject = SimpleNamespace(
            inject=Mock(
                side_effect=phenomena_pipeline.InjectionRejected(
                    "no candidate"
                )
            ),
            summary_fields=(),
        )

        with self.assertRaises(
            phenomena_pipeline.UnexpectedInjectionErrors
        ) as raised:
            self._run_with_injectors(
                {"crash": crash, "reject": reject},
                [
                    {"type": "crash", "params": {}},
                    {"type": "reject", "params": {}},
                ],
            )

        crash.inject.assert_called_once()
        reject.inject.assert_called_once()
        self.assertEqual(len(raised.exception.failures), 1)
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)

    def test_runner_collects_injection_errors_across_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            summaries_dir = Path(tmp) / "summaries"
            summaries_dir.mkdir()
            first = summaries_dir / "first.json"
            second = summaries_dir / "second.json"
            first.write_text(
                json.dumps({"dataset": "first.csv"}),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps({"dataset": "second.csv"}),
                encoding="utf-8",
            )
            cause = RuntimeError("broken injector")
            failure = (first, "broken", cause)
            cfg = SimpleNamespace(
                summary=None,
                summaries_dir=summaries_dir,
                seed=42,
                templates_dir=Path("templates"),
                templates=None,
                question_type="ds",
                instances_dir=Path(tmp) / "instances",
                run_id="test",
            )

            with patch.object(
                run_experiment,
                "build_generation_metadata",
                return_value={
                    "fingerprint": "test",
                    "datasets": ["first", "second"],
                },
            ), patch.object(
                phenomena_pipeline,
                "build_instance",
                side_effect=[
                    phenomena_pipeline.UnexpectedInjectionErrors([failure]),
                    [],
                ],
            ) as build, redirect_stdout(io.StringIO()):
                with self.assertRaises(
                    phenomena_pipeline.UnexpectedInjectionErrors
                ) as raised:
                    run_experiment.stage_phenomena(cfg)

        self.assertEqual(build.call_count, 2)
        self.assertEqual(raised.exception.failures, [failure])


class AnswerFailurePropagationTests(unittest.TestCase):
    def test_compute_failure_clears_stale_answer_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifests = []
            for name, template_id in (
                ("first", "bad_template"),
                ("second", "good_template"),
            ):
                instance_dir = root / name
                instance_dir.mkdir()
                (instance_dir / "table.csv").write_text(
                    "feature,target\n1,2\n",
                    encoding="utf-8",
                )
                manifest_path = instance_dir / "manifest.json"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "dataset_name": "dataset",
                            "phenomenon": {
                                "injector_type": "injector",
                                "effects": {},
                            },
                            "validation": {"passed": True},
                            "qa_pairs": [
                                {
                                    "template_id": template_id,
                                    "slot_assignments": {},
                                    "answer": "stale",
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                manifests.append(manifest_path)

            bad_compute = Mock(
                side_effect=RuntimeError("broken answer computer")
            )
            good_compute = Mock(return_value=42)
            registry = {
                "bad_template": SimpleNamespace(
                    compute_answers={"bad_template": bad_compute}
                ),
                "good_template": SimpleNamespace(
                    compute_answers={"good_template": good_compute}
                ),
            }

            with patch.dict(
                answer_pipeline.TEMPLATE_TO_PHENOMENON,
                registry,
                clear=True,
            ), redirect_stdout(io.StringIO()):
                with self.assertRaises(
                    answer_pipeline.AnswerComputationErrors
                ) as raised:
                    answer_pipeline.run(
                        root,
                        manifest_paths=manifests,
                    )

            failed = json.loads(manifests[0].read_text(encoding="utf-8"))
            succeeded = json.loads(
                manifests[1].read_text(encoding="utf-8")
            )

        self.assertIsNone(failed["qa_pairs"][0]["answer"])
        self.assertEqual(succeeded["qa_pairs"][0]["answer"], 42)
        self.assertIs(raised.exception.__cause__, bad_compute.side_effect)
        good_compute.assert_called_once()

    def test_answer_errors_finish_manifest_batch_then_raise(self):
        manifests = [Path("first.json"), Path("second.json")]
        error_rows = [
            ["dataset", "injector", "template_a", "None", "ERROR"]
        ]
        cause = RuntimeError("broken answer computer")
        first_error = answer_pipeline.AnswerComputationErrors(
            [(manifests[0], "template_a", cause)],
            error_rows,
        )
        process_results = [
            first_error,
            [["dataset", "injector", "template_b", "'ok'", "OK"]],
        ]

        with patch.object(
            answer_pipeline,
            "process_instance",
            side_effect=process_results,
        ) as process, redirect_stdout(io.StringIO()) as stdout:
            with self.assertRaisesRegex(
                answer_pipeline.AnswerComputationErrors,
                "Answer computation failed for 1 QA pair",
            ) as raised:
                answer_pipeline.run(
                    Path("instances"),
                    manifest_paths=manifests,
                )

        self.assertEqual(process.call_count, 2)
        self.assertIs(raised.exception.__cause__, cause)
        self.assertIn("1 ERROR", stdout.getvalue())
        self.assertIn("1 OK", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
