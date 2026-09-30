from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import eval_pipeline  # noqa: E402
import grade_pipeline  # noqa: E402
import manual_pipeline  # noqa: E402
import phenomena_pipeline  # noqa: E402
from io_utils import reset_generated_instance_dir  # noqa: E402
from shared.path_utils import (  # noqa: E402
    GENERATED_ARTIFACT_NAME_RESERVE_UNITS,
    windows_utf16_units,
)


class InstanceArtifactLifecycleTests(unittest.TestCase):
    def test_generated_instance_name_preserves_artifact_path_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = (
                Path(tmp)
                / ("instances_" + "\N{GRINNING FACE}" * 20)
                / "dataset"
                / "seed_42"
            )
            parent_units = windows_utf16_units(parent.absolute())
            max_name_units = 39
            max_path_units = (
                parent_units
                + 2
                + max_name_units
                + GENERATED_ARTIFACT_NAME_RESERVE_UNITS
            )
            long_name = (
                "fc_pairwise_interaction_antagonistic_reversal_"
                "CO2_tCO2__Lagging_Current_Power_Factor"
            )
            identity = (
                "fc_pairwise_interaction",
                (("pattern", "antagonistic_reversal"),),
            )

            bounded = phenomena_pipeline._fit_instance_name_to_path(
                long_name,
                parent,
                identity,
                max_path_units=max_path_units,
            )
            repeated = phenomena_pipeline._fit_instance_name_to_path(
                long_name,
                parent,
                identity,
                max_path_units=max_path_units,
            )
            other = phenomena_pipeline._fit_instance_name_to_path(
                long_name,
                parent,
                (identity, "other"),
                max_path_units=max_path_units,
            )

            self.assertEqual(bounded, repeated)
            self.assertNotEqual(bounded, other)
            self.assertEqual(windows_utf16_units(bounded), max_name_units)
            self.assertRegex(bounded, r"__[0-9a-f]{12}$")
            self.assertLessEqual(
                parent_units
                + 2
                + windows_utf16_units(bounded)
                + GENERATED_ARTIFACT_NAME_RESERVE_UNITS,
                max_path_units,
            )

    def test_generated_instance_name_keeps_short_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "instances" / "dataset" / "seed_42"
            short_name = "fc_threshold_value"
            max_path_units = (
                windows_utf16_units(parent.absolute())
                + 2
                + windows_utf16_units(short_name)
                + GENERATED_ARTIFACT_NAME_RESERVE_UNITS
            )

            self.assertEqual(
                phenomena_pipeline._fit_instance_name_to_path(
                    short_name,
                    parent,
                    ("fc_threshold_value", ()),
                    max_path_units=max_path_units,
                ),
                short_name,
            )

    def test_generated_instance_name_rejects_an_unusable_parent_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "instances" / "dataset" / "seed_42"
            max_path_units = (
                windows_utf16_units(parent.absolute())
                + 2
                + GENERATED_ARTIFACT_NAME_RESERVE_UNITS
                + 14
            )

            with self.assertRaisesRegex(
                OSError,
                "parent path is too long",
            ):
                phenomena_pipeline._fit_instance_name_to_path(
                    "a_very_long_generated_instance_name",
                    parent,
                    ("fake", ()),
                    max_path_units=max_path_units,
                )

    @unittest.skipUnless(os.name == "nt", "Windows extended paths only")
    def test_generated_instance_name_preserves_an_explicit_extended_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(
                "\\\\?\\" + str(Path(tmp).absolute())
            ) / ("nested" * 30)
            instance_name = (
                "fc_pairwise_interaction_antagonistic_reversal_"
                "CO2_tCO2__Lagging_Current_Power_Factor"
            )

            self.assertGreater(
                windows_utf16_units(parent / instance_name / "artifact.json"),
                259,
            )
            self.assertEqual(
                phenomena_pipeline._fit_instance_name_to_path(
                    instance_name,
                    parent,
                    ("fc_pairwise_interaction", ()),
                ),
                instance_name,
            )

    def test_real_steel_pairwise_instances_fit_the_artifact_budget(self):
        dataset_name = "steel_industry_data_1000"
        # Seed 42 makes both pairwise instances pick long column names, so the
        # legacy "<injector>_<label>" directory name would exceed MAX_PATH.
        seed = 42
        seed_name = f"seed_{seed}"
        artifact_name = "generated_output_artifact_data.json"
        summary_path = (
            REPO_ROOT
            / "data"
            / "standardized"
            / "summaries"
            / f"{dataset_name}.json"
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "instances"
            short_parent = (
                output_root / "runs" / "x" / dataset_name / seed_name
            )
            run_id_units = (
                154 - windows_utf16_units(short_parent.absolute()) + 1
            )
            self.assertGreaterEqual(run_id_units, 1)
            self.assertLessEqual(run_id_units, 128)
            run_id = "r" * run_id_units
            fake_windows_os = SimpleNamespace(name="nt", path=os.path)

            with (
                patch.object(phenomena_pipeline, "os", fake_windows_os),
                redirect_stdout(io.StringIO()),
            ):
                output_dirs = phenomena_pipeline.build_instance(
                    summary_path,
                    seed=seed,
                    templates_dir=REPO_ROOT / "templates",
                    template_filter=[
                        "fc_pairwise_antagonistic_reversal_v0",
                        "fc_pairwise_compensatory_reversal_v0",
                    ],
                    output_root=output_root,
                    run_id=run_id,
                )

            self.assertEqual(len(output_dirs), 2)
            for instance_dir in output_dirs:
                self.assertEqual(
                    windows_utf16_units(instance_dir.parent.absolute()),
                    154,
                )
                self.assertEqual(windows_utf16_units(instance_dir.name), 39)
                self.assertRegex(instance_dir.name, r"__[0-9a-f]{12}$")
                self.assertTrue((instance_dir / "table.csv").is_file())
                manifest_path = instance_dir / "manifest.json"
                self.assertTrue(manifest_path.is_file())

                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                injector = manifest["phenomenon"]["injector_type"]
                effects = manifest["phenomenon"]["effects"]
                label = phenomena_pipeline._slugify(
                    phenomena_pipeline.PHENOMENA[injector].label(effects) or ""
                )
                old_name = f"{injector}_{label}"
                old_artifact = instance_dir.parent / old_name / artifact_name
                self.assertGreater(
                    windows_utf16_units(old_artifact.absolute()),
                    259,
                )

                artifact_path = instance_dir / artifact_name
                self.assertEqual(
                    windows_utf16_units(artifact_path.absolute()),
                    230,
                )
                artifact_path.write_text("{}\n", encoding="utf-8")
                self.assertTrue(artifact_path.is_file())

    def test_standalone_generation_rejects_a_run_id_namespace_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summaries = root / "standardized" / "summaries"
            summaries.mkdir(parents=True)
            summary_path = summaries / "summary.json"
            summary_path.write_text(
                json.dumps({"dataset": "dataset.csv"}),
                encoding="utf-8",
            )
            output_root = root / "instances"
            (output_root / "runs" / "Baseline").mkdir(parents=True)

            with self.assertRaisesRegex(
                ValueError,
                "existing namespace",
            ):
                phenomena_pipeline.build_instance(
                    summary_path,
                    seed=42,
                    output_root=output_root,
                    run_id="baseline",
                )

    def test_batch_rejects_portably_colliding_dataset_names_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(
                json.dumps({"dataset": "Dataset.csv"}),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps({"dataset": "dataset.csv"}),
                encoding="utf-8",
            )

            with patch.object(phenomena_pipeline, "build_instance") as build:
                with self.assertRaisesRegex(
                    ValueError,
                    "collide on a portable filesystem",
                ):
                    phenomena_pipeline.build_instances(
                        [first, second],
                        seed=42,
                    )

            build.assert_not_called()

    def test_reset_rejects_paths_outside_or_too_close_to_the_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instances_root = root / "instances"
            broad_target = instances_root / "dataset"
            broad_target.mkdir(parents=True)
            broad_sentinel = broad_target / "sentinel.txt"
            broad_sentinel.write_text("keep", encoding="utf-8")

            outside_target = root / "outside" / "dataset" / "seed_42" / "fake"
            outside_target.mkdir(parents=True)
            outside_sentinel = outside_target / "sentinel.txt"
            outside_sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaises(ValueError):
                reset_generated_instance_dir(
                    broad_target,
                    instances_root,
                )
            with self.assertRaises(ValueError):
                reset_generated_instance_dir(
                    outside_target,
                    instances_root,
                )

            self.assertTrue(broad_sentinel.exists())
            self.assertTrue(outside_sentinel.exists())

    def test_reset_rejects_an_intermediate_directory_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instances_root = root / "instances"
            real_instance = (
                instances_root
                / "real_dataset"
                / "seed_42"
                / "fake"
            )
            real_instance.mkdir(parents=True)
            sentinel = real_instance / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")
            alias = instances_root / "alias"
            try:
                alias.symlink_to(
                    instances_root / "real_dataset",
                    target_is_directory=True,
                )
            except OSError as exc:
                self.skipTest(f"directory symlinks unavailable: {exc}")

            with self.assertRaisesRegex(
                ValueError,
                "intermediate symlink or junction",
            ):
                reset_generated_instance_dir(
                    alias / "seed_42" / "fake",
                    instances_root,
                )

            self.assertTrue(sentinel.exists())

    def test_reset_allows_the_configured_root_itself_to_be_a_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_root = root / "real_instances"
            instance = real_root / "dataset" / "seed_42" / "fake"
            instance.mkdir(parents=True)
            (instance / "stale.txt").write_text("stale", encoding="utf-8")
            linked_root = root / "instances"
            try:
                linked_root.symlink_to(real_root, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks unavailable: {exc}")

            rebuilt = reset_generated_instance_dir(
                linked_root / "dataset" / "seed_42" / "fake",
                linked_root,
            )

            self.assertEqual(
                rebuilt,
                linked_root / "dataset" / "seed_42" / "fake",
            )
            self.assertTrue(rebuilt.is_dir())
            self.assertFalse((rebuilt / "stale.txt").exists())

    def test_manual_force_recreates_the_instance_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table_path = root / "source.csv"
            table_path.write_text(
                "feature,target\n1,2\n",
                encoding="utf-8",
            )
            config = {
                "dataset_name": "manual_dataset",
                "table": table_path,
                "target": "target",
                "questions": [
                    {
                        "id": "manual_lifecycle_value",
                        "question": "What is the target?",
                        "answer": 2,
                        "answer_format": "value",
                    }
                ],
            }
            instances_root = root / "instances"

            instance_dir = manual_pipeline.build_instance(
                config,
                instances_root,
                force=False,
            )
            stale_artifact = instance_dir / "stale_data.bin"
            stale_artifact.write_bytes(b"stale")
            table_path.write_text(
                "feature,target\n3,4\n",
                encoding="utf-8",
            )

            rebuilt_dir = manual_pipeline.build_instance(
                config,
                instances_root,
                force=True,
            )

            self.assertEqual(rebuilt_dir, instance_dir)
            self.assertFalse(stale_artifact.exists())
            self.assertEqual(
                (instance_dir / "table.csv").read_text(encoding="utf-8"),
                "feature,target\n3,4\n",
            )

    def test_build_instance_recreates_only_the_target_instance_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            standardized = root / "data" / "standardized"
            summaries = standardized / "summaries"
            summaries.mkdir(parents=True)
            csv_path = standardized / "dataset_100.csv"
            pd.DataFrame(
                {"feature": [1.0, 2.0], "target": [3.0, 4.0]}
            ).to_csv(csv_path, index=False)
            summary_path = summaries / "dataset_100.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "dataset": csv_path.name,
                        "target": "target",
                        "by_kind": {},
                    }
                ),
                encoding="utf-8",
            )

            match = phenomena_pipeline.TemplateMatch(
                template={
                    "template_id": "fake_v0",
                    "category": "test",
                    "answer_format": "value",
                    "phenomena": [{"injector": "fake", "param_mapping": {}}],
                },
                template_path=root / "fake.json",
                is_compatible=True,
                reasons=[],
                slot_assignments={},
                feature_pool=[],
                rendered_question="What is the value?",
            )

            def inject(df, params, rng):
                return df.copy(), {
                    "type": "fake",
                    "params": params,
                    "effects": {},
                }

            fake_phenomenon = SimpleNamespace(
                inject=inject,
                summary_fields=(),
                label=None,
            )

            output_root = root / "instances"
            instance_dir = (
                output_root / "dataset_100" / "seed_42" / "fake"
            )
            stale_nested = instance_dir / "old" / "stale_data.bin"
            stale_nested.parent.mkdir(parents=True)
            stale_nested.write_bytes(b"stale")
            sibling_sentinel = (
                output_root
                / "dataset_100"
                / "seed_42"
                / "sibling"
                / "sentinel.txt"
            )
            sibling_sentinel.parent.mkdir(parents=True)
            sibling_sentinel.write_text("keep", encoding="utf-8")

            with (
                patch.object(
                    phenomena_pipeline,
                    "get_template_matches",
                    return_value=[match],
                ),
                patch.dict(
                    phenomena_pipeline.PHENOMENA,
                    {"fake": fake_phenomenon},
                    clear=True,
                ),
                redirect_stdout(io.StringIO()),
            ):
                output_dirs = phenomena_pipeline.build_instance(
                    summary_path,
                    42,
                    root / "templates",
                    output_root=output_root,
                )

            self.assertEqual(output_dirs, [instance_dir])
            self.assertFalse(stale_nested.exists())
            self.assertTrue((instance_dir / "table.csv").exists())
            self.assertTrue((instance_dir / "manifest.json").exists())
            self.assertEqual(
                sibling_sentinel.read_text(encoding="utf-8"),
                "keep",
            )


class EmptyOutputLifecycleTests(unittest.TestCase):
    def test_eval_overwrites_stale_output_when_all_qas_are_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance_dir = (
                root / "instances" / "dataset_100" / "seed_42" / "fake"
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
                        "phenomenon": {"injector_type": "fake"},
                        "validation": {"passed": True},
                        "qa_pairs": [
                            {
                                "template_id": "other",
                                "answer": "2",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output_path = root / "results" / "eval_results.json"
            output_path.parent.mkdir()
            output_path.write_text(
                '[{"stale": true}]',
                encoding="utf-8",
            )

            query_mock = Mock(side_effect=AssertionError("query must not run"))
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
                    {"openai": query_mock},
                ),
                redirect_stdout(io.StringIO()),
            ):
                results = eval_pipeline.run_eval(
                    ["gpt-test"],
                    root / "instances",
                    output_path,
                    template_ids=["wanted"],
                    manifest_paths=[manifest_path],
                )

            self.assertEqual(results, [])
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                [],
            )
            query_mock.assert_not_called()

    def test_grade_overwrites_stale_output_for_empty_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "eval_results.json"
            input_path.write_text("[]", encoding="utf-8")
            output_path = root / "graded.json"
            output_path.write_text(
                '[{"stale": true}]',
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
                patch.object(grade_pipeline, "grade_entry") as grade_entry,
                redirect_stdout(io.StringIO()),
            ):
                written_path = grade_pipeline.run(
                    input_path,
                    output_path,
                )

            self.assertEqual(written_path, output_path)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                [],
            )
            grade_entry.assert_not_called()


if __name__ == "__main__":
    unittest.main()
