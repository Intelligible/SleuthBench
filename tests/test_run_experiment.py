from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_experiment  # noqa: E402


def _write_run_manifest(
    instances_dir: Path,
    injector: str,
    template_ids: list[str] | None,
) -> Path:
    instance_dir = (
        instances_dir
        / "runs"
        / "test-run"
        / "dataset_100"
        / "seed_42"
        / injector
    )
    instance_dir.mkdir(parents=True)
    manifest_path = instance_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({
            "run_id": "test-run",
            "seed": 42,
            "generation": {"fingerprint": "test-fingerprint"},
            "dataset_name": "dataset_100",
            "phenomenon": {"injector_type": injector},
            "qa_pairs": (
                [
                    {"template_id": template_id}
                    for template_id in template_ids
                ]
                if template_ids is not None
                else None
            ),
        }),
        encoding="utf-8",
    )
    return manifest_path


def _config(instances_dir: Path, templates: list[str] | None):
    return SimpleNamespace(
        instances_dir=instances_dir,
        run_id="test-run",
        seed=42,
        dataset_filter=None,
        injector_filter=None,
        templates=templates,
    )


def _load_config_data(data: dict) -> run_experiment.ExperimentConfig:
    with patch.object(
        Path,
        "read_text",
        return_value=json.dumps(data),
    ):
        return run_experiment.load_config(Path("experiment.yaml"), False)


class ConfigValidationTests(unittest.TestCase):
    def test_runner_tool_validation_uses_eval_pipeline_registry(self):
        self.assertEqual(
            run_experiment.VALID_TOOLS,
            frozenset(run_experiment.eval_pipeline.SUPPORTED_TOOLS),
        )

    def test_run_id_cannot_alias_an_existing_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for field in ("instances_dir", "results_dir"):
                with self.subTest(field=field):
                    instances_dir = root / f"{field}_instances"
                    results_dir = root / f"{field}_results"
                    selected_root = (
                        instances_dir
                        if field == "instances_dir"
                        else results_dir
                    )
                    (selected_root / "runs" / "Baseline").mkdir(
                        parents=True,
                    )
                    data = {
                        "seed": 42,
                        "summary": "summary.json",
                        "run_id": "baseline",
                        "instances_dir": str(instances_dir),
                        "results_dir": str(results_dir),
                    }

                    with self.assertRaisesRegex(
                        ValueError,
                        "existing namespace",
                    ):
                        _load_config_data(data)

    def test_derived_dataset_filter_is_unicode_normalized(self):
        config = _load_config_data({
            "seed": 42,
            "summary": "cafe\N{COMBINING ACUTE ACCENT}.json",
        })

        self.assertEqual(
            config.dataset_filter,
            "caf\N{LATIN SMALL LETTER E WITH ACUTE}",
        )

    def test_fresh_generation_cannot_skip_eval_prerequisites(self):
        for skipped_stage in ("skip_validate", "skip_answer"):
            with self.subTest(skipped_stage=skipped_stage):
                data = {
                    "seed": 42,
                    "summary": "summary.json",
                    skipped_stage: True,
                }

                with self.assertRaisesRegex(ValueError, skipped_stage):
                    _load_config_data(data)

    def test_reused_completed_manifests_may_skip_eval_prerequisites(self):
        config = _load_config_data({
            "seed": 42,
            "summary": "summary.json",
            "skip_phenomena": True,
            "skip_validate": True,
            "skip_answer": True,
        })

        self.assertTrue(config.skip_phenomena)
        self.assertTrue(config.skip_validate)
        self.assertTrue(config.skip_answer)
        self.assertFalse(config.skip_eval)

    def test_partial_fresh_pipeline_may_stop_before_eval(self):
        config = _load_config_data({
            "seed": 42,
            "summary": "summary.json",
            "skip_validate": True,
            "skip_answer": True,
            "skip_eval": True,
        })

        self.assertTrue(config.skip_validate)
        self.assertTrue(config.skip_answer)
        self.assertTrue(config.skip_eval)

    def test_run_names_cannot_collide_case_insensitively(self):
        data = {
            "seed": 42,
            "summary": "summary.json",
            "runs": [
                {"name": "Baseline"},
                {"name": "baseline"},
            ],
        }

        with self.assertRaisesRegex(
            ValueError,
            "case-insensitive filesystems",
        ):
            _load_config_data(data)

    def test_run_names_cannot_collide_after_unicode_normalization(self):
        data = {
            "seed": 42,
            "summary": "summary.json",
            "runs": [
                {"name": "\N{LATIN SMALL LETTER E WITH ACUTE}"},
                {"name": "e\N{COMBINING ACUTE ACCENT}"},
            ],
        }

        with self.assertRaisesRegex(ValueError, "Unicode normalization"):
            _load_config_data(data)

    def test_top_level_models_cannot_contain_duplicates(self):
        data = {
            "seed": 42,
            "summary": "summary.json",
            "models": ["gpt-test", "gpt-test"],
        }

        with self.assertRaisesRegex(
            ValueError,
            r"models contains duplicate model name 'gpt-test'",
        ):
            _load_config_data(data)

    def test_per_run_models_cannot_contain_duplicates(self):
        data = {
            "seed": 42,
            "summary": "summary.json",
            "models": ["gpt-default"],
            "runs": [{
                "name": "tools",
                "models": ["gpt-test", "gpt-test"],
            }],
        }

        with self.assertRaisesRegex(
            ValueError,
            r"runs\[0\]\.models .*duplicate model name 'gpt-test'",
        ):
            _load_config_data(data)


class GenerationMetadataValidationTests(unittest.TestCase):
    def test_dataset_names_cannot_collide_on_portable_filesystems(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            standardized = root / "standardized"
            summaries_dir = standardized / "summaries"
            templates_dir = root / "templates"
            summaries_dir.mkdir(parents=True)
            templates_dir.mkdir()

            upper_table = standardized / "Dataset.csv"
            lower_table = standardized / "dataset.csv"
            upper_table.write_text("x,y\n1,2\n", encoding="utf-8")
            if not lower_table.exists():
                lower_table.write_text("x,y\n1,2\n", encoding="utf-8")

            first = summaries_dir / "first.json"
            second = summaries_dir / "second.json"
            first.write_text(
                json.dumps({"dataset": "Dataset.csv"}),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps({"dataset": "dataset.csv"}),
                encoding="utf-8",
            )
            (templates_dir / "template.json").write_text(
                json.dumps({"template_id": "fake_v0"}),
                encoding="utf-8",
            )
            config = SimpleNamespace(
                seed=42,
                question_type="ds",
                templates=None,
                templates_dir=templates_dir,
            )

            with self.assertRaisesRegex(
                ValueError,
                "Unicode-normalizing filesystems",
            ):
                run_experiment.build_generation_metadata(
                    config,
                    [first, second],
                )


class DiscoverRunManifestsTests(unittest.TestCase):
    def test_template_filter_keeps_shared_injector_when_any_qa_pair_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            instances_dir = Path(tmp)
            shared_manifest = _write_run_manifest(
                instances_dir,
                "fc_interaction_dominant",
                [
                    "fc_interaction_dominant_v0",
                    "fc_interaction_direction_v0",
                ],
            )
            _write_run_manifest(
                instances_dir,
                "fc_noise_feature",
                ["fc_noise_feature_v0"],
            )

            with patch.object(
                run_experiment,
                "build_generation_metadata",
                return_value={"fingerprint": "test-fingerprint"},
            ):
                manifests = run_experiment.discover_run_manifests(
                    _config(instances_dir, ["fc_interaction_direction_v0"])
                )

            self.assertEqual(manifests, [shared_manifest])

    def test_empty_template_filter_keeps_all_matching_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            instances_dir = Path(tmp)
            first_manifest = _write_run_manifest(
                instances_dir,
                "fc_noise_feature",
                ["fc_noise_feature_v0"],
            )
            second_manifest = _write_run_manifest(
                instances_dir,
                "fc_threshold_value",
                ["fc_threshold_value_v0"],
            )

            for templates in (None, []):
                with self.subTest(templates=templates):
                    with patch.object(
                        run_experiment,
                        "build_generation_metadata",
                        return_value={"fingerprint": "test-fingerprint"},
                    ):
                        manifests = run_experiment.discover_run_manifests(
                            _config(instances_dir, templates)
                        )
                    self.assertEqual(
                        set(manifests),
                        {first_manifest, second_manifest},
                    )

    def test_template_filter_treats_null_qa_pairs_as_no_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            instances_dir = Path(tmp)
            _write_run_manifest(instances_dir, "legacy_injector", None)
            matching_manifest = _write_run_manifest(
                instances_dir,
                "fc_noise_feature",
                ["fc_noise_feature_v0"],
            )

            with patch.object(
                run_experiment,
                "build_generation_metadata",
                return_value={"fingerprint": "test-fingerprint"},
            ):
                manifests = run_experiment.discover_run_manifests(
                    _config(instances_dir, ["fc_noise_feature_v0"])
                )

            self.assertEqual(manifests, [matching_manifest])


if __name__ == "__main__":
    unittest.main()
