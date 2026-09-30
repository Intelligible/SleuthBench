from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import answer_pipeline  # noqa: E402
import phenomena_pipeline  # noqa: E402
import run_experiment  # noqa: E402
import validate_pipeline  # noqa: E402


def _write_manifest(
    instance_dir: Path,
    *,
    injector: str = "test_injector",
    category: str = "test",
    validation: dict | None = None,
    answer: object = "stale",
    run_id: str | None = None,
    seed: int = 42,
    dataset_name: str = "dataset_100",
    fingerprint: str | None = None,
) -> Path:
    instance_dir.mkdir(parents=True, exist_ok=True)
    (instance_dir / "table.csv").write_text(
        "feature,target\n1,2\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "seed": seed,
        "dataset_name": dataset_name,
        "target": "target",
        "phenomenon": {
            "injector_type": injector,
            "effects": {},
        },
        "qa_pairs": [
            {
                "template_id": "test_template_v0",
                "category": category,
                "slot_assignments": {},
                "answer": answer,
            }
        ],
    }
    if validation is not None:
        manifest["validation"] = validation
    if fingerprint is not None:
        manifest["generation"] = {"fingerprint": fingerprint}
    manifest_path = instance_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class RegistryInvalidationTests(unittest.TestCase):
    def test_missing_validator_invalidates_old_pass_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_manifest(
                root / "dataset_100" / "seed_42" / "test_injector",
                validation={"passed": True, "checks": []},
            )

            with (
                patch.dict(validate_pipeline.PHENOMENA, {}, clear=True),
                redirect_stdout(io.StringIO()),
            ):
                validate_pipeline.run(
                    root,
                    dataset_filter=None,
                    injector_filter=None,
                    force=False,
                    manifest_paths=[manifest_path],
                )

            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIsNone(saved["validation"])

    def test_revalidation_invalidates_old_pass_before_validator_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_manifest(
                root / "dataset_100" / "seed_42" / "test_injector",
                validation={"passed": True, "checks": []},
            )
            crashing = SimpleNamespace(
                validate=Mock(side_effect=RuntimeError("validator crashed"))
            )

            with patch.dict(
                validate_pipeline.PHENOMENA,
                {"test_injector": crashing},
                clear=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "validator crashed"):
                    validate_pipeline.validate_instance(manifest_path)

            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIsNone(saved["validation"])

    def test_missing_and_unimplemented_answer_computers_clear_old_answers(self):
        for mode in ("missing", "not_implemented"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest_path = _write_manifest(
                    root / "dataset_100" / "seed_42" / "test_injector",
                    validation={"passed": True, "checks": []},
                )
                registry = {}
                if mode == "not_implemented":
                    registry["test_template_v0"] = SimpleNamespace(
                        compute_answers={
                            "test_template_v0": Mock(
                                side_effect=NotImplementedError
                            )
                        }
                    )

                with (
                    patch.dict(
                        answer_pipeline.TEMPLATE_TO_PHENOMENON,
                        registry,
                        clear=True,
                    ),
                    redirect_stdout(io.StringIO()),
                ):
                    answer_pipeline.run(
                        root,
                        manifest_paths=[manifest_path],
                    )

                saved = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                self.assertIsNone(saved["qa_pairs"][0]["answer"])

    def test_standalone_validate_and_answer_preserve_manual_authored_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation = {
                "passed": True,
                "checks": [{"name": "manual_instance", "passed": True}],
            }
            manifest_path = _write_manifest(
                root / "manual_dataset" / "seed_0" / "manual",
                injector="manual",
                category="manual",
                validation=validation,
                answer=42,
                seed=0,
                dataset_name="manual_dataset",
            )

            with (
                patch.dict(validate_pipeline.PHENOMENA, {}, clear=True),
                patch.dict(
                    answer_pipeline.TEMPLATE_TO_PHENOMENON,
                    {},
                    clear=True,
                ),
                redirect_stdout(io.StringIO()),
            ):
                validate_pipeline.run(
                    root,
                    dataset_filter=None,
                    injector_filter=None,
                    force=True,
                    manifest_paths=[manifest_path],
                )
                answer_pipeline.run(root, manifest_paths=[manifest_path])

            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["validation"], validation)
            self.assertEqual(saved["qa_pairs"][0]["answer"], 42)


def _experiment_fixture(root: Path):
    standardized = root / "standardized"
    summaries_dir = standardized / "summaries"
    summaries_dir.mkdir(parents=True)
    table_path = standardized / "dataset_100.csv"
    table_path.write_text("feature,target\n1,2\n", encoding="utf-8")
    summary_path = summaries_dir / "dataset_100.json"
    summary_path.write_text(
        json.dumps(
            {
                "dataset": table_path.name,
                "target": "target",
                "columns": {},
            }
        ),
        encoding="utf-8",
    )
    templates_dir = root / "templates"
    templates_dir.mkdir()
    template_path = templates_dir / "test_template.json"
    template_path.write_text(
        json.dumps({"template_id": "test_template_v0", "ds_question": "Q?"}),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        summary=summary_path,
        summaries_dir=None,
        templates_dir=templates_dir,
        templates=["test_template_v0"],
        seed=42,
        question_type="ds",
        instances_dir=root / "instances",
        run_id="test-run",
        dataset_filter=None,
        injector_filter=None,
    )
    return cfg, summary_path, table_path, template_path


class RunBatchLifecycleTests(unittest.TestCase):
    def test_generation_fingerprint_covers_question_summary_table_and_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, summary_path, table_path, template_path = (
                _experiment_fixture(Path(tmp))
            )
            baseline = run_experiment.build_generation_metadata(cfg)[
                "fingerprint"
            ]

            cfg.question_type = "business"
            self.assertNotEqual(
                baseline,
                run_experiment.build_generation_metadata(cfg)["fingerprint"],
            )
            cfg.question_type = "ds"

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["description"] = "changed"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            self.assertNotEqual(
                baseline,
                run_experiment.build_generation_metadata(cfg)["fingerprint"],
            )
            summary.pop("description")
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            table_path.write_text("feature,target\n3,4\n", encoding="utf-8")
            self.assertNotEqual(
                baseline,
                run_experiment.build_generation_metadata(cfg)["fingerprint"],
            )
            table_path.write_text("feature,target\n1,2\n", encoding="utf-8")

            template = json.loads(template_path.read_text(encoding="utf-8"))
            template["ds_question"] = "Changed?"
            template_path.write_text(json.dumps(template), encoding="utf-8")
            self.assertNotEqual(
                baseline,
                run_experiment.build_generation_metadata(cfg)["fingerprint"],
            )

    def test_discovery_rejects_legacy_manifest_without_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, _, _, _ = _experiment_fixture(root)
            _write_manifest(
                cfg.instances_dir
                / "runs"
                / cfg.run_id
                / "dataset_100"
                / "seed_42"
                / "test_injector",
                run_id=cfg.run_id,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "different Stage 1 inputs",
            ):
                run_experiment.discover_run_manifests(cfg)

    def test_cleanup_removes_only_same_run_and_seed_orphans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, _, _, _ = _experiment_fixture(root)
            metadata = run_experiment.build_generation_metadata(cfg)
            run_root = cfg.instances_dir / "runs" / cfg.run_id
            expected = run_root / "dataset_100" / "seed_42" / "expected"
            orphan = run_root / "dataset_100" / "seed_42" / "orphan"
            removed_dataset = (
                run_root / "removed_dataset" / "seed_42" / "orphan"
            )
            other_seed = (
                run_root / "dataset_100" / "seed_7" / "other_seed"
            )
            foreign = run_root / "dataset_100" / "seed_42" / "foreign"

            _write_manifest(
                expected,
                run_id=cfg.run_id,
                fingerprint=metadata["fingerprint"],
            )
            _write_manifest(orphan, run_id=cfg.run_id)
            _write_manifest(
                removed_dataset,
                run_id=cfg.run_id,
                dataset_name="removed_dataset",
            )
            _write_manifest(other_seed, run_id=cfg.run_id, seed=7)
            _write_manifest(foreign, run_id="different-run")

            removed = run_experiment._cleanup_orphan_run_instances(
                cfg,
                metadata,
                [expected],
            )

            self.assertEqual(removed, 2)
            self.assertTrue(expected.exists())
            self.assertFalse(orphan.exists())
            self.assertFalse(removed_dataset.exists())
            self.assertTrue(other_seed.exists())
            self.assertTrue(foreign.exists())

    def test_failed_stage_does_not_reconcile_old_orphans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, _, _, _ = _experiment_fixture(root)
            orphan = (
                cfg.instances_dir
                / "runs"
                / cfg.run_id
                / "dataset_100"
                / "seed_42"
                / "old"
            )
            _write_manifest(orphan, run_id=cfg.run_id)

            with (
                patch.object(
                    phenomena_pipeline,
                    "build_instances",
                    side_effect=RuntimeError("injection failed"),
                ),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "injection failed"):
                    run_experiment.stage_phenomena(cfg)

            self.assertTrue(orphan.exists())


if __name__ == "__main__":
    unittest.main()
