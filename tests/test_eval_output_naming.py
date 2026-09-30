from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import eval_pipeline  # noqa: E402


class EvalOutputNamingTests(unittest.TestCase):
    def _path(self, **overrides) -> Path:
        values = {
            "models": ["gpt-test"],
            "instances_dir": Path("data/instances"),
            "tools": ["run_python"],
            "columns_only": False,
            "table_in_prompt": False,
            "run_id": None,
            "dataset": None,
            "injectors": None,
            "templates": None,
        }
        values.update(overrides)
        return eval_pipeline._default_eval_output_path(**values)

    def test_identical_conditions_have_the_same_path(self):
        self.assertEqual(
            self._path(
                tools=["run_python", "load_data"],
                injectors=["b", "a"],
                templates=["second", "first"],
            ),
            self._path(
                tools=["load_data", "run_python"],
                injectors=["a", "b"],
                templates=["first", "second"],
            ),
        )

    def test_each_material_filter_changes_the_path(self):
        baseline = self._path()
        variants = {
            self._path(dataset="_100"),
            self._path(injectors=["fc_noise_feature"]),
            self._path(templates=["fc_noise_feature_v0"]),
            self._path(models=["gpt-other"]),
            self._path(models=["gpt-test", "gpt-other"]),
            self._path(columns_only=True),
            self._path(table_in_prompt=True),
            self._path(instances_dir=Path("other/instances")),
        }
        self.assertNotIn(baseline, variants)
        self.assertEqual(len(variants), 8)

    def test_run_id_scopes_the_hashed_output(self):
        path = self._path(run_id="experiment")
        self.assertEqual(
            path.parent,
            Path("data/results/runs/experiment"),
        )
        self.assertRegex(path.name, r"^eval_results_gpt-test_[0-9a-f]{12}\.json$")

    def test_model_label_cannot_escape_the_results_directory(self):
        path = self._path(models=[r"gpt-x\..\..\victim"])
        self.assertEqual(path.parent, Path("data/results"))
        self.assertNotIn("\\", path.name)
        self.assertNotIn("..", path.name)


if __name__ == "__main__":
    unittest.main()
