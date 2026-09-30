from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import find_applicable_templates as matcher  # noqa: E402


def _summary(dataset: str = "table.csv", extra_column: bool = False) -> dict:
    columns = {
        "group": {
            "dtype": "object",
            "kind": "categorical",
            "unique_count": 100,
        },
        "target": {
            "dtype": "float64",
            "kind": "regression",
            "unique_count": 100,
        },
    }
    if extra_column:
        columns["another_feature"] = {
            "dtype": "float64",
            "kind": "regression",
            "unique_count": 100,
        }
    return {
        "dataset": dataset,
        "target": "target",
        "columns": columns,
        "by_kind": {
            "categorical": ["group"],
            "regression": ["target"],
        },
    }


def _template(template_id: str) -> dict:
    return {
        "template_id": template_id,
        "ds_question": "Which values: {VALUE_A}, {VALUE_B}?",
        "slots": {
            "GROUP_COL": {
                "type": "column",
                "kind_any_of": ["categorical"],
            },
            "VALUE_A": {
                "type": "string",
                "source": "column_values",
                "source_column": "GROUP_COL",
                "position": "random",
            },
            "VALUE_B": {
                "type": "string",
                "source": "column_values",
                "source_column": "GROUP_COL",
                "position": "random",
            },
        },
    }


class TemplateMatchingReproducibilityTests(unittest.TestCase):
    def _make_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        templates_dir = root / "templates"
        templates_dir.mkdir()
        summary_path = root / "summary.json"
        dataset_path = root / "table.csv"
        summary_path.write_text(json.dumps(_summary()), encoding="utf-8")
        rows = ["group,target"]
        rows.extend(f"value_{index},{index}" for index in range(100))
        dataset_path.write_text("\n".join(rows), encoding="utf-8")
        return summary_path, dataset_path, templates_dir

    def test_template_loading_order_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            templates_dir = Path(tmp)
            path_a = templates_dir / "a.json"
            path_b = templates_dir / "b.json"
            path_a.write_text(json.dumps(_template("a_v0")), encoding="utf-8")
            path_b.write_text(json.dumps(_template("b_v0")), encoding="utf-8")

            with patch.object(
                Path,
                "rglob",
                return_value=iter([path_b, path_a]),
            ):
                loaded = matcher.load_templates(templates_dir)

        self.assertEqual(
            [template["template_id"] for _, template in loaded],
            ["a_v0", "b_v0"],
        )

    def test_assignments_do_not_depend_on_summary_path_spelling(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            root = Path(tmp)
            summary_path, dataset_path, templates_dir = self._make_fixture(root)
            (templates_dir / "template.json").write_text(
                json.dumps(_template("path_test_v0")),
                encoding="utf-8",
            )
            relative_summary = summary_path.relative_to(REPO_ROOT)
            relative_dataset = dataset_path.relative_to(REPO_ROOT)
            relative_templates = templates_dir.relative_to(REPO_ROOT)

            absolute_match = matcher.get_template_matches(
                summary_path.resolve(),
                dataset_path.resolve(),
                templates_dir.resolve(),
                seed=42,
            )[0]
            relative_match = matcher.get_template_matches(
                relative_summary,
                relative_dataset,
                relative_templates,
                seed=42,
            )[0]

        self.assertEqual(
            absolute_match.slot_assignments,
            relative_match.slot_assignments,
        )

    def test_each_template_has_an_independent_random_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path, dataset_path, templates_dir = self._make_fixture(root)
            template_a = _template("a_v0")
            template_b = _template("b_v0")
            path_a = templates_dir / "a.json"
            path_b = templates_dir / "b.json"

            with patch.object(
                matcher,
                "load_templates",
                return_value=[(path_a, template_a), (path_b, template_b)],
            ):
                forward = matcher.get_template_matches(
                    summary_path,
                    dataset_path,
                    templates_dir,
                    seed=42,
                )
            with patch.object(
                matcher,
                "load_templates",
                return_value=[(path_b, template_b), (path_a, template_a)],
            ):
                reversed_matches = matcher.get_template_matches(
                    summary_path,
                    dataset_path,
                    templates_dir,
                    seed=42,
                )

        forward_assignments = {
            match.template["template_id"]: match.slot_assignments
            for match in forward
        }
        reversed_assignments = {
            match.template["template_id"]: match.slot_assignments
            for match in reversed_matches
        }
        self.assertEqual(forward_assignments, reversed_assignments)

    def test_seed_and_summary_content_change_the_random_stream(self):
        template = _template("stream_test_v0")
        baseline = matcher._template_rng(42, _summary(), template).random()

        self.assertNotEqual(
            baseline,
            matcher._template_rng(43, _summary(), template).random(),
        )
        self.assertNotEqual(
            baseline,
            matcher._template_rng(
                42,
                _summary(extra_column=True),
                template,
            ).random(),
        )


if __name__ == "__main__":
    unittest.main()
