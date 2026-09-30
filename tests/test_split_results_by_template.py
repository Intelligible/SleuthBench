from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "split_results_by_template.py"
SPEC = importlib.util.spec_from_file_location(
    "split_results_by_template",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
splitter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(splitter)


class SplitResultsByTemplateTests(unittest.TestCase):
    def _split(self, root: Path, template_ids: list[str]):
        input_path = root / "results.json"
        outdir = root / "split"
        input_path.write_text(
            json.dumps(
                [
                    {"template_id": template_id, "value": index}
                    for index, template_id in enumerate(template_ids)
                ]
            ),
            encoding="utf-8",
        )
        with redirect_stdout(io.StringIO()):
            paths = splitter.split_results(input_path, outdir)
        return outdir, paths

    def test_traversal_ids_stay_inside_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template_ids = [
                "../../victim",
                r"..\..\victim",
                "/absolute/victim",
                "normal_v0",
            ]

            outdir, paths = self._split(root, template_ids)
            resolved_outdir = outdir.resolve()

            self.assertEqual(set(paths), set(template_ids))
            self.assertEqual(len({path.name.casefold() for path in paths.values()}), 4)
            for template_id, path in paths.items():
                self.assertEqual(path.resolve().parent, resolved_outdir)
                self.assertNotIn("/", path.name)
                self.assertNotIn("\\", path.name)
                self.assertEqual(
                    json.loads(path.read_text(encoding="utf-8"))[0][
                        "template_id"
                    ],
                    template_id,
                )
            self.assertFalse((root / "victim.json").exists())

    def test_sanitization_and_case_collisions_get_unique_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template_ids = ["a/b", r"a\b", "a_b", "A_B", "CON"]

            _, paths = self._split(root, template_ids)

            names = [path.name for path in paths.values()]
            self.assertEqual(len(set(name.casefold() for name in names)), 5)
            self.assertTrue(
                all(name.endswith(".json") for name in names)
            )
            self.assertTrue(
                all(".." not in name for name in names)
            )

    def test_digest_collision_has_deterministic_counter_fallback(self):
        with patch.object(
            splitter,
            "_template_digest",
            return_value="0" * 64,
        ):
            first = splitter._allocate_output_filenames(["a/b", r"a\b"])
            second = splitter._allocate_output_filenames([r"a\b", "a/b"])

        self.assertEqual(first, second)
        self.assertEqual(
            len({name.casefold() for name in first.values()}),
            2,
        )
        self.assertTrue(any("__2.json" in name for name in first.values()))

    def test_containment_guard_rejects_direct_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            with self.assertRaisesRegex(ValueError, "unsafe output filename"):
                splitter._contained_output_path(outdir, "../victim.json")

    def test_cli_reconfigures_a_narrow_stdout_before_unicode_logging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "results.json"
            outdir = root / "split"
            input_path.write_text(
                json.dumps(
                    [{"template_id": "template_→_v0"}],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            stdout_buffer = io.BytesIO()
            stdout = io.TextIOWrapper(stdout_buffer, encoding="ascii")

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "split_results_by_template.py",
                        str(input_path),
                        str(outdir),
                    ],
                ),
                patch.object(sys, "stdout", stdout),
            ):
                splitter.main()
                stdout.flush()

            output = stdout_buffer.getvalue().decode("utf-8")
            self.assertIn("template_→_v0", output)
            self.assertEqual(len(list(outdir.glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
