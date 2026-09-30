from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from shared.path_utils import (  # noqa: E402
    ensure_portable_child_namespace,
    normalize_path_text,
    portable_path_key,
    safe_path_component,
    truncate_to_utf16_units,
    windows_utf16_units,
)


class PortablePathTextTests(unittest.TestCase):
    def test_windows_path_length_counts_utf16_code_units(self):
        emoji = "\N{GRINNING FACE}"

        self.assertEqual(windows_utf16_units(f"a{emoji}b"), 4)
        self.assertEqual(
            truncate_to_utf16_units(f"ab{emoji}cd", 4),
            f"ab{emoji}",
        )
        self.assertEqual(
            truncate_to_utf16_units(f"ab{emoji}cd", 3),
            "ab",
        )

    def test_safe_component_rejects_a_common_filesystem_unit_overflow(self):
        emoji = "\N{GRINNING FACE}"

        self.assertEqual(
            safe_path_component(emoji * 63, "name"),
            emoji * 63,
        )
        with self.assertRaisesRegex(ValueError, "portable filesystem name"):
            safe_path_component(emoji * 64, "name")

        with self.assertRaisesRegex(ValueError, "portable filesystem name"):
            safe_path_component("bad\ud800name", "name")

    def test_safe_components_are_normalized_to_nfc(self):
        decomposed = "cafe\N{COMBINING ACUTE ACCENT}"
        composed = "caf\N{LATIN SMALL LETTER E WITH ACUTE}"

        self.assertEqual(
            safe_path_component(decomposed, "name"),
            composed,
        )
        self.assertEqual(normalize_path_text(decomposed), composed)

    def test_collision_key_folds_case_and_unicode_form(self):
        self.assertEqual(
            portable_path_key("CAFÉ"),
            portable_path_key("cafe\N{COMBINING ACUTE ACCENT}"),
        )

    def test_existing_namespace_rejects_a_case_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            (parent / "Baseline").mkdir()

            with self.assertRaisesRegex(ValueError, "existing namespace"):
                ensure_portable_child_namespace(
                    parent,
                    "baseline",
                    "run_id",
                )

    def test_canonical_alias_is_allowed_only_when_it_is_the_same_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            decomposed = parent / "cafe\N{COMBINING ACUTE ACCENT}"
            composed_name = "caf\N{LATIN SMALL LETTER E WITH ACUTE}"
            decomposed.mkdir()
            composed = parent / composed_name

            if composed.exists() and decomposed.samefile(composed):
                self.assertEqual(
                    ensure_portable_child_namespace(
                        parent,
                        composed_name,
                        "run_id",
                    ),
                    composed_name,
                )
            else:
                with self.assertRaisesRegex(
                    ValueError,
                    "existing namespace",
                ):
                    ensure_portable_child_namespace(
                        parent,
                        composed_name,
                        "run_id",
                    )


if __name__ == "__main__":
    unittest.main()
