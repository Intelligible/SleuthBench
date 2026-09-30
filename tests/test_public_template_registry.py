from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phenomena import PHENOMENA, TEMPLATE_TO_PHENOMENON  # noqa: E402

PUBLIC_TEMPLATE_IDS = {
    "dq_bad_row_indicator_v0",
    "dq_bad_row_indicator_v1",
    "dq_categorical_target_outlier_v0",
    "dq_conditional_bad_rows_v0",
    "dq_group_evidence_underpowered_v0",
    "dq_missing_label_semantic_v0",
    "dq_missing_label_target_dependent_v0",
    "dq_unreliable_feature_v0",
    "fc_noise_feature_v0",
    "fc_interaction_direction_v0",
    "fc_interaction_dominant_v0",
    "fc_pairwise_antagonistic_reversal_v0",
    "fc_pairwise_compensatory_reversal_v0",
    "fc_pairwise_positive_synergy_v0",
    "fc_monotone_classify_v0",
    "fc_nonmonotone_peak_v0",
    "fc_threshold_value_v0",
}


class PublicTemplateRegistryTests(unittest.TestCase):
    def test_public_templates_and_registries_are_consistent(self):
        template_paths = sorted(
            path
            for path in (REPO_ROOT / "templates").rglob("*.json")
            if path.name != "template_schema.json"
        )
        templates = [
            (path, json.loads(path.read_text(encoding="utf-8")))
            for path in template_paths
        ]
        template_ids = [template["template_id"] for _, template in templates]

        self.assertEqual(
            len(template_ids),
            len(set(template_ids)),
            "public template IDs must be unique",
        )
        self.assertSetEqual(set(template_ids), PUBLIC_TEMPLATE_IDS)
        self.assertEqual(
            len(PUBLIC_TEMPLATE_IDS),
            17,
            "the public benchmark ships exactly 17 templates",
        )
        self.assertSetEqual(
            set(TEMPLATE_TO_PHENOMENON),
            PUBLIC_TEMPLATE_IDS,
            "the answer registry must match the public template set exactly",
        )

        referenced_injectors: set[str] = set()
        for path, template in templates:
            template_id = template["template_id"]
            injectors = {
                specification["injector"]
                for specification in template.get("phenomena", [])
            }
            relative_path = path.relative_to(REPO_ROOT)
            with self.subTest(template=template_id, path=str(relative_path)):
                self.assertTrue(injectors, "template must reference an injector")
                self.assertTrue(
                    injectors <= set(PHENOMENA),
                    f"unregistered injectors: {sorted(injectors - set(PHENOMENA))}",
                )
                self.assertIn(
                    TEMPLATE_TO_PHENOMENON[template_id].name,
                    injectors,
                    "the answer computer must belong to a referenced injector",
                )
            referenced_injectors.update(injectors)

        self.assertSetEqual(
            set(PHENOMENA),
            referenced_injectors,
            "every registered injector must be referenced by a public template",
        )
        self.assertEqual(
            len(PHENOMENA),
            14,
            "the public benchmark ships exactly 14 injectors",
        )


if __name__ == "__main__":
    unittest.main()
