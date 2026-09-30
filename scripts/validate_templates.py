"""Validate all template JSON files against templates/template_schema.json."""

import json
import sys
from pathlib import Path

import jsonschema

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shared.cli import configure_cli_streams  # noqa: E402

SCHEMA_PATH = REPO_ROOT / "templates" / "template_schema.json"
TEMPLATES_DIR = REPO_ROOT / "templates"


def main() -> int:
    configure_cli_streams()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)

    passed, failed = 0, 0
    for template_file in sorted(TEMPLATES_DIR.rglob("*.json")):
        if template_file.name == "template_schema.json":
            continue
        template = json.loads(template_file.read_text(encoding="utf-8"))
        errors = list(validator.iter_errors(template))
        rel = template_file.relative_to(REPO_ROOT)
        if errors:
            failed += 1
            print(f"FAIL  {rel}")
            for e in errors:
                print(f"      {e.json_path}: {e.message}")
        else:
            passed += 1
            print(f"PASS  {rel}")

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
