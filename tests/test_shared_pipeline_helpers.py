from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from shared.files import sha256_file  # noqa: E402
from shared.manifests import resolve_manifest_paths  # noqa: E402
from io_utils import remove_generated_instance_dir  # noqa: E402


class ManifestResolutionTests(unittest.TestCase):
    def test_explicit_empty_batch_does_not_fall_back_to_tree_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "dataset" / "seed_42" / "fake" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}", encoding="utf-8")

            self.assertEqual(resolve_manifest_paths(root, []), [])
            self.assertEqual(resolve_manifest_paths(root), [manifest])

    def test_explicit_batch_is_normalized_and_sorted(self):
        paths = [Path("z/manifest.json"), Path("a/manifest.json")]

        self.assertEqual(
            resolve_manifest_paths(Path("unused"), paths),
            list(reversed(paths)),
        )


class SharedFileTests(unittest.TestCase):
    def test_sha256_file_streams_the_expected_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.bin"
            payload = b"SleuthBench\n" * 200_000
            path.write_bytes(payload)

            self.assertEqual(
                sha256_file(path),
                hashlib.sha256(payload).hexdigest(),
            )


class GeneratedPathRemovalTests(unittest.TestCase):
    def test_remove_deletes_one_instance_without_recreating_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "instances"
            instance = root / "dataset" / "seed_42" / "fake"
            instance.mkdir(parents=True)
            (instance / "manifest.json").write_text("{}", encoding="utf-8")

            removed = remove_generated_instance_dir(instance, root)

            self.assertEqual(removed, instance)
            self.assertFalse(instance.exists())

    def test_remove_rejects_broad_and_outside_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = workspace / "instances"
            broad = root / "dataset"
            outside = workspace / "outside" / "seed_42" / "fake"
            broad.mkdir(parents=True)
            outside.mkdir(parents=True)
            (broad / "keep.txt").write_text("keep", encoding="utf-8")
            (outside / "keep.txt").write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "broad instance path"):
                remove_generated_instance_dir(broad, root)
            with self.assertRaisesRegex(ValueError, "under instances root"):
                remove_generated_instance_dir(outside, root)

            self.assertTrue((broad / "keep.txt").exists())
            self.assertTrue((outside / "keep.txt").exists())


if __name__ == "__main__":
    unittest.main()
