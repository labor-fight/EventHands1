"""Archive safety tests run entirely in temporary directories."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.xcl.archive_legacy import archive, apply_deletion, digest, scope


class ArchiveSafetyTests(unittest.TestCase):
    def fixture(self, root):
        for name, content in {"outputs/xag/protocol.json": "{}", "outputs/xag/sources/seq/data.npy": "KEEP",
                              "outputs/xag/source_manifest.json": "{}", "outputs/xag/runs/old/weight.ckpt": "DELETE_LATER",
                              "outputs/xag/reviews/c1_deep_audit_20261017/evidence.json": "{}",
                              "outputs/xag_paired/20261018/weight.ckpt": "KEEP",
                              "tools/xag/train_audit.py": "value = 1\n",
                              "semkine/train.py": "from tools.xag.train_audit import value\n"}.items():
            filename = root / name; filename.parent.mkdir(parents=True, exist_ok=True); filename.write_text(content)

    def test_default_archive_deletes_nothing_and_preserves_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root)
            destination = root / "archive"
            report = archive(root, destination)
            self.assertTrue((root / "outputs/xag/runs/old/weight.ckpt").is_file())
            self.assertTrue((destination / "files/outputs/xag/protocol.json").is_file())
            self.assertFalse((destination / "files/outputs/xag/runs/old/weight.ckpt").exists())
            self.assertTrue(report["dependency_scan"]["references"])
            removed = {p.relative_to(root).as_posix() for p in scope(root)}
            self.assertNotIn("outputs/xag/sources", removed)
            self.assertNotIn("outputs/xag_paired", removed)
            self.assertNotIn("outputs/xag/reviews/c1_deep_audit_20261017", removed)

    def test_apply_rejects_absent_migration_without_touching_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root); destination = root / "archive"
            archive(root, destination)
            receipt = root / "migration.json"; receipt.write_text(json.dumps({"status": "PASS"}))
            with self.assertRaisesRegex(ValueError, "exact dependency scan"):
                apply_deletion(root, destination / "inventory.json", receipt)
            self.assertTrue((root / "outputs/xag/runs/old/weight.ckpt").is_file())

    def test_symlinks_are_inventoried_without_traversing_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root)
            link = root / "outputs/xag/runs/preserved_link"
            link.symlink_to(root / "outputs/xag_paired", target_is_directory=True)
            report = archive(root, root / "archive")
            names = {row["path"] for row in report["records"]}
            self.assertIn("outputs/xag/runs/preserved_link", names)
            self.assertFalse(any("preserved_link/" in name for name in names))

    def test_changed_legacy_file_rejected_after_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root); destination = root / "archive"
            report = archive(root, destination); manifest = destination / "inventory.json"
            migration = {"status": "PASS", "dependency_scan_sha256": report["dependency_scan"]["sha256"],
                         "inventory_sha256": digest(manifest), "new_xcl_runtime_independently_verified": True}
            receipt = root / "migration.json"; receipt.write_text(json.dumps(migration))
            (root / "outputs/xag/runs/old/weight.ckpt").write_text("CHANGED")
            with self.assertRaisesRegex(ValueError, "changed since inventory"):
                apply_deletion(root, manifest, receipt)


if __name__ == "__main__":
    unittest.main()
