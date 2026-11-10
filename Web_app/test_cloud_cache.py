"""Offline tests for cloud-side test tile extraction."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location("asset_cache", Path(__file__).with_name("asset_cache.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CloudCacheTests(unittest.TestCase):
    def make_zip(self, root, extra=None):
        path = root / "fixture.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("BEN_14k/BigEarthNet-S1/test/a.tif", b"sar")
            z.writestr("BEN_14k/BigEarthNet-S2/test/a.tif", b"optical")
            z.writestr("BEN_14k/BigEarthNet-S1/train/a.tif", b"train")
            if extra:
                z.writestr(extra, b"unsafe")
        return path

    def test_extract_and_reuse(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = self.make_zip(root)
            result = module.extract_test_tiles(path, root / "cache")
            self.assertEqual((result / "BigEarthNet-S1/test/a.tif").read_bytes(), b"sar")
            self.assertEqual((result / "BigEarthNet-S2/test/a.tif").read_bytes(), b"optical")
            self.assertFalse((result / "BigEarthNet-S1/train").exists())
            self.assertTrue((result / ".complete").exists())
            self.assertEqual(module.extract_test_tiles(path, root / "cache"), result)

    def test_reject_traversal(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = self.make_zip(root, "BigEarthNet-S1/test/../../escape.tif")
            with self.assertRaises(ValueError):
                module.extract_test_tiles(path, root / "cache")
            self.assertFalse((root / "escape.tif").exists())

    def test_interrupted_and_conflicting_cache(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = self.make_zip(root)
            result = module.extract_test_tiles(path, root / "cache")
            (result / ".complete").unlink()
            self.assertEqual(module.extract_test_tiles(path, root / "cache"), result)
            (result / ".complete").unlink()
            (result / "BigEarthNet-S1/test/a.tif").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                module.extract_test_tiles(path, root / "cache")
            self.assertEqual((result / "BigEarthNet-S1/test/a.tif").read_bytes(), b"changed")


if __name__ == "__main__":
    unittest.main()
