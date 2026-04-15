from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import hct_runtime as rt


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.prev_base = os.environ.get("HCT_BASE_DIR")
        os.environ["HCT_BASE_DIR"] = str(self.base)
        self.addCleanup(self.restore_env)
        rt.ensure_layout()

    def restore_env(self) -> None:
        if self.prev_base is None:
            os.environ.pop("HCT_BASE_DIR", None)
        else:
            os.environ["HCT_BASE_DIR"] = self.prev_base

    def write_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def test_discover_datasets_and_defaults(self) -> None:
        raw_dir = rt.dataset_raw_dir("F2_001")
        raw_dir.mkdir(parents=True)
        (raw_dir / "frame_0000.jpg").write_bytes(b"")
        self.write_json(
            rt.dataset_crops_path("F2_001"),
            {
                "dataset_id": "F2_001",
                "raw_dir": str(raw_dir),
                "frames": ["frame_0000.jpg"],
                "plants": [{"id": "g1_r01", "genotype": 1, "replicate": 1, "bbox": [0, 0, 1, 1], "crop_uid": "u1"}],
            },
        )
        infos = rt.discover_datasets()
        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0].dataset_id, "F2_001")

        rt.write_defaults({"last_dataset": "F2_001"})
        self.assertEqual(rt.get_default("last_dataset"), "F2_001")


if __name__ == "__main__":
    unittest.main()
