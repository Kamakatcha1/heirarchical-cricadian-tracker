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

    def test_load_json_warns_and_returns_default_for_malformed_file(self) -> None:
        broken = self.base / "data" / "broken.json"
        broken.write_text("{not valid json")
        self.assertEqual(rt.load_json(broken, {"ok": True}), {"ok": True})

    def test_parse_name_mapping_and_annotation_remap(self) -> None:
        mapping = rt.parse_name_mapping("0=WT,4=Mut")
        self.assertEqual(mapping, {0: "WT", 4: "Mut"})

        plants = [
            {"id": "g0_r01", "genotype": 0, "replicate": 1, "bbox": [1, 2, 3, 4], "crop_uid": "u1"},
            {"id": "g4_r01", "genotype": 4, "replicate": 1, "bbox": [5, 6, 7, 8], "crop_uid": "u2"},
        ]
        annotations = [
            {"crop_uid": "u1", "plant_id": "old", "frame_index": 0, "genotype": 9, "replicate": 9, "crop_bbox": [1, 2, 3, 4]},
            {"crop_bbox": [5, 6, 7, 8], "plant_id": "old2", "frame_index": 1, "genotype": 4, "replicate": 2},
        ]
        remapped, dropped = rt.remap_annotations_to_plants(plants, annotations)
        self.assertEqual(dropped, 0)
        self.assertEqual([ann["plant_id"] for ann in remapped], ["g0_r01", "g4_r01"])
        self.assertEqual([ann["crop_uid"] for ann in remapped], ["u1", "u2"])

    def test_discover_measurement_runs(self) -> None:
        raw_dir = rt.dataset_raw_dir("F2_002")
        raw_dir.mkdir(parents=True)
        self.write_json(
            rt.dataset_crops_path("F2_002"),
            {
                "dataset_id": "F2_002",
                "raw_dir": str(raw_dir),
                "frames": ["frame_0000.jpg"],
                "plants": [{"id": "g0_r01", "genotype": 0, "replicate": 1, "bbox": [0, 0, 1, 1], "crop_uid": "u1"}],
            },
        )
        run_dir = rt.dataset_measurement_run_dir("F2_002", "model_a")
        run_dir.mkdir(parents=True)
        (run_dir / "tip_distances.csv").write_text(
            "dataset_id,plant_id,label,genotype,replicate,frame_index,frame_filename,tip_distance_px\n"
            "F2_002,g0_r01,g0_1,0,1,0,frame_0000.jpg,1.0\n"
        )
        self.write_json(run_dir / "measure_info.json", {"model_name": "model_a", "measured": "2026-04-16T12:00:00"})

        infos = rt.discover_datasets()
        self.assertTrue(infos[0].has_measurements)

        runs = rt.discover_measurement_runs("F2_002")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].name, "model_a")
        self.assertEqual(runs[0].model_name, "model_a")


if __name__ == "__main__":
    unittest.main()
