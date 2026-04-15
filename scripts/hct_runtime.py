from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

DEFAULTS_PATH_NAME = ".defaults.json"

CROP_DEFAULTS = {
    "blend_method": "max",
    "expected_plants": 0,
    "max_display": 1100,
    "max_frames": 180,
}

ANNOTATE_DEFAULTS = {
    "images_per_plant": 5,
    "display_scale": 2,
}

MASK_DEFAULTS = {
    "sigma": 4,
    "augment": True,
    "augmentations_per_image": 4,
    "aug_max_rotate": 15.0,
    "aug_min_scale": 0.9,
    "aug_max_scale": 1.1,
    "aug_max_shift": 0.05,
    "aug_hflip_prob": 0.5,
    "aug_vflip_prob": 0.2,
    "aug_brightness_alpha_min": 0.9,
    "aug_brightness_alpha_max": 1.1,
    "aug_brightness_beta_min": -10.0,
    "aug_brightness_beta_max": 10.0,
    "aug_seed": 0,
}

TRAIN_DEFAULTS = {
    "img_size": 128,
    "batch_size": 8,
    "epochs": 30,
    "learning_rate": 1e-4,
    "val_split": 0.2,
    "patience": 10,
    "train_seed": 1337,
    "wmse_alpha": 50.0,
    "dice_weight": 0.5,
    "fn_weight": 2.0,
}

MEASURE_DEFAULTS = {
    "max_frames": 0,
    "num_tips": 2,
    "min_dist": 20,
    "interval_min": 30,
}

EXPORT_DEFAULTS = {
    "interval_min": 30,
    "genotype_names": {idx: f"G{idx}" for idx in range(10)},
}


@dataclass
class DatasetInfo:
    dataset_id: str
    dataset_dir: Path
    raw_dir: Path
    raw_frame_count: int
    crop_path: Path
    annotation_path: Path
    output_dir: Path
    has_crops: bool
    has_annotations: bool
    has_measurements: bool
    plants: list[dict[str, Any]]
    frames: list[str]
    annotations: list[dict[str, Any]]

    @property
    def plant_count(self) -> int:
        return len(self.plants)

    @property
    def annotation_count(self) -> int:
        return len(self.annotations)

    @property
    def annotation_target(self) -> int:
        if not self.has_annotations:
            return 0
        data = load_json(self.annotation_path)
        return int(data.get("images_per_folder", data.get("images_per_plant", 0)) or 0)

    @property
    def fully_annotated_plants(self) -> int:
        if not self.plants or not self.annotations:
            return 0
        target = self.annotation_target
        counts: dict[str, int] = {}
        for ann in self.annotations:
            uid = str(ann.get("crop_uid") or ann.get("plant_id") or "")
            if uid:
                counts[uid] = counts.get(uid, 0) + 1
        full = 0
        for plant in self.plants:
            uid = str(plant.get("crop_uid") or plant.get("id") or "")
            if target > 0:
                if counts.get(uid, 0) >= target:
                    full += 1
            elif counts.get(uid, 0) > 0:
                full += 1
        return full


@dataclass
class ModelInfo:
    name: str
    model_dir: Path
    model_path: Path
    training_info_path: Path | None
    val_dice: float | None
    datasets: list[str]
    created: str | None


def repo_root() -> Path:
    configured = os.environ.get("HCT_BASE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    return repo_root() / "data"


def datasets_dir() -> Path:
    return data_dir() / "datasets"


def training_dir() -> Path:
    return data_dir() / "training"


def models_dir() -> Path:
    return data_dir() / "models"


def exports_dir() -> Path:
    return data_dir() / "exports"


def defaults_path() -> Path:
    return data_dir() / DEFAULTS_PATH_NAME


def dataset_dir(dataset_id: str) -> Path:
    return datasets_dir() / dataset_id


def dataset_raw_dir(dataset_id: str) -> Path:
    return dataset_dir(dataset_id) / "raw"


def dataset_crops_path(dataset_id: str) -> Path:
    return dataset_dir(dataset_id) / "crops.json"


def dataset_annotations_path(dataset_id: str) -> Path:
    return dataset_dir(dataset_id) / "annotations.json"


def dataset_output_dir(dataset_id: str) -> Path:
    return dataset_dir(dataset_id) / "output"


def dataset_measurements_path(dataset_id: str) -> Path:
    return dataset_output_dir(dataset_id) / "tip_distances.csv"


def ensure_layout() -> None:
    for path in (data_dir(), datasets_dir(), training_dir(), models_dir(), exports_dir()):
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def image_files(raw_dir: Path, max_frames: int = 0) -> list[Path]:
    if not raw_dir.exists():
        return []
    paths = sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    if max_frames > 0:
        return paths[:max_frames]
    return paths


def read_defaults() -> dict[str, Any]:
    path = defaults_path()
    if not path.exists():
        return {}
    data = load_json(path, {})
    if not isinstance(data, dict):
        return {}
    return data


def write_defaults(values: dict[str, Any]) -> None:
    current = read_defaults()
    current.update(values)
    write_json(defaults_path(), current)


def get_default(key: str, fallback: Any = None) -> Any:
    return read_defaults().get(key, fallback)


def discover_datasets() -> list[DatasetInfo]:
    ensure_layout()
    infos: list[DatasetInfo] = []
    for ds_dir in sorted(path for path in datasets_dir().iterdir() if path.is_dir()):
        ds_id = ds_dir.name
        raw_dir = ds_dir / "raw"
        crop_path = ds_dir / "crops.json"
        annotation_path = ds_dir / "annotations.json"
        output_dir = ds_dir / "output"
        crop_data = load_json(crop_path, {}) if crop_path.exists() else {}
        annotation_data = load_json(annotation_path, {}) if annotation_path.exists() else {}
        frames = crop_data.get("frames", []) if isinstance(crop_data, dict) else []
        plants = crop_data.get("plants", []) if isinstance(crop_data, dict) else []
        annotations = annotation_data.get("annotations", []) if isinstance(annotation_data, dict) else []
        infos.append(
            DatasetInfo(
                dataset_id=ds_id,
                dataset_dir=ds_dir,
                raw_dir=raw_dir,
                raw_frame_count=len(image_files(raw_dir)),
                crop_path=crop_path,
                annotation_path=annotation_path,
                output_dir=output_dir,
                has_crops=crop_path.exists(),
                has_annotations=annotation_path.exists(),
                has_measurements=(output_dir / "tip_distances.csv").exists(),
                plants=plants if isinstance(plants, list) else [],
                frames=frames if isinstance(frames, list) else [],
                annotations=annotations if isinstance(annotations, list) else [],
            )
        )
    return infos


def discover_models() -> list[ModelInfo]:
    ensure_layout()
    infos: list[ModelInfo] = []
    for model_dir in sorted(path for path in models_dir().iterdir() if path.is_dir()):
        model_path = model_dir / "best.keras"
        if not model_path.exists():
            continue
        info_path = model_dir / "training_info.json"
        info = load_json(info_path, {}) if info_path.exists() else {}
        val_dice = info.get("best_val_dice")
        datasets = []
        if isinstance(info.get("training_manifest"), dict):
            datasets = list(info["training_manifest"].get("datasets", []))
        elif isinstance(info.get("training_data"), dict):
            datasets = list(info["training_data"].get("datasets", []))
        created = info.get("trained") or info.get("generated") or info.get("created")
        infos.append(
            ModelInfo(
                name=model_dir.name,
                model_dir=model_dir,
                model_path=model_path,
                training_info_path=info_path if info_path.exists() else None,
                val_dice=float(val_dice) if isinstance(val_dice, (int, float)) else None,
                datasets=datasets,
                created=str(created) if created else None,
            )
        )
    return infos


def require_batch_value(value: Any, flag_name: str) -> Any:
    if value is None:
        raise SystemExit(f"--batch requires {flag_name}")
    return value


def prompt_input(message: str) -> str:
    return input(message).strip()


def prompt_with_default(
    message: str,
    default: Any | None = None,
    parser: Callable[[str], Any] | None = None,
    allow_empty: bool = False,
) -> Any:
    suffix = f" [{default}]" if default not in (None, "") else ""
    while True:
        raw = prompt_input(f"{message}{suffix} > ")
        if raw == "":
            if default not in (None, ""):
                return default
            if allow_empty:
                return ""
            print("A value is required.")
            continue
        if parser is None:
            return raw
        try:
            return parser(raw)
        except Exception as exc:  # pragma: no cover - interactive feedback
            print(f"Invalid value: {exc}")


def prompt_yes_no(message: str, default: bool = False) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        raw = prompt_input(f"{message} [{default_text}] > ").lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Enter y or n.")


def prompt_select_one(
    title: str,
    items: Sequence[Any],
    render: Callable[[Any], str],
    default_index: int | None = None,
) -> Any:
    if not items:
        raise SystemExit(f"No choices available for {title.lower()}.")
    print(title)
    for idx, item in enumerate(items, start=1):
        marker = " (default)" if default_index == idx - 1 else ""
        print(f"  [{idx}] {render(item)}{marker}")
    while True:
        raw = prompt_input("Which one? > ")
        if raw == "" and default_index is not None:
            return items[default_index]
        try:
            index = int(raw)
        except ValueError:
            print("Enter the number of the choice.")
            continue
        if 1 <= index <= len(items):
            return items[index - 1]
        print("Choice out of range.")


def parse_csv_items(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def prompt_select_many(
    title: str,
    items: Sequence[Any],
    render: Callable[[Any], str],
    default_indices: Sequence[int] | None = None,
) -> list[Any]:
    if not items:
        raise SystemExit(f"No choices available for {title.lower()}.")
    print(title)
    for idx, item in enumerate(items, start=1):
        default_marker = ""
        if default_indices and idx - 1 in set(default_indices):
            default_marker = " (default)"
        print(f"  [{idx}] {render(item)}{default_marker}")
    default_text = ""
    if default_indices:
        default_text = ",".join(str(i + 1) for i in default_indices)
    while True:
        suffix = f" [{default_text}]" if default_text else ""
        raw = prompt_input(f"Which ones? (comma-separated){suffix} > ")
        if raw == "" and default_indices:
            return [items[i] for i in default_indices]
        parts = parse_csv_items(raw)
        if not parts:
            print("Select at least one option.")
            continue
        try:
            indices = [int(part) for part in parts]
        except ValueError:
            print("Enter comma-separated numbers.")
            continue
        if any(index < 1 or index > len(items) for index in indices):
            print("Choice out of range.")
            continue
        deduped: list[Any] = []
        seen = set()
        for index in indices:
            if index not in seen:
                deduped.append(items[index - 1])
                seen.add(index)
        return deduped


def parse_dataset_ids_arg(text: str) -> list[str]:
    items = parse_csv_items(text)
    if not items:
        raise argparse.ArgumentTypeError("expected one or more dataset ids")
    return items


def parse_int_csv(text: str) -> list[int]:
    values = []
    for part in parse_csv_items(text):
        values.append(int(part))
    return values


def parse_name_mapping(text: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for item in parse_csv_items(text):
        if "=" not in item:
            raise argparse.ArgumentTypeError("expected genotype mappings like 1=M82,2=Penelli")
        key, value = item.split("=", 1)
        mapping[int(key.strip())] = value.strip()
    if not mapping:
        raise argparse.ArgumentTypeError("expected at least one genotype mapping")
    return mapping


def parse_bool_flag(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def parse_model_name(name: str) -> ModelInfo:
    matches = [model for model in discover_models() if model.name == name]
    if not matches:
        raise SystemExit(f"Model not found: {name}")
    return matches[0]


def model_name_from_manifest(manifest: dict[str, Any]) -> str:
    datasets = list(manifest.get("datasets", []))
    if not datasets:
        raise SystemExit("Training manifest is missing dataset provenance.")
    joined = "+".join(datasets)
    stamp = datetime.now().strftime("%Y%m%d")
    return f"{joined}_{stamp}"


def count_tip_distance_rows(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    plant_ids = set()
    frames = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            plant_ids.add(row.get("plant_id", ""))
            frames.add(row.get("frame_index", ""))
    return len({p for p in plant_ids if p}), len({f for f in frames if f != ""})


def save_defaults_after_success(**values: Any) -> None:
    filtered = {key: value for key, value in values.items() if value is not None}
    if filtered:
        write_defaults(filtered)


def prompt_or_batch(
    *,
    batch: bool,
    value: Any,
    flag_name: str,
    prompt: str,
    default: Any | None = None,
    parser: Callable[[str], Any] | None = None,
    allow_empty: bool = False,
) -> Any:
    if value is not None:
        return value
    if batch:
        return require_batch_value(value, flag_name)
    return prompt_with_default(prompt, default=default, parser=parser, allow_empty=allow_empty)
