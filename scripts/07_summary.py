from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hct_runtime as rt


USE_STD = True
COLORS = {"M82": "tab:blue", "Penelli": "tab:orange", "Pimpi": "tab:green"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate summary plots from a Biodare detrended CSV.")
    parser.add_argument("--batch", action="store_true", help="Disable prompts and require explicit CLI values.")
    parser.add_argument("--input", help="Specific detrended CSV filename or path.")
    parser.add_argument("--exclude-samples", help='Comma-separated sample numbers like "3,15,22".')
    return parser


def parse_biodare_csv(path: Path, exclude_samples: list[int]) -> tuple[np.ndarray, dict[str, list[np.ndarray]]]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    label_row = None
    label_idx = None
    for idx, row in enumerate(rows):
        if row and row[0].strip().lower().startswith("label"):
            label_row = row
            label_idx = idx
            break
    if label_row is None or label_idx is None:
        raise ValueError("Could not find 'Label:' row in CSV.")

    labels = label_row[1:]
    genotype_names = []
    col_indices = []
    for col_offset, label in enumerate(labels):
        label = label.strip()
        if not label:
            continue
        match = re.match(r"(\d+)\.", label)
        sample_num = int(match.group(1)) if match else col_offset + 1
        if sample_num in exclude_samples:
            continue
        col_indices.append(col_offset + 1)
        m = re.search(r"\]\s*(.+)$", label)
        genotype_names.append(m.group(1).strip() if m else label)

    times = []
    traces: list[list[float]] = [[] for _ in genotype_names]
    for row in rows[label_idx + 1 :]:
        if not row or not row[0].strip():
            continue
        try:
            t = float(row[0])
        except ValueError:
            continue
        times.append(t)
        for idx, col_index in enumerate(col_indices):
            cell = row[col_index].strip() if col_index < len(row) else ""
            try:
                traces[idx].append(float(cell))
            except ValueError:
                traces[idx].append(float("nan"))

    grouped: dict[str, list[np.ndarray]] = {}
    for name, values in zip(genotype_names, traces):
        grouped.setdefault(name, []).append(np.array(values))
    return np.array(times), grouped


def summarize(replicates: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.array(replicates)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    if USE_STD:
        return mean, std
    n = np.sum(np.isfinite(stacked), axis=0).astype(float)
    n[n == 0] = 1
    return mean, std / np.sqrt(n)


def plot_one(name: str, hours: np.ndarray, mean: np.ndarray, spread: np.ndarray, color: str, n_reps: int, out_dir: Path) -> None:
    spread_label = "STD" if USE_STD else "SEM"
    plt.figure(figsize=(9, 4))
    plt.plot(hours, mean, color=color, linewidth=2, label=name)
    plt.fill_between(hours, mean - spread, mean + spread, color=color, alpha=0.2)
    plt.xlabel("Hours")
    plt.ylabel("Detrended value")
    plt.title(f"{name}: mean +/- {spread_label} (n={n_reps})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / f"{name.lower()}_summary.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  {out_path.name}")


def plot_all(all_data: dict[str, tuple[np.ndarray, np.ndarray, int]], hours: np.ndarray, out_dir: Path) -> None:
    spread_label = "STD" if USE_STD else "SEM"
    plt.figure(figsize=(10, 5))
    for name, (mean, spread, n_reps) in all_data.items():
        color = COLORS.get(name, "tab:gray")
        plt.plot(hours, mean, color=color, linewidth=2, label=f"{name} (n={n_reps})")
        plt.fill_between(hours, mean - spread, mean + spread, color=color, alpha=0.2)
    plt.xlabel("Hours")
    plt.ylabel("Detrended value")
    plt.title(f"All genotypes: mean +/- {spread_label}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / "all_genotypes_summary.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  {out_path.name}")


def resolve_input(args: argparse.Namespace) -> Path:
    exports_root = rt.exports_dir()
    if args.input:
        candidate = Path(args.input)
        if not candidate.is_absolute():
            candidate = exports_root / args.input
        if not candidate.exists():
            raise SystemExit(f"Input CSV not found: {candidate}")
        return candidate

    candidates = sorted(path for path in exports_root.glob("*.csv") if "detrended" in path.name.lower())
    if not candidates:
        raise FileNotFoundError(f"No detrended CSV found in {exports_root}")
    if args.batch:
        raise SystemExit("--batch requires --input when multiple or implicit detrended CSV selection would be needed.")
    if len(candidates) == 1:
        return candidates[0]
    return rt.prompt_select_one("Available detrended CSV files:", candidates, lambda path: path.name)


def main() -> None:
    args = build_parser().parse_args()
    rt.ensure_layout()

    csv_path = resolve_input(args)
    exclude_samples = rt.parse_int_csv(args.exclude_samples) if args.exclude_samples else []

    print(f"Input: {csv_path.name}")
    if exclude_samples:
        print(f"Excluding samples: {sorted(exclude_samples)}")

    summary_dir = rt.exports_dir() / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    hours, grouped = parse_biodare_csv(csv_path, exclude_samples)
    print(f"Timepoints: {len(hours)} ({hours[0]:.1f}h - {hours[-1]:.1f}h)")

    all_data: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for name, reps in grouped.items():
        n_reps = len(reps)
        print(f"\n  {name}: {n_reps} replicates")
        mean, spread = summarize(reps)
        all_data[name] = (mean, spread, n_reps)
        plot_one(name, hours, mean, spread, COLORS.get(name, "tab:gray"), n_reps, summary_dir)

    print()
    plot_all(all_data, hours, summary_dir)
    print(f"\nSaved: {summary_dir}")


if __name__ == "__main__":
    main()
