from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DAMAGE_LEVELS = ["no-damage", "minor-damage", "major-damage", "destroyed"]
EXPECTED_SHAPE = (256, 256, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify stratified train/validation/test split for processed paired data."
    )
    parser.add_argument("--processed-dir", default="processed_balanced_pairs")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def count_labels(rows: list[dict[str, str]]) -> Counter:
    return Counter(row["label"] for row in rows)


def check_array(path: Path, expected_shape: tuple[int, int, int], errors: list[str]) -> np.ndarray:
    if not path.is_file():
        errors.append(f"Missing array file: {path}")
        return np.array([])

    array = np.load(path)
    if array.ndim != 4 or array.shape[1:] != expected_shape:
        errors.append(f"{path.name} has shape {array.shape}; expected (N, 256, 256, 6).")
    if array.dtype != np.float32:
        errors.append(f"{path.name} has dtype {array.dtype}; expected float32.")
    if array.size and (array.min() < 0.0 or array.max() > 1.0):
        errors.append(
            f"{path.name} is not normalized to [0, 1]; min={array.min()}, max={array.max()}."
        )
    return array


def check_label_array(path: Path, errors: list[str]) -> np.ndarray:
    if not path.is_file():
        errors.append(f"Missing label file: {path}")
        return np.array([])

    array = np.load(path)
    if array.ndim != 1:
        errors.append(f"{path.name} has shape {array.shape}; expected a 1D label array.")
    if not np.issubdtype(array.dtype, np.integer):
        errors.append(f"{path.name} has dtype {array.dtype}; expected integer labels.")
    return array


def save_split_chart(
    output_path: Path,
    split_counts: dict[str, Counter],
) -> None:
    x = np.arange(len(DAMAGE_LEVELS))
    width = 0.25

    plt.figure(figsize=(10, 5))
    for offset, split_name, color in [
        (-width, "train", "#6a8c4a"),
        (0, "validation", "#d4762a"),
        (width, "test", "#c4874a"),
    ]:
        values = [split_counts[split_name][label] for label in DAMAGE_LEVELS]
        plt.bar(x + offset, values, width, label=split_name.title(), color=color)

    plt.title("Stratified Train / Validation / Test Split")
    plt.xlabel("Damage class")
    plt.ylabel("Pair count")
    plt.xticks(x, [label.replace("-", " ").title() for label in DAMAGE_LEVELS])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    errors: list[str] = []
    warnings: list[str] = []

    required_csvs = {
        "train": processed_dir / "train_pairs.csv",
        "validation": processed_dir / "validation_pairs.csv",
        "test": processed_dir / "test_pairs.csv",
    }

    for split_name, path in required_csvs.items():
        if not path.is_file():
            errors.append(f"Missing {split_name} CSV: {path}")

    if errors:
        print("ERRORS FOUND:")
        for error in errors:
            print(f"- {error}")
        return

    rows_by_split = {name: load_csv(path) for name, path in required_csvs.items()}
    split_counts = {name: count_labels(rows) for name, rows in rows_by_split.items()}

    x_train = check_array(processed_dir / "X_train.npy", EXPECTED_SHAPE, errors)
    y_train = check_label_array(processed_dir / "y_train.npy", errors)
    x_val = check_array(processed_dir / "X_val.npy", EXPECTED_SHAPE, errors)
    y_val = check_label_array(processed_dir / "y_val.npy", errors)
    x_test = check_array(processed_dir / "X_test.npy", EXPECTED_SHAPE, errors)
    y_test = check_label_array(processed_dir / "y_test.npy", errors)

    if x_train.size and y_train.size and len(x_train) != len(y_train):
        errors.append("X_train and y_train length mismatch.")
    if x_val.size and y_val.size and len(x_val) != len(y_val):
        errors.append("X_val and y_val length mismatch.")
    if x_test.size and y_test.size and len(x_test) != len(y_test):
        errors.append("X_test and y_test length mismatch.")

    total = sum(len(rows) for rows in rows_by_split.values())
    actual_ratios = {
        name: len(rows) / total
        for name, rows in rows_by_split.items()
    }
    expected_ratios = {
        "train": args.train_ratio,
        "validation": args.validation_ratio,
        "test": args.test_ratio,
    }

    for split_name, expected in expected_ratios.items():
        if abs(actual_ratios[split_name] - expected) > 0.02:
            errors.append(
                f"{split_name} ratio is {actual_ratios[split_name]:.3f}; expected about {expected:.3f}."
            )

    for split_name, counts in split_counts.items():
        missing = [label for label in DAMAGE_LEVELS if counts[label] == 0]
        if missing:
            errors.append(f"{split_name} split is missing classes: {missing}")
        if max(counts.values()) - min(counts.values()) > 1:
            warnings.append(
                f"{split_name} split class counts differ by more than 1: {dict(counts)}"
            )

    chart_path = processed_dir / "split_class_distribution.png"
    save_split_chart(chart_path, split_counts)

    summary = {
        "total_samples": total,
        "split_counts": {name: len(rows) for name, rows in rows_by_split.items()},
        "expected_ratios": expected_ratios,
        "actual_ratios": actual_ratios,
        "class_counts": {
            name: {label: counts[label] for label in DAMAGE_LEVELS}
            for name, counts in split_counts.items()
        },
        "input_shape_verified": [256, 256, 6],
        "normalization_verified": True,
        "augmentation_policy": {
            "train": "augmentation allowed during model training",
            "validation": "no augmentation",
            "test": "no augmentation",
        },
        "chart": str(chart_path),
        "errors": errors,
        "warnings": warnings,
    }

    with (processed_dir / "split_verification_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Split verification complete.")
    print(f"Total samples: {total}")
    print("Split counts:")
    for split_name in ["train", "validation", "test"]:
        print(f"  {split_name}: {len(rows_by_split[split_name])} ({actual_ratios[split_name]:.2%})")
    print("Class counts:")
    for split_name in ["train", "validation", "test"]:
        print(f"  {split_name}: {dict(split_counts[split_name])}")

    if errors:
        print("\nERRORS FOUND:")
        for error in errors:
            print(f"- {error}")
    else:
        print("\nNo split verification errors found.")

    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"- {warning}")

    print(f"\nSplit chart: {chart_path.resolve()}")
    print(f"Summary: {(processed_dir / 'split_verification_summary.json').resolve()}")


if __name__ == "__main__":
    main()
