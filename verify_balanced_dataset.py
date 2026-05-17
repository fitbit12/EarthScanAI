from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


DAMAGE_LEVELS = ["no-damage", "minor-damage", "major-damage", "destroyed"]
CLASS_DIR_NAMES = {
    "no-damage": "no_damage",
    "minor-damage": "minor_damage",
    "major-damage": "major_damage",
    "destroyed": "destroyed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and document the balanced paired xView2 subset."
    )
    parser.add_argument(
        "--pairs-csv",
        default="dataset_work/all_pairs_index.csv",
        help="Pre/post pair index CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset_work",
        help="Folder where verification files and charts will be saved.",
    )
    return parser.parse_args()


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def pair_id_from_name(filename: str) -> str:
    return filename.replace("_pre_disaster.png", "").replace("_post_disaster.png", "")


def write_class_csvs(rows: list[dict[str, str]], output_dir: Path) -> dict[str, str]:
    class_dir = output_dir / "balanced_class_indexes"
    class_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    if not rows:
        return outputs

    fieldnames = list(rows[0].keys())
    for label in DAMAGE_LEVELS:
        class_rows = [row for row in rows if row["label"] == label]
        output_path = class_dir / f"{CLASS_DIR_NAMES[label]}.csv"
        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(class_rows)
        outputs[label] = str(output_path)

    return outputs


def save_bar_chart(
    title: str,
    labels: list[str],
    values: list[int],
    output_path: Path,
    color: str = "#6a8c4a",
) -> None:
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, color=color)
    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("Pair count")
    plt.xticks(rotation=25, ha="right")
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    pairs_csv = Path(args.pairs_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pairs_csv.is_file():
        raise FileNotFoundError(f"Balanced pairs CSV not found: {pairs_csv}")

    rows = read_pairs(pairs_csv)
    errors: list[str] = []
    warnings: list[str] = []

    pair_ids = [row["pair_id"] for row in rows]
    duplicates = [pair_id for pair_id, count in Counter(pair_ids).items() if count > 1]
    if duplicates:
        errors.append(f"Duplicate pair IDs found: {duplicates[:10]}")

    class_counts = Counter(row["label"] for row in rows)
    disaster_type_counts = Counter(row["disaster_type"] for row in rows)
    disaster_counts = Counter(row["disaster"] for row in rows)

    expected_count = min(class_counts.values()) if class_counts else 0
    for label in DAMAGE_LEVELS:
        if class_counts[label] != expected_count:
            warnings.append(
                f"Class {label} has {class_counts[label]} samples; expected {expected_count}."
            )

    for index, row in enumerate(rows, start=2):
        pre_path = Path(row["pre_image"])
        post_path = Path(row["post_image"])
        pre_json = Path(row["pre_json"])
        post_json = Path(row["post_json"])

        for label, path in [
            ("pre image", pre_path),
            ("post image", post_path),
            ("pre JSON", pre_json),
            ("post JSON", post_json),
        ]:
            if not path.is_file():
                errors.append(f"Missing {label} on CSV row {index}: {path}")

        if pre_path.is_file() and pair_id_from_name(pre_path.name) != row["pair_id"]:
            errors.append(f"Pre image pair ID mismatch on CSV row {index}: {pre_path.name}")
        if post_path.is_file() and pair_id_from_name(post_path.name) != row["pair_id"]:
            errors.append(f"Post image pair ID mismatch on CSV row {index}: {post_path.name}")

    if len(disaster_type_counts) < 2:
        warnings.append("Balanced subset contains only one disaster type.")

    class_csv_outputs = write_class_csvs(rows, output_dir)

    class_chart = output_dir / "balanced_class_distribution.png"
    save_bar_chart(
        "Balanced Damage Class Distribution",
        [label.replace("-", " ").title() for label in DAMAGE_LEVELS],
        [class_counts[label] for label in DAMAGE_LEVELS],
        class_chart,
        color="#6a8c4a",
    )

    disaster_type_chart = output_dir / "balanced_disaster_type_distribution.png"
    disaster_type_labels = [item[0] for item in disaster_type_counts.most_common()]
    disaster_type_values = [item[1] for item in disaster_type_counts.most_common()]
    save_bar_chart(
        "Disaster Type Mix in Balanced Subset",
        disaster_type_labels,
        disaster_type_values,
        disaster_type_chart,
        color="#d4762a",
    )

    summary = {
        "balanced_pairs_csv": str(pairs_csv),
        "total_pairs": len(rows),
        "expected_pairs_per_class": expected_count,
        "class_counts": dict(class_counts),
        "disaster_type_counts": dict(disaster_type_counts),
        "disaster_counts": dict(disaster_counts),
        "class_csv_outputs": class_csv_outputs,
        "charts": {
            "class_distribution": str(class_chart),
            "disaster_type_distribution": str(disaster_type_chart),
        },
        "errors": errors,
        "warnings": warnings,
    }

    with (output_dir / "balanced_dataset_verification.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Balanced dataset verification complete.")
    print(f"Total pairs: {len(rows)}")
    print("Class counts:")
    for label in DAMAGE_LEVELS:
        print(f"  {label}: {class_counts[label]}")
    print("Disaster type counts:")
    for label, count in disaster_type_counts.most_common():
        print(f"  {label}: {count}")

    if errors:
        print("\nERRORS FOUND:")
        for error in errors:
            print(f"- {error}")
    else:
        print("\nNo balanced dataset errors found.")

    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"- {warning}")

    print(f"\nVerification summary: {(output_dir / 'balanced_dataset_verification.json').resolve()}")
    print(f"Class chart: {class_chart.resolve()}")
    print(f"Disaster type chart: {disaster_type_chart.resolve()}")


if __name__ == "__main__":
    main()
