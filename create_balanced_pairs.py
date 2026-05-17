from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


DAMAGE_LEVELS = ["no-damage", "minor-damage", "major-damage", "destroyed"]
SEVERITY = {label: index for index, label in enumerate(DAMAGE_LEVELS)}
CLASS_DIR_NAMES = {
    "no-damage": "no_damage",
    "minor-damage": "minor_damage",
    "major-damage": "major_damage",
    "destroyed": "destroyed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a balanced pre/post pair index from the full xView2 dataset."
    )
    parser.add_argument(
        "--dataset-root",
        default="Full dataset",
        help="Root folder containing train/train/images and train/train/labels.",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset_work",
        help="Folder where the balanced pair index and summary will be saved.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help="Pairs per class. If omitted, uses the smallest available class count.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible balanced sampling.",
    )
    parser.add_argument(
        "--labeling-method",
        choices=["most-severe", "majority", "weighted-score"],
        default="weighted-score",
        help=(
            "How to assign a single damage label to each image tile. "
            "'most-severe' picks the highest damage found (original method). "
            "'majority' picks the most common damage type. "
            "'weighted-score' computes a severity-weighted proportion and bins it."
        ),
    )
    return parser.parse_args()


def pair_id_from_name(filename: str) -> str:
    return filename.replace("_pre_disaster.png", "").replace("_post_disaster.png", "")


def image_phase(filename: str) -> str | None:
    if "_pre_disaster" in filename:
        return "pre"
    if "_post_disaster" in filename:
        return "post"
    return None


def post_image_label(
    label_path: Path, method: str = "weighted-score"
) -> tuple[str, int, Counter]:
    """Assign a single damage label to an image tile.

    Methods:
    - most-severe: picks the highest severity damage found in any building.
    - majority: picks the most common damage type across buildings.
    - weighted-score: computes a severity-weighted average and bins it.
    """
    with label_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    building_counts: Counter = Counter()
    for feature in data.get("features", {}).get("xy", []):
        subtype = feature.get("properties", {}).get("subtype")
        if subtype in SEVERITY:
            building_counts[subtype] += 1

    if not building_counts:
        return "no-damage", 0, building_counts

    total_buildings = sum(building_counts.values())

    if method == "most-severe":
        label = max(building_counts, key=lambda l: SEVERITY[l])
    elif method == "majority":
        label = max(building_counts, key=lambda l: building_counts[l])
    elif method == "weighted-score":
        # Weighted severity: 0=no-damage, 1=minor, 2=major, 3=destroyed
        score = sum(
            SEVERITY[lvl] * building_counts[lvl] for lvl in building_counts
        ) / total_buildings
        # Bin the continuous score into discrete classes
        if score < 0.5:
            label = "no-damage"
        elif score < 1.5:
            label = "minor-damage"
        elif score < 2.5:
            label = "major-damage"
        else:
            label = "destroyed"
    else:
        raise ValueError(f"Unknown labeling method: {method}")

    return label, total_buildings, building_counts


def read_metadata(label_path: Path) -> dict:
    with label_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data.get("metadata", {})


def collect_all_pairs(
    dataset_root: Path, labeling_method: str = "weighted-score"
) -> list[dict[str, object]]:
    images_dir = dataset_root / "train" / "train" / "images"
    labels_dir = dataset_root / "train" / "train" / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    pre_files: dict[str, Path] = {}
    post_files: dict[str, Path] = {}
    pre_labels: dict[str, Path] = {}
    post_labels: dict[str, Path] = {}

    for image_path in images_dir.glob("*.png"):
        phase = image_phase(image_path.name)
        if phase == "pre":
            pre_files[pair_id_from_name(image_path.name)] = image_path
        elif phase == "post":
            post_files[pair_id_from_name(image_path.name)] = image_path

    for label_path in labels_dir.glob("*.json"):
        phase = image_phase(label_path.name)
        image_name = label_path.with_suffix(".png").name
        if phase == "pre":
            pre_labels[pair_id_from_name(image_name)] = label_path
        elif phase == "post":
            post_labels[pair_id_from_name(image_name)] = label_path

    complete_ids = sorted(set(pre_files) & set(post_files) & set(pre_labels) & set(post_labels))
    pairs: list[dict[str, object]] = []

    for pair_id in complete_ids:
        post_json = post_labels[pair_id]
        meta = read_metadata(post_json)
        label, building_count, building_counts = post_image_label(
            post_json, method=labeling_method
        )

        pairs.append(
            {
                "pair_id": pair_id,
                "pre_image": str(pre_files[pair_id]),
                "post_image": str(post_files[pair_id]),
                "pre_json": str(pre_labels[pair_id]),
                "post_json": str(post_json),
                "label": label,
                "class_dir": CLASS_DIR_NAMES[label],
                "disaster": meta.get("disaster", "unknown"),
                "disaster_type": meta.get("disaster_type", "unknown"),
                "building_count": building_count,
                "no_damage_buildings": building_counts["no-damage"],
                "minor_damage_buildings": building_counts["minor-damage"],
                "major_damage_buildings": building_counts["major-damage"],
                "destroyed_buildings": building_counts["destroyed"],
            }
        )

    return pairs


def balanced_sample(
    pairs: list[dict[str, object]],
    samples_per_class: int | None,
    random_state: int,
) -> tuple[list[dict[str, object]], int]:
    rng = random.Random(random_state)
    by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for pair in pairs:
        by_label[str(pair["label"])].append(pair)

    missing_classes = [label for label in DAMAGE_LEVELS if not by_label[label]]
    if missing_classes:
        raise ValueError(f"Missing classes in dataset: {missing_classes}")

    available_min = min(len(by_label[label]) for label in DAMAGE_LEVELS)
    target = samples_per_class or available_min
    if target > available_min:
        raise ValueError(
            f"Requested {target} samples per class, but the smallest class has {available_min}."
        )

    selected: list[dict[str, object]] = []
    for label in DAMAGE_LEVELS:
        class_pairs = by_label[label][:]
        rng.shuffle(class_pairs)
        selected.extend(class_pairs[:target])

    rng.shuffle(selected)
    return selected, target


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "pair_id",
        "label",
        "class_dir",
        "disaster",
        "disaster_type",
        "pre_image",
        "post_image",
        "pre_json",
        "post_json",
        "building_count",
        "no_damage_buildings",
        "minor_damage_buildings",
        "major_damage_buildings",
        "destroyed_buildings",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = collect_all_pairs(dataset_root, args.labeling_method)
    balanced_pairs, target = balanced_sample(
        all_pairs,
        args.samples_per_class,
        args.random_state,
    )

    all_counts = Counter(pair["label"] for pair in all_pairs)
    balanced_counts = Counter(pair["label"] for pair in balanced_pairs)
    disaster_type_counts = Counter(pair["disaster_type"] for pair in balanced_pairs)
    disaster_counts = Counter(pair["disaster"] for pair in balanced_pairs)

    write_csv(output_dir / "all_pairs_index.csv", all_pairs)
    write_csv(output_dir / "balanced_pairs_index.csv", balanced_pairs)

    summary = {
        "dataset_root": str(dataset_root),
        "pairing_method": "matched pre/post files by shared image ID",
        "labeling_method": "most severe building damage subtype from post-disaster JSON",
        "model_input_plan": {
            "pre_image": "256 x 256 x 3",
            "post_image": "256 x 256 x 3",
            "combined_pair": "256 x 256 x 6",
            "channel_layout": "channels 0-2 = pre RGB, channels 3-5 = post RGB",
        },
        "total_complete_pairs": len(all_pairs),
        "all_class_counts": dict(all_counts),
        "balanced_samples_per_class": target,
        "balanced_total_pairs": len(balanced_pairs),
        "balanced_class_counts": dict(balanced_counts),
        "balanced_disaster_type_counts": dict(disaster_type_counts),
        "balanced_disaster_counts": dict(disaster_counts),
        "random_state": args.random_state,
        "outputs": {
            "all_pairs_index": str(output_dir / "all_pairs_index.csv"),
            "balanced_pairs_index": str(output_dir / "balanced_pairs_index.csv"),
        },
    }

    with (output_dir / "balanced_pairs_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Balanced pair index created.")
    print(f"Complete pre/post pairs found: {len(all_pairs)}")
    print(f"Samples per class selected: {target}")
    print(f"Balanced total pairs: {len(balanced_pairs)}")
    print("Balanced class counts:")
    for label in DAMAGE_LEVELS:
        print(f"  {label}: {balanced_counts[label]}")
    print(f"Output folder: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
