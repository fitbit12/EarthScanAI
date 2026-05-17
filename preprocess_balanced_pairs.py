from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None
    from PIL import Image


DAMAGE_LEVELS = ["no-damage", "minor-damage", "major-damage", "destroyed"]
LABEL_TO_ID = {label: index for index, label in enumerate(DAMAGE_LEVELS)}
CLASS_DIR_NAMES = {
    "no-damage": "no_damage",
    "minor-damage": "minor_damage",
    "major-damage": "major_damage",
    "destroyed": "destroyed",
}
IMAGE_SIZE = (256, 256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess pre/post pairs into 256x256x6 arrays."
    )
    parser.add_argument(
        "--pairs-csv",
        default="dataset_work/all_pairs_index.csv",
        help="All pair index CSV created by create_balanced_pairs.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="processed_balanced_pairs",
        help="Folder where processed arrays and split metadata will be saved.",
    )
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_image(image_path: Path) -> np.ndarray:
    if cv2 is not None:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    else:
        image = Image.open(image_path).convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        image = np.asarray(image)

    return image.astype(np.float32) / 255.0


def make_pair_array(row: dict[str, str]) -> np.ndarray:
    pre = read_image(Path(row["pre_image"]))
    post = read_image(Path(row["post_image"]))
    return np.concatenate([pre, post], axis=-1)


def split_rows(
    rows: list[dict[str, str]],
    test_size: float,
    validation_size: float,
    random_state: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    labels = [row["label"] for row in rows]
    train_val_rows, test_rows = train_test_split(
        rows,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    train_val_labels = [row["label"] for row in train_val_rows]
    adjusted_validation_size = validation_size / (1.0 - test_size)
    train_rows, val_rows = train_test_split(
        train_val_rows,
        test_size=adjusted_validation_size,
        random_state=random_state,
        stratify=train_val_labels,
    )

    return train_rows, val_rows, test_rows


def rows_to_arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    x_values = []
    y_values = []
    for row in rows:
        x_values.append(make_pair_array(row))
        y_values.append(LABEL_TO_ID[row["label"]])
    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.int64)


def write_split_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "pair_id",
        "label",
        "label_id",
        "class_dir",
        "disaster",
        "disaster_type",
        "pre_image",
        "post_image",
        "pre_json",
        "post_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = {key: row[key] for key in fieldnames if key in row}
            output["label_id"] = LABEL_TO_ID[row["label"]]
            writer.writerow(output)


def save_class_distribution_csv(path: Path, splits: dict[str, list[dict[str, str]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "class", "count"])
        for split_name, rows in splits.items():
            counts = Counter(row["label"] for row in rows)
            for label in DAMAGE_LEVELS:
                writer.writerow([split_name, label, counts[label]])


def save_preview(rows: list[dict[str, str]], output_path: Path) -> None:
    selected = []
    for label in DAMAGE_LEVELS:
        match = next((row for row in rows if row["label"] == label), None)
        if match:
            selected.append(match)

    if not selected:
        return

    fig, axes = plt.subplots(len(selected), 2, figsize=(6, 3 * len(selected)))
    if len(selected) == 1:
        axes = np.asarray([axes])

    for row_index, row in enumerate(selected):
        pre = read_image(Path(row["pre_image"]))[:, :, :3]
        post = read_image(Path(row["post_image"]))[:, :, :3]
        axes[row_index, 0].imshow(pre)
        axes[row_index, 0].set_title(f"{row['label']} - pre")
        axes[row_index, 0].axis("off")
        axes[row_index, 1].imshow(post)
        axes[row_index, 1].set_title(f"{row['label']} - post")
        axes[row_index, 1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def array_stats(array: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
    }


def main() -> None:
    args = parse_args()
    pairs_csv = Path(args.pairs_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pairs_csv.is_file():
        raise FileNotFoundError(f"Balanced pairs CSV not found: {pairs_csv}")

    rows = read_pairs(pairs_csv)
    train_rows, val_rows, test_rows = split_rows(
        rows,
        args.test_size,
        args.validation_size,
        args.random_state,
    )

    x_train, y_train = rows_to_arrays(train_rows)
    x_val, y_val = rows_to_arrays(val_rows)
    x_test, y_test = rows_to_arrays(test_rows)

    np.save(output_dir / "X_train.npy", x_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", x_val)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "X_test.npy", x_test)
    np.save(output_dir / "y_test.npy", y_test)

    write_split_csv(output_dir / "train_pairs.csv", train_rows)
    write_split_csv(output_dir / "validation_pairs.csv", val_rows)
    write_split_csv(output_dir / "test_pairs.csv", test_rows)
    save_class_distribution_csv(
        output_dir / "class_distribution.csv",
        {"train": train_rows, "validation": val_rows, "test": test_rows},
    )
    save_preview(rows, output_dir / "sample_pair_preview.png")

    splits = {
        "train": train_rows,
        "validation": val_rows,
        "test": test_rows,
    }
    summary = {
        "source_pairs_csv": str(pairs_csv),
        "input_shape": [256, 256, 6],
        "pre_image_shape": [256, 256, 3],
        "post_image_shape": [256, 256, 3],
        "channel_layout": "channels 0-2 = pre RGB, channels 3-5 = post RGB",
        "normalization": "float32 pixel values divided by 255.0, range [0, 1]",
        "raw_images_modified": False,
        "split_ratios": {
            "train": 1.0 - args.validation_size - args.test_size,
            "validation": args.validation_size,
            "test": args.test_size,
        },
        "split_counts": {name: len(split_rows) for name, split_rows in splits.items()},
        "split_class_counts": {
            name: dict(Counter(row["label"] for row in split_rows))
            for name, split_rows in splits.items()
        },
        "array_stats": {
            "X_train": array_stats(x_train),
            "X_val": array_stats(x_val),
            "X_test": array_stats(x_test),
        },
        "outputs": {
            "X_train": str(output_dir / "X_train.npy"),
            "y_train": str(output_dir / "y_train.npy"),
            "X_val": str(output_dir / "X_val.npy"),
            "y_val": str(output_dir / "y_val.npy"),
            "X_test": str(output_dir / "X_test.npy"),
            "y_test": str(output_dir / "y_test.npy"),
            "metadata_pairs": [
                str(output_dir / "train_pairs.csv"),
                str(output_dir / "validation_pairs.csv"),
                str(output_dir / "test_pairs.csv"),
            ],
            "class_distribution": str(output_dir / "class_distribution.csv"),
            "sample_pair_preview": str(output_dir / "sample_pair_preview.png"),
        },
    }

    with (output_dir / "preprocessing_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Balanced pair preprocessing complete.")
    print(f"Train shape: {x_train.shape}")
    print(f"Validation shape: {x_val.shape}")
    print(f"Test shape: {x_test.shape}")
    print(f"Train min/max: {x_train.min():.4f}/{x_train.max():.4f}")
    print(f"Validation min/max: {x_val.min():.4f}/{x_val.max():.4f}")
    print(f"Test min/max: {x_test.min():.4f}/{x_test.max():.4f}")
    print(f"Output folder: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
