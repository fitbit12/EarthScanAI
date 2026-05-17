from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.utils import Sequence  # type: ignore


DEFAULT_AUGMENTATION_CONFIG = {
    "rotation_range": 90,
    "width_shift_range": 0.15,
    "height_shift_range": 0.15,
    "zoom_range": 0.2,
    "shear_range": 10,
    "horizontal_flip": True,
    "vertical_flip": True,
    "brightness_range": (0.75, 1.25),
    "channel_shift_range": 15.0,
    "fill_mode": "nearest",
}


class PairedAugmentationSequence(Sequence):
    """Keras sequence that applies identical spatial transforms to pre/post images."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        datagen: ImageDataGenerator | None = None,
        seed: int = 42,
        shuffle: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.datagen = datagen or ImageDataGenerator(**DEFAULT_AUGMENTATION_CONFIG)
        self.seed = seed
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indexes = np.arange(len(x))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.x) / self.batch_size))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        start = batch_index * self.batch_size
        end = min(start + self.batch_size, len(self.x))
        batch_indexes = self.indexes[start:end]
        batch_x = np.empty((len(batch_indexes), 256, 256, 6), dtype=np.float32)
        batch_y = self.y[batch_indexes]

        for output_index, sample_index in enumerate(batch_indexes):
            batch_x[output_index] = augment_pair(
                self.x[sample_index],
                self.datagen,
            )

        return batch_x, batch_y


def augment_pair(pair: np.ndarray, datagen: ImageDataGenerator) -> np.ndarray:
    # ImageDataGenerator's brightness/channel transforms are designed around
    # image-like 0-255 values, so we temporarily unnormalize and normalize again.
    pre = pair[:, :, :3] * 255.0
    post = pair[:, :, 3:] * 255.0

    # One transform parameter set is reused for both images to preserve alignment.
    transform = datagen.get_random_transform(pre.shape)
    pre_aug = datagen.apply_transform(pre, transform)
    post_aug = datagen.apply_transform(post, transform)

    paired_aug = np.concatenate([pre_aug, post_aug], axis=-1)
    return (np.clip(paired_aug, 0.0, 255.0) / 255.0).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify paired training augmentation and save a preview image."
    )
    parser.add_argument("--processed-dir", default="processed_balanced_pairs")
    parser.add_argument("--output-dir", default="augmentation_checks")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_augmentation_preview(
    x_train: np.ndarray,
    output_path: Path,
    samples: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    datagen = ImageDataGenerator(**DEFAULT_AUGMENTATION_CONFIG)
    sample_indexes = rng.choice(len(x_train), size=min(samples, len(x_train)), replace=False)

    fig, axes = plt.subplots(len(sample_indexes), 4, figsize=(12, 3 * len(sample_indexes)))
    if len(sample_indexes) == 1:
        axes = np.asarray([axes])

    for row_index, sample_index in enumerate(sample_indexes):
        original = x_train[sample_index]
        augmented = augment_pair(original, datagen)

        axes[row_index, 0].imshow(original[:, :, :3])
        axes[row_index, 0].set_title("Original pre")
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(original[:, :, 3:])
        axes[row_index, 1].set_title("Original post")
        axes[row_index, 1].axis("off")

        axes[row_index, 2].imshow(augmented[:, :, :3])
        axes[row_index, 2].set_title("Augmented pre")
        axes[row_index, 2].axis("off")

        axes[row_index, 3].imshow(augmented[:, :, 3:])
        axes[row_index, 3].set_title("Augmented post")
        axes[row_index, 3].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train_path = processed_dir / "X_train.npy"
    y_train_path = processed_dir / "y_train.npy"
    if not x_train_path.is_file() or not y_train_path.is_file():
        raise FileNotFoundError("X_train.npy and y_train.npy are required before augmentation verification.")

    x_train = np.load(x_train_path)
    y_train = np.load(y_train_path)

    if x_train.shape[1:] != (256, 256, 6):
        raise ValueError(f"Expected X_train shape (N, 256, 256, 6), got {x_train.shape}")
    if len(x_train) != len(y_train):
        raise ValueError("X_train and y_train length mismatch.")

    sequence = PairedAugmentationSequence(
        x_train,
        y_train,
        batch_size=8,
        seed=args.seed,
    )
    batch_x, batch_y = sequence[0]

    preview_path = output_dir / "paired_augmentation_preview.png"
    save_augmentation_preview(x_train, preview_path, args.samples, args.seed)

    summary = {
        "augmentation_applies_to": "training pairs only",
        "validation_augmentation": False,
        "test_augmentation": False,
        "same_transform_for_pre_and_post": True,
        "input_shape": [256, 256, 6],
        "batch_shape_example": list(batch_x.shape),
        "batch_label_shape_example": list(batch_y.shape),
        "batch_min": float(batch_x.min()),
        "batch_max": float(batch_x.max()),
        "augmentation_config": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in DEFAULT_AUGMENTATION_CONFIG.items()
        },
        "preview": str(preview_path),
    }

    with (output_dir / "paired_augmentation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Paired augmentation verification complete.")
    print(f"Example augmented batch shape: {batch_x.shape}")
    print(f"Example batch min/max: {batch_x.min():.4f}/{batch_x.max():.4f}")
    print(f"Preview saved to: {preview_path.resolve()}")
    print(f"Summary saved to: {(output_dir / 'paired_augmentation_summary.json').resolve()}")


if __name__ == "__main__":
    main()
