from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

from paired_augmentation import PairedAugmentationSequence


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LABELS = ["no-damage", "minor-damage", "major-damage", "destroyed"]
INPUT_SHAPE = (256, 256, 6)
CONSERVATIVE_AUGMENTATION_CONFIG = {
    "rotation_range": 30,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "zoom_range": 0.1,
    "shear_range": 5,
    "horizontal_flip": True,
    "vertical_flip": True,
    "brightness_range": (0.9, 1.1),
    "channel_shift_range": 0.05,
    "fill_mode": "nearest",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate CNN and CNN+KNN models on balanced xView2 pairs."
    )
    parser.add_argument("--processed-dir", default="processed_balanced_pairs")
    parser.add_argument("--models-dir", default="models_siamese")
    parser.add_argument("--visualizations-dir", default="visualizations_siamese")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--base-seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_arrays(processed_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = []
    for name in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
        path = processed_dir / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")
        arrays.append(np.load(path))

    x_train, y_train, x_val, y_val, x_test, y_test = arrays
    for split_name, x in [("train", x_train), ("validation", x_val), ("test", x_test)]:
        if x.shape[1:] != INPUT_SHAPE:
            raise ValueError(f"{split_name} input shape must be (N, 256, 256, 6), got {x.shape}")
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if x.min() < 0 or x.max() > 1:
            raise ValueError(f"{split_name} input values must be normalized to [0, 1].")

    return x_train, y_train, x_val, y_val, x_test, y_test


def focal_loss(gamma: float = 2.0, label_smoothing: float = 0.05):
    """Create a focal loss function for multi-class classification.

    Focal loss down-weights easy examples so the model focuses on hard,
    misclassified ones. Combined with label smoothing to prevent
    overconfidence on a small dataset.
    """

    def _focal_loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(y_true, num_classes)
        if label_smoothing > 0:
            y_true_one_hot = (
                y_true_one_hot * (1.0 - label_smoothing)
                + label_smoothing / tf.cast(num_classes, tf.float32)
            )
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = y_true_one_hot * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=-1))

    _focal_loss.__name__ = "focal_loss"
    return _focal_loss


def build_cnn(seed: int) -> tf.keras.Model:
    """Build a Siamese CNN with a pretrained MobileNetV2 backbone.

    Architecture:
    - Splits 6-channel input into pre (ch 0-2) and post (ch 3-5) images
    - Processes each through a shared MobileNetV2 backbone (ImageNet weights)
    - Computes temporal difference features (post - pre) for change detection
    - Fuses [pre, post, diff] features through a BatchNorm classification head
    - Named 'features' layer preserved for CNN+KNN pipeline
    """
    set_seed(seed)

    input_layer = layers.Input(shape=INPUT_SHAPE, name="paired_input")

    # Split pre-disaster (channels 0-2) and post-disaster (channels 3-5)
    pre = layers.Lambda(lambda x: x[..., :3], name="split_pre")(input_layer)
    post = layers.Lambda(lambda x: x[..., 3:], name="split_post")(input_layer)

    # Scale from [0, 1] to [-1, 1] for MobileNetV2 expected input range
    pre_scaled = layers.Lambda(lambda x: x * 2.0 - 1.0, name="scale_pre")(pre)
    post_scaled = layers.Lambda(lambda x: x * 2.0 - 1.0, name="scale_post")(post)

    # Shared pretrained backbone — same weights process both temporal images
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(256, 256, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone._name = "shared_backbone"

    # Fine-tune later layers only; freeze early generic feature extractors
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    feat_pre = backbone(pre_scaled)
    feat_post = backbone(post_scaled)

    # Explicit change detection: spatial temporal difference features
    diff = layers.Subtract(name="temporal_diff")([feat_post, feat_pre])

    # Feature fusion: [pre_features | post_features | change_features]
    fused = layers.Concatenate(name="fusion")([feat_pre, feat_post, diff])

    # Process spatial features with a Conv layer before pooling
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="spatial_fusion")(fused)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D(name="global_pooling")(x)

    # Classification head with BatchNorm for stable training
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(name="features")(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(len(LABELS), activation="softmax", name="damage_softmax")(x)

    model = models.Model(
        inputs=input_layer, outputs=output, name="siamese_damage_cnn"
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=focal_loss(gamma=2.0, label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model


def get_class_weights(y_train: np.ndarray) -> dict[int, float]:
    classes = np.array(sorted(np.unique(y_train)))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}


def get_dataset_for_training(
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    seed: int,
    use_augmentation: bool,
) -> tuple[object, int]:
    if use_augmentation:
        datagen = ImageDataGenerator(**CONSERVATIVE_AUGMENTATION_CONFIG)
        sequence = PairedAugmentationSequence(
            x_train,
            y_train,
            batch_size=batch_size,
            datagen=datagen,
            seed=seed,
        )
        return sequence, len(sequence)
    return (x_train, y_train), int(np.ceil(len(x_train) / batch_size))


def make_feature_extractor(cnn: tf.keras.Model) -> tf.keras.Model:
    return tf.keras.Model(
        inputs=cnn.inputs,
        outputs=cnn.get_layer("features").output,
        name="cnn_feature_extractor",
    )


def weighted_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def save_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    keys = list(history.history.keys())
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", *keys])
        for index in range(len(history.history[keys[0]])):
            writer.writerow([index + 1, *[history.history[key][index] for key in keys]])


def plot_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history.get("accuracy", []), label="train")
    axes[0].plot(history.history.get("val_accuracy", []), label="validation")
    axes[0].set_title("CNN Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history.get("loss", []), label="train")
    axes[1].plot(history.history.get("val_loss", []), label="validation")
    axes[1].set_title("CNN Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(LABELS)), labels=LABELS, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(LABELS)), labels=LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, int(matrix[row, column]), ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_mean_std(summary: dict[str, dict[str, dict[str, float]]], output_path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, model_key in [(-width / 2, "cnn"), (width / 2, "cnn_knn")]:
        means = [summary[model_key][metric]["mean"] for metric in metrics]
        stds = [summary[model_key][metric]["std"] for metric in metrics]
        ax.bar(x + offset, means, width, yerr=stds, capsize=4, label=model_key.upper())

    ax.set_xticks(x, labels=["Accuracy", "Precision", "Recall", "F1"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Five-run Mean and Standard Deviation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def summarize_runs(run_metrics: list[dict[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for model_key in ["cnn", "cnn_knn"]:
        summary[model_key] = {}
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            values = np.array([run[model_key][metric] for run in run_metrics], dtype=np.float32)
            summary[model_key][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    return summary


def print_mean_std_summary(summary: dict[str, dict[str, dict[str, float]]]) -> None:
    metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-score",
    }
    model_labels = {
        "cnn": "CNN",
        "cnn_knn": "CNN+KNN",
    }

    print("\nFinal mean +/- standard deviation:")
    for model_key in ["cnn", "cnn_knn"]:
        print(f"\n{model_labels[model_key]}:")
        for metric_key, metric_label in metric_labels.items():
            mean = summary[model_key][metric_key]["mean"] * 100
            std = summary[model_key][metric_key]["std"] * 100
            print(f"  {metric_label:<9}: {mean:6.2f}% +/- {std:.2f}%")


def train_one_run(
    run_number: int,
    seed: int,
    args: argparse.Namespace,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, object]:
    x_train, y_train, x_val, y_val, x_test, y_test = data
    run_dir = Path(args.models_dir) / f"run_{run_number:02d}"
    run_viz_dir = Path(args.visualizations_dir) / f"run_{run_number:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_viz_dir.mkdir(parents=True, exist_ok=True)

    class_weights = get_class_weights(y_train)
    cnn = build_cnn(seed)
    train_data, steps_per_epoch = get_dataset_for_training(
        x_train,
        y_train,
        args.batch_size,
        seed,
        use_augmentation=not args.no_augmentation,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            min_delta=0.001,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    if isinstance(train_data, tuple):
        history = cnn.fit(
            train_data[0],
            train_data[1],
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        history = cnn.fit(
            train_data,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

    cnn_probabilities = cnn.predict(x_test, batch_size=args.batch_size, verbose=0)
    cnn_predictions = np.argmax(cnn_probabilities, axis=1)
    cnn_metrics = weighted_metrics(y_test, cnn_predictions)

    feature_extractor = make_feature_extractor(cnn)
    train_features = feature_extractor.predict(x_train, batch_size=args.batch_size, verbose=0)
    test_features = feature_extractor.predict(x_test, batch_size=args.batch_size, verbose=0)

    cnn_knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=args.knn_k, weights="distance"),
    )
    cnn_knn.fit(train_features, y_train)
    knn_predictions = cnn_knn.predict(test_features)
    knn_metrics = weighted_metrics(y_test, knn_predictions)

    cnn.save(run_dir / "cnn_model.keras")
    feature_extractor.save(run_dir / "cnn_feature_extractor.keras")
    joblib.dump(cnn_knn, run_dir / "cnn_knn.joblib")
    save_history(history, run_dir / "history.csv")
    plot_history(history, run_viz_dir / "training_accuracy_loss.png")

    cnn_cm = confusion_matrix(y_test, cnn_predictions, labels=list(range(len(LABELS))))
    knn_cm = confusion_matrix(y_test, knn_predictions, labels=list(range(len(LABELS))))
    plot_confusion_matrix(cnn_cm, "CNN Confusion Matrix", run_viz_dir / "cnn_confusion_matrix.png")
    plot_confusion_matrix(knn_cm, "CNN+KNN Confusion Matrix", run_viz_dir / "cnn_knn_confusion_matrix.png")

    run_result = {
        "run": run_number,
        "seed": seed,
        "class_weights": class_weights,
        "cnn": {
            **cnn_metrics,
            "classification_report": classification_report(
                y_test,
                cnn_predictions,
                target_names=LABELS,
                zero_division=0,
                output_dict=True,
            ),
        },
        "cnn_knn": {
            **knn_metrics,
            "k": args.knn_k,
            "classification_report": classification_report(
                y_test,
                knn_predictions,
                target_names=LABELS,
                zero_division=0,
                output_dict=True,
            ),
        },
        "outputs": {
            "cnn_model": str(run_dir / "cnn_model.keras"),
            "feature_extractor": str(run_dir / "cnn_feature_extractor.keras"),
            "cnn_knn": str(run_dir / "cnn_knn.joblib"),
            "history_csv": str(run_dir / "history.csv"),
            "training_plot": str(run_viz_dir / "training_accuracy_loss.png"),
            "cnn_confusion_matrix": str(run_viz_dir / "cnn_confusion_matrix.png"),
            "cnn_knn_confusion_matrix": str(run_viz_dir / "cnn_knn_confusion_matrix.png"),
        },
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(run_result, file, indent=2)

    return run_result


def save_runs_csv(run_metrics: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["run", "seed", "model", "accuracy", "precision", "recall", "f1_score"])
        for run in run_metrics:
            for model_key in ["cnn", "cnn_knn"]:
                writer.writerow(
                    [
                        run["run"],
                        run["seed"],
                        model_key,
                        run[model_key]["accuracy"],
                        run[model_key]["precision"],
                        run[model_key]["recall"],
                        run[model_key]["f1_score"],
                    ]
                )


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    models_dir = Path(args.models_dir)
    visualizations_dir = Path(args.visualizations_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    data = load_arrays(processed_dir)
    y_train = data[1]
    class_weights = get_class_weights(y_train)
    with (models_dir / "class_weights.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "label_order": LABELS,
                "class_weights": {LABELS[class_id]: weight for class_id, weight in class_weights.items()},
                "reason": "Computed from the training split to reduce class imbalance risk.",
            },
            file,
            indent=2,
        )

    run_metrics = []
    for run_index in range(args.runs):
        seed = args.base_seed + run_index
        print(f"\nStarting run {run_index + 1}/{args.runs} with seed {seed}")
        run_metrics.append(train_one_run(run_index + 1, seed, args, data))

    summary = summarize_runs(run_metrics)
    aggregate = {
        "project_title": "Building Damage Assessment from Pre/Post Satellite Images using CNN and CNN-KNN",
        "label_order": LABELS,
        "input_shape": list(INPUT_SHAPE),
        "runs": args.runs,
        "epochs_requested": args.epochs,
        "augmentation_used_for_training": not args.no_augmentation,
        "augmentation_config": (
            {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in CONSERVATIVE_AUGMENTATION_CONFIG.items()
            }
            if not args.no_augmentation
            else None
        ),
        "validation_augmentation": False,
        "test_augmentation": False,
        "class_weights": {LABELS[class_id]: weight for class_id, weight in class_weights.items()},
        "mean_std": summary,
        "run_metrics": run_metrics,
    }

    with (models_dir / "five_run_evaluation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(aggregate, file, indent=2)

    save_runs_csv(run_metrics, models_dir / "five_run_metrics.csv")
    plot_mean_std(summary, visualizations_dir / "five_run_mean_std.png")

    print("\nTraining and evaluation complete.")
    print_mean_std_summary(summary)
    print(f"Class weights: {class_weights}")
    print(f"Summary saved to: {(models_dir / 'five_run_evaluation_summary.json').resolve()}")
    print(f"Metrics CSV saved to: {(models_dir / 'five_run_metrics.csv').resolve()}")
    print(f"Mean/std chart saved to: {(visualizations_dir / 'five_run_mean_std.png').resolve()}")


if __name__ == "__main__":
    main()
