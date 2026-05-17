from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_DIR = PROJECT_DIR / "models_final_aug_256_15ep_b16" / "run_02"
SUMMARY_PATH = PROJECT_DIR / "models_final_aug_256_15ep_b16" / "five_run_evaluation_summary.json"
RUN_METRICS_PATH = MODEL_DIR / "metrics.json"

CNN_MODEL_PATH = MODEL_DIR / "cnn_model.keras"
FEATURE_EXTRACTOR_PATH = MODEL_DIR / "cnn_feature_extractor.keras"
KNN_MODEL_PATH = MODEL_DIR / "cnn_knn.joblib"

LABELS = ["no-damage", "minor-damage", "major-damage", "destroyed"]
DISPLAY_LABELS = ["No damage", "Minor damage", "Major damage", "Destroyed"]
IMAGE_SIZE = (256, 256)
MODEL_DISPLAY_NAMES = {
    "cnn": "CNN",
    "cnn_knn": "CNN+KNN",
}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
CORS(app)

_models: dict[str, Any] | None = None
_summary: dict[str, Any] | None = None
_run_metrics: dict[str, Any] | None = None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_summary() -> dict[str, Any]:
    global _summary
    if _summary is None:
        _summary = load_json(SUMMARY_PATH)
    return _summary


def get_run_metrics() -> dict[str, Any]:
    global _run_metrics
    if _run_metrics is None:
        _run_metrics = load_json(RUN_METRICS_PATH)
    return _run_metrics


def get_models() -> dict[str, Any]:
    global _models
    if _models is None:
        _models = {
            "cnn": tf.keras.models.load_model(
                CNN_MODEL_PATH,
                compile=False,
                safe_mode=False,
            ),
            "feature_extractor": tf.keras.models.load_model(
                FEATURE_EXTRACTOR_PATH,
                compile=False,
                safe_mode=False,
            ),
            "cnn_knn": joblib.load(KNN_MODEL_PATH),
        }
    return _models


def read_uploaded_image(file_storage) -> np.ndarray:
    image_bytes = np.frombuffer(file_storage.read(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read uploaded image: {file_storage.filename}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def build_pair_input(pre_file, post_file) -> np.ndarray:
    pre = read_uploaded_image(pre_file)
    post = read_uploaded_image(post_file)
    pair = np.concatenate([pre, post], axis=-1)
    return np.expand_dims(pair, axis=0).astype(np.float32)


def build_pair_input_with_images(pre_file, post_file) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre = read_uploaded_image(pre_file)
    post = read_uploaded_image(post_file)
    pair = np.concatenate([pre, post], axis=-1)
    return np.expand_dims(pair, axis=0).astype(np.float32), pre, post


def make_change_heatmap(pre: np.ndarray, post: np.ndarray) -> dict[str, Any]:
    difference = np.mean(np.abs(post - pre), axis=-1)
    difference = cv2.GaussianBlur(difference, (17, 17), 0)

    if float(difference.max()) > float(difference.min()):
        normalized = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX)
    else:
        normalized = np.zeros_like(difference, dtype=np.float32)

    heatmap_gray = normalized.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    post_uint8 = np.clip(post * 255.0, 0, 255).astype(np.uint8)
    overlay = cv2.addWeighted(post_uint8, 0.58, heatmap_color, 0.42, 0)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", overlay_bgr)
    if not success:
        raise ValueError("Could not encode generated heatmap.")

    return {
        "image_data_url": "data:image/png;base64,"
        + base64.b64encode(encoded.tobytes()).decode("ascii"),
        "method": "absolute RGB change overlay",
        "description": (
            "Highlights the strongest visual changes between the pre- and post-disaster images. "
            "It is a change visualization, not pixel-level ground-truth damage segmentation."
        ),
        "mean_change": round(float(difference.mean()), 4),
        "max_change": round(float(difference.max()), 4),
    }


def probabilities_to_response(probabilities: np.ndarray) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    values = values / values.sum() if values.sum() > 0 else values
    prediction_id = int(np.argmax(values))

    class_scores = []
    for class_id, score in enumerate(values):
        class_scores.append(
            {
                "id": class_id,
                "label": LABELS[class_id],
                "display_label": DISPLAY_LABELS[class_id],
                "probability": float(score),
                "percentage": round(float(score) * 100, 2),
            }
        )

    return {
        "prediction": {
            "id": prediction_id,
            "label": LABELS[prediction_id],
            "display_label": DISPLAY_LABELS[prediction_id],
            "confidence": float(values[prediction_id]),
            "confidence_percentage": round(float(values[prediction_id]) * 100, 2),
        },
        "class_scores": class_scores,
    }


def knn_probabilities(pipeline, features: np.ndarray) -> np.ndarray:
    raw_probabilities = pipeline.predict_proba(features)[0]
    classifier = pipeline.named_steps["kneighborsclassifier"]
    probabilities = np.zeros(len(LABELS), dtype=np.float32)
    for index, class_id in enumerate(classifier.classes_):
        probabilities[int(class_id)] = raw_probabilities[index]
    return probabilities


def metric_percent(value: float) -> float:
    return round(float(value) * 100, 2)


def model_metrics_payload(model_key: str) -> dict[str, Any]:
    summary = get_summary()
    run_metrics = get_run_metrics()
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]

    return {
        "selected_run": {
            key: metric_percent(run_metrics[model_key][key])
            for key in metric_keys
        },
        "five_run_mean_std": {
            key: {
                "mean": metric_percent(summary["mean_std"][model_key][key]["mean"]),
                "std": metric_percent(summary["mean_std"][model_key][key]["std"]),
            }
            for key in metric_keys
        },
    }


def performance_chart_payload() -> dict[str, Any]:
    summary = get_summary()
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]

    return {
        "labels": metric_labels,
        "datasets": [
            {
                "label": MODEL_DISPLAY_NAMES[model_key],
                "mean": [
                    metric_percent(summary["mean_std"][model_key][metric]["mean"])
                    for metric in metric_keys
                ],
                "std": [
                    metric_percent(summary["mean_std"][model_key][metric]["std"])
                    for metric in metric_keys
                ],
            }
            for model_key in ["cnn", "cnn_knn"]
        ],
    }


def confidence_chart_payload(cnn_response: dict[str, Any], knn_response: dict[str, Any]) -> dict[str, Any]:
    return {
        "labels": DISPLAY_LABELS,
        "datasets": [
            {
                "label": "CNN",
                "data": [item["percentage"] for item in cnn_response["class_scores"]],
            },
            {
                "label": "CNN+KNN",
                "data": [item["percentage"] for item in knn_response["class_scores"]],
            },
        ],
    }


def selected_model_from_request() -> str:
    selected = request.form.get("model", "both").strip().lower()
    aliases = {
        "cnnknn": "cnn_knn",
        "cnn+knn": "cnn_knn",
        "knn": "cnn_knn",
    }
    selected = aliases.get(selected, selected)
    if selected not in {"both", "cnn", "cnn_knn"}:
        return "both"
    return selected


def model_files_status() -> dict[str, bool]:
    return {
        "cnn_model": CNN_MODEL_PATH.is_file(),
        "feature_extractor": FEATURE_EXTRACTOR_PATH.is_file(),
        "cnn_knn": KNN_MODEL_PATH.is_file(),
        "summary": SUMMARY_PATH.is_file(),
        "run_metrics": RUN_METRICS_PATH.is_file(),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    status = model_files_status()
    return jsonify(
        {
            "status": "ready" if all(status.values()) else "missing_files",
            "model_dir": str(MODEL_DIR),
            "selected_run": 2,
            "input_shape": [256, 256, 6],
            "labels": LABELS,
            "files": status,
        }
    )


@app.get("/api/metrics")
def metrics():
    summary = get_summary()
    run_metrics = get_run_metrics()
    return jsonify(
        {
            "project_title": summary["project_title"],
            "selected_backend_run": run_metrics["run"],
            "runs": summary["runs"],
            "epochs_requested": summary["epochs_requested"],
            "batch_size": 16,
            "augmentation_used_for_training": summary["augmentation_used_for_training"],
            "heatmap_supported": True,
            "heatmap_method": "absolute RGB change overlay",
            "label_order": summary["label_order"],
            "input_shape": summary["input_shape"],
            "models": {
                "cnn": model_metrics_payload("cnn"),
                "cnn_knn": model_metrics_payload("cnn_knn"),
            },
            "charts": {
                "performance": performance_chart_payload(),
            },
        }
    )


@app.post("/api/predict")
def predict():
    try:
        pre_file = request.files.get("pre_image") or request.files.get("pre")
        post_file = request.files.get("post_image") or request.files.get("post")
        if pre_file is None or post_file is None:
            return jsonify({"error": "Both pre_image and post_image files are required."}), 400

        selected_model = selected_model_from_request()
        pair_input, pre_image, post_image = build_pair_input_with_images(pre_file, post_file)
        models = get_models()

        cnn_probabilities = models["cnn"].predict(pair_input, verbose=0)[0]
        features = models["feature_extractor"].predict(pair_input, verbose=0)
        knn_probabilities_values = knn_probabilities(models["cnn_knn"], features)

        cnn_response = probabilities_to_response(cnn_probabilities)
        knn_response = probabilities_to_response(knn_probabilities_values)
        agreement = (
            cnn_response["prediction"]["id"]
            == knn_response["prediction"]["id"]
        )

        return jsonify(
            {
                "selected_model": selected_model,
                "input": {
                    "pre_filename": pre_file.filename,
                    "post_filename": post_file.filename,
                    "processed_shape": [256, 256, 6],
                    "normalization": "float32 pixel values scaled to [0, 1]",
                },
                "labels": [
                    {"id": index, "label": label, "display_label": DISPLAY_LABELS[index]}
                    for index, label in enumerate(LABELS)
                ],
                "models": {
                    "cnn": {
                        **cnn_response,
                        "metrics": model_metrics_payload("cnn"),
                    },
                    "cnn_knn": {
                        **knn_response,
                        "metrics": model_metrics_payload("cnn_knn"),
                    },
                },
                "comparison": {
                    "agreement": agreement,
                    "message": (
                        "Both models predicted the same damage class."
                        if agreement
                        else "Models disagreed on the damage class."
                    ),
                    "recommended_display": (
                        knn_response["prediction"]["display_label"]
                        if selected_model in {"both", "cnn_knn"}
                        else cnn_response["prediction"]["display_label"]
                    ),
                },
                "charts": {
                    "confidence": confidence_chart_payload(cnn_response, knn_response),
                    "performance": performance_chart_payload(),
                },
                "heatmap": make_change_heatmap(pre_image, post_image),
                "evaluation": {
                    "backend_model_run": 2,
                    "backend_model_reason": "Run 02 had the best CNN+KNN accuracy among the five final runs.",
                    "final_results_should_report": "Use five-run mean +/- standard deviation for report and viva.",
                    "runs": get_summary()["runs"],
                    "epochs_requested": get_summary()["epochs_requested"],
                    "batch_size": 16,
                    "augmentation_used_for_training": get_summary()["augmentation_used_for_training"],
                },
            }
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {error}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
