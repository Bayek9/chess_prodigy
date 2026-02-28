#!/usr/bin/env python3
"""
Train a board/no-board classifier from directory datasets and export TFLite.

Expected directory layout (one or more roots):
  datasets/real_scan/board_binary/
    board/
    no_board/
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "Missing dependency: tensorflow\n"
        "Install with: pip install -r tools/dataset/requirements-train.txt"
    ) from exc

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    Image = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_TO_INDEX = {"no_board": 0, "board": 1}
INDEX_TO_CLASS = {0: "no_board", 1: "board"}


@dataclass(frozen=True)
class Sample:
    path: str
    label: int


@dataclass(frozen=True)
class ThresholdStats:
    threshold: float
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    tpr: float
    tnr: float
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train board/no-board classifier and export keras + tflite models."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        action="append",
        default=None,
        help=(
            "Dataset dir(s). Supports: "
            "(a) root dirs containing board/ and/or no_board/ subdirs, "
            "(b) direct class dirs like no_board_screen/ (negative-only). "
            "Can be passed multiple times. "
            "Do not pass *_aug/*augmented* dirs to avoid train/val leakage."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="models/board_binary")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-val-per-class",
        type=int,
        default=1,
        help="Minimum validation samples per class when possible.",
    )
    parser.add_argument(
        "--max-train-per-class",
        type=int,
        default=0,
        help="Optional cap per class in train split (0 = no cap).",
    )
    parser.add_argument(
        "--max-val-per-class",
        type=int,
        default=0,
        help="Optional cap per class in val split (0 = no cap).",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.30,
        help="Minimum threshold for calibration grid.",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.90,
        help="Maximum threshold for calibration grid.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Threshold step for calibration grid.",
    )
    parser.add_argument(
        "--threshold-board-weight",
        type=float,
        default=0.45,
        help="Weight for board recall (TPR) in threshold optimization.",
    )
    parser.add_argument(
        "--threshold-no-board-weight",
        type=float,
        default=0.55,
        help="Weight for no_board specificity (TNR) in threshold optimization.",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable dynamic-range quantization for TFLite export.",
    )
    return parser.parse_args()


def list_images(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(p)
    return out


def infer_direct_class_label(root: Path) -> int | None:
    name = root.name.lower()
    if name == "board":
        return CLASS_TO_INDEX["board"]
    if name == "no_board" or name.startswith("no_board_"):
        return CLASS_TO_INDEX["no_board"]
    return None


def _is_augmented_data_root(path: Path) -> bool:
    text = "/".join(part.lower() for part in path.parts)
    return "_aug" in text or "augmented" in text


def collect_samples(data_roots: list[Path]) -> list[Sample]:
    samples: list[Sample] = []
    for root in data_roots:
        direct_label = infer_direct_class_label(root)
        if direct_label is not None:
            for path in list_images(root):
                samples.append(Sample(path=str(path), label=direct_label))
            continue

        for class_name, label in CLASS_TO_INDEX.items():
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for path in list_images(class_dir):
                samples.append(Sample(path=str(path), label=label))
    return samples


def split_stratified(
    samples: list[Sample],
    val_split: float,
    seed: int,
    min_val_per_class: int,
) -> tuple[list[Sample], list[Sample]]:
    by_class: dict[int, list[Sample]] = {0: [], 1: []}
    for sample in samples:
        by_class[sample.label].append(sample)

    rng = random.Random(seed)
    train: list[Sample] = []
    val: list[Sample] = []

    for label, class_samples in by_class.items():
        if not class_samples:
            raise SystemExit(f"No samples found for class: {INDEX_TO_CLASS[label]}")

        rng.shuffle(class_samples)
        if len(class_samples) == 1:
            raise SystemExit(
                f"Not enough samples for class {INDEX_TO_CLASS[label]}: 1 sample only."
            )

        val_count = max(min_val_per_class, int(round(len(class_samples) * val_split)))
        val_count = min(val_count, len(class_samples) - 1)
        val.extend(class_samples[:val_count])
        train.extend(class_samples[val_count:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def cap_samples_per_class(samples: list[Sample], max_per_class: int, seed: int) -> list[Sample]:
    if max_per_class <= 0:
        return samples

    by_class: dict[int, list[Sample]] = {0: [], 1: []}
    for sample in samples:
        by_class[sample.label].append(sample)

    rng = random.Random(seed)
    out: list[Sample] = []
    for label in (0, 1):
        group = by_class[label]
        if len(group) <= max_per_class:
            out.extend(group)
        else:
            out.extend(rng.sample(group, max_per_class))

    rng.shuffle(out)
    return out




def filter_decodable_samples(samples: list[Sample]) -> tuple[list[Sample], int]:
    if Image is None:
        return samples, 0

    valid: list[Sample] = []
    removed = 0
    for sample in samples:
        try:
            with Image.open(sample.path) as im:
                im.verify()
            valid.append(sample)
        except Exception:
            removed += 1
    return valid, removed

def count_by_class(samples: list[Sample]) -> dict[str, int]:
    counts = {"no_board": 0, "board": 0}
    for sample in samples:
        if sample.label == 0:
            counts["no_board"] += 1
        else:
            counts["board"] += 1
    return counts


def decode_and_resize(path: tf.Tensor, label: tf.Tensor, image_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    data = tf.io.read_file(path)
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size], antialias=True)
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.cast(label, tf.float32)


def build_dataset(
    samples: list[Sample],
    image_size: int,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    paths = [s.path for s in samples]
    labels = [s.label for s in samples]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(samples), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, y: decode_and_resize(p, y, image_size=image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(image_size: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.04)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)
    x = tf.keras.layers.RandomContrast(0.1)(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(192, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="board_probability")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="board_binary_cnn")


def class_weights_from_samples(samples: list[Sample]) -> dict[int, float]:
    counts = {0: 0, 1: 0}
    for sample in samples:
        counts[sample.label] += 1

    total = counts[0] + counts[1]
    weights: dict[int, float] = {}
    for label in (0, 1):
        if counts[label] == 0:
            weights[label] = 1.0
        else:
            weights[label] = total / (2.0 * counts[label])
    return weights


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def confusion_from_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, int]:
    y_pred = (probs >= threshold).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_from_confusion(conf: dict[str, int]) -> dict[str, float]:
    tp = conf["tp"]
    tn = conf["tn"]
    fp = conf["fp"]
    fn = conf["fn"]
    total = tp + tn + fp + fn

    accuracy = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    tpr = recall
    tnr = _safe_div(tn, tn + fp)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": tpr,
        "tnr": tnr,
    }


def evaluate_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    board_weight: float,
    no_board_weight: float,
) -> list[ThresholdStats]:
    if threshold_step <= 0:
        raise SystemExit("threshold_step must be > 0")

    stats: list[ThresholdStats] = []
    t = threshold_min
    while t <= threshold_max + 1e-12:
        thr = round(float(t), 6)
        conf = confusion_from_probs(y_true=y_true, probs=probs, threshold=thr)
        m = metrics_from_confusion(conf)
        score = (board_weight * m["tpr"]) + (no_board_weight * m["tnr"])
        stats.append(
            ThresholdStats(
                threshold=thr,
                tp=conf["tp"],
                tn=conf["tn"],
                fp=conf["fp"],
                fn=conf["fn"],
                accuracy=m["accuracy"],
                precision=m["precision"],
                recall=m["recall"],
                f1=m["f1"],
                tpr=m["tpr"],
                tnr=m["tnr"],
                score=score,
            )
        )
        t += threshold_step
    return stats


def select_recommended_threshold(points: list[ThresholdStats]) -> ThresholdStats:
    if not points:
        raise SystemExit("No threshold points computed.")

    # score desc, then balanced tie-breakers
    return sorted(
        points,
        key=lambda p: (p.score, p.f1, p.tnr, p.tpr, p.accuracy),
        reverse=True,
    )[0]


def main() -> None:
    args = parse_args()
    if not (0.0 < args.val_split < 0.5):
        raise SystemExit("val_split must be > 0 and < 0.5")
    if not (0.0 <= args.threshold_min < args.threshold_max <= 1.0):
        raise SystemExit("threshold range must satisfy 0 <= min < max <= 1")

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_dir_args = args.data_dir if args.data_dir else ["datasets/real_scan/board_binary"]
    # Deduplicate while preserving order.
    data_roots = [Path(p) for p in dict.fromkeys(data_dir_args)]
    for root in data_roots:
        if not root.exists():
            raise SystemExit(f"data-dir not found: {root}")

    augmented_roots = [root for root in data_roots if _is_augmented_data_root(root)]
    if augmented_roots:
        joined = ", ".join(str(p) for p in augmented_roots)
        raise SystemExit(
            "Refusing augmented dirs for training to avoid train/val leakage. "
            "Use original dirs only (rely on on-the-fly augmentation layers). "
            f"Found: {joined}"
        )

    original_samples = collect_samples(data_roots)
    if not original_samples:
        raise SystemExit("No images found in provided data dirs.")

    original_samples, dropped_invalid = filter_decodable_samples(original_samples)
    if dropped_invalid > 0:
        print(f"[train-board] dropped_invalid_images={dropped_invalid}")
    if not original_samples:
        raise SystemExit("No decodable non-augmented images left after filtering invalid files.")

    train_samples, val_samples = split_stratified(
        samples=original_samples,
        val_split=args.val_split,
        seed=args.seed,
        min_val_per_class=args.min_val_per_class,
    )
    train_samples = cap_samples_per_class(
        train_samples,
        max_per_class=args.max_train_per_class,
        seed=args.seed,
    )
    val_samples = cap_samples_per_class(
        val_samples,
        max_per_class=args.max_val_per_class,
        seed=args.seed + 1,
    )
    original_counts = count_by_class(original_samples)
    train_counts = count_by_class(train_samples)
    val_counts = count_by_class(val_samples)
    train_ds = build_dataset(
        train_samples, image_size=args.image_size, batch_size=args.batch_size, training=True, seed=args.seed
    )
    val_ds = build_dataset(
        val_samples, image_size=args.image_size, batch_size=args.batch_size, training=False, seed=args.seed
    )

    model = build_model(image_size=args.image_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / "best.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
        ),
    ]

    class_weights = class_weights_from_samples(train_samples)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    final_model_path = out_dir / "board_binary.keras"
    model.save(final_model_path)

    eval_map = {
        k: float(v) for k, v in model.evaluate(val_ds, verbose=0, return_dict=True).items()
    }

    y_true = np.array([s.label for s in val_samples], dtype=np.int32)
    probs = model.predict(val_ds, verbose=0).reshape(-1)

    default_conf = confusion_from_probs(y_true=y_true, probs=probs, threshold=0.5)
    default_metrics = metrics_from_confusion(default_conf)

    points = evaluate_thresholds(
        y_true=y_true,
        probs=probs,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        board_weight=args.threshold_board_weight,
        no_board_weight=args.threshold_no_board_weight,
    )
    recommended = select_recommended_threshold(points)

    labels_path = out_dir / "labels.txt"
    labels_path.write_text("no_board\nboard\n", encoding="utf-8")

    tflite_path = out_dir / "board_binary.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if not args.no_quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    tflite_path.write_bytes(tflite_bytes)

    top_points = sorted(points, key=lambda p: p.score, reverse=True)[:5]
    metrics_path = out_dir / "metrics.json"
    metrics = {
        "original_count": len(original_samples),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "original_class_counts": original_counts,
        "train_class_counts": train_counts,
        "val_class_counts": val_counts,
        "class_weights": class_weights,
        "keras_eval_metrics": eval_map,
        "threshold_policy": {
            "board_weight": args.threshold_board_weight,
            "no_board_weight": args.threshold_no_board_weight,
            "min": args.threshold_min,
            "max": args.threshold_max,
            "step": args.threshold_step,
        },
        "default_threshold": 0.5,
        "default_threshold_metrics": {
            **default_metrics,
            **default_conf,
        },
        "recommended_threshold": recommended.threshold,
        "recommended_threshold_metrics": {
            "accuracy": recommended.accuracy,
            "precision": recommended.precision,
            "recall": recommended.recall,
            "f1": recommended.f1,
            "tpr": recommended.tpr,
            "tnr": recommended.tnr,
            "tp": recommended.tp,
            "tn": recommended.tn,
            "fp": recommended.fp,
            "fn": recommended.fn,
            "score": recommended.score,
        },
        "top_thresholds": [
            {
                "threshold": p.threshold,
                "score": p.score,
                "accuracy": p.accuracy,
                "precision": p.precision,
                "recall": p.recall,
                "f1": p.f1,
                "tpr": p.tpr,
                "tnr": p.tnr,
                "tp": p.tp,
                "tn": p.tn,
                "fp": p.fp,
                "fn": p.fn,
            }
            for p in top_points
        ],
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs_ran": len(history.history.get("loss", [])),
        "output_files": {
            "keras_best": str(best_model_path),
            "keras_final": str(final_model_path),
            "tflite": str(tflite_path),
            "labels": str(labels_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        "[train-board] original_total="
        f"{len(original_samples)} train={len(train_samples)} val={len(val_samples)}"
    )
    print(f"[train-board] data_roots={len(data_roots)}")
    print(f"[train-board] original_counts={original_counts}")
    print(f"[train-board] train_counts={train_counts} val_counts={val_counts}")
    print(f"[train-board] class_weights={class_weights}")
    print(f"[train-board] keras_eval={eval_map}")
    print(f"[train-board] default@0.50={default_metrics} conf={default_conf}")
    print(
        "[train-board] recommended_threshold="
        f"{recommended.threshold:.3f} score={recommended.score:.4f} "
        f"tp={recommended.tp} tn={recommended.tn} fp={recommended.fp} fn={recommended.fn}"
    )
    print(f"[train-board] saved keras={final_model_path}")
    print(f"[train-board] saved tflite={tflite_path} ({len(tflite_bytes)} bytes)")
    print(f"[train-board] saved metrics={metrics_path}")


if __name__ == "__main__":
    main()
