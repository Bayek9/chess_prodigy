#!/usr/bin/env python3
"""
Calibrate board/no_board threshold for an existing Keras model.

This script avoids retraining when the classifier is badly calibrated
around 0.5 and finds a better operating threshold for your use case.
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
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: tensorflow\n"
        "Install with: pip install -r tools/dataset/requirements-train.txt"
    ) from exc

try:
    from PIL import Image
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: Pillow\n"
        "Install with: pip install -r tools/dataset/requirements-train.txt"
    ) from exc


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
        description="Calibrate recommended threshold for board/no_board model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/board_binary/board_binary.keras",
        help="Path to saved Keras model.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        action="append",
        required=True,
        help="Root dir containing board/ and/or no_board/ subdirs. Can repeat.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--calib-split",
        type=float,
        default=0.2,
        help="Stratified calibration split ratio per class.",
    )
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--threshold-board-weight", type=float, default=0.45)
    parser.add_argument("--threshold-no-board-weight", type=float, default=0.55)
    parser.add_argument(
        "--output-json",
        type=str,
        default="models/board_binary/threshold_calibration.json",
    )
    return parser.parse_args()


def list_images(root: Path) -> list[Path]:
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def collect_samples(data_roots: list[Path]) -> list[Sample]:
    samples: list[Sample] = []
    for root in data_roots:
        for class_name, label in CLASS_TO_INDEX.items():
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for path in list_images(class_dir):
                samples.append(Sample(path=str(path), label=label))
    return samples


def filter_decodable(samples: list[Sample]) -> tuple[list[Sample], int]:
    valid: list[Sample] = []
    dropped = 0
    for sample in samples:
        try:
            with Image.open(sample.path) as im:
                im.verify()
            valid.append(sample)
        except Exception:
            dropped += 1
    return valid, dropped


def split_calib(samples: list[Sample], ratio: float, seed: int) -> list[Sample]:
    by_class: dict[int, list[Sample]] = {0: [], 1: []}
    for sample in samples:
        by_class[sample.label].append(sample)
    rng = random.Random(seed)
    calib: list[Sample] = []
    for label in (0, 1):
        cls = by_class[label]
        if not cls:
            raise SystemExit(f"No samples for class {INDEX_TO_CLASS[label]}")
        rng.shuffle(cls)
        count = max(1, int(round(len(cls) * ratio)))
        count = min(count, len(cls))
        calib.extend(cls[:count])
    rng.shuffle(calib)
    return calib


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
    points: list[ThresholdStats] = []
    t = threshold_min
    while t <= threshold_max + 1e-12:
        thr = round(float(t), 6)
        conf = confusion_from_probs(y_true=y_true, probs=probs, threshold=thr)
        m = metrics_from_confusion(conf)
        score = (board_weight * m["tpr"]) + (no_board_weight * m["tnr"])
        points.append(
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
    return points


def select_best(points: list[ThresholdStats]) -> ThresholdStats:
    if not points:
        raise SystemExit("No threshold points.")
    return sorted(
        points,
        key=lambda p: (p.score, p.f1, p.tnr, p.tpr, p.accuracy),
        reverse=True,
    )[0]


def batch_predict(
    model: tf.keras.Model,
    samples: list[Sample],
    image_size: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([s.label for s in samples], dtype=np.int32)
    probs_all: list[np.ndarray] = []

    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        batch = np.zeros((len(chunk), image_size, image_size, 3), dtype=np.float32)
        for j, sample in enumerate(chunk):
            with Image.open(sample.path) as im:
                rgb = im.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
            batch[j] = arr
        probs = model.predict(batch, verbose=0).reshape(-1)
        probs_all.append(probs)

    return y_true, np.concatenate(probs_all, axis=0)


def main() -> None:
    args = parse_args()
    if not (0.0 < args.calib_split <= 1.0):
        raise SystemExit("calib_split must be in (0,1].")
    if not (0.0 <= args.threshold_min < args.threshold_max <= 1.0):
        raise SystemExit("threshold range must satisfy 0 <= min < max <= 1.")
    if args.threshold_step <= 0:
        raise SystemExit("threshold_step must be > 0.")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    data_roots = [Path(p) for p in dict.fromkeys(args.data_dir)]
    for root in data_roots:
        if not root.exists():
            raise SystemExit(f"data-dir not found: {root}")

    samples = collect_samples(data_roots)
    if not samples:
        raise SystemExit("No samples found.")

    samples, dropped = filter_decodable(samples)
    if not samples:
        raise SystemExit("No decodable samples found.")

    calib_samples = split_calib(samples=samples, ratio=args.calib_split, seed=args.seed)
    counts = {
        "board": sum(1 for s in calib_samples if s.label == 1),
        "no_board": sum(1 for s in calib_samples if s.label == 0),
    }

    model = tf.keras.models.load_model(model_path)
    y_true, probs = batch_predict(
        model=model,
        samples=calib_samples,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

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
    best = select_best(points)
    top = sorted(points, key=lambda p: p.score, reverse=True)[:5]

    out = {
        "model_path": str(model_path),
        "data_dirs": [str(p) for p in data_roots],
        "dropped_invalid_images": dropped,
        "calibration_samples": len(calib_samples),
        "calibration_class_counts": counts,
        "default_threshold": 0.5,
        "default_threshold_metrics": {**default_metrics, **default_conf},
        "recommended_threshold": best.threshold,
        "recommended_threshold_metrics": {
            "accuracy": best.accuracy,
            "precision": best.precision,
            "recall": best.recall,
            "f1": best.f1,
            "tpr": best.tpr,
            "tnr": best.tnr,
            "tp": best.tp,
            "tn": best.tn,
            "fp": best.fp,
            "fn": best.fn,
            "score": best.score,
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
            for p in top
        ],
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[calibrate-board] samples={len(calib_samples)} counts={counts} dropped_invalid={dropped}")
    print(f"[calibrate-board] default@0.50={default_metrics} conf={default_conf}")
    print(
        "[calibrate-board] recommended_threshold="
        f"{best.threshold:.3f} score={best.score:.4f} "
        f"tp={best.tp} tn={best.tn} fp={best.fp} fn={best.fn}"
    )
    print(f"[calibrate-board] report={out_path}")


if __name__ == "__main__":
    main()
