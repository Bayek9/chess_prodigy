#!/usr/bin/env python3
"""
Calibrate board/no_board thresholds for an existing Keras model.

Key guarantees:
- proper split: train / val / test (threshold search on val, report on test)
- inference logic aligned with app gate classifier:
  - 5 crops (center + 4 corners)
  - same crop sampling strategy as Flutter implementation
  - strong score = top-2 mean
  - fallback score = max
- optional hard example export (hard_fp / hard_fn) with quotas
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
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
        description="Calibrate hysteresis thresholds for board/no_board model."
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
        help=(
            "Dataset dir(s). Supports: "
            "(a) root dirs containing board/ and/or no_board/ subdirs, "
            "(b) direct class dirs like no_board_screen/ (negative-only). "
            "Can repeat."
        ),
    )
    parser.add_argument(
        "--allow-augmented-data",
        action="store_true",
        help=(
            "Allow *_aug/*augmented* dirs in calibration data. "
            "Disabled by default to avoid train/val/test leakage."
        ),
    )
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Stratified validation split ratio per class.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Stratified test split ratio per class.",
    )
    parser.add_argument(
        "--crop-size-fraction",
        type=float,
        default=0.70,
        help="Crop size fraction used in 5-crop inference (must match app).",
    )

    # Reject threshold tuning (fallback score = max crop probability).
    parser.add_argument("--reject-threshold-min", type=float, default=0.05)
    parser.add_argument("--reject-threshold-max", type=float, default=0.60)
    parser.add_argument("--reject-threshold-step", type=float, default=0.01)
    parser.add_argument("--reject-board-weight", type=float, default=0.75)
    parser.add_argument("--reject-no-board-weight", type=float, default=0.25)

    # Accept threshold tuning (strong score = top2 mean probability).
    parser.add_argument("--accept-threshold-min", type=float, default=0.50)
    parser.add_argument("--accept-threshold-max", type=float, default=0.99)
    parser.add_argument("--accept-threshold-step", type=float, default=0.01)
    parser.add_argument("--accept-board-weight", type=float, default=0.30)
    parser.add_argument("--accept-no-board-weight", type=float, default=0.70)

    parser.add_argument(
        "--hysteresis-min-gap",
        type=float,
        default=0.05,
        help="Require accept_threshold >= reject_threshold + gap.",
    )

    # Backward-compatibility aliases from previous script.
    parser.add_argument("--threshold-min", type=float, default=None)
    parser.add_argument("--threshold-max", type=float, default=None)
    parser.add_argument("--threshold-step", type=float, default=None)
    parser.add_argument("--threshold-board-weight", type=float, default=None)
    parser.add_argument("--threshold-no-board-weight", type=float, default=None)

    # Optional hard-example export.
    parser.add_argument(
        "--hard-negatives-dir",
        type=str,
        default=None,
        help="Optional output dir to copy hard_fp/hard_fn samples.",
    )
    parser.add_argument("--max-hard-fp", type=int, default=200)
    parser.add_argument("--max-hard-fn", type=int, default=200)
    parser.add_argument(
        "--hard-copy-stride",
        type=int,
        default=1,
        help="Copy one sample every N mistakes (1 = keep all, subject to max).",
    )
    parser.add_argument(
        "--hard-purge-existing",
        action="store_true",
        help="Delete previous hard_fp_*/hard_fn_* files before export.",
    )

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


def infer_direct_class_label(root: Path) -> int | None:
    name = root.name.lower()
    if name == "board":
        return CLASS_TO_INDEX["board"]
    if name == "no_board" or name.startswith("no_board_"):
        return CLASS_TO_INDEX["no_board"]
    return None


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

def _is_augmented_data_root(path: Path) -> bool:
    text = "/".join(part.lower() for part in path.parts)
    return "_aug" in text or "augmented" in text


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


def count_by_class(samples: list[Sample]) -> dict[str, int]:
    return {
        "board": sum(1 for s in samples if s.label == 1),
        "no_board": sum(1 for s in samples if s.label == 0),
    }


def split_train_val_test(
    samples: list[Sample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    by_class: dict[int, list[Sample]] = {0: [], 1: []}
    for sample in samples:
        by_class[sample.label].append(sample)

    rng = random.Random(seed)
    train: list[Sample] = []
    val: list[Sample] = []
    test: list[Sample] = []

    for label in (0, 1):
        cls = by_class[label]
        if len(cls) < 3:
            raise SystemExit(
                f"Need at least 3 samples for class {INDEX_TO_CLASS[label]} "
                f"to build train/val/test split. Found={len(cls)}"
            )
        rng.shuffle(cls)

        val_n = max(1, int(round(len(cls) * val_ratio)))
        test_n = max(1, int(round(len(cls) * test_ratio)))

        # Keep at least one sample in train remainder.
        max_eval = len(cls) - 1
        if val_n + test_n > max_eval:
            overflow = (val_n + test_n) - max_eval
            reduce_test = min(overflow, max(0, test_n - 1))
            test_n -= reduce_test
            overflow -= reduce_test
            if overflow > 0:
                val_n = max(1, val_n - overflow)

        val.extend(cls[:val_n])
        test.extend(cls[val_n : val_n + test_n])
        train.extend(cls[val_n + test_n :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def confusion_from_probs(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> dict[str, int]:
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


def threshold_to_dict(point: ThresholdStats) -> dict[str, float | int]:
    return {
        "threshold": point.threshold,
        "score": point.score,
        "accuracy": point.accuracy,
        "precision": point.precision,
        "recall": point.recall,
        "f1": point.f1,
        "tpr": point.tpr,
        "tnr": point.tnr,
        "tp": point.tp,
        "tn": point.tn,
        "fp": point.fp,
        "fn": point.fn,
    }


def build_crop_windows(crop_size_fraction: float) -> list[tuple[float, float, float, float]]:
    size = float(np.clip(crop_size_fraction, 0.50, 1.0))
    offset = float(np.clip(1.0 - size, 0.0, 0.5))
    center_offset = offset * 0.5
    return [
        (center_offset, center_offset, size, size),
        (0.0, 0.0, size, size),
        (offset, 0.0, size, size),
        (0.0, offset, size, size),
        (offset, offset, size, size),
    ]


def load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def crop_to_tensor(
    rgb: np.ndarray,
    crop: tuple[float, float, float, float],
    image_size: int,
) -> np.ndarray:
    source_height, source_width = rgb.shape[0], rgb.shape[1]
    left, top, crop_width, crop_height = crop

    source_left = left * max(1, source_width - 1)
    source_top = top * max(1, source_height - 1)
    source_crop_width = max(1.0, crop_width * source_width)
    source_crop_height = max(1.0, crop_height * source_height)
    source_right = min(float(source_width), source_left + source_crop_width)
    source_bottom = min(float(source_height), source_top + source_crop_height)

    sampled_width = max(1.0, source_right - source_left)
    sampled_height = max(1.0, source_bottom - source_top)

    ys = source_top + ((np.arange(image_size, dtype=np.float64) + 0.5) * sampled_height / image_size)
    xs = source_left + ((np.arange(image_size, dtype=np.float64) + 0.5) * sampled_width / image_size)

    ys = np.clip(np.floor(ys).astype(np.int32), 0, source_height - 1)
    xs = np.clip(np.floor(xs).astype(np.int32), 0, source_width - 1)

    crop_rgb = rgb[np.ix_(ys, xs)]
    return crop_rgb.astype(np.float32) / 255.0


def batch_predict_scores(
    model: tf.keras.Model,
    samples: list[Sample],
    image_size: int,
    batch_size: int,
    crop_size_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.array([s.label for s in samples], dtype=np.int32)
    strong_scores: list[float] = []
    fallback_scores: list[float] = []
    mean_scores: list[float] = []

    crops = build_crop_windows(crop_size_fraction)
    crop_count = len(crops)

    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        crop_batch: list[np.ndarray] = []
        for sample in chunk:
            rgb = load_rgb(sample.path)
            for crop in crops:
                crop_batch.append(crop_to_tensor(rgb=rgb, crop=crop, image_size=image_size))

        if not crop_batch:
            continue

        model_input = np.stack(crop_batch, axis=0).astype(np.float32)
        probs = model.predict(model_input, verbose=0).reshape(-1)
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        probs = np.clip(probs, 0.0, 1.0)

        for j in range(len(chunk)):
            start = j * crop_count
            p = probs[start : start + crop_count]
            if p.size == 0:
                strong_scores.append(0.0)
                fallback_scores.append(0.0)
                mean_scores.append(0.0)
                continue

            ordered = np.sort(p)
            top_k = ordered[-2:] if ordered.size >= 2 else ordered[-1:]
            strong_scores.append(float(np.mean(top_k)))
            fallback_scores.append(float(ordered[-1]))
            mean_scores.append(float(np.mean(p)))

    return (
        y_true,
        np.array(strong_scores, dtype=np.float32),
        np.array(fallback_scores, dtype=np.float32),
        np.array(mean_scores, dtype=np.float32),
    )


def summarize_hysteresis(
    y_true: np.ndarray,
    strong_scores: np.ndarray,
    fallback_scores: np.ndarray,
    accept_threshold: float,
    reject_threshold: float,
) -> dict[str, object]:
    strong_accept = strong_scores >= accept_threshold
    strong_reject = fallback_scores < reject_threshold
    gray = ~(strong_accept | strong_reject)

    # Gate output: reject blocks, accept+gray pass to detector fallback path.
    pass_mask = ~strong_reject
    gate_pred = pass_mask.astype(np.int32)
    conf = confusion_from_probs(y_true=y_true, probs=gate_pred.astype(np.float32), threshold=0.5)
    metrics = metrics_from_confusion(conf)

    def region_counts(mask: np.ndarray) -> dict[str, int]:
        board_n = int(np.sum((y_true == 1) & mask))
        no_board_n = int(np.sum((y_true == 0) & mask))
        return {
            "board": board_n,
            "no_board": no_board_n,
            "total": board_n + no_board_n,
        }

    board_total = int(np.sum(y_true == 1))
    no_board_total = int(np.sum(y_true == 0))

    strong_accept_tp = int(np.sum((y_true == 1) & strong_accept))
    strong_accept_fp = int(np.sum((y_true == 0) & strong_accept))
    strong_reject_tn = int(np.sum((y_true == 0) & strong_reject))
    strong_reject_fn = int(np.sum((y_true == 1) & strong_reject))

    strong_accept_precision = _safe_div(strong_accept_tp, strong_accept_tp + strong_accept_fp)
    strong_accept_board_recall = _safe_div(strong_accept_tp, board_total)
    strong_reject_no_board_precision = _safe_div(strong_reject_tn, strong_reject_tn + strong_reject_fn)
    strong_reject_no_board_recall = _safe_div(strong_reject_tn, no_board_total)

    return {
        "gate_pass_vs_block_metrics": {**metrics, **conf},
        "regions": {
            "strong_accept": region_counts(strong_accept),
            "gray_zone": region_counts(gray),
            "strong_reject": region_counts(strong_reject),
        },
        "strong_accept_metrics": {
            "precision": strong_accept_precision,
            "board_recall": strong_accept_board_recall,
            "tp": strong_accept_tp,
            "fp": strong_accept_fp,
        },
        "strong_reject_metrics": {
            "no_board_precision": strong_reject_no_board_precision,
            "no_board_recall": strong_reject_no_board_recall,
            "tn": strong_reject_tn,
            "fn": strong_reject_fn,
        },
    }


def export_hard_examples(
    out_dir: Path,
    samples: list[Sample],
    y_true: np.ndarray,
    fallback_scores: np.ndarray,
    reject_threshold: float,
    max_fp: int,
    max_fn: int,
    stride: int,
    purge_existing: bool,
) -> dict[str, object]:
    board_dir = out_dir / "board"
    no_board_dir = out_dir / "no_board"
    board_dir.mkdir(parents=True, exist_ok=True)
    no_board_dir.mkdir(parents=True, exist_ok=True)

    if purge_existing:
        for pattern in ("hard_fp_*", "hard_fn_*"):
            for p in board_dir.glob(pattern):
                if p.is_file():
                    p.unlink(missing_ok=True)
            for p in no_board_dir.glob(pattern):
                if p.is_file():
                    p.unlink(missing_ok=True)

    strong_reject = fallback_scores < reject_threshold
    fp_indices = np.where((y_true == 0) & (~strong_reject))[0].tolist()  # no_board passed by gate
    fn_indices = np.where((y_true == 1) & strong_reject)[0].tolist()     # board blocked by gate

    stride = max(1, stride)
    fp_indices = [idx for pos, idx in enumerate(fp_indices) if pos % stride == 0][: max(0, max_fp)]
    fn_indices = [idx for pos, idx in enumerate(fn_indices) if pos % stride == 0][: max(0, max_fn)]

    copied_fp = 0
    copied_fn = 0

    for rank, idx in enumerate(fp_indices):
        src = Path(samples[idx].path)
        dst = no_board_dir / f"hard_fp_{rank:04d}_{src.name}"
        shutil.copy2(src, dst)
        copied_fp += 1

    for rank, idx in enumerate(fn_indices):
        src = Path(samples[idx].path)
        dst = board_dir / f"hard_fn_{rank:04d}_{src.name}"
        shutil.copy2(src, dst)
        copied_fn += 1

    return {
        "out_dir": str(out_dir),
        "copied": {
            "hard_fp": copied_fp,
            "hard_fn": copied_fn,
        },
        "limits": {
            "max_hard_fp": max_fp,
            "max_hard_fn": max_fn,
            "stride": stride,
            "purged_existing": bool(purge_existing),
        },
    }


def main() -> None:
    args = parse_args()

    if not (0.0 < args.val_split < 0.5):
        raise SystemExit("val_split must be > 0 and < 0.5")
    if not (0.0 < args.test_split < 0.5):
        raise SystemExit("test_split must be > 0 and < 0.5")
    if args.val_split + args.test_split >= 0.95:
        raise SystemExit("val_split + test_split must be < 0.95")
    if args.batch_size <= 0:
        raise SystemExit("batch_size must be > 0")
    if args.image_size <= 16:
        raise SystemExit("image_size must be > 16")
    if args.hysteresis_min_gap < 0:
        raise SystemExit("hysteresis_min_gap must be >= 0")

    # Backward-compat alias mapping.
    reject_min = args.reject_threshold_min
    reject_max = args.reject_threshold_max
    reject_step = args.reject_threshold_step
    accept_min = args.accept_threshold_min
    accept_max = args.accept_threshold_max
    accept_step = args.accept_threshold_step
    reject_board_weight = args.reject_board_weight
    reject_no_board_weight = args.reject_no_board_weight

    if args.threshold_min is not None:
        reject_min = args.threshold_min
        accept_min = args.threshold_min
    if args.threshold_max is not None:
        reject_max = args.threshold_max
        accept_max = args.threshold_max
    if args.threshold_step is not None:
        reject_step = args.threshold_step
        accept_step = args.threshold_step
    if args.threshold_board_weight is not None:
        reject_board_weight = args.threshold_board_weight
    if args.threshold_no_board_weight is not None:
        reject_no_board_weight = args.threshold_no_board_weight

    if not (0.0 <= reject_min < reject_max <= 1.0):
        raise SystemExit("reject threshold range must satisfy 0 <= min < max <= 1")
    if not (0.0 <= accept_min < accept_max <= 1.0):
        raise SystemExit("accept threshold range must satisfy 0 <= min < max <= 1")
    if reject_step <= 0 or accept_step <= 0:
        raise SystemExit("threshold steps must be > 0")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    data_roots = [Path(p) for p in dict.fromkeys(args.data_dir)]
    for root in data_roots:
        if not root.exists():
            raise SystemExit(f"data-dir not found: {root}")

    augmented_roots = [root for root in data_roots if _is_augmented_data_root(root)]
    if augmented_roots and not args.allow_augmented_data:
        joined = ", ".join(str(p) for p in augmented_roots)
        raise SystemExit(
            "Refusing augmented dirs for calibration (val/test leakage risk). "
            "Use original data only, or pass --allow-augmented-data to override. "
            f"Found: {joined}"
        )

    samples = collect_samples(data_roots)
    if not samples:
        raise SystemExit("No samples found.")

    samples, dropped = filter_decodable(samples)
    if not samples:
        raise SystemExit("No decodable samples found.")

    train_samples, val_samples, test_samples = split_train_val_test(
        samples=samples,
        val_ratio=args.val_split,
        test_ratio=args.test_split,
        seed=args.seed,
    )

    model = tf.keras.models.load_model(model_path)

    val_y, val_strong, val_fallback, val_mean = batch_predict_scores(
        model=model,
        samples=val_samples,
        image_size=args.image_size,
        batch_size=args.batch_size,
        crop_size_fraction=args.crop_size_fraction,
    )
    test_y, test_strong, test_fallback, test_mean = batch_predict_scores(
        model=model,
        samples=test_samples,
        image_size=args.image_size,
        batch_size=args.batch_size,
        crop_size_fraction=args.crop_size_fraction,
    )

    reject_points = evaluate_thresholds(
        y_true=val_y,
        probs=val_fallback,
        threshold_min=reject_min,
        threshold_max=reject_max,
        threshold_step=reject_step,
        board_weight=reject_board_weight,
        no_board_weight=reject_no_board_weight,
    )
    best_reject = select_best(reject_points)

    accept_points = evaluate_thresholds(
        y_true=val_y,
        probs=val_strong,
        threshold_min=accept_min,
        threshold_max=accept_max,
        threshold_step=accept_step,
        board_weight=args.accept_board_weight,
        no_board_weight=args.accept_no_board_weight,
    )

    min_accept = best_reject.threshold + args.hysteresis_min_gap
    accept_candidates = [p for p in accept_points if p.threshold >= min_accept]
    accept_constraint_relaxed = False
    if not accept_candidates:
        accept_constraint_relaxed = True
        accept_candidates = accept_points
    best_accept = select_best(accept_candidates)

    val_default_conf = confusion_from_probs(y_true=val_y, probs=val_strong, threshold=0.5)
    val_default_metrics = metrics_from_confusion(val_default_conf)
    test_default_conf = confusion_from_probs(y_true=test_y, probs=test_strong, threshold=0.5)
    test_default_metrics = metrics_from_confusion(test_default_conf)

    val_accept_conf = confusion_from_probs(y_true=val_y, probs=val_strong, threshold=best_accept.threshold)
    val_accept_metrics = metrics_from_confusion(val_accept_conf)
    test_accept_conf = confusion_from_probs(y_true=test_y, probs=test_strong, threshold=best_accept.threshold)
    test_accept_metrics = metrics_from_confusion(test_accept_conf)

    val_reject_conf = confusion_from_probs(y_true=val_y, probs=val_fallback, threshold=best_reject.threshold)
    val_reject_metrics = metrics_from_confusion(val_reject_conf)
    test_reject_conf = confusion_from_probs(y_true=test_y, probs=test_fallback, threshold=best_reject.threshold)
    test_reject_metrics = metrics_from_confusion(test_reject_conf)

    val_hysteresis = summarize_hysteresis(
        y_true=val_y,
        strong_scores=val_strong,
        fallback_scores=val_fallback,
        accept_threshold=best_accept.threshold,
        reject_threshold=best_reject.threshold,
    )
    test_hysteresis = summarize_hysteresis(
        y_true=test_y,
        strong_scores=test_strong,
        fallback_scores=test_fallback,
        accept_threshold=best_accept.threshold,
        reject_threshold=best_reject.threshold,
    )

    hard_export: dict[str, object] | None = None
    if args.hard_negatives_dir:
        hard_export = export_hard_examples(
            out_dir=Path(args.hard_negatives_dir),
            samples=test_samples,
            y_true=test_y,
            fallback_scores=test_fallback,
            reject_threshold=best_reject.threshold,
            max_fp=args.max_hard_fp,
            max_fn=args.max_hard_fn,
            stride=args.hard_copy_stride,
            purge_existing=args.hard_purge_existing,
        )

    top_reject = sorted(reject_points, key=lambda p: p.score, reverse=True)[:5]
    top_accept = sorted(accept_points, key=lambda p: p.score, reverse=True)[:5]

    out = {
        "model_path": str(model_path),
        "data_dirs": [str(p) for p in data_roots],
        "dropped_invalid_images": dropped,
        "split_policy": {
            "val_split": args.val_split,
            "test_split": args.test_split,
            "seed": args.seed,
        },
        "split_counts": {
            "train": {
                "samples": len(train_samples),
                "class_counts": count_by_class(train_samples),
            },
            "val": {
                "samples": len(val_samples),
                "class_counts": count_by_class(val_samples),
            },
            "test": {
                "samples": len(test_samples),
                "class_counts": count_by_class(test_samples),
            },
        },
        "inference_policy": {
            "image_size": args.image_size,
            "crop_size_fraction": args.crop_size_fraction,
            "crop_count": 5,
            "crop_order": [
                "center",
                "top_left",
                "top_right",
                "bottom_left",
                "bottom_right",
            ],
            "strong_score": "top2_mean",
            "fallback_score": "max",
        },
        "default_threshold": 0.5,
        "default_threshold_metrics_val": {**val_default_metrics, **val_default_conf},
        "default_threshold_metrics_test": {**test_default_metrics, **test_default_conf},
        "reject_threshold_policy": {
            "min": reject_min,
            "max": reject_max,
            "step": reject_step,
            "board_weight": reject_board_weight,
            "no_board_weight": reject_no_board_weight,
        },
        "accept_threshold_policy": {
            "min": accept_min,
            "max": accept_max,
            "step": accept_step,
            "board_weight": args.accept_board_weight,
            "no_board_weight": args.accept_no_board_weight,
            "hysteresis_min_gap": args.hysteresis_min_gap,
            "constraint_relaxed": accept_constraint_relaxed,
        },
        "recommended_reject_threshold": best_reject.threshold,
        "recommended_reject_threshold_metrics_val": {
            **val_reject_metrics,
            **val_reject_conf,
            "score": best_reject.score,
        },
        "recommended_reject_threshold_metrics_test": {
            **test_reject_metrics,
            **test_reject_conf,
        },
        "recommended_accept_threshold": best_accept.threshold,
        "recommended_accept_threshold_metrics_val": {
            **val_accept_metrics,
            **val_accept_conf,
            "score": best_accept.score,
        },
        "recommended_accept_threshold_metrics_test": {
            **test_accept_metrics,
            **test_accept_conf,
        },
        "recommended_threshold": best_accept.threshold,
        "recommended_threshold_metrics": {
            **test_accept_metrics,
            **test_accept_conf,
        },
        "recommended_hysteresis": {
            "accept_threshold": best_accept.threshold,
            "reject_threshold": best_reject.threshold,
            "gray_zone": {
                "min_inclusive": best_reject.threshold,
                "max_exclusive": best_accept.threshold,
            },
        },
        "hysteresis_val_summary": val_hysteresis,
        "hysteresis_test_summary": test_hysteresis,
        "score_distributions": {
            "val": {
                "strong_mean": float(np.mean(val_strong)) if len(val_strong) else 0.0,
                "fallback_mean": float(np.mean(val_fallback)) if len(val_fallback) else 0.0,
                "crop_mean": float(np.mean(val_mean)) if len(val_mean) else 0.0,
            },
            "test": {
                "strong_mean": float(np.mean(test_strong)) if len(test_strong) else 0.0,
                "fallback_mean": float(np.mean(test_fallback)) if len(test_fallback) else 0.0,
                "crop_mean": float(np.mean(test_mean)) if len(test_mean) else 0.0,
            },
        },
        "top_reject_thresholds": [threshold_to_dict(p) for p in top_reject],
        "top_accept_thresholds": [threshold_to_dict(p) for p in top_accept],
        "top_thresholds": [threshold_to_dict(p) for p in top_accept],
        "hard_examples_export": hard_export,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(
        "[calibrate-board] split "
        f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)} "
        f"dropped_invalid={dropped}"
    )
    print(
        "[calibrate-board] reject_threshold "
        f"val={best_reject.threshold:.3f} "
        f"metrics_val={{recall:{val_reject_metrics['recall']:.4f},tnr:{val_reject_metrics['tnr']:.4f}}}"
    )
    print(
        "[calibrate-board] accept_threshold "
        f"val={best_accept.threshold:.3f} "
        f"metrics_val={{recall:{val_accept_metrics['recall']:.4f},tnr:{val_accept_metrics['tnr']:.4f}}}"
    )
    print(
        "[calibrate-board] hysteresis_test "
        f"accept={best_accept.threshold:.3f} reject={best_reject.threshold:.3f} "
        f"gate_metrics={test_hysteresis['gate_pass_vs_block_metrics']}"
    )
    if hard_export is not None:
        copied = hard_export["copied"]
        print(
            "[calibrate-board] hard_examples "
            f"fp={copied['hard_fp']} fn={copied['hard_fn']} out={hard_export['out_dir']}"
        )
    print(f"[calibrate-board] report={out_path}")


if __name__ == "__main__":
    main()


