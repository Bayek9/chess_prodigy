#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train 13-class chess piece classifier and export TFLite models.

Expected manifest format (from import_piece_archives.py):
  source,archive,origin_path,split,label,row,col,output_path

`output_path` is relative to --dataset-dir and points to the crop image file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


CLASSES_13 = [
    "empty",
    "wP",
    "wN",
    "wB",
    "wR",
    "wQ",
    "wK",
    "bP",
    "bN",
    "bB",
    "bR",
    "bQ",
    "bK",
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES_13)}


def normalize_label(raw: str) -> str:
    """Accept empty / wP..bK / single-char FEN symbols."""
    if raw is None:
        return "empty"
    s = str(raw).strip()
    if not s or s.lower() in {"empty", "blank", "none"}:
        return "empty"

    if len(s) == 1 and s in "PNBRQKpnbrqk":
        return ("w" + s.upper()) if s.isupper() else ("b" + s.upper())

    if len(s) == 2 and s[0].lower() in {"w", "b"} and s[1].upper() in "PNBRQK":
        return s[0].lower() + s[1].upper()

    raise ValueError(f"Unknown label: {raw}")


def find_manifest(dataset_dir: Path) -> Path:
    preferred = dataset_dir / "labels" / "piece_samples.csv"
    if preferred.exists():
        return preferred

    for candidate in dataset_dir.rglob("*.csv"):
        if "piece_samples" in candidate.name.lower() or "manifest" in candidate.name.lower():
            return candidate

    raise FileNotFoundError(
        f"Manifest not found in {dataset_dir}. Expected labels/piece_samples.csv"
    )


def load_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return pd.DataFrame(rows)

    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        rows = obj.get("rows", obj)
        return pd.DataFrame(rows)

    raise ValueError(f"Unsupported manifest extension: {path}")


def resolve_columns(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    cols = {c.lower(): c for c in df.columns}

    path_col = cols.get("output_path") or cols.get("path") or cols.get("image_path")
    label_col = cols.get("label") or cols.get("class")
    split_col = cols.get("split")
    source_col = cols.get("source")

    if not path_col:
        raise KeyError(f"Path column not found. Columns: {list(df.columns)}")
    if not label_col:
        raise KeyError(f"Label column not found. Columns: {list(df.columns)}")
    if not split_col:
        raise KeyError(f"Split column not found. Columns: {list(df.columns)}")
    if not source_col:
        raise KeyError(f"Source column not found. Columns: {list(df.columns)}")

    return path_col, label_col, split_col, source_col


def normalize_source(raw: str) -> str:
    s = str(raw).strip().lower()
    if s in {"yolo", "yolo_2d", "coco_2d", "coco_2d_grid", "2d", "screen_2d"}:
        return "yolo"
    if s in {"real", "real_pieces", "photo"}:
        return "real"
    return "unknown"


def normalize_split(raw: str) -> str:
    s = str(raw).strip().lower()
    if s == "valid":
        return "val"
    if s in {"train", "val", "test"}:
        return s
    return "train"


def build_df(dataset_dir: Path, source: str) -> pd.DataFrame:
    manifest = find_manifest(dataset_dir)
    df = load_manifest(manifest)
    path_col, label_col, split_col, source_col = resolve_columns(df)

    out = pd.DataFrame()
    out["__path"] = df[path_col].astype(str)
    out["__label"] = df[label_col].astype(str).apply(normalize_label)
    out["__split"] = df[split_col].astype(str).apply(normalize_split)
    out["__source"] = df[source_col].astype(str).apply(normalize_source)

    if source != "all":
        out = out[out["__source"].eq(source)].copy()

    out["__abs_path"] = out["__path"].apply(
        lambda rel: str((dataset_dir / rel).resolve()) if not Path(rel).is_absolute() else rel
    )
    out["__y"] = out["__label"].map(CLASS_TO_ID)

    missing = out[out["__y"].isna()]
    if not missing.empty:
        raise ValueError(f"Unmapped labels found: {missing['__label'].unique()[:10]}")

    exists_mask = out["__abs_path"].apply(lambda p: Path(p).exists())
    dropped = int((~exists_mask).sum())
    if dropped > 0:
        print(f"[warn] dropping {dropped} rows with missing files")
        out = out[exists_mask].copy()

    return out


def write_clean_manifest(df: pd.DataFrame, out_path: Path) -> None:
    """Write the post-normalization filtered manifest used by training."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean = df[["__source", "__split", "__label", "__path", "__abs_path", "__y"]].copy()
    clean = clean.rename(
        columns={
            "__source": "source",
            "__split": "split",
            "__label": "label",
            "__path": "output_path",
            "__abs_path": "abs_path",
            "__y": "class_id",
        }
    )
    clean.to_csv(out_path, index=False)
    print(f"[info] wrote clean manifest: {out_path} ({len(clean)} rows)")


def make_tf_dataset(
    df: pd.DataFrame,
    img_size: int,
    batch: int,
    augment: bool,
    shuffle: bool,
    repeat: bool = False,
    ignore_decode_errors: bool = False,
) -> tf.data.Dataset:
    paths = df["__abs_path"].values.astype(str)
    labels = df["__y"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(df), 50000), reshuffle_each_iteration=True)

    def _load(path: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [img_size, img_size], method="bilinear")
        return img, y

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if ignore_decode_errors:
        # Skip corrupt/unreadable images instead of aborting the full training run.
        ds = ds.ignore_errors()

    if augment:
        aug = tf.keras.Sequential(
            [
                tf.keras.layers.RandomBrightness(0.15),
                tf.keras.layers.RandomContrast(0.15),
                tf.keras.layers.RandomTranslation(0.02, 0.02),
                tf.keras.layers.RandomZoom(0.05),
            ],
            name="augment",
        )

        def _aug(img: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            return aug(img, training=True), y

        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(img_size: int, num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs * 255.0)
    x = backbone(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

def initialize_from_checkpoint(model: tf.keras.Model, init_path: Path) -> None:
    """Initialize model weights from a previous run (.keras/.h5/weights)."""
    if not init_path.exists():
        raise FileNotFoundError(f"init checkpoint not found: {init_path}")

    # Keras v3 full-model format (.keras or SavedModel dir).
    if init_path.suffix.lower() == ".keras" or init_path.is_dir():
        loaded = tf.keras.models.load_model(str(init_path), compile=False)
        model.set_weights(loaded.get_weights())
        print(f"[info] initialized from model: {init_path}")
        return

    # Weight-only files (e.g. .h5/.ckpt).
    model.load_weights(str(init_path))
    print(f"[info] initialized from weights: {init_path}")



def compute_class_weights(
    train_df: pd.DataFrame,
    mode: str,
    cap: Optional[float],
) -> Optional[Dict[int, float]]:
    if mode == "none":
        return None

    counts = train_df["__y"].value_counts().to_dict()
    total = float(sum(counts.values()))

    raw: Dict[int, float] = {}
    for cls_id in range(len(CLASSES_13)):
        c = float(counts.get(cls_id, 1))
        inv = total / (float(len(CLASSES_13)) * c)
        if mode == "inverse":
            w = inv
        elif mode == "sqrt":
            w = float(np.sqrt(inv))
        else:
            raise ValueError(f"Unsupported class weight mode: {mode}")
        raw[cls_id] = w

    # Normalize to mean=1 so lr/optimizer behavior stays comparable across modes.
    mean_w = sum(raw.values()) / float(len(raw))
    if mean_w > 0:
        raw = {k: (v / mean_w) for k, v in raw.items()}

    if cap is not None and cap > 0:
        raw = {k: min(v, cap) for k, v in raw.items()}

    return raw


def sparse_focal_loss(
    num_classes: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
    from_logits: bool = False,
):
    """Sparse-label focal loss using one-hot expansion inside the loss."""

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

        if from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        y_true_oh = tf.one_hot(y_true, depth=num_classes)
        ce = tf.keras.losses.categorical_crossentropy(y_true_oh, y_pred)
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
        fl = alpha * tf.pow(1.0 - p_t, gamma) * ce
        return tf.reduce_mean(fl)

    return loss


def export_tflite(
    saved_model_dir: Path,
    out_dir: Path,
    want_int8: bool,
    rep_ds: tf.data.Dataset,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    tflite_fp32 = converter.convert()
    (out_dir / "piece_13cls_fp32.tflite").write_bytes(tflite_fp32)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter.convert()
    (out_dir / "piece_13cls_fp16.tflite").write_bytes(tflite_fp16)

    if want_int8:

        def rep_gen() -> Iterable[list[np.ndarray]]:
            for batch_imgs, _ in rep_ds.take(200):
                yield [batch_imgs]

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_int8 = converter.convert()
        (out_dir / "piece_13cls_int8.tflite").write_bytes(tflite_int8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--source", choices=["yolo", "real", "all"], default="yolo")
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--export-int8", action="store_true")
    parser.add_argument("--loss", choices=["sparse_ce", "sparse_focal"], default="sparse_ce")
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--class-weight-mode", choices=["none", "sqrt", "inverse"], default="sqrt")
    parser.add_argument("--class-weight-cap", type=float, default=5.0)
    parser.add_argument("--clean-manifest-out", type=Path)
    parser.add_argument("--init-from", type=Path, help="Optional checkpoint/model to initialize before training")
    parser.add_argument("--no-class-weights", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--ignore-decode-errors",
        action="store_true",
        help="Drop unreadable/corrupt images during tf.data loading.",
    )
    args = parser.parse_args()

    df = build_df(args.dataset_dir, args.source)

    if args.clean_manifest_out:
        write_clean_manifest(df, args.clean_manifest_out)

    train_df = df[df["__split"].eq("train")].copy()
    val_df = df[df["__split"].eq("val")].copy()
    test_df = df[df["__split"].eq("test")].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("Train or validation split is empty. Check manifest/splits.")

    print(
        f"Loaded source={args.source}: "
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
    )

    train_ds = make_tf_dataset(
        train_df,
        args.img_size,
        args.batch,
        augment=True,
        shuffle=True,
        repeat=True,
        ignore_decode_errors=args.ignore_decode_errors,
    )
    val_ds = make_tf_dataset(
        val_df,
        args.img_size,
        args.batch,
        augment=False,
        shuffle=False,
        repeat=False,
        ignore_decode_errors=args.ignore_decode_errors,
    )
    test_ds = make_tf_dataset(
        test_df,
        args.img_size,
        args.batch,
        augment=False,
        shuffle=False,
        repeat=False,
        ignore_decode_errors=args.ignore_decode_errors,
    )

    train_steps = max(1, int(np.ceil(len(train_df) / args.batch)))

    model = build_model(args.img_size, len(CLASSES_13))
    if args.init_from:
        initialize_from_checkpoint(model, args.init_from)

    if args.loss == "sparse_focal":
        loss_fn = sparse_focal_loss(
            num_classes=len(CLASSES_13),
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
            from_logits=False,
        )
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=loss_fn,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    class_weight_mode = "none" if args.no_class_weights else args.class_weight_mode
    if args.loss == "sparse_focal" and class_weight_mode != "none":
        print("[info] disabling class weights for sparse_focal loss")
        class_weight_mode = "none"
    class_weights = compute_class_weights(train_df, class_weight_mode, args.class_weight_cap)
    print(
        f"[info] loss={args.loss} focal_alpha={args.focal_alpha} focal_gamma={args.focal_gamma}"
    )
    print(f"[info] class_weight_mode={class_weight_mode} cap={args.class_weight_cap}")

    ckpt_dir = args.out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best.keras"),
            monitor="val_acc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_acc",
            mode="max",
            factor=0.5,
            patience=2,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=train_steps,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    if args.fine_tune_epochs > 0:
        # Fine-tune
        backbone = model.layers[1]
        backbone.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.fine_tune_lr),
            loss=loss_fn,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            steps_per_epoch=train_steps,
            class_weight=class_weights,
            callbacks=callbacks,
        )
    else:
        print("[info] fine_tune_epochs=0, skipping fine-tune stage")

    if len(test_df) > 0:
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        print({"test_loss": float(test_loss), "test_acc": float(test_acc)})
    else:
        print("[warn] test split empty; skipping test evaluation")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    saved_model_dir = args.out_dir / "saved_model"
    model.export(str(saved_model_dir))

    tflite_dir = args.out_dir / "tflite"
    export_tflite(saved_model_dir, tflite_dir, args.export_int8, train_ds)

    (args.out_dir / "labels_13.txt").write_text("\n".join(CLASSES_13) + "\n", encoding="utf-8")
    print(f"Exported to: {args.out_dir}")


if __name__ == "__main__":
    main()
