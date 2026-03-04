#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate 13-class TFLite model on top-down board images and compute:
- per-square accuracy
- exact piece-placement FEN match
- board-level error stats
- per-class accuracy and top confusions

Input jsonl format:
  {"image":"path/to/board.png","fen":"piece_placement_only"}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import chess
import numpy as np
from PIL import Image
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
ID_TO_CLASS = {i: c for i, c in enumerate(CLASSES_13)}
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES_13)}


def cls_to_fen_char(cls_name: str) -> str:
    if cls_name == "empty":
        return ""
    color = cls_name[0]
    piece = cls_name[1]
    return piece.upper() if color == "w" else piece.lower()


def load_positions(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def square_name(r: int, c: int) -> str:
    return f"{chr(ord('a') + c)}{8 - r}"


def slice_board(img: Image.Image) -> List[Image.Image]:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))

    cell = s / 8.0
    squares: List[Image.Image] = []
    for r in range(8):
        for c in range(8):
            x0 = int(round(c * cell))
            y0 = int(round(r * cell))
            x1 = int(round((c + 1) * cell))
            y1 = int(round((r + 1) * cell))
            squares.append(img.crop((x0, y0, x1, y1)))
    return squares


def build_piece_placement(pred_classes: List[str]) -> str:
    rows: List[str] = []
    idx = 0
    for _ in range(8):
        empties = 0
        fen_row = ""
        for _ in range(8):
            ch = cls_to_fen_char(pred_classes[idx])
            idx += 1
            if ch == "":
                empties += 1
            else:
                if empties:
                    fen_row += str(empties)
                    empties = 0
                fen_row += ch
        if empties:
            fen_row += str(empties)
        rows.append(fen_row)
    return "/".join(rows)


def expected_classes_from_board_fen(board_fen: str) -> List[str]:
    board = chess.Board(None)
    board.set_board_fen(board_fen)

    expected: List[str] = []
    for rank in range(8, 0, -1):
        for file_idx in range(8):
            sq = chess.square(file_idx, rank - 1)
            piece = board.piece_at(sq)
            if piece is None:
                expected.append("empty")
            else:
                color = "w" if piece.color == chess.WHITE else "b"
                expected.append(color + piece.symbol().upper())
    return expected


def per_square_acc(expected_placement: str, pred_classes: List[str]) -> float:
    expected = expected_classes_from_board_fen(expected_placement)
    correct = sum(1 for a, b in zip(expected, pred_classes) if a == b)
    return correct / 64.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite", required=True, type=Path)
    parser.add_argument("--positions", required=True, type=Path)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("."),
        help="Base directory for relative image paths in positions jsonl.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        help="Optional output CSV with one row per board square prediction.",
    )
    parser.add_argument(
        "--dump-mismatches-dir",
        type=Path,
        help="Optional directory to dump mismatched square crops.",
    )
    args = parser.parse_args()

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    if args.dump_mismatches_dir:
        args.dump_mismatches_dir.mkdir(parents=True, exist_ok=True)

    interpreter = tf.lite.Interpreter(model_path=str(args.tflite))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    in_scale, in_zero = in_det.get("quantization", (0.0, 0))
    out_scale, out_zero = out_det.get("quantization", (0.0, 0))

    def prepare_input(batch: np.ndarray, dtype: np.dtype) -> np.ndarray:
        if dtype == np.float32:
            return batch.astype(np.float32)
        if dtype in (np.int8, np.uint8):
            if in_scale and in_scale > 0:
                quant = np.round(batch / in_scale + in_zero)
            else:
                quant = np.round(batch * 255.0)
            if dtype == np.int8:
                quant = np.clip(quant, -128, 127)
            else:
                quant = np.clip(quant, 0, 255)
            return quant.astype(dtype)
        raise ValueError(f"Unsupported TFLite input dtype: {dtype}")

    def decode_output(raw: np.ndarray, dtype: np.dtype) -> np.ndarray:
        if dtype == np.float32:
            return raw.astype(np.float32)
        if dtype in (np.int8, np.uint8):
            if out_scale and out_scale > 0:
                return (raw.astype(np.float32) - out_zero) * out_scale
            return raw.astype(np.float32)
        return raw.astype(np.float32)

    def run_one(batch: np.ndarray) -> np.ndarray:
        prepared = prepare_input(batch, in_det["dtype"])
        interpreter.resize_tensor_input(in_det["index"], prepared.shape, strict=True)
        interpreter.allocate_tensors()

        cur_in = interpreter.get_input_details()[0]
        cur_out = interpreter.get_output_details()[0]
        interpreter.set_tensor(cur_in["index"], prepared)
        interpreter.invoke()
        raw_out = interpreter.get_tensor(cur_out["index"])
        return decode_output(raw_out, cur_out["dtype"])

    rows = load_positions(args.positions)
    if not rows:
        raise ValueError("No position rows found in positions file")

    csv_file = None
    csv_writer = None
    if args.csv_out:
        csv_file = args.csv_out.open("w", encoding="utf-8", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "board_idx",
                "image",
                "r",
                "c",
                "square",
                "true",
                "pred",
                "prob_pred",
                "prob_true",
                "top2",
                "margin",
            ],
        )
        csv_writer.writeheader()

    exact = 0
    sq_accs: List[float] = []
    wrong_squares_per_board: List[int] = []
    all_true_ids: List[int] = []
    all_pred_ids: List[int] = []
    skipped_images: List[Dict[str, str]] = []

    for board_idx, row in enumerate(rows):
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            image_path = (args.root_dir / image_path).resolve()
        expected = str(row["fen"]).strip()

        try:
            img = Image.open(image_path).convert("RGB")
            squares = slice_board(img)

            batch = []
            for square_img in squares:
                square_img = square_img.resize((args.img_size, args.img_size), Image.BILINEAR)
                arr = np.asarray(square_img).astype(np.float32) / 255.0
                batch.append(arr)
            batch_np = np.stack(batch, axis=0)
        except Exception as e:
            skipped_images.append({"board_idx": str(board_idx), "image": image_path.as_posix(), "error": str(e)})
            continue

        probs = run_one(batch_np)
        pred_ids = probs.argmax(axis=1).tolist()
        pred_classes = [ID_TO_CLASS[i] for i in pred_ids]
        expected_classes = expected_classes_from_board_fen(expected)
        expected_ids = [CLASS_TO_ID[c] for c in expected_classes]

        pred_placement = build_piece_placement(pred_classes)
        if pred_placement == expected:
            exact += 1

        wrong_count = sum(1 for t, p in zip(expected_ids, pred_ids) if t != p)
        wrong_squares_per_board.append(wrong_count)
        all_true_ids.extend(expected_ids)
        all_pred_ids.extend(pred_ids)
        sq_accs.append(per_square_acc(expected, pred_classes))

        if csv_writer or args.dump_mismatches_dir:
            image_name = image_path.as_posix()
            stem = image_path.stem
            for sq_idx, (true_id, pred_id) in enumerate(zip(expected_ids, pred_ids)):
                row_probs = probs[sq_idx].astype(np.float32)
                top2_ids = np.argsort(row_probs)[::-1][:2]
                top1_prob = float(row_probs[top2_ids[0]])
                top2_prob = float(row_probs[top2_ids[1]]) if len(top2_ids) > 1 else 0.0
                prob_true = float(row_probs[true_id])
                margin = top1_prob - top2_prob
                r = sq_idx // 8
                c = sq_idx % 8

                if csv_writer:
                    csv_writer.writerow(
                        {
                            "board_idx": board_idx,
                            "image": image_name,
                            "r": r,
                            "c": c,
                            "square": square_name(r, c),
                            "true": ID_TO_CLASS[true_id],
                            "pred": ID_TO_CLASS[pred_id],
                            "prob_pred": round(top1_prob, 6),
                            "prob_true": round(prob_true, 6),
                            "top2": f"{ID_TO_CLASS[top2_ids[0]]}:{top1_prob:.6f}|{ID_TO_CLASS[top2_ids[1]]}:{top2_prob:.6f}",
                            "margin": round(margin, 6),
                        }
                    )

                if args.dump_mismatches_dir and true_id != pred_id:
                    out_name = (
                        f"b{board_idx:03d}_{stem}_{square_name(r, c)}"
                        f"_true-{ID_TO_CLASS[true_id]}_pred-{ID_TO_CLASS[pred_id]}.png"
                    )
                    squares[sq_idx].save(args.dump_mismatches_dir / out_name)

    if csv_file:
        csv_file.close()

    processed_boards = len(wrong_squares_per_board)
    if processed_boards == 0:
        raise RuntimeError("No boards processed successfully (all images failed to decode/load)")

    fen_exact = exact / processed_boards
    per_sq = float(np.mean(sq_accs)) if sq_accs else 0.0
    avg_wrong = float(np.mean(wrong_squares_per_board)) if wrong_squares_per_board else 0.0

    board_count = processed_boards
    boards_le_1 = sum(1 for w in wrong_squares_per_board if w <= 1) / board_count
    boards_le_2 = sum(1 for w in wrong_squares_per_board if w <= 2) / board_count
    boards_le_4 = sum(1 for w in wrong_squares_per_board if w <= 4) / board_count

    conf_mat = np.zeros((len(CLASSES_13), len(CLASSES_13)), dtype=np.int64)
    for true_id, pred_id in zip(all_true_ids, all_pred_ids):
        conf_mat[true_id, pred_id] += 1

    per_class_acc: List[Dict[str, object]] = []
    for i, cls_name in enumerate(CLASSES_13):
        support = int(conf_mat[i, :].sum())
        correct = int(conf_mat[i, i])
        acc = (correct / support * 100.0) if support > 0 else None
        per_class_acc.append(
            {
                "class": cls_name,
                "support": support,
                "acc_pct": None if acc is None else round(acc, 2),
            }
        )

    confusion_rows = []
    for true_id in range(len(CLASSES_13)):
        for pred_id in range(len(CLASSES_13)):
            if true_id == pred_id:
                continue
            count = int(conf_mat[true_id, pred_id])
            if count <= 0:
                continue
            confusion_rows.append(
                {
                    "true": ID_TO_CLASS[true_id],
                    "pred": ID_TO_CLASS[pred_id],
                    "count": count,
                }
            )
    confusion_rows.sort(key=lambda x: x["count"], reverse=True)

    print(
        {
            "boards": processed_boards,
            "fen_exact_match": round(fen_exact * 100.0, 1),
            "per_square_acc": round(per_sq * 100.0, 2),
            "avg_wrong_squares_per_board": round(avg_wrong, 2),
            "boards_le_1_error_pct": round(boards_le_1 * 100.0, 1),
            "boards_le_2_error_pct": round(boards_le_2 * 100.0, 1),
            "boards_le_4_error_pct": round(boards_le_4 * 100.0, 1),
            "top_confusions": confusion_rows[:5],
            "per_class_acc": per_class_acc,
            "csv_out": None if not args.csv_out else str(args.csv_out),
            "dump_mismatches_dir": None if not args.dump_mismatches_dir else str(args.dump_mismatches_dir),
            "skipped_images": len(skipped_images),
            "skipped_examples": skipped_images[:5],
        }
    )


if __name__ == "__main__":
    main()
