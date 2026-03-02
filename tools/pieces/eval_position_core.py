#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate 13-class TFLite model on top-down board images and compute:
- per-square accuracy
- exact piece-placement FEN match

Input jsonl format:
  {"image":"path/to/board.png","fen":"piece_placement_only"}
"""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()

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
                # Fallback if quantization metadata is missing.
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

    exact = 0
    sq_accs: List[float] = []

    for row in rows:
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            image_path = (args.root_dir / image_path).resolve()
        expected = str(row["fen"]).strip()

        img = Image.open(image_path).convert("RGB")
        squares = slice_board(img)

        batch = []
        for square_img in squares:
            square_img = square_img.resize((args.img_size, args.img_size), Image.BILINEAR)
            arr = np.asarray(square_img).astype(np.float32) / 255.0
            batch.append(arr)
        batch_np = np.stack(batch, axis=0)

        probs = run_one(batch_np)
        pred_ids = probs.argmax(axis=1).tolist()
        pred_classes = [ID_TO_CLASS[i] for i in pred_ids]

        pred_placement = build_piece_placement(pred_classes)
        if pred_placement == expected:
            exact += 1

        sq_accs.append(per_square_acc(expected, pred_classes))

    fen_exact = exact / max(1, len(rows))
    per_sq = float(np.mean(sq_accs)) if sq_accs else 0.0

    print(
        {
            "boards": len(rows),
            "fen_exact_match": round(fen_exact * 100.0, 1),
            "per_square_acc": round(per_sq * 100.0, 2),
        }
    )


if __name__ == "__main__":
    main()
