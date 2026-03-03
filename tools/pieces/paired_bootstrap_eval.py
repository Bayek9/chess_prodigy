#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired bootstrap comparison for board-level eval CSV outputs.

Expected CSV format (from eval_position_core.py):
  board_idx,image,r,c,square,true,pred,prob_pred,prob_true,top2,margin
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


class BoardStats:
    def __init__(self, board_idx: int, wrong: int, total: int) -> None:
        self.board_idx = board_idx
        self.wrong = wrong
        self.total = total


def load_board_stats(csv_path: Path) -> Dict[int, BoardStats]:
    rows: Dict[int, BoardStats] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["board_idx"])
            is_wrong = row["true"] != row["pred"]
            if idx not in rows:
                rows[idx] = BoardStats(board_idx=idx, wrong=0, total=0)
            rows[idx].total += 1
            if is_wrong:
                rows[idx].wrong += 1
    return rows


def metrics_for_indices(data: Dict[int, BoardStats], indices: np.ndarray) -> dict:
    wrongs = np.array([data[int(i)].wrong for i in indices], dtype=np.float64)
    totals = np.array([data[int(i)].total for i in indices], dtype=np.float64)
    total_cases = float(totals.sum())
    if total_cases <= 0:
        raise ValueError("No samples found for selected indices")

    total_wrong = float(wrongs.sum())
    per_square = 1.0 - (total_wrong / total_cases)
    avg_wrong = float(np.mean(wrongs))
    le2 = float(np.mean(wrongs <= 2.0))
    le4 = float(np.mean(wrongs <= 4.0))

    return {
        "per_square": per_square,
        "avg_wrong": avg_wrong,
        "boards_le_2": le2,
        "boards_le_4": le4,
    }


def ci_percentile(values: np.ndarray, low_q: float = 2.5, high_q: float = 97.5) -> tuple[float, float]:
    return float(np.percentile(values, low_q)), float(np.percentile(values, high_q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-a", required=True, type=Path, help="Baseline CSV")
    parser.add_argument("--csv-b", required=True, type=Path, help="Candidate CSV")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    a = load_board_stats(args.csv_a)
    b = load_board_stats(args.csv_b)

    common = sorted(set(a.keys()) & set(b.keys()))
    if not common:
        raise ValueError("No common board_idx between both CSV files")

    dropped_a = sorted(set(a.keys()) - set(common))
    dropped_b = sorted(set(b.keys()) - set(common))

    base_a = metrics_for_indices(a, np.array(common, dtype=np.int64))
    base_b = metrics_for_indices(b, np.array(common, dtype=np.int64))

    rng = np.random.default_rng(args.seed)
    n = len(common)
    index_array = np.array(common, dtype=np.int64)

    deltas_per_square = np.zeros(args.iters, dtype=np.float64)
    deltas_avg_wrong = np.zeros(args.iters, dtype=np.float64)
    deltas_le2 = np.zeros(args.iters, dtype=np.float64)
    deltas_le4 = np.zeros(args.iters, dtype=np.float64)

    for i in range(args.iters):
        sample = rng.choice(index_array, size=n, replace=True)
        m_a = metrics_for_indices(a, sample)
        m_b = metrics_for_indices(b, sample)
        deltas_per_square[i] = m_b["per_square"] - m_a["per_square"]
        deltas_avg_wrong[i] = m_b["avg_wrong"] - m_a["avg_wrong"]
        deltas_le2[i] = m_b["boards_le_2"] - m_a["boards_le_2"]
        deltas_le4[i] = m_b["boards_le_4"] - m_a["boards_le_4"]

    ci_ps = ci_percentile(deltas_per_square)
    ci_aw = ci_percentile(deltas_avg_wrong)
    ci_l2 = ci_percentile(deltas_le2)
    ci_l4 = ci_percentile(deltas_le4)

    result = {
        "boards_common": n,
        "dropped_only_a": dropped_a,
        "dropped_only_b": dropped_b,
        "baseline": {
            "label": args.label_a,
            "per_square_pct": round(base_a["per_square"] * 100.0, 3),
            "avg_wrong": round(base_a["avg_wrong"], 3),
            "boards_le_2_pct": round(base_a["boards_le_2"] * 100.0, 3),
            "boards_le_4_pct": round(base_a["boards_le_4"] * 100.0, 3),
        },
        "candidate": {
            "label": args.label_b,
            "per_square_pct": round(base_b["per_square"] * 100.0, 3),
            "avg_wrong": round(base_b["avg_wrong"], 3),
            "boards_le_2_pct": round(base_b["boards_le_2"] * 100.0, 3),
            "boards_le_4_pct": round(base_b["boards_le_4"] * 100.0, 3),
        },
        "delta_b_minus_a": {
            "per_square_pct": {
                "mean": round(float(np.mean(deltas_per_square)) * 100.0, 3),
                "ci95": [round(ci_ps[0] * 100.0, 3), round(ci_ps[1] * 100.0, 3)],
            },
            "avg_wrong": {
                "mean": round(float(np.mean(deltas_avg_wrong)), 3),
                "ci95": [round(ci_aw[0], 3), round(ci_aw[1], 3)],
            },
            "boards_le_2_pct": {
                "mean": round(float(np.mean(deltas_le2)) * 100.0, 3),
                "ci95": [round(ci_l2[0] * 100.0, 3), round(ci_l2[1] * 100.0, 3)],
            },
            "boards_le_4_pct": {
                "mean": round(float(np.mean(deltas_le4)) * 100.0, 3),
                "ci95": [round(ci_l4[0] * 100.0, 3), round(ci_l4[1] * 100.0, 3)],
            },
        },
        "iters": args.iters,
        "seed": args.seed,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
