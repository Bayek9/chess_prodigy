#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create K-fold splits for position-core jsonl files at board level.

Input format (one JSON per line):
  {"image": "...", "fen": "..."}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positions",
        type=Path,
        default=Path("tools/pieces/position_core_screen_gt_v1.jsonl"),
        help="Source JSONL with one board per line.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of folds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible shuffling.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tools/pieces/gt_v1_folds"),
        help="Output directory for fold jsonl files.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.positions)
    n = len(rows)
    if n == 0:
        raise ValueError(f"No rows found in {args.positions}")

    if args.k < 2:
        raise ValueError("--k must be >= 2")
    if args.k > n:
        raise ValueError(f"--k ({args.k}) must be <= number of boards ({n})")

    rng = np.random.default_rng(args.seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, args.k)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "source": str(args.positions),
        "total_boards": n,
        "k": args.k,
        "seed": args.seed,
        "folds": [],
    }

    for fold_idx, hold_idx in enumerate(folds):
        hold = set(int(x) for x in hold_idx.tolist())
        train_rows = [rows[i] for i in range(n) if i not in hold]
        hold_rows = [rows[i] for i in range(n) if i in hold]

        hold_path = args.out_dir / f"holdout_fold{fold_idx}.jsonl"
        train_path = args.out_dir / f"train_fold{fold_idx}.jsonl"

        dump_jsonl(hold_path, hold_rows)
        dump_jsonl(train_path, train_rows)

        manifest["folds"].append(
            {
                "fold": fold_idx,
                "train_boards": len(train_rows),
                "holdout_boards": len(hold_rows),
                "train_path": str(train_path),
                "holdout_path": str(hold_path),
            }
        )

    manifest_path = args.out_dir / "folds_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    hold_sizes = [len(f.tolist()) for f in folds]
    print(
        json.dumps(
            {
                "total_boards": n,
                "k": args.k,
                "holdout_sizes": hold_sizes,
                "out_dir": str(args.out_dir),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
