#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_DATASETS = [
    {
        "name": "archive",
        "domain": "photo",
        "screen_csv": "models/eval_suite_byfen/ft1b/archive/position_core_cases.csv",
        "photo_csv": "models/eval_suite_byfen/real_fen_all_plus_pseudo5_v1_from_real_fen_all_e1/archive/position_core_cases.csv",
    },
    {
        "name": "samryan",
        "domain": "photo",
        "screen_csv": "models/eval_suite_byfen/ft1b/samryan/position_core_cases.csv",
        "photo_csv": "models/eval_suite_byfen/real_fen_all_plus_pseudo5_v1_from_real_fen_all_e1/samryan/position_core_cases.csv",
    },
    {
        "name": "screen_gt_v1",
        "domain": "screen",
        "screen_csv": "models/eval_suite_screen_gtv1/ft1b/position_core_cases.csv",
        "photo_csv": "models/eval_suite_screen_gtv1/real_fen_all_plus_pseudo5_v1_from_real_fen_all_e1/position_core_cases.csv",
    },
]


@dataclass
class BoardStats:
    wrong: int
    total: int
    errors: int
    pieces: int
    avg_margin: float


def _validate_prediction_board(group: pd.DataFrame) -> Tuple[int, int]:
    white_kings = 0
    black_kings = 0
    white_pawns = 0
    black_pawns = 0
    pieces = 0
    errors = 0

    for _, row in group.iterrows():
        pred = str(row["pred"])
        if pred == "empty":
            continue

        pieces += 1
        square = str(row.get("square", ""))
        rank = square[1] if len(square) >= 2 else ""
        color = pred[0]
        ptype = pred[1] if len(pred) >= 2 else ""

        if ptype == "K":
            if color == "w":
                white_kings += 1
            elif color == "b":
                black_kings += 1

        if ptype == "P":
            if rank in {"1", "8"}:
                errors += 1
            if color == "w":
                white_pawns += 1
            elif color == "b":
                black_pawns += 1

    if white_kings != 1:
        errors += 1
    if black_kings != 1:
        errors += 1
    if white_pawns > 8:
        errors += 1
    if black_pawns > 8:
        errors += 1
    if pieces > 32:
        errors += 1

    return errors, pieces


def _load_board_stats(csv_path: Path) -> Dict[int, BoardStats]:
    df = pd.read_csv(csv_path)
    df["margin"] = pd.to_numeric(df["margin"], errors="coerce").fillna(0.0)
    out: Dict[int, BoardStats] = {}
    for board_idx, group in df.groupby("board_idx"):
        wrong = int((group["true"] != group["pred"]).sum())
        total = int(len(group))
        errors, pieces = _validate_prediction_board(group)
        avg_margin = float(group["margin"].mean())
        out[int(board_idx)] = BoardStats(
            wrong=wrong,
            total=total,
            errors=errors,
            pieces=pieces,
            avg_margin=avg_margin,
        )
    return out


def _should_retry(primary: BoardStats, min_avg_margin: float, min_pieces: int, max_pieces: int) -> bool:
    plausible = min_pieces <= primary.pieces <= max_pieces
    low_conf = primary.avg_margin < min_avg_margin
    return (not plausible) or low_conf


def _should_switch(primary: BoardStats, alternate: BoardStats, min_pieces: int, max_pieces: int) -> bool:
    primary_plausible = min_pieces <= primary.pieces <= max_pieces
    alternate_plausible = min_pieces <= alternate.pieces <= max_pieces

    if alternate.errors < primary.errors:
        return True
    if alternate.errors > primary.errors:
        return False

    if alternate_plausible and not primary_plausible:
        return True
    if alternate_plausible != primary_plausible:
        return False

    return alternate.avg_margin > (primary.avg_margin + 1e-6)


def _evaluate(
    datasets: List[dict],
    min_avg_margin: float,
    min_pieces: int,
    max_pieces: int,
    enable_fallback: bool,
) -> Tuple[dict, List[dict]]:
    total_wrong = 0
    total_cells = 0
    total_boards = 0
    total_retries = 0
    total_switches = 0
    per_dataset_rows: List[dict] = []

    for ds in datasets:
        name = ds["name"]
        domain = ds["domain"]
        screen = ds["screen_stats"]
        photo = ds["photo_stats"]
        keys = sorted(set(screen.keys()) & set(photo.keys()))

        wrong = 0
        cells = 0
        boards = 0
        retries = 0
        switches = 0

        for key in keys:
            screen_stats = screen[key]
            photo_stats = photo[key]
            primary = screen_stats if domain == "screen" else photo_stats
            alternate = photo_stats if domain == "screen" else screen_stats

            use_alternate = False
            if enable_fallback and _should_retry(primary, min_avg_margin, min_pieces, max_pieces):
                retries += 1
                use_alternate = _should_switch(primary, alternate, min_pieces, max_pieces)
                if use_alternate:
                    switches += 1

            chosen = alternate if use_alternate else primary
            wrong += chosen.wrong
            cells += chosen.total
            boards += 1

        per_square = 100.0 * (1.0 - (wrong / max(1, cells)))
        avg_wrong = wrong / max(1, boards)
        retry_rate = retries / max(1, boards)
        switch_rate = switches / max(1, boards)

        per_dataset_rows.append(
            {
                "dataset": name,
                "domain": domain,
                "per_square": round(per_square, 4),
                "avg_wrong": round(avg_wrong, 4),
                "boards": boards,
                "retry_rate": round(retry_rate, 4),
                "switch_rate": round(switch_rate, 4),
            }
        )

        total_wrong += wrong
        total_cells += cells
        total_boards += boards
        total_retries += retries
        total_switches += switches

    global_metrics = {
        "per_square": round(100.0 * (1.0 - (total_wrong / max(1, total_cells))), 4),
        "avg_wrong": round(total_wrong / max(1, total_boards), 4),
        "boards": total_boards,
        "retry_rate": round(total_retries / max(1, total_boards), 4),
        "switch_rate": round(total_switches / max(1, total_boards), 4),
    }
    return global_metrics, per_dataset_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate dual-model fallback thresholds from eval CSVs.")
    ap.add_argument("--datasets-json", type=Path, default=None, help="Optional JSON file overriding default datasets list.")
    ap.add_argument("--min-avg-margins", default="0.04,0.06,0.08,0.10,0.12")
    ap.add_argument("--min-pieces", default="14,16,18")
    ap.add_argument("--max-pieces", default="36,40,44")
    ap.add_argument("--out-dir", type=Path, default=Path("models/eval_suite_byfen"))
    args = ap.parse_args()

    datasets_cfg = DEFAULT_DATASETS
    if args.datasets_json is not None:
        datasets_cfg = json.loads(args.datasets_json.read_text(encoding="utf-8"))

    datasets: List[dict] = []
    for ds in datasets_cfg:
        screen_csv = Path(ds["screen_csv"])
        photo_csv = Path(ds["photo_csv"])
        if not screen_csv.exists() or not photo_csv.exists():
            raise FileNotFoundError(f"Missing CSV for dataset {ds['name']}: {screen_csv} | {photo_csv}")
        datasets.append(
            {
                "name": ds["name"],
                "domain": ds["domain"],
                "screen_csv": str(screen_csv),
                "photo_csv": str(photo_csv),
                "screen_stats": _load_board_stats(screen_csv),
                "photo_stats": _load_board_stats(photo_csv),
            }
        )

    margin_values = [float(x.strip()) for x in args.min_avg_margins.split(",") if x.strip()]
    min_piece_values = [int(x.strip()) for x in args.min_pieces.split(",") if x.strip()]
    max_piece_values = [int(x.strip()) for x in args.max_pieces.split(",") if x.strip()]

    grid_rows: List[dict] = []
    best = None

    for min_avg_margin in margin_values:
        for min_pieces in min_piece_values:
            for max_pieces in max_piece_values:
                if min_pieces >= max_pieces:
                    continue
                global_metrics, per_dataset = _evaluate(
                    datasets=datasets,
                    min_avg_margin=min_avg_margin,
                    min_pieces=min_pieces,
                    max_pieces=max_pieces,
                    enable_fallback=True,
                )
                row = {
                    "min_avg_margin": min_avg_margin,
                    "min_pieces": min_pieces,
                    "max_pieces": max_pieces,
                    **global_metrics,
                }
                grid_rows.append(row)

                candidate = (
                    row["avg_wrong"],
                    -row["per_square"],
                    row["retry_rate"],
                    row["switch_rate"],
                )
                if best is None or candidate < best[0]:
                    best = (candidate, row, per_dataset)

    assert best is not None
    _, best_row, best_per_dataset = best

    baseline_global, baseline_per_dataset = _evaluate(
        datasets=datasets,
        min_avg_margin=best_row["min_avg_margin"],
        min_pieces=int(best_row["min_pieces"]),
        max_pieces=int(best_row["max_pieces"]),
        enable_fallback=False,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_df = pd.DataFrame(grid_rows).sort_values(["avg_wrong", "per_square", "retry_rate"], ascending=[True, False, True])
    grid_csv = out_dir / "fallback_threshold_grid.csv"
    grid_df.to_csv(grid_csv, index=False)

    summary = {
        "datasets": [
            {
                "name": d["name"],
                "domain": d["domain"],
                "screen_csv": d["screen_csv"],
                "photo_csv": d["photo_csv"],
                "boards_common": len(set(d["screen_stats"].keys()) & set(d["photo_stats"].keys())),
            }
            for d in datasets
        ],
        "best_thresholds": {
            "min_avg_margin": best_row["min_avg_margin"],
            "min_pieces": int(best_row["min_pieces"]),
            "max_pieces": int(best_row["max_pieces"]),
        },
        "best_global": {
            "per_square": best_row["per_square"],
            "avg_wrong": best_row["avg_wrong"],
            "boards": int(best_row["boards"]),
            "retry_rate": best_row["retry_rate"],
            "switch_rate": best_row["switch_rate"],
        },
        "best_per_dataset": best_per_dataset,
        "baseline_primary_only": {
            "global": baseline_global,
            "per_dataset": baseline_per_dataset,
        },
        "grid_csv": str(grid_csv.as_posix()),
    }

    summary_json = out_dir / "fallback_threshold_best.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

