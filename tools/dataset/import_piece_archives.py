#!/usr/bin/env python3
"""
Import external chess-piece archives into a unified 13-class square dataset.

Supported inputs:
1) YOLO board archive (2D boards + labels):
   - chess_yolo_data/images/train/*.png
   - chess_yolo_data/labels/train/*.txt
   Each board is split into 64 square crops (empty + 12 piece classes).

2) Real piece archive (already square crops by class):
   - train/<class>/*.png
   - valid/<class>/*.png
   (optional test/<class>/*.png)

Output layout:
  <output>/piece_crops/{train,val,test}/{label}/*.png
  <output>/labels/piece_samples.csv
  <output>/labels/import_summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: Pillow\n"
        "Install with: pip install -r tools/dataset/requirements.txt"
    ) from exc

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGET_LABELS = [
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
TARGET_LABEL_SET = set(TARGET_LABELS)

REAL_CLASS_TO_LABEL = {
    "empty": "empty",
    "white_pawn": "wP",
    "white_knight": "wN",
    "white_bishop": "wB",
    "white_rook": "wR",
    "white_queen": "wQ",
    "white_king": "wK",
    "black_pawn": "bP",
    "black_knight": "bN",
    "black_bishop": "bB",
    "black_rook": "bR",
    "black_queen": "bQ",
    "black_king": "bK",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import 2D YOLO and real piece archives into a unified "
            "13-class piece-crop dataset."
        )
    )
    parser.add_argument(
        "--yolo-archive",
        action="append",
        default=[],
        help="Path to YOLO board archive (can be repeated).",
    )
    parser.add_argument(
        "--real-archive",
        action="append",
        default=[],
        help="Path to real piece-crops archive (can be repeated).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/piece_classifier_external",
        help="Output directory.",
    )
    parser.add_argument(
        "--max-yolo-boards",
        type=int,
        default=2500,
        help="Max YOLO boards imported per archive (0 = all).",
    )
    parser.add_argument(
        "--max-real-per-class",
        type=int,
        default=0,
        help="Max real crops imported per class per split per archive (0 = all).",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=512,
        help="Final normalized board size before 8x8 crop extraction.",
    )
    parser.add_argument(
        "--yolo-val-ratio",
        type=float,
        default=0.10,
        help="Validation ratio for YOLO-derived square crops.",
    )
    parser.add_argument(
        "--yolo-test-ratio",
        type=float,
        default=0.02,
        help="Test ratio for YOLO-derived square crops.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used in deterministic split hashing.",
    )
    parser.add_argument(
        "--yolo-class-map",
        type=str,
        default="bB,bK,bN,bP,bQ,bR,wB,wK,wN,wP,wQ,wR",
        help=(
            "Comma-separated mapping for YOLO class ids 0..N-1. "
            "Example: bB,bK,bN,bP,bQ,bR,wB,wK,wN,wP,wQ,wR"
        ),
    )
    return parser.parse_args()


def is_image_member(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def parse_yolo_class_map(raw: str) -> list[str]:
    labels = [x.strip() for x in raw.split(",") if x.strip()]
    if not labels:
        raise SystemExit("--yolo-class-map is empty.")
    unknown = [x for x in labels if x not in TARGET_LABEL_SET or x == "empty"]
    if unknown:
        raise SystemExit(
            "Invalid labels in --yolo-class-map: "
            + ", ".join(unknown)
            + "\nAllowed piece labels: wP,wN,wB,wR,wQ,wK,bP,bN,bB,bR,bQ,bK"
        )
    return labels


def deterministic_split(key: str, val_ratio: float, test_ratio: float, seed: int) -> str:
    payload = f"{seed}:{key}".encode("utf-8", errors="ignore")
    h = hashlib.sha1(payload).hexdigest()
    value = int(h[:8], 16) / 0xFFFFFFFF
    if value < test_ratio:
        return "test"
    if value < (test_ratio + val_ratio):
        return "val"
    return "train"


def ensure_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        for label in TARGET_LABELS:
            (root / "piece_crops" / split / label).mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)


def decode_image_to_rgb(raw: bytes) -> Image.Image | None:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def normalize_board_square(image: Image.Image, board_size: int) -> Image.Image:
    width, height = image.size
    if width != height:
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
    if image.size != (board_size, board_size):
        image = image.resize((board_size, board_size), Image.Resampling.BILINEAR)
    return image


def parse_yolo_annotations(raw_text: str, class_map: list[str]) -> dict[tuple[int, int], str]:
    grid: dict[tuple[int, int], str] = {}
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
        except ValueError:
            continue
        if class_id < 0 or class_id >= len(class_map):
            continue

        col = max(0, min(7, int(x_center * 8.0)))
        row = max(0, min(7, int(y_center * 8.0)))
        grid[(row, col)] = class_map[class_id]
    return grid


def crop_square(image: Image.Image, row: int, col: int) -> Image.Image:
    side = image.size[0]  # square image
    x0 = round(col * side / 8.0)
    x1 = round((col + 1) * side / 8.0)
    y0 = round(row * side / 8.0)
    y1 = round((row + 1) * side / 8.0)
    return image.crop((x0, y0, x1, y1))


def pick_members(members: list[str], max_count: int) -> list[str]:
    if max_count <= 0 or len(members) <= max_count:
        return members
    # Deterministic subset: lexical first max_count to keep reproducible runs.
    return sorted(members)[:max_count]


def load_yolo_image_members(names: Iterable[str]) -> list[str]:
    out = []
    for name in names:
        if "/images/" not in name:
            continue
        if not is_image_member(name):
            continue
        out.append(name)
    return out


def yolo_label_path(image_member: str) -> str:
    path = Path(image_member)
    # .../images/<split>/<id>.png -> .../labels/<split>/<id>.txt
    parent_parts = list(path.parts)
    try:
        image_idx = parent_parts.index("images")
        parent_parts[image_idx] = "labels"
    except ValueError:
        return ""
    parent_parts[-1] = f"{path.stem}.txt"
    return "/".join(parent_parts)


def write_crop(
    out_root: Path,
    split: str,
    label: str,
    stem: str,
    image: Image.Image,
) -> Path:
    out_dir = out_root / "piece_crops" / split / label
    out_path = out_dir / f"{stem}.png"
    image.save(out_path, format="PNG")
    return out_path


def import_yolo_archive(
    archive_path: Path,
    out_root: Path,
    class_map: list[str],
    board_size: int,
    max_boards: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    manifest_rows: list[dict[str, str]],
    split_counts: dict[str, Counter],
) -> dict[str, object]:
    stats: dict[str, object] = {
        "archive": str(archive_path),
        "type": "yolo_2d",
        "found": archive_path.exists(),
        "boards_candidates": 0,
        "boards_selected": 0,
        "boards_processed": 0,
        "boards_skipped_no_label": 0,
        "boards_skipped_decode": 0,
        "crops_written": 0,
    }
    if not archive_path.exists():
        return stats

    with zipfile.ZipFile(archive_path, "r") as zf:
        names = zf.namelist()
        name_set = set(names)
        images = sorted(load_yolo_image_members(names))
        stats["boards_candidates"] = len(images)

        images_with_labels = [
            image_member
            for image_member in images
            if (label_member := yolo_label_path(image_member)) and label_member in name_set
        ]
        stats["boards_skipped_no_label"] = len(images) - len(images_with_labels)

        selected = pick_members(images_with_labels, max_boards)
        stats["boards_selected"] = len(selected)

        for index, image_member in enumerate(selected):
            label_member = yolo_label_path(image_member)
            if not label_member or label_member not in name_set:
                continue

            raw_image = zf.read(image_member)
            image = decode_image_to_rgb(raw_image)
            if image is None:
                stats["boards_skipped_decode"] = int(stats["boards_skipped_decode"]) + 1
                continue
            image = normalize_board_square(image, board_size)

            raw_label = zf.read(label_member).decode("utf-8", errors="ignore")
            occupied = parse_yolo_annotations(raw_label, class_map)
            board_key = f"{archive_path.name}:{Path(image_member).stem}"
            split = deterministic_split(
                board_key,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )

            for row in range(8):
                for col in range(8):
                    label = occupied.get((row, col), "empty")
                    crop = crop_square(image, row=row, col=col)
                    stem = (
                        f"y2d_{Path(archive_path).stem}_{Path(image_member).stem}"
                        f"_r{row}c{col}_{index:06d}"
                    )
                    out_path = write_crop(out_root, split, label, stem, crop)
                    split_counts[split][label] += 1
                    stats["crops_written"] = int(stats["crops_written"]) + 1
                    manifest_rows.append(
                        {
                            "source": "yolo_2d",
                            "archive": str(archive_path),
                            "origin_path": image_member,
                            "split": split,
                            "label": label,
                            "row": str(row),
                            "col": str(col),
                            "output_path": str(out_path.relative_to(out_root)),
                        }
                    )
            stats["boards_processed"] = int(stats["boards_processed"]) + 1

    return stats


def find_real_members(zf: zipfile.ZipFile) -> list[tuple[str, str, str]]:
    """Return tuples of (split, class_name, member_path)."""
    found: list[tuple[str, str, str]] = []
    for name in zf.namelist():
        if not is_image_member(name):
            continue
        parts = Path(name).parts
        if len(parts) < 3:
            continue
        split = parts[0].lower()
        class_name = parts[1].lower()
        if split not in {"train", "valid", "val", "test"}:
            continue
        found.append((split, class_name, name))
    return found


def normalize_real_split(split: str) -> str:
    if split == "valid":
        return "val"
    return split


def import_real_archive(
    archive_path: Path,
    out_root: Path,
    max_per_class: int,
    manifest_rows: list[dict[str, str]],
    split_counts: dict[str, Counter],
) -> dict[str, object]:
    stats: dict[str, object] = {
        "archive": str(archive_path),
        "type": "real_pieces",
        "found": archive_path.exists(),
        "images_candidates": 0,
        "images_processed": 0,
        "images_skipped_unknown_class": 0,
        "images_skipped_decode": 0,
        "crops_written": 0,
    }
    if not archive_path.exists():
        return stats

    per_bucket_counter: dict[tuple[str, str], int] = defaultdict(int)

    with zipfile.ZipFile(archive_path, "r") as zf:
        members = sorted(find_real_members(zf))
        stats["images_candidates"] = len(members)

        for idx, (raw_split, class_name, member) in enumerate(members):
            split = normalize_real_split(raw_split)
            if split not in {"train", "val", "test"}:
                continue

            label = REAL_CLASS_TO_LABEL.get(class_name)
            if label is None:
                stats["images_skipped_unknown_class"] = int(stats["images_skipped_unknown_class"]) + 1
                continue

            bucket_key = (split, label)
            if max_per_class > 0 and per_bucket_counter[bucket_key] >= max_per_class:
                continue

            raw_image = zf.read(member)
            image = decode_image_to_rgb(raw_image)
            if image is None:
                stats["images_skipped_decode"] = int(stats["images_skipped_decode"]) + 1
                continue

            stem = f"real_{Path(archive_path).stem}_{Path(member).stem}_{idx:06d}"
            out_path = write_crop(out_root, split, label, stem, image)
            per_bucket_counter[bucket_key] += 1
            split_counts[split][label] += 1
            stats["images_processed"] = int(stats["images_processed"]) + 1
            stats["crops_written"] = int(stats["crops_written"]) + 1

            manifest_rows.append(
                {
                    "source": "real_pieces",
                    "archive": str(archive_path),
                    "origin_path": member,
                    "split": split,
                    "label": label,
                    "row": "",
                    "col": "",
                    "output_path": str(out_path.relative_to(out_root)),
                }
            )

    return stats


def write_manifest_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "source",
        "archive",
        "origin_path",
        "split",
        "label",
        "row",
        "col",
        "output_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if not args.yolo_archive and not args.real_archive:
        raise SystemExit("Nothing to import: provide --yolo-archive and/or --real-archive.")

    class_map = parse_yolo_class_map(args.yolo_class_map)
    out_root = Path(args.output_dir)
    ensure_dirs(out_root)

    split_counts: dict[str, Counter] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }
    manifest_rows: list[dict[str, str]] = []

    yolo_stats = []
    for raw in args.yolo_archive:
        stats = import_yolo_archive(
            archive_path=Path(raw),
            out_root=out_root,
            class_map=class_map,
            board_size=args.board_size,
            max_boards=args.max_yolo_boards,
            val_ratio=args.yolo_val_ratio,
            test_ratio=args.yolo_test_ratio,
            seed=args.seed,
            manifest_rows=manifest_rows,
            split_counts=split_counts,
        )
        yolo_stats.append(stats)

    real_stats = []
    for raw in args.real_archive:
        stats = import_real_archive(
            archive_path=Path(raw),
            out_root=out_root,
            max_per_class=args.max_real_per_class,
            manifest_rows=manifest_rows,
            split_counts=split_counts,
        )
        real_stats.append(stats)

    manifest_path = out_root / "labels" / "piece_samples.csv"
    write_manifest_csv(manifest_path, manifest_rows)

    summary = {
        "output_dir": str(out_root),
        "yolo_class_map": class_map,
        "max_yolo_boards": args.max_yolo_boards,
        "max_real_per_class": args.max_real_per_class,
        "seed": args.seed,
        "archives": {
            "yolo": yolo_stats,
            "real": real_stats,
        },
        "split_counts": {
            split: dict(counter)
            for split, counter in split_counts.items()
        },
        "totals": {
            split: int(sum(counter.values()))
            for split, counter in split_counts.items()
        },
        "manifest_rows": len(manifest_rows),
    }

    summary_path = out_root / "labels" / "import_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[import-piece] done")
    print(f"[import-piece] output={out_root}")
    print(f"[import-piece] manifest={manifest_path}")
    print(f"[import-piece] summary={summary_path}")
    totals = summary["totals"]
    print(
        "[import-piece] totals "
        f"train={totals['train']} val={totals['val']} test={totals['test']}"
    )


if __name__ == "__main__":
    main()
