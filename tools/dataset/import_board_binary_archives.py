#!/usr/bin/env python3
"""
Import external archives into a strict board/no_board folder layout.

Goal:
- keep real board photos separate from synthetic 2D datasets
- build board/no_board folders usable by train_board_binary_classifier.py

Examples (PowerShell):
  python tools/dataset/import_board_binary_archives.py `
    --real-archive "C:\\Users\\samib\\Downloads\\archive.zip" `
    --real-archive "C:\\Users\\samib\\Downloads\\Chess Pieces.v24-416x416_aug.coco.zip" `
    --no-board-archive "C:\\Users\\samib\\Downloads\\val2017.zip" `
    --output-root datasets/external `
    --max-no-board-per-archive 2000
"""

from __future__ import annotations

import argparse
import io
import json
import random
import zipfile
from pathlib import Path

try:
    from PIL import Image
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: Pillow\n"
        "Install with: pip install -r tools/dataset/requirements.txt"
    ) from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import board/no_board images from zip archives into a strict layout "
            "without mixing domains."
        )
    )
    parser.add_argument(
        "--real-archive",
        action="append",
        default=[],
        help="Archive containing REAL board photos (board=YES). Can be repeated.",
    )
    parser.add_argument(
        "--no-board-archive",
        action="append",
        default=[],
        help="Archive containing NON-board images (board=NO). Can be repeated.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="datasets/external",
        help="Root output directory.",
    )
    parser.add_argument(
        "--max-real-per-archive",
        type=int,
        default=0,
        help="Max imported images per real archive (0 = all).",
    )
    parser.add_argument(
        "--max-no-board-per-archive",
        type=int,
        default=1500,
        help="Max imported images per no-board archive (0 = all).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _is_image_member(name: str) -> bool:
    suffix = Path(name).suffix.lower()
    return suffix in IMAGE_EXTENSIONS


def _list_image_members(zf: zipfile.ZipFile) -> list[str]:
    return [n for n in zf.namelist() if _is_image_member(n)]


def _pick_members(
    members: list[str], max_count: int, rng: random.Random
) -> list[str]:
    if max_count <= 0 or len(members) <= max_count:
        return members
    return rng.sample(members, max_count)


def _safe_stem(path: str) -> str:
    stem = Path(path).stem
    cleaned = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in stem)
    return cleaned.strip("_") or "archive"


def _write_rgb_png(raw: bytes, out_path: Path) -> bool:
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        image.save(out_path, format="PNG")
        return True
    except Exception:
        return False


def _import_archive(
    archive_path: Path,
    out_dir: Path,
    prefix: str,
    max_count: int,
    rng: random.Random,
) -> dict[str, int]:
    if not archive_path.exists():
        return {"archive_found": 0, "candidates": 0, "selected": 0, "saved": 0}

    saved = 0
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = _list_image_members(zf)
        selected = _pick_members(members, max_count=max_count, rng=rng)
        for idx, member in enumerate(selected):
            try:
                raw = zf.read(member)
            except Exception:
                continue
            name = f"{prefix}_{idx:06d}.png"
            out_path = out_dir / name
            if _write_rgb_png(raw, out_path):
                saved += 1

    return {
        "archive_found": 1,
        "candidates": len(members) if "members" in locals() else 0,
        "selected": len(selected) if "selected" in locals() else 0,
        "saved": saved,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if not args.real_archive and not args.no_board_archive:
        raise SystemExit("Nothing to import: provide --real-archive and/or --no-board-archive.")

    output_root = Path(args.output_root)
    real_board_dir = output_root / "board_binary_real" / "board"
    no_board_dir = output_root / "board_binary_no_board" / "no_board"
    real_board_dir.mkdir(parents=True, exist_ok=True)
    no_board_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "output_root": str(output_root),
        "seed": args.seed,
        "real_archives": [],
        "no_board_archives": [],
    }

    for i, raw_path in enumerate(args.real_archive):
        archive_path = Path(raw_path)
        prefix = f"real_{i:02d}_{_safe_stem(archive_path.name)}"
        stats = _import_archive(
            archive_path=archive_path,
            out_dir=real_board_dir,
            prefix=prefix,
            max_count=args.max_real_per_archive,
            rng=rng,
        )
        summary["real_archives"].append(
            {"archive": str(archive_path), "prefix": prefix, **stats}
        )

    for i, raw_path in enumerate(args.no_board_archive):
        archive_path = Path(raw_path)
        prefix = f"noboard_{i:02d}_{_safe_stem(archive_path.name)}"
        stats = _import_archive(
            archive_path=archive_path,
            out_dir=no_board_dir,
            prefix=prefix,
            max_count=args.max_no_board_per_archive,
            rng=rng,
        )
        summary["no_board_archives"].append(
            {"archive": str(archive_path), "prefix": prefix, **stats}
        )

    board_count = len(list(real_board_dir.glob("*.png")))
    no_board_count = len(list(no_board_dir.glob("*.png")))
    summary["final_counts"] = {"board": board_count, "no_board": no_board_count}

    summary_path = output_root / "board_binary_import_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[import-board-binary] board={board_count} no_board={no_board_count}")
    print(f"[import-board-binary] summary={summary_path}")
    print(f"[import-board-binary] board_dir={real_board_dir}")
    print(f"[import-board-binary] no_board_dir={no_board_dir}")


if __name__ == "__main__":
    main()

