#!/usr/bin/env python3
"""
Extract real-data subsets from scan_test_cases.json.

Outputs:
1) Binary board/no-board dataset from expected.board_detected.
2) Warped full-board images when corners are available.
3) Optional labeled real piece crops when expected.fen is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

try:
    import chess
except ModuleNotFoundError:  # pragma: no cover - runtime dependency
    chess = None

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover - runtime dependency
    Image = None


PIECE_LABELS = [
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

DOMAIN_VALUES = ("photo_real", "photo_print", "photo_screen", "screenshot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract real board/no-board samples and optional labeled real piece crops "
            "from assets/scan_samples/scan_test_cases.json."
        )
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        default="assets/scan_samples/scan_test_cases.json",
        help="Path to scan validation dataset json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/real_scan",
        help="Output directory.",
    )
    parser.add_argument(
        "--warp-size",
        type=int,
        default=512,
        help="Target size for board warp (square).",
    )
    parser.add_argument(
        "--split-domains",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also copy board/no_board into domain-specific folders to avoid mixing "
            "photo_real vs photo_print vs photo_screen vs screenshot."
        ),
    )
    return parser.parse_args()


def ensure_runtime_deps() -> None:
    if Image is None:
        raise SystemExit(
            "Missing dependency: Pillow\n"
            "Install with: pip install -r tools/dataset/requirements.txt"
        )


def _resample_bicubic() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BICUBIC
    return Image.BICUBIC


def _transform_quad() -> int:
    if hasattr(Image, "Transform"):
        return Image.Transform.QUAD
    return Image.QUAD


def as_bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def parse_corners(raw: Any) -> list[tuple[float, float]] | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    parsed: list[tuple[float, float]] = []
    for point in raw:
        if not isinstance(point, dict):
            return None
        if "x" not in point or "y" not in point:
            return None
        parsed.append((float(point["x"]), float(point["y"])))
    return parsed


def piece_to_label(piece: chess.Piece | None) -> str:
    if piece is None:
        return "empty"
    side = "w" if piece.color == chess.WHITE else "b"
    symbol = piece.symbol().upper()
    return f"{side}{symbol}"


def warp_with_corners(image: Image.Image, corners: list[tuple[float, float]], size: int) -> Image.Image:
    # Internal convention is tl,tr,br,bl.
    # Pillow QUAD expects source points as UL,LL,LR,UR.
    # So we remap to: tl, bl, br, tr.
    quad = (
        corners[0][0],
        corners[0][1],
        corners[3][0],
        corners[3][1],
        corners[2][0],
        corners[2][1],
        corners[1][0],
        corners[1][1],
    )
    return image.transform(
        (size, size),
        _transform_quad(),
        quad,
        resample=_resample_bicubic(),
    )


def crop_square(image: Image.Image, row: int, col: int) -> Image.Image:
    size = image.width // 8
    left = col * size
    top = row * size
    return image.crop((left, top, left + size, top + size))


def resolve_path(repo_root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return repo_root / path


def normalize_capture_domain(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "photo_real": "photo_real",
        "real_photo": "photo_real",
        "real_board": "photo_real",
        "board_photo": "photo_real",
        "photo_print": "photo_print",
        "photo_book": "photo_print",
        "book_photo": "photo_print",
        "printed_board": "photo_print",
        "print_board": "photo_print",
        "photo_screen": "photo_screen",
        "screen_photo": "photo_screen",
        "photo_of_screen": "photo_screen",
        "screen": "photo_screen",
        "screenshot": "screenshot",
    }
    return aliases.get(key)


def infer_capture_domain(case_id: str, image_rel: str, case_type: Any, explicit_domain: Any) -> str:
    domain = normalize_capture_domain(explicit_domain)
    if domain in DOMAIN_VALUES:
        return domain

    type_key = str(case_type).strip().lower()
    if type_key == "screenshot":
        return "screenshot"

    if type_key in {"photo_print", "photo_book", "book", "print"}:
        return "photo_print"

    haystack = f"{case_id} {image_rel}".lower()
    print_keywords = (
        "book",
        "livre",
        "printed",
        "print",
        "paper",
        "magazine",
        "manual",
    )
    if any(word in haystack for word in print_keywords):
        return "photo_print"

    screen_keywords = (
        "screen",
        "ecran",
        "monitor",
        "display",
        "laptop",
        "phone",
        "mobile",
        "reddit",
        "youtube",
        "lichess",
        "chess.com",
    )
    if any(word in haystack for word in screen_keywords):
        return "photo_screen"

    if type_key == "photo":
        return "photo_real"

    # Keep output in the declared domain layout.
    return "photo_real"


def main() -> None:
    args = parse_args()
    ensure_runtime_deps()

    repo_root = Path(__file__).resolve().parents[2]
    dataset_path = resolve_path(repo_root, args.dataset_json)
    output_dir = resolve_path(repo_root, args.output_dir)

    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    # Accept UTF-8 JSON files with or without BOM (common on Windows editors).
    with dataset_path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)

    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise SystemExit("Invalid dataset json: `cases` is missing or not a list.")

    board_binary_dir = output_dir / "board_binary"
    board_binary_domain_dir = output_dir / "board_binary_domain"
    warped_dir = output_dir / "warped_boards"
    labels_dir = output_dir / "labels"
    real_piece_dir = output_dir / "piece_crops_real"
    for name in ("board", "no_board"):
        (board_binary_dir / name).mkdir(parents=True, exist_ok=True)
        for domain in DOMAIN_VALUES:
            (board_binary_domain_dir / domain / name).mkdir(parents=True, exist_ok=True)
    warped_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for label in PIECE_LABELS:
        (real_piece_dir / label).mkdir(parents=True, exist_ok=True)

    piece_manifest_path = labels_dir / "real_piece_samples.csv"
    summary_path = labels_dir / "real_extract_summary.json"

    counters = {
        "total_cases": len(cases),
        "copied_board_binary": 0,
        "copied_board_binary_by_domain": {
            domain: {"board": 0, "no_board": 0} for domain in DOMAIN_VALUES
        },
        "warped_boards": 0,
        "real_piece_crops": 0,
        "cases_with_fen": 0,
        "cases_skipped_missing_image": 0,
        "cases_skipped_invalid_corners": 0,
    }

    with piece_manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "image_path",
                "label",
                "case_id",
                "square",
                "fen",
                "source_warp_path",
            ],
        )
        writer.writeheader()

        for case in cases:
            case_id = str(case.get("id", "unknown_case"))
            image_rel = case.get("image")
            expected = case.get("expected", {})
            if not isinstance(image_rel, str):
                continue
            case_type = case.get("type")
            capture_domain = infer_capture_domain(
                case_id=case_id,
                image_rel=image_rel,
                case_type=case_type,
                explicit_domain=case.get("capture_domain"),
            )

            source_path = resolve_path(repo_root, image_rel)
            if not source_path.exists():
                counters["cases_skipped_missing_image"] += 1
                continue

            board_detected = as_bool_or_none(expected.get("board_detected"))
            if board_detected is not None:
                binary_cls = "board" if board_detected else "no_board"
                dst = board_binary_dir / binary_cls / f"{case_id}{source_path.suffix.lower()}"
                shutil.copy2(source_path, dst)
                if args.split_domains:
                    domain_dst = (
                        board_binary_domain_dir
                        / capture_domain
                        / binary_cls
                        / f"{case_id}{source_path.suffix.lower()}"
                    )
                    shutil.copy2(source_path, domain_dst)
                    counters["copied_board_binary_by_domain"][capture_domain][binary_cls] += 1
                counters["copied_board_binary"] += 1

            corners = parse_corners(expected.get("corners"))
            if not board_detected or corners is None:
                if board_detected and corners is None:
                    counters["cases_skipped_invalid_corners"] += 1
                continue

            image = Image.open(source_path).convert("RGB")
            warped = warp_with_corners(image, corners, size=args.warp_size)
            warped_path = warped_dir / f"{case_id}.png"
            warped.save(warped_path, format="PNG")
            counters["warped_boards"] += 1

            fen = expected.get("fen")
            if not fen:
                continue
            if chess is None:
                continue

            counters["cases_with_fen"] += 1
            board = chess.Board(str(fen))
            for row in range(8):
                for col in range(8):
                    square = chess.square(col, 7 - row)
                    square_name = chess.square_name(square)
                    label = piece_to_label(board.piece_at(square))
                    crop = crop_square(warped, row=row, col=col)
                    rel = Path("piece_crops_real") / label / f"{case_id}_{square_name}.png"
                    crop_path = output_dir / rel
                    crop.save(crop_path, format="PNG")
                    counters["real_piece_crops"] += 1
                    writer.writerow(
                        {
                            "image_path": rel.as_posix(),
                            "label": label,
                            "case_id": case_id,
                            "square": square_name,
                            "fen": fen,
                            "source_warp_path": (Path("warped_boards") / f"{case_id}.png").as_posix(),
                        }
                    )

    summary_path.write_text(json.dumps(counters, indent=2), encoding="utf-8")

    print(f"[real] dataset={dataset_path}")
    print(f"[real] copied_board_binary={counters['copied_board_binary']}")
    if args.split_domains:
        print("[real] copied_board_binary_by_domain:")
        for domain in DOMAIN_VALUES:
            board_n = counters["copied_board_binary_by_domain"][domain]["board"]
            no_board_n = counters["copied_board_binary_by_domain"][domain]["no_board"]
            print(f"[real]   {domain}: board={board_n} no_board={no_board_n}")
    print(f"[real] warped_boards={counters['warped_boards']}")
    print(f"[real] real_piece_crops={counters['real_piece_crops']}")
    print(f"[real] summary={summary_path}")
    print(f"[real] manifest={piece_manifest_path}")


if __name__ == "__main__":
    main()
