#!/usr/bin/env python3
"""
Generate a synthetic chess piece classification dataset (13 classes).

Pipeline:
1) Load positions from FEN and/or PGN.
2) Render board SVG with python-chess (coordinates disabled).
3) Convert SVG to PNG with CairoSVG.
4) Apply screenshot/camera-like augmentations.
5) Crop 64 squares and auto-label with exact piece labels.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import chess
    import chess.pgn
    import chess.svg
except ModuleNotFoundError:  # pragma: no cover - runtime dependency
    chess = None

try:
    import cairosvg
except ModuleNotFoundError:  # pragma: no cover - runtime dependency
    cairosvg = None

try:
    from PIL import Image, ImageEnhance, ImageFilter
except ModuleNotFoundError:  # pragma: no cover - runtime dependency
    Image = None
    ImageEnhance = None
    ImageFilter = None


THEMES = [
    {"id": "green", "light": "#EEEED2", "dark": "#769656"},
    {"id": "brown", "light": "#F0D9B5", "dark": "#B58863"},
    {"id": "blue", "light": "#DEE3E6", "dark": "#8CA2AD"},
    {"id": "gray", "light": "#E6E6E6", "dark": "#9A9A9A"},
    {"id": "sepia", "light": "#F7E7CE", "dark": "#A67B5B"},
    {"id": "soft", "light": "#DCE9D5", "dark": "#6B8B6F"},
]

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


@dataclass(frozen=True)
class GeneratedBoard:
    fen: str
    board_index: int
    variant_index: int
    split: str
    board_image_relpath: str
    theme_id: str
    augment_tag: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic piece-classifier dataset from FEN/PGN. "
            "Labels are auto-generated from board state."
        )
    )
    parser.add_argument("--fen-file", type=str, default=None, help="Text file with one FEN per line.")
    parser.add_argument("--pgn-file", type=str, default=None, help="PGN file for realistic positions.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/piece_classifier",
        help="Output directory.",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=1500,
        help="Maximum number of unique positions to generate.",
    )
    parser.add_argument(
        "--variants-per-position",
        type=int,
        default=4,
        help="How many rendered/augmented boards per position.",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=512,
        help="Rendered board size in pixels (square).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train split ratio (0..1).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.08,
        help="Validation split ratio (0..1). Remaining goes to test.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--keep-clean-variant",
        action="store_true",
        help="If set, variant #0 stays un-augmented (clean board).",
    )
    return parser.parse_args()


def ensure_runtime_deps() -> None:
    missing = []
    if chess is None:
        missing.append("python-chess")
    if cairosvg is None:
        missing.append("cairosvg")
    if Image is None:
        missing.append("Pillow")
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing dependencies: {joined}\n"
            "Install with: pip install -r tools/dataset/requirements.txt"
        )


def load_fens_from_file(path: Path) -> list[str]:
    fens: list[str] = []
    if not path.exists():
        raise FileNotFoundError(f"FEN file not found: {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fens.append(line)
    return fens


def load_fens_from_pgn(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"PGN file not found: {path}")

    fens: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            board = game.board()
            fens.append(board.fen())
            for move in game.mainline_moves():
                board.push(move)
                fens.append(board.fen())
    return fens


def dedupe_keep_order(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def choose_split(rng: random.Random, train_ratio: float, val_ratio: float) -> str:
    r = rng.random()
    if r < train_ratio:
        return "train"
    if r < train_ratio + val_ratio:
        return "val"
    return "test"


def piece_to_label(piece: chess.Piece | None) -> str:
    if piece is None:
        return "empty"
    side = "w" if piece.color == chess.WHITE else "b"
    symbol = piece.symbol().upper()
    return f"{side}{symbol}"


def svg_board(board: chess.Board, size: int, theme: dict[str, str]) -> str:
    style = (
        f".square.light {{ fill: {theme['light']}; }} "
        f".square.dark {{ fill: {theme['dark']}; }}"
    )
    return chess.svg.board(
        board=board,
        size=size,
        coordinates=False,
        style=style,
    )


def render_svg_to_image(svg: str) -> Image.Image:
    png_bytes = cairosvg.svg2png(bytestring=svg.encode("utf-8"))
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def _resample_bicubic() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BICUBIC
    return Image.BICUBIC


def _transform_quad() -> int:
    if hasattr(Image, "Transform"):
        return Image.Transform.QUAD
    return Image.QUAD


def perspective_jitter(image: Image.Image, rng: random.Random, ratio: float = 0.035) -> Image.Image:
    width, height = image.size
    jitter = ratio * min(width, height)

    # Internal corners in tl,tr,br,bl.
    tl = (rng.uniform(0.0, jitter), rng.uniform(0.0, jitter))
    tr = (width - rng.uniform(0.0, jitter), rng.uniform(0.0, jitter))
    br = (width - rng.uniform(0.0, jitter), height - rng.uniform(0.0, jitter))
    bl = (rng.uniform(0.0, jitter), height - rng.uniform(0.0, jitter))

    # Pillow QUAD expects UL,LL,LR,UR -> tl,bl,br,tr.
    quad = (
        tl[0],
        tl[1],
        bl[0],
        bl[1],
        br[0],
        br[1],
        tr[0],
        tr[1],
    )
    return image.transform(
        (width, height),
        _transform_quad(),
        quad,
        resample=_resample_bicubic(),
    )


def add_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    amplitude = rng.uniform(5.0, 12.0)
    alpha = rng.uniform(0.03, 0.08)
    noise = Image.effect_noise(image.size, amplitude).convert("L")
    noise_rgb = Image.merge("RGB", (noise, noise, noise))
    return Image.blend(image, noise_rgb, alpha)


def add_jpeg_artifact(image: Image.Image, rng: random.Random) -> Image.Image:
    quality = rng.randint(38, 92)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def augment_board(image: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    out = image.copy()
    tags: list[str] = []

    if rng.random() < 0.75:
        out = perspective_jitter(out, rng)
        tags.append("persp")

    if rng.random() < 0.55:
        angle = rng.uniform(-2.6, 2.6)
        out = out.rotate(angle, resample=_resample_bicubic(), fillcolor=(127, 127, 127))
        tags.append("rot")

    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.88, 1.12))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.86, 1.18))
    out = ImageEnhance.Color(out).enhance(rng.uniform(0.9, 1.2))

    if rng.random() < 0.40:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.1)))
        tags.append("blur")

    if rng.random() < 0.50:
        out = add_jpeg_artifact(out, rng)
        tags.append("jpeg")

    if rng.random() < 0.45:
        out = add_noise(out, rng)
        tags.append("noise")

    return out, "+".join(tags) if tags else "light"


def crop_square(image: Image.Image, row: int, col: int) -> Image.Image:
    size = image.width // 8
    left = col * size
    top = row * size
    return image.crop((left, top, left + size, top + size))


def main() -> None:
    args = parse_args()
    ensure_runtime_deps()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("Invalid split ratios. Require: train_ratio > 0, val_ratio >= 0, train+val < 1.")

    if not args.fen_file and not args.pgn_file:
        raise SystemExit("Provide at least one source: --fen-file and/or --pgn-file.")

    rng = random.Random(args.seed)

    source_fens: list[str] = []
    if args.fen_file:
        source_fens.extend(load_fens_from_file(Path(args.fen_file)))
    if args.pgn_file:
        source_fens.extend(load_fens_from_pgn(Path(args.pgn_file)))

    unique_fens = dedupe_keep_order(source_fens)
    if not unique_fens:
        raise SystemExit("No positions found in the provided sources.")

    max_positions = min(args.max_positions, len(unique_fens))
    shuffled_fens = list(unique_fens)
    rng.shuffle(shuffled_fens)
    selected_fens = shuffled_fens[:max_positions]

    output_dir = Path(args.output_dir)
    boards_dir = output_dir / "boards"
    crops_dir = output_dir / "piece_crops"
    labels_dir = output_dir / "labels"

    boards_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for label in PIECE_LABELS:
            (crops_dir / split / label).mkdir(parents=True, exist_ok=True)

    manifest_path = labels_dir / "piece_samples.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "image_path",
                "split",
                "label",
                "square",
                "fen",
                "board_index",
                "variant_index",
                "board_image_path",
                "theme_id",
                "augment_tag",
            ],
        )
        writer.writeheader()

        board_count = 0
        crop_count = 0

        for board_index, fen in enumerate(selected_fens):
            board = chess.Board(fen=fen)
            split = choose_split(rng, args.train_ratio, args.val_ratio)

            for variant_index in range(args.variants_per_position):
                theme = rng.choice(THEMES)
                rendered = render_svg_to_image(svg_board(board, args.board_size, theme))

                if args.keep_clean_variant and variant_index == 0:
                    board_img = rendered
                    augment_tag = "clean"
                else:
                    board_img, augment_tag = augment_board(rendered, rng)

                board_file = f"{board_index:06d}_{variant_index:02d}.png"
                board_path = boards_dir / board_file
                board_img.save(board_path, format="PNG")

                generated = GeneratedBoard(
                    fen=fen,
                    board_index=board_index,
                    variant_index=variant_index,
                    split=split,
                    board_image_relpath=str(Path("boards") / board_file),
                    theme_id=theme["id"],
                    augment_tag=augment_tag,
                )

                board_count += 1
                for row in range(8):
                    for col in range(8):
                        square = chess.square(col, 7 - row)
                        square_name = chess.square_name(square)
                        label = piece_to_label(board.piece_at(square))

                        crop = crop_square(board_img, row=row, col=col)
                        crop_rel = Path("piece_crops") / generated.split / label / (
                            f"{generated.board_index:06d}_{generated.variant_index:02d}_{square_name}.png"
                        )
                        crop_path = output_dir / crop_rel
                        crop.save(crop_path, format="PNG")
                        crop_count += 1

                        writer.writerow(
                            {
                                "image_path": crop_rel.as_posix(),
                                "split": generated.split,
                                "label": label,
                                "square": square_name,
                                "fen": generated.fen,
                                "board_index": generated.board_index,
                                "variant_index": generated.variant_index,
                                "board_image_path": generated.board_image_relpath,
                                "theme_id": generated.theme_id,
                                "augment_tag": generated.augment_tag,
                            }
                        )

    print(f"[synthetic] positions={len(selected_fens)} variants={args.variants_per_position}")
    print(f"[synthetic] boards={board_count} crops={crop_count}")
    print(f"[synthetic] manifest={manifest_path}")
    print("[synthetic] labels=" + ", ".join(PIECE_LABELS))


if __name__ == "__main__":
    main()
