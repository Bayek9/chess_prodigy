#!/usr/bin/env python3
"""
Augment a class folder (board or no_board) to increase samples for training.
"""

from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

try:
    from PIL import Image, ImageEnhance, ImageFilter
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: Pillow\n"
        "Install with: pip install -r tools/dataset/requirements.txt"
    ) from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate extra augmented samples in a class folder."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="datasets/real_scan/board_binary/no_board",
        help="Directory with base images (single class).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/real_scan/board_binary/no_board",
        help="Directory to write augmented no_board images.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Filename prefix for generated images (default: folder name + '_aug').",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=120,
        help="Total number of images wanted in output dir after augmentation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _resample_bicubic() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BICUBIC
    return Image.BICUBIC


def list_images(path: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def add_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    amp = rng.uniform(4.0, 14.0)
    alpha = rng.uniform(0.03, 0.10)
    noise = Image.effect_noise(image.size, amp).convert("L")
    noise_rgb = Image.merge("RGB", (noise, noise, noise))
    return Image.blend(image, noise_rgb, alpha)


def add_jpeg_artifact(image: Image.Image, rng: random.Random) -> Image.Image:
    quality = rng.randint(35, 90)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def random_crop_zoom(image: Image.Image, rng: random.Random) -> Image.Image:
    w, h = image.size
    scale = rng.uniform(0.58, 0.95)
    cw = int(w * scale)
    ch = int(h * scale)
    if cw <= 8 or ch <= 8:
        return image
    x0 = rng.randint(0, max(0, w - cw))
    y0 = rng.randint(0, max(0, h - ch))
    crop = image.crop((x0, y0, x0 + cw, y0 + ch))
    return crop.resize((w, h), resample=_resample_bicubic())


def augment(image: Image.Image, rng: random.Random) -> Image.Image:
    out = image.convert("RGB")

    if rng.random() < 0.8:
        out = random_crop_zoom(out, rng)
    if rng.random() < 0.6:
        angle = rng.uniform(-15.0, 15.0)
        out = out.rotate(angle, resample=_resample_bicubic(), fillcolor=(30, 30, 30))
    if rng.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.25:
        out = out.transpose(Image.FLIP_TOP_BOTTOM)

    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.75, 1.25))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.70, 1.35))
    out = ImageEnhance.Color(out).enhance(rng.uniform(0.65, 1.35))

    if rng.random() < 0.55:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 2.0)))
    if rng.random() < 0.60:
        out = add_jpeg_artifact(out, rng)
    if rng.random() < 0.55:
        out = add_noise(out, rng)

    return out


def _sanitize_prefix(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text.strip())
    return cleaned.strip("_") or "aug"


def next_output_name(out_dir: Path, prefix: str, idx: int) -> Path:
    return out_dir / f"{prefix}_{idx:05d}.png"


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = _sanitize_prefix(args.prefix if args.prefix else f"{out_dir.name}_aug")

    input_images = list_images(input_dir)
    if not input_images:
        raise SystemExit(f"No input images found in: {input_dir}")

    existing_images = list_images(out_dir)
    if len(existing_images) >= args.target_count:
        print(
            f"[augment-no-board] already have {len(existing_images)} images, "
            f"target={args.target_count}, nothing to do."
        )
        return

    needed = args.target_count - len(existing_images)
    next_idx = 0
    while next_output_name(out_dir, prefix, next_idx).exists():
        next_idx += 1

    generated = 0
    for _ in range(needed):
        src = rng.choice(input_images)
        image = Image.open(src).convert("RGB")
        aug = augment(image, rng)
        out_path = next_output_name(out_dir, prefix, next_idx)
        aug.save(out_path, format="PNG")
        generated += 1
        next_idx += 1
        while next_output_name(out_dir, prefix, next_idx).exists():
            next_idx += 1

    total = len(list_images(out_dir))
    print(f"[augment-class] base={len(input_images)} generated={generated} total={total}")
    print(f"[augment-class] output_dir={out_dir}")
    print(f"[augment-class] prefix={prefix}")


if __name__ == "__main__":
    main()
