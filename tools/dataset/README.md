# Dataset Tools (2D-first)

This folder contains scripts to build datasets for:

1. Piece classifier (13 classes: `empty`, `wP..wK`, `bP..bK`)
2. Board/no-board classifier (binary)

The workflow is designed for your current scan pipeline:

- 80-95% synthetic from FEN/PGN
- 5-20% real images from `assets/scan_samples/scan_test_cases.json`

## 1) Install Python deps

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r tools/dataset/requirements.txt
```

If `cairosvg` fails on Windows with `no library called "cairo-2" was found`,
use Inkscape CLI renderer (section 2).
If both Cairo and Inkscape are unavailable, use `--svg-renderer pillow_symbols`
as a pure-Python fallback.

## 2) Generate synthetic piece dataset (FEN/PGN -> SVG -> PNG -> 64 crops + labels)

From PGN (recommended, realistic positions):

```powershell
python tools/dataset/generate_synthetic_piece_dataset.py `
  --pgn-file data\games.pgn `
  --output-dir datasets\piece_classifier `
  --max-positions 3000 `
  --variants-per-position 4 `
  --board-size 512 `
  --keep-clean-variant
```

Windows fallback (no Cairo runtime):

```powershell
python tools/dataset/generate_synthetic_piece_dataset.py `
  --pgn-file data\games.pgn `
  --output-dir datasets\piece_classifier `
  --max-positions 3000 `
  --variants-per-position 4 `
  --board-size 512 `
  --keep-clean-variant `
  --svg-renderer inkscape `
  --inkscape-path "C:\Program Files\Inkscape\bin\inkscape.exe"
```

Pure-Python fallback (no Cairo, no Inkscape):

```powershell
python tools/dataset/generate_synthetic_piece_dataset.py `
  --fen-file data\fens.txt `
  --output-dir datasets\piece_classifier `
  --max-positions 200 `
  --variants-per-position 2 `
  --board-size 512 `
  --svg-renderer pillow_symbols
```

From FEN list:

```powershell
python tools/dataset/generate_synthetic_piece_dataset.py `
  --fen-file data\fens.txt `
  --output-dir datasets\piece_classifier `
  --max-positions 2000 `
  --variants-per-position 5
```

Output:

- `datasets/piece_classifier/boards/*.png`
- `datasets/piece_classifier/piece_crops/{train,val,test}/{label}/*.png`
- `datasets/piece_classifier/labels/piece_samples.csv`

Notes:

- `coordinates=False` is enforced for clean square crops.
- Augmentations include perspective jitter, small rotation, blur, JPEG artifacts, and noise.
- Renderer backend is `auto` by default: CairoSVG, then Inkscape, then Pillow symbols.

## 3) Extract real data from scan dataset JSON

```powershell
python tools/dataset/extract_real_scan_dataset.py `
  --dataset-json assets\scan_samples\scan_test_cases.json `
  --output-dir datasets\real_scan `
  --warp-size 512
```

Output:

- `datasets/real_scan/board_binary/{board,no_board}/*`
- `datasets/real_scan/board_binary_domain/{photo_real,photo_print,photo_screen,screenshot}/{board,no_board}/*`
- `datasets/real_scan/warped_boards/*.png` (for cases with corners)
- `datasets/real_scan/piece_crops_real/{label}/*.png` only if `expected.fen` exists
- `datasets/real_scan/labels/real_piece_samples.csv`
- `datasets/real_scan/labels/real_extract_summary.json`

Notes:

- Domain separation is enabled by default (`--split-domains`).
- You can override per-case domain in JSON with:
  - `"capture_domain": "photo_real"` or `"photo_print"` or `"photo_screen"` or `"screenshot"`.
- This avoids mixing true board photos with screen photos during training.

## 4) Training strategy

- Train piece classifier mostly on synthetic crops.
- Add real labeled crops when you have reliable FEN on real cases.
- Keep validation/test sets fully real to measure true performance.
- For board detection, use `board_binary` with your `no_board*` and `photo_demi_board*` negatives.
- Do not mix 2D diagrams/screenshots with photo domains (`photo_real` / `photo_print`) for board detection.
  Keep them in separate folders and train separately.

## 4.1) Strict import for board/no_board (no 2D mix)

Use this helper to import external archives into a strict layout:

- `datasets/external/board_binary_real/board` (real board photos only)
- `datasets/external/board_binary_no_board/no_board` (negative images only)

```powershell
python tools/dataset/import_board_binary_archives.py `
  --real-archive "C:\Users\samib\Downloads\archive.zip" `
  --real-archive "C:\Users\samib\Downloads\Chess Pieces.v24-416x416_aug.coco.zip" `
  --no-board-archive "C:\Users\samib\Downloads\val2017.zip" `
  --output-root datasets/external `
  --max-no-board-per-archive 2000
```

## 5) Train board/no-board and export TFLite

Install training dependencies:

```powershell
pip install -r tools/dataset/requirements-train.txt
```

Train:

```powershell
python tools/dataset/train_board_binary_classifier.py `
  --data-dir datasets/real_scan/board_binary_domain/photo_real `
  --data-dir datasets/real_scan/board_binary_domain/photo_print `
  --data-dir datasets/external/board_binary_real `
  --data-dir datasets/external/board_binary_no_board `
  --output-dir models/board_binary `
  --image-size 192 `
  --batch-size 32 `
  --epochs 6 `
  --val-split 0.2 `
  --max-train-per-class 400 `
  --max-val-per-class 100
```

Outputs:

- `models/board_binary/best.keras`
- `models/board_binary/board_binary.keras`
- `models/board_binary/board_binary.tflite`
- `models/board_binary/labels.txt`
- `models/board_binary/metrics.json`

Notes:

- Current real set is small (27 images). Add more `no_board` and half-board negatives first.
- You can pass multiple `--data-dir` arguments to merge datasets.



## 5.1) Calibrate hysteresis thresholds (important)

After training, calibrate on the real data mix with a strict split:

- `val`: threshold search/calibration
- `test`: final reporting only (no threshold selection on test)

```powershell
python tools/dataset/calibrate_board_threshold.py `
  --model-path models/board_binary/board_binary.keras `
  --data-dir datasets/real_scan/board_binary_domain/photo_real `
  --data-dir datasets/real_scan/board_binary_domain/photo_print `
  --data-dir datasets/external/board_binary_real `
  --data-dir datasets/external/board_binary_no_board `
  --val-split 0.2 `
  --test-split 0.2 `
  --reject-board-weight 0.75 `
  --reject-no-board-weight 0.25 `
  --accept-board-weight 0.30 `
  --accept-no-board-weight 0.70 `
  --hysteresis-min-gap 0.05 `
  --output-json models/board_binary/threshold_calibration.json
```

Optional hard-negative export (opt-in, quota-limited):

```powershell
python tools/dataset/calibrate_board_threshold.py `
  --model-path models/board_binary/board_binary.keras `
  --data-dir datasets/real_scan/board_binary_domain/photo_real `
  --data-dir datasets/real_scan/board_binary_domain/photo_print `
  --data-dir datasets/external/board_binary_real `
  --data-dir datasets/external/board_binary_no_board `
  --hard-negatives-dir datasets/real_scan/hard_examples `
  --max-hard-fp 120 `
  --max-hard-fn 120 `
  --hard-copy-stride 2 `
  --hard-purge-existing
```

Use from JSON in Flutter:

- `recommended_accept_threshold` -> `boardPresenceThreshold`
- `recommended_reject_threshold` -> `boardPresenceRejectThreshold`

The script calibrates with the same classifier logic as the app gate:
5 crops, `strong=top2_mean`, `fallback=max`.

## 5.2) Screen-domain calibration (no source leakage)

For 2D screen routing (`screenshot` / `photo_screen`), do not use real-world negatives
(e.g. `datasets/external/board_binary_no_board`) during calibration.
Use same-domain negatives only.

Recommended split for screen negatives:

- Originals: `datasets/real_scan/board_binary/no_board_screen/`
- Augmented copies (train-only): `datasets/real_scan/board_binary/no_board_screen_aug/`

```powershell
New-Item -ItemType Directory -Force datasets/real_scan/board_binary/no_board_screen | Out-Null
New-Item -ItemType Directory -Force datasets/real_scan/board_binary/no_board_screen_aug | Out-Null
```

Training command (includes augmented negatives):

```powershell
python tools/dataset/train_board_binary_classifier.py `
  --data-dir datasets/real_scan/board_binary_domain/photo_screen `
  --data-dir datasets/real_scan/board_binary_domain/screenshot `
  --data-dir datasets/real_scan/board_binary/no_board_screen `
  --data-dir datasets/real_scan/board_binary/no_board_screen_aug `
  --output-dir models/board_binary_screen
```

Calibration command (original negatives only; no `_aug`):

```powershell
python tools/dataset/calibrate_board_threshold.py `
  --model-path models/board_binary_screen/board_binary.keras `
  --data-dir datasets/real_scan/board_binary_domain/photo_screen `
  --data-dir datasets/real_scan/board_binary_domain/screenshot `
  --data-dir datasets/real_scan/board_binary/no_board_screen `
  --val-split 0.2 `
  --test-split 0.2 `
  --output-json models/board_binary_screen/threshold_calibration_screen.json
```

`train_board_binary_classifier.py` and `calibrate_board_threshold.py` accept direct
class directories like `no_board_screen/` as negative-only inputs.
`calibrate_board_threshold.py` now blocks `*_aug/*augmented*` by default to avoid leakage
(use `--allow-augmented-data` only if you intentionally override this).

## 6) Increase no_board / board counts quickly

Generate extra augmented negatives:

```powershell
python tools/dataset/augment_no_board_samples.py `
  --input-dir datasets/real_scan/board_binary/no_board `
  --output-dir datasets/real_scan/board_binary/no_board `
  --target-count 120 `
  --seed 42
```

Generate extra augmented positives:

```powershell
python tools/dataset/augment_no_board_samples.py `
  --input-dir datasets/real_scan/board_binary/board `
  --output-dir datasets/real_scan/board_binary/board `
  --target-count 120 `
  --prefix board_aug `
  --seed 43
```

## 7) Ship the model into Flutter assets

Copy trained model and labels into app assets:

```powershell
New-Item -ItemType Directory -Force assets/scan_models | Out-Null
Copy-Item models/board_binary/board_binary.tflite assets/scan_models/board_binary.tflite -Force
Copy-Item models/board_binary/labels.txt assets/scan_models/board_binary_labels.txt -Force
```

Then run:

```powershell
flutter pub get
```


