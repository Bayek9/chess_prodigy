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

## 3) Extract real data from scan dataset JSON

```powershell
python tools/dataset/extract_real_scan_dataset.py `
  --dataset-json assets\scan_samples\scan_test_cases.json `
  --output-dir datasets\real_scan `
  --warp-size 512
```

Output:

- `datasets/real_scan/board_binary/{board,no_board}/*`
- `datasets/real_scan/warped_boards/*.png` (for cases with corners)
- `datasets/real_scan/piece_crops_real/{label}/*.png` only if `expected.fen` exists
- `datasets/real_scan/labels/real_piece_samples.csv`
- `datasets/real_scan/labels/real_extract_summary.json`

## 4) Training strategy

- Train piece classifier mostly on synthetic crops.
- Add real labeled crops when you have reliable FEN on real cases.
- Keep validation/test sets fully real to measure true performance.
- For board detection, use `board_binary` with your `no_board*` and `photo_demi_board*` negatives.
