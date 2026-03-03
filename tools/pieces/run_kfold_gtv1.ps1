param(
  [string]$Py = ".\.venv\Scripts\python.exe",
  [int]$K = 4,
  [int]$ImgSize = 128,
  [int]$BootstrapIters = 5000,
  [int]$Seed = 0
)

if (!(Test-Path $Py)) { $Py = "python" }

$AName = "ft1"
$ATfl = "models\piece13_yolo_hn_os_ft1\tflite\piece_13cls_fp16.tflite"

$BName = "focal_ft1b"
$BTfl = "models\piece13_yolo_focal_ft1b\tflite\piece_13cls_fp16.tflite"

if (!(Test-Path $ATfl)) { throw "Missing model A: $ATfl" }
if (!(Test-Path $BTfl)) { throw "Missing model B: $BTfl" }

for ($i=0; $i -lt $K; $i++) {
  $pos = "tools\pieces\gt_v1_folds_k4\holdout_fold$i.jsonl"
  if (!(Test-Path $pos)) { throw "Missing fold file: $pos" }

  $outA = "models\piece13_yolo_hn_os_ft1\eval_gt_v1_k4_fold$i"
  $outB = "models\piece13_yolo_focal_ft1b\eval_gt_v1_k4_fold$i"

  New-Item -ItemType Directory -Force -Path $outA | Out-Null
  New-Item -ItemType Directory -Force -Path $outB | Out-Null

  Write-Host "[kfold] fold=$i eval $AName"
  & $Py tools\pieces\eval_position_core.py `
    --tflite $ATfl `
    --positions $pos `
    --img-size $ImgSize `
    --csv-out "$outA\position_core_cases.csv" `
    --dump-mismatches-dir "$outA\mismatches"
  if ($LASTEXITCODE -ne 0) { throw "eval failed for $AName fold $i" }

  Write-Host "[kfold] fold=$i eval $BName"
  & $Py tools\pieces\eval_position_core.py `
    --tflite $BTfl `
    --positions $pos `
    --img-size $ImgSize `
    --csv-out "$outB\position_core_cases.csv" `
    --dump-mismatches-dir "$outB\mismatches"
  if ($LASTEXITCODE -ne 0) { throw "eval failed for $BName fold $i" }

  Write-Host "[kfold] fold=$i bootstrap"
  & $Py tools\pieces\paired_bootstrap_eval.py `
    --csv-a "$outA\position_core_cases.csv" --label-a $AName `
    --csv-b "$outB\position_core_cases.csv" --label-b $BName `
    --iters $BootstrapIters --seed $Seed |
    Tee-Object -FilePath "models\kfold_gtv1_fold$i.bootstrap.txt"
  if ($LASTEXITCODE -ne 0) { throw "bootstrap failed fold $i" }
}

Write-Host "[kfold] done"
