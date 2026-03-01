# Regression Scan Set

This folder stores a fixed mini regression set for scan tuning.

## Layout

- `tools/regression/images/`: source images for regression
- `tools/regression/cases.json`: source-of-truth case list
- `tools/regression/parse_report.ps1`: parse copied field report text
- `assets/regression/`: runtime copy used by integration tests
- `assets/regression/images/`: image assets consumed by `rootBundle`

## Case format

`cases.json` accepts either:

- an array of cases, or
- an object `{ "version": 1, "cases": [...] }`

Case schema:

```json
{
  "id": "fp_001",
  "path": "tools/regression/images/fp_001.jpg",
  "expected_domain": "photo_real",
  "expected_class": "no_board",
  "acquisition": "gallery",
  "notes": "carrelage"
}
```

## Recommended set

- 10 FP (`expected_class=no_board` but app currently detects board)
- 10 FN (`expected_class=board` but app currently misses board)
- 5 TP + 5 TN controls

## Sync source -> assets

After editing `tools/regression/cases.json` / `tools/regression/images/*`:

```powershell
.\tools\regression\sync_to_assets.ps1
```

## Run integration regression

```powershell
flutter test integration_test/regression_scan_test.dart -d <device-id>
```

The test prints:

- per-case results with routing/gate/final stage
- one compact JSON line prefixed with `[regression][json]`

## Parse field protocol report text

```powershell
.\tools\regression\parse_report.ps1 -Path .\report.txt
```

This outputs:

- TP/TN/FP/FN
- retry rate (`t_alt_ms > 0`)
- medians for `t_primary_ms` and `t_alt_ms`
- gate raw distribution

## Execution modes

### Correctness / regression (debug)

Use this for fast iteration on FP/FN and stage diagnostics:

```powershell
flutter test integration_test/regression_scan_test.dart -d <device-id>
```

Or run all integration tests:

```powershell
flutter test integration_test -d <device-id>
```

### Performance profiling (profile)

For representative timings/jank, run with `flutter drive` in profile mode:

```powershell
flutter drive --driver=test_driver/perf_driver.dart --target=integration_test/regression_scan_test.dart -d <device-id> --profile --no-dds
```

Then open DevTools Performance/Timeline during the run.

## Save regression JSON output

The runner prints one compact line prefixed with `[regression][json]`.

Example capture from terminal output:

```powershell
flutter test integration_test/regression_scan_test.dart -d <device-id> | Tee-Object -FilePath regression_run.log
Select-String -Path .\regression_run.log -Pattern "\[regression\]\[json\]" | Select-Object -Last 1 | ForEach-Object { $_.Line } | Set-Content .\report.txt
```

Then parse metrics:

```powershell
.\tools\regression\parse_report.ps1 -Path .\report.txt
```
