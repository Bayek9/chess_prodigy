param(
  [string]$SourceDir = "tools/regression",
  [string]$AssetsDir = "assets/regression"
)

$srcCases = Join-Path $SourceDir "cases.json"
$srcImages = Join-Path $SourceDir "images"
$dstCases = Join-Path $AssetsDir "cases.json"
$dstImages = Join-Path $AssetsDir "images"

if (-not (Test-Path $srcCases)) { throw "Missing $srcCases" }
if (-not (Test-Path $srcImages)) { throw "Missing $srcImages" }

New-Item -ItemType Directory -Force -Path $dstImages | Out-Null

$raw = Get-Content $srcCases -Raw
$data = $raw | ConvertFrom-Json

$cases = @()
$version = 1

if ($data -is [System.Collections.IEnumerable] -and -not ($data.PSObject.Properties.Name -contains 'cases')) {
  $cases = @($data)
} else {
  if ($null -ne $data.version) {
    $version = [int]$data.version
  }
  $cases = @($data.cases)
}

$payload = [ordered]@{ version = $version; cases = @() }

foreach ($c in $cases) {
  $id = [string]$c.id
  $path = [string]$c.path
  if ([string]::IsNullOrWhiteSpace($id) -or [string]::IsNullOrWhiteSpace($path)) {
    continue
  }

  $fileName = [System.IO.Path]::GetFileName($path)
  $srcFile = Join-Path $srcImages $fileName
  if (-not (Test-Path $srcFile)) {
    Write-Warning ("Skip {0}: image not found at {1}" -f $id, $srcFile)
    continue
  }

  Copy-Item -Path $srcFile -Destination (Join-Path $dstImages $fileName) -Force

  $payload.cases += [ordered]@{
    id = $id
    path = "assets/regression/images/$fileName"
    expected_domain = [string]$c.expected_domain
    expected_class = [string]$c.expected_class
    acquisition = [string]$c.acquisition
    notes = [string]$c.notes
  }
}

$payload | ConvertTo-Json -Depth 8 | Set-Content -Path $dstCases
Write-Host ("Synced {0} case(s) to {1}" -f $payload.cases.Count, $AssetsDir)
