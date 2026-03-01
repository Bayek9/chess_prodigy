# usage:
#   .\tools\regression\parse_report.ps1 -Path .\report.txt
param(
  [Parameter(Mandatory = $true)]
  [string]$Path
)

function Median([double[]]$xs) {
  if (-not $xs -or $xs.Count -eq 0) { return $null }
  $s = $xs | Sort-Object
  $n = $s.Count
  if ($n % 2 -eq 1) { return $s[[int]($n / 2)] }
  return ($s[$n / 2 - 1] + $s[$n / 2]) / 2.0
}

$lines = Get-Content $Path

# report line example:
# #38 t=... expected=... source=... chosen=... gate_decision_raw=... t_primary_ms=... t_alt_ms=... decision=... boardDetected=... outcome=FP
$entryRe = [regex]'^\#(?<idx>\d+)\s+.*\boutcome=(?<outcome>TP|TN|FP|FN)\b.*$'
$tPrimaryRe = [regex]'\bt_primary_ms=(?<ms>\d+)\b'
$tAltRe = [regex]'\bt_alt_ms=(?<ms>\d+)\b'
$gateRawRe = [regex]'\bgate_decision_raw=(?<g>[a-z_]+)\b'

$outcomes = @()
$tPrimary = New-Object System.Collections.Generic.List[double]
$tAlt = New-Object System.Collections.Generic.List[double]
$gateCounts = @{}

foreach ($l in $lines) {
  $m = $entryRe.Match($l)
  if ($m.Success) {
    $outcomes += $m.Groups['outcome'].Value
  }

  $mp = $tPrimaryRe.Match($l)
  if ($mp.Success) { $tPrimary.Add([double]$mp.Groups['ms'].Value) }

  $ma = $tAltRe.Match($l)
  if ($ma.Success) { $tAlt.Add([double]$ma.Groups['ms'].Value) }

  $mg = $gateRawRe.Match($l)
  if ($mg.Success) {
    $g = $mg.Groups['g'].Value
    if (-not $gateCounts.ContainsKey($g)) { $gateCounts[$g] = 0 }
    $gateCounts[$g]++
  }
}

$tp = ($outcomes | Where-Object { $_ -eq 'TP' }).Count
$tn = ($outcomes | Where-Object { $_ -eq 'TN' }).Count
$fp = ($outcomes | Where-Object { $_ -eq 'FP' }).Count
$fn = ($outcomes | Where-Object { $_ -eq 'FN' }).Count
$total = $outcomes.Count

$retryCount = ($tAlt | Where-Object { $_ -gt 0 }).Count
$retryRate = if ($total -gt 0) { [math]::Round(100.0 * $retryCount / $total, 1) } else { 0 }

$medPrimary = Median($tPrimary.ToArray())
$medAltAll = Median($tAlt.ToArray())
$medAltRetry = Median(($tAlt | Where-Object { $_ -gt 0 }))

Write-Host "total=$total  TP=$tp  TN=$tn  FP=$fp  FN=$fn"
Write-Host "retry_rate=$retryRate%  (t_alt_ms>0: $retryCount)"
if ($medPrimary -ne $null) { Write-Host ("median_t_primary_ms={0:N1}" -f $medPrimary) }
if ($medAltAll -ne $null) { Write-Host ("median_t_alt_ms_all={0:N1}" -f $medAltAll) }
if ($medAltRetry -ne $null) { Write-Host ("median_t_alt_ms_retryOnly={0:N1}" -f $medAltRetry) }

if ($gateCounts.Keys.Count -gt 0) {
  Write-Host "`nGate raw distribution:"
  $gateCounts.GetEnumerator() |
    Sort-Object -Property Value -Descending |
    ForEach-Object { Write-Host ("  {0} = {1}" -f $_.Key, $_.Value) }
}
