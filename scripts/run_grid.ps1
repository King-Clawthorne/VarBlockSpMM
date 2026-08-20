param(
  [string]$Exe = "$PSScriptRoot\..\build\Release\vbsr_benchmark.exe",
  [string]$Output = "$PSScriptRoot\..\data\regime_map.csv",
  [int]$Rows = 4096,
  [int]$Reps = 20,
  [int]$Warmup = 5,
  [int]$Seed = 1
)
$ErrorActionPreference = "Stop"
$cudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin"
if (Test-Path $cudaBin) { $env:PATH = "$cudaBin;$env:PATH" }
$Exe = (Resolve-Path $Exe).Path
$first = $true
foreach ($distribution in @("uniform", "low", "high", "bimodal")) {
  foreach ($locality in @("local", "random")) {
    foreach ($degree in @(2, 4, 8, 16)) {
      foreach ($rhs in @(8, 16, 32, 64)) {
        $lines = & $Exe --rows $Rows --degree $degree --rhs $rhs --distribution $distribution --locality $locality --reps $Reps --warmup $Warmup --seed $Seed 2>&1
        if ($LASTEXITCODE -ne 0) { throw "benchmark failed: $lines" }
        if ($first) { $lines | Set-Content -Encoding utf8 $Output; $first = $false }
        else { $lines | Select-Object -Skip 1 | Add-Content -Encoding utf8 $Output }
      }
    }
  }
}
Write-Host "Wrote $Output"
