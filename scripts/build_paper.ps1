$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath("$PSScriptRoot/..")
$output = Join-Path $projectRoot 'build/paper'
New-Item -ItemType Directory -Force $output | Out-Null
python "$PSScriptRoot/analyze_revision.py"
if ($LASTEXITCODE -ne 0) { throw 'Result validation failed' }
Push-Location (Join-Path $projectRoot 'research')
try {
  foreach ($pass in 1..2) {
    & pdflatex -interaction=nonstopmode -halt-on-error "-output-directory=$output" paper.tex
    if ($LASTEXITCODE -ne 0) { throw 'Paper build failed' }
  }
  Copy-Item -LiteralPath (Join-Path $output 'paper.pdf') -Destination (Join-Path $projectRoot 'research/paper.pdf')
} finally {
  Pop-Location
}
