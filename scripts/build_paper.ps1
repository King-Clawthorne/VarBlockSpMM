param([string]$ResultsRoot = '')
$ErrorActionPreference = 'Stop'
function Assert-PaperPunctuation([string]$Path) {
  $text = [IO.File]::ReadAllText($Path)
  foreach ($codePoint in @(0x2013, 0x2014, 0x003B)) {
    if ($text.Contains([string][char]$codePoint)) {
      throw ('Forbidden paper punctuation U+{0:X4} in {1}' -f $codePoint, $Path)
    }
  }
}
$projectRoot = [IO.Path]::GetFullPath("$PSScriptRoot/..")
$output = Join-Path $projectRoot 'build/paper'
New-Item -ItemType Directory -Force $output | Out-Null
$missingApplication = 13..15 | Where-Object { -not (Test-Path -LiteralPath (Join-Path $projectRoot "data/application/bcsstk$_.bin")) }
if ($missingApplication) {
  python "$PSScriptRoot/prepare_application.py"
  if ($LASTEXITCODE -ne 0) { throw 'Published matrix download failed' }
}
if ($ResultsRoot) {
  python "$PSScriptRoot/analyze_revision.py" --revision (Join-Path $ResultsRoot 'synthetic') --application (Join-Path $ResultsRoot 'published')
} else {
  python "$PSScriptRoot/analyze_revision.py"
}
if ($LASTEXITCODE -ne 0) { throw 'Result validation failed' }
$missingNative = Get-ChildItem -LiteralPath (Join-Path $projectRoot 'data/native') -Filter 'covariance_*.json' |
  Where-Object { -not (Test-Path -LiteralPath ([IO.Path]::ChangeExtension($_.FullName, '.bin'))) }
if ($missingNative) {
  python "$PSScriptRoot/prepare_native.py"
  if ($LASTEXITCODE -ne 0) { throw 'Native input reconstruction failed' }
}
if ($ResultsRoot) {
  python "$PSScriptRoot/analyze_supplement.py" --native (Join-Path $ResultsRoot 'native') --ablation (Join-Path $ResultsRoot 'ablation')
} else {
  python "$PSScriptRoot/analyze_supplement.py"
}
if ($LASTEXITCODE -ne 0) { throw 'Supplementary result validation failed' }
$mainSummary = Get-Content -Raw -LiteralPath (Join-Path $projectRoot 'research/generated/summary.json') | ConvertFrom-Json
$supplementSummary = Get-Content -Raw -LiteralPath (Join-Path $projectRoot 'research/generated/supplement-summary.json') | ConvertFrom-Json
if ($mainSummary.provenance.kernel_sha256 -ne $supplementSummary.kernel_sha256) {
  throw 'Main and supplementary campaigns use different direct kernels'
}
if ($mainSummary.provenance.library_sha256 -ne $supplementSummary.library_sha256) {
  throw 'Main and supplementary campaigns use different library sources'
}
python "$PSScriptRoot/analyze_kernel.py"
if ($LASTEXITCODE -ne 0) { throw 'Kernel result validation failed' }
python "$PSScriptRoot/analyze_tuning.py"
if ($LASTEXITCODE -ne 0) { throw 'Narrow development validation failed' }
if ($ResultsRoot) {
  python "$PSScriptRoot/export_robustness.py" --input (Join-Path $ResultsRoot 'robustness')
} else {
  python "$PSScriptRoot/export_robustness.py"
}
if ($LASTEXITCODE -ne 0) { throw 'Robustness result validation failed' }
if ($ResultsRoot) {
  python "$PSScriptRoot/export_relevance.py" --input (Join-Path $ResultsRoot 'relevance')
} else {
  python "$PSScriptRoot/export_relevance.py"
}
if ($LASTEXITCODE -ne 0) { throw 'Focused follow-up validation failed' }
if ($ResultsRoot) {
  python "$PSScriptRoot/export_dg.py" --input (Join-Path $ResultsRoot 'dg')
} else {
  python "$PSScriptRoot/export_dg.py"
}
if ($LASTEXITCODE -ne 0) { throw 'Specialized transport comparison validation failed' }
Assert-PaperPunctuation (Join-Path $projectRoot 'research/paper.tex')
Assert-PaperPunctuation (Join-Path $projectRoot 'research/relevance.tex')
Assert-PaperPunctuation (Join-Path $projectRoot 'research/supplement.tex')
Get-ChildItem -LiteralPath (Join-Path $projectRoot 'research/generated') -Filter '*.tex' |
  ForEach-Object { Assert-PaperPunctuation $_.FullName }
Push-Location (Join-Path $projectRoot 'research')
try {
  foreach ($pass in 1..2) {
    & pdflatex -interaction=nonstopmode -halt-on-error "-output-directory=$output" paper.tex
    if ($LASTEXITCODE -ne 0) { throw 'Paper build failed' }
  }
  # Check rendered text as well: TeX can turn two ASCII hyphens into a dash.
  & pdftotext -enc UTF-8 (Join-Path $output 'paper.pdf') (Join-Path $output 'paper-text.txt')
  if ($LASTEXITCODE -ne 0) { throw 'PDF text extraction failed' }
  Assert-PaperPunctuation (Join-Path $output 'paper-text.txt')
  foreach ($pass in 1..2) {
    & pdflatex -interaction=nonstopmode -halt-on-error "-output-directory=$output" supplement.tex
    if ($LASTEXITCODE -ne 0) { throw 'Supplement build failed' }
  }
  & pdftotext -enc UTF-8 (Join-Path $output 'supplement.pdf') (Join-Path $output 'supplement-text.txt')
  if ($LASTEXITCODE -ne 0) { throw 'Supplement text extraction failed' }
  Assert-PaperPunctuation (Join-Path $output 'supplement-text.txt')
  # A mistyped control sequence such as "Section~ef{sec:relevance}" compiles
  # without error and only shows up in the rendered page, so check the text.
  python "$PSScriptRoot/check_references.py"
  if ($LASTEXITCODE -ne 0) { throw 'Rendered reference check failed' }
  Copy-Item -LiteralPath (Join-Path $output 'paper.pdf') -Destination (Join-Path $projectRoot 'research/paper.pdf')
  Copy-Item -LiteralPath (Join-Path $output 'supplement.pdf') -Destination (Join-Path $projectRoot 'research/supplement.pdf')
} finally {
  Pop-Location
}
