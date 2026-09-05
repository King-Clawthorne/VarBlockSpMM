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
python "$PSScriptRoot/analyze_revision.py"
if ($LASTEXITCODE -ne 0) { throw 'Result validation failed' }
python "$PSScriptRoot/analyze_kernel.py"
if ($LASTEXITCODE -ne 0) { throw 'Kernel result validation failed' }
Assert-PaperPunctuation (Join-Path $projectRoot 'research/paper.tex')
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
  Copy-Item -LiteralPath (Join-Path $output 'paper.pdf') -Destination (Join-Path $projectRoot 'research/paper.pdf')
} finally {
  Pop-Location
}
