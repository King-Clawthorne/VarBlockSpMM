$ErrorActionPreference = 'Stop'
function Assert-PaperPunctuation([string]$Path) {
    $text = [IO.File]::ReadAllText($Path)
    foreach ($codePoint in @(0x2013, 0x2014, 0x003B)) {
        if ($text.Contains([string][char]$codePoint)) {
            throw ('Forbidden paper punctuation U+{0:X4} in {1}' -f $codePoint, $Path)
        }
    }
}
$projectRoot = [IO.Path]::GetFullPath("$PSScriptRoot")
$output = Join-Path $projectRoot 'build/paper'
New-Item -ItemType Directory -Force $output | Out-Null
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
    Copy-Item -LiteralPath (Join-Path $output 'paper.pdf') -Destination (Join-Path $projectRoot 'research/paper.pdf')
    Copy-Item -LiteralPath (Join-Path $output 'supplement.pdf') -Destination (Join-Path $projectRoot 'research/supplement.pdf')
} finally {
    Pop-Location
}
