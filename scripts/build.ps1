[CmdletBinding()]
param(
  [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
  [string]$Configuration = "Release",
  [string]$BuildDirectory = "$PSScriptRoot\..\build",
  [string]$Generator = "Visual Studio 17 2022",
  [string]$Architecture = "x64",
  [string]$CudaArchitectures = "native",
  [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path -LiteralPath "$PSScriptRoot\..").Path
$buildPath = [System.IO.Path]::GetFullPath($BuildDirectory)

Write-Host "Configuring VarBlockSpMM in $buildPath"
cmake `
  --fresh `
  -S $projectRoot `
  -B $buildPath `
  -G $Generator `
  -A $Architecture `
  "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
if ($LASTEXITCODE -ne 0) {
  throw "CMake configuration failed with exit code $LASTEXITCODE."
}

Write-Host "Building $Configuration configuration"
cmake --build $buildPath --config $Configuration --parallel
if ($LASTEXITCODE -ne 0) {
  throw "Build failed with exit code $LASTEXITCODE."
}

if (-not $SkipTests) {
  Write-Host "Running tests"
  ctest --test-dir $buildPath -C $Configuration --output-on-failure
  if ($LASTEXITCODE -ne 0) {
    throw "Tests failed with exit code $LASTEXITCODE."
  }
}

Write-Host "VarBlockSpMM $Configuration build completed successfully."
