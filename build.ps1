[CmdletBinding()]
param(
  [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
  [string]$Configuration = "Release",
  [string]$BuildDirectory = "$PSScriptRoot\build",
  [string]$Generator = "Visual Studio 17 2022",
  [string]$Architecture = "x64",
  [string]$CudaArchitectures = "native"
)

# Configure and build the library with the selected Visual Studio generator,
# architecture, build configuration, and CUDA target architecture.
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$buildPath = [System.IO.Path]::GetFullPath($BuildDirectory)

# --fresh clears stale CMake cache state so generator and toolchain changes
# cannot silently reuse an incompatible configuration.
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

# Build the requested configuration and stop immediately on a native build error.
Write-Host "Building $Configuration configuration"
cmake --build $buildPath --config $Configuration --parallel
if ($LASTEXITCODE -ne 0) {
  throw "Build failed with exit code $LASTEXITCODE."
}

Write-Host "VarBlockSpMM $Configuration build completed successfully."
