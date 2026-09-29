[CmdletBinding()]
param(
  [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
  [string]$Configuration = "Release",
  [string]$BuildDirectory = "$PSScriptRoot\build",
  [string]$WslDistribution = "Ubuntu",
  [string]$WslEnvironment = "vbsr-cuda134",
  [string]$CudaArchitectures = "native"
)

# Configure and build with the Linux CUDA toolchain in WSL. The WSL
# micromamba environment supplies CUDA Toolkit, CMake, and the host compiler.
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$buildPath = [System.IO.Path]::GetFullPath($BuildDirectory)

Write-Host "Resolving WSL paths for $WslDistribution"
$wslHome = (& wsl.exe --distribution $WslDistribution --exec bash -lc 'printf %s "$HOME"').Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($wslHome)) {
  throw "Could not resolve the WSL home directory for distribution '$WslDistribution'."
}

$wslProjectRoot = (& wsl.exe --distribution $WslDistribution --exec wslpath -a $projectRoot).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($wslProjectRoot)) {
  throw "Could not translate project path to WSL: $projectRoot"
}

$wslBuildPath = (& wsl.exe --distribution $WslDistribution --exec wslpath -a $buildPath).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($wslBuildPath)) {
  throw "Could not translate build path to WSL: $buildPath"
}

$wslPrefix = @(
  "--distribution", $WslDistribution,
  "--exec", "env", "-i",
  "HOME=$wslHome",
  "PATH=/usr/bin:/bin:/usr/sbin:/sbin",
  "MAMBA_ROOT_PREFIX=$wslHome/micromamba",
  "$wslHome/.local/bin/micromamba",
  "run", "-n", $WslEnvironment
)

Write-Host "Configuring host C++26 and CUDA C++23 in $wslBuildPath"
$configureArgs = $wslPrefix + @(
  "cmake", "--fresh",
  "-S", $wslProjectRoot,
  "-B", $wslBuildPath,
  "-DCMAKE_BUILD_TYPE=$Configuration",
  "-DCMAKE_PROJECT_INCLUDE_BEFORE=$wslProjectRoot/cmake/EnableCuda23.cmake",
  "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
)
& wsl.exe @configureArgs
if ($LASTEXITCODE -ne 0) {
  throw "WSL CMake configuration failed with exit code $LASTEXITCODE."
}

Write-Host "Building $Configuration configuration in WSL"
$buildArgs = $wslPrefix + @("cmake", "--build", $wslBuildPath, "--parallel")
& wsl.exe @buildArgs
if ($LASTEXITCODE -ne 0) {
  throw "WSL build failed with exit code $LASTEXITCODE."
}

Write-Host "VarBlockSpMM $Configuration WSL build completed successfully."
