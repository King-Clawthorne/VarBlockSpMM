[CmdletBinding()]
param(
  [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
  [string]$Configuration = "Release",
  [string]$BuildDirectory = "$PSScriptRoot\build",
  [string]$WslDistribution = "Ubuntu",
  [string]$WslEnvironment = "vbsr-cuda134",
  [string]$CudaArchitectures = "native"
)

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

$sourceExtensions = @(".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh")
$sourceFiles = [System.Collections.Generic.List[System.IO.FileInfo]]::new()
$directoriesToScan = [System.Collections.Generic.Stack[string]]::new()
$directoriesToScan.Push($projectRoot)
$buildRoot = [System.IO.Path]::GetFullPath($buildPath).TrimEnd([char[]]@("\", "/"))

while ($directoriesToScan.Count -gt 0) {
  $currentDirectory = $directoriesToScan.Pop()
  foreach ($file in Get-ChildItem -LiteralPath $currentDirectory -File) {
    if ($sourceExtensions -contains $file.Extension.ToLowerInvariant()) {
      $sourceFiles.Add($file)
    }
  }

  foreach ($directory in Get-ChildItem -LiteralPath $currentDirectory -Directory -Force) {
    $directoryPath = [System.IO.Path]::GetFullPath($directory.FullName).TrimEnd([char[]]@("\", "/"))
    if ($directory.Name -eq ".git" -or $directory.Name -eq "build" -or
        $directoryPath.Equals($buildRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
      continue
    }
    $directoriesToScan.Push($directory.FullName)
  }
}

$sourceFiles = @($sourceFiles | Sort-Object FullName)
if ($sourceFiles.Count -eq 0) {
  throw "No C or C++ source files were found under $projectRoot."
}

$wslSourceFiles = @($sourceFiles | ForEach-Object {
  $relativePath = $_.FullName.Substring($projectRoot.Length).TrimStart("\", "/").Replace("\", "/")
  "$wslProjectRoot/$relativePath"
})

$wslPrefix = @(
  "--distribution", $WslDistribution,
  "--exec", "env", "-i",
  "HOME=$wslHome",
  "PATH=/usr/bin:/bin:/usr/sbin:/sbin",
  "MAMBA_ROOT_PREFIX=$wslHome/micromamba",
  "$wslHome/.local/bin/micromamba",
  "run", "-n", $WslEnvironment
)

$toolCheckArgs = $wslPrefix + @(
  "bash", "-c",
  'for tool in doxygen clang clang-format clang-tidy run-clang-tidy; do command -v "$tool" >/dev/null || exit 1; done'
)
& wsl.exe @toolCheckArgs
if ($LASTEXITCODE -ne 0) {
  throw "The WSL environment '$WslEnvironment' must provide Doxygen and LLVM clang/clang-tools (clang, clang-format, clang-tidy, and run-clang-tidy)."
}

Write-Host "Formatting $($sourceFiles.Count) C and C++ files"
$formatArgs = $wslPrefix + @("clang-format", "--style=file", "-i") + $wslSourceFiles
& wsl.exe @formatArgs
if ($LASTEXITCODE -ne 0) {
  throw "clang-format failed with exit code $LASTEXITCODE."
}

Write-Host "Configuring host C++26 and CUDA C++23 in $wslBuildPath"
$configureArgs = $wslPrefix + @(
  "cmake", "--fresh",
  "-S", $wslProjectRoot,
  "-B", $wslBuildPath,
  "-DCMAKE_BUILD_TYPE=$Configuration",
  "-DCMAKE_PROJECT_INCLUDE_BEFORE=$wslProjectRoot/cmake/EnableCuda23.cmake",
  "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures",
  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
)
& wsl.exe @configureArgs
if ($LASTEXITCODE -ne 0) {
  throw "WSL CMake configuration failed with exit code $LASTEXITCODE."
}

Write-Host "Generating Doxygen HTML documentation"
$doxygenArgs = $wslPrefix + @("bash", "-c", 'cd "$1" && doxygen Doxyfile', "build.ps1", $wslProjectRoot)
& wsl.exe @doxygenArgs
if ($LASTEXITCODE -ne 0) {
  throw "Doxygen failed with exit code $LASTEXITCODE."
}

Write-Host "Running clang-tidy on each translation unit in parallel across available CPU cores"
$wslToolkitPath = "$wslHome/micromamba/envs/$WslEnvironment"
$gccIncludeArgs = $wslPrefix + @("x86_64-conda-linux-gnu-c++", "-print-file-name=include")
$wslGccBuiltinInclude = (& wsl.exe @gccIncludeArgs).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($wslGccBuiltinInclude) -or
    $wslGccBuiltinInclude -eq "include") {
  throw "Could not resolve the GCC builtin include directory in WSL environment '$WslEnvironment'."
}

$tidyArgs = $wslPrefix + @(
  "run-clang-tidy",
  "-p", $wslBuildPath,
  "-j", "0",
  "-quiet",
  # Translate the NVCC compile command into arguments understood by clang-tidy.
  "-removed-arg=-forward-unknown-to-host-compiler",
  "-removed-arg=--options-file",
  "-removed-arg=CMakeFiles/varblockspmm.dir/includes_CUDA.rsp",
  "-removed-arg=-arch=native",
  "-removed-arg=--use_fast_math",
  "-removed-arg=-Xcompiler=-std=c++2c",
  "-removed-arg=-x",
  "-removed-arg=cu",
  "-extra-arg=--sysroot=$wslToolkitPath/x86_64-conda-linux-gnu/sysroot",
  "-extra-arg=--cuda-path=$wslToolkitPath",
  "-extra-arg=--cuda-gpu-arch=sm_70",
  "-extra-arg=-isystem",
  "-extra-arg=$wslGccBuiltinInclude",
  "-extra-arg=-I$wslProjectRoot/src",
  "-extra-arg=-I$wslToolkitPath/targets/x86_64-linux/include",
  "-extra-arg=-I$wslToolkitPath/targets/x86_64-linux/include/cccl"
)
& wsl.exe @tidyArgs
if ($LASTEXITCODE -ne 0) {
  throw "clang-tidy failed with exit code $LASTEXITCODE."
}

Write-Host "Building $Configuration configuration in WSL"
$buildArgs = $wslPrefix + @("cmake", "--build", $wslBuildPath, "--parallel")
& wsl.exe @buildArgs
if ($LASTEXITCODE -ne 0) {
  throw "WSL build failed with exit code $LASTEXITCODE."
}

Write-Host "VarBlockSpMM $Configuration WSL build completed successfully."
