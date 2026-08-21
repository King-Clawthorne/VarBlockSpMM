param(
  [string]$InputDirectory = "$PSScriptRoot\..\data\publication",
  [int]$BootstrapIterations = 10000,
  [int]$BootstrapSeed = 20260821
)
$ErrorActionPreference = "Stop"
$files = Get-ChildItem -LiteralPath $InputDirectory -Filter "regime_map_seed*.csv" | Sort-Object Name
if ($files.Count -lt 2) { throw "Expected at least two seeded sweep files in $InputDirectory" }

$allRatios = @()
"level,seed,comparator,geometric_mean_speedup,ci_low,ci_high,wins,total,minimum_speedup"
foreach ($file in $files) {
  $data = Import-Csv -LiteralPath $file.FullName
  $groups = $data | Group-Object block_rows,degree,distribution,locality,rhs
  $ratios = @{ scalar = @(); cusparse = @(); grouped = @() }
  foreach ($group in $groups) {
    $time = @{}
    foreach ($row in $group.Group) { $time[$row.method] = [double]$row.gpu_median_ms }
    $ratios.scalar += $time.row_owned_scalar / $time.row_owned_hybrid
    $ratios.cusparse += $time.scalar_csr_cusparse / $time.row_owned_hybrid
    $ratios.grouped += $time.slot_grouped_cublas / $time.row_owned_hybrid
    $config = ($group.Name -replace ', ', '|')
    $allRatios += [pscustomobject]@{ config=$config; seed=[int]$group.Group[0].seed; comparator='scalar'; ratio=$time.row_owned_scalar/$time.row_owned_hybrid }
    $allRatios += [pscustomobject]@{ config=$config; seed=[int]$group.Group[0].seed; comparator='cusparse'; ratio=$time.scalar_csr_cusparse/$time.row_owned_hybrid }
    $allRatios += [pscustomobject]@{ config=$config; seed=[int]$group.Group[0].seed; comparator='grouped'; ratio=$time.slot_grouped_cublas/$time.row_owned_hybrid }
  }
  $seed = [int](($data | Select-Object -First 1).seed)
  foreach ($name in "scalar", "cusparse", "grouped") {
    $values = $ratios[$name]
    $logMean = ($values | ForEach-Object { [Math]::Log($_) } | Measure-Object -Average).Average
    $geomean = [Math]::Exp($logMean)
    $wins = ($values | Where-Object { $_ -gt 1.0 }).Count
    $minimum = ($values | Measure-Object -Minimum).Minimum
    "seed,$seed,$name,$($geomean.ToString('F6',[Globalization.CultureInfo]::InvariantCulture)),NA,NA,$wins,$($values.Count),$($minimum.ToString('F6',[Globalization.CultureInfo]::InvariantCulture))"
  }
}

# Cluster bootstrap over the 128 workload configurations. Each sampled cluster
# retains and averages all five matrix seeds, so seeds are not treated as 640
# independent observations. The interval describes sensitivity to the tested
# workload-regime population, not raw timing noise or other GPU architectures.
$rng = [Random]::new($BootstrapSeed)
foreach ($name in "scalar", "cusparse", "grouped") {
  $clusterLogs = @($allRatios | Where-Object comparator -eq $name |
    Group-Object config | ForEach-Object {
      (($_.Group | ForEach-Object { [Math]::Log($_.ratio) }) | Measure-Object -Average).Average
    })
  $observed = [Math]::Exp(($clusterLogs | Measure-Object -Average).Average)
  $boot = [double[]]::new($BootstrapIterations)
  for ($b = 0; $b -lt $BootstrapIterations; ++$b) {
    $sum = 0.0
    for ($i = 0; $i -lt $clusterLogs.Count; ++$i) { $sum += $clusterLogs[$rng.Next($clusterLogs.Count)] }
    $boot[$b] = [Math]::Exp($sum / $clusterLogs.Count)
  }
  [Array]::Sort($boot)
  $low = $boot[[Math]::Floor(0.025 * ($BootstrapIterations - 1))]
  $high = $boot[[Math]::Ceiling(0.975 * ($BootstrapIterations - 1))]
  $values = @($allRatios | Where-Object comparator -eq $name | ForEach-Object ratio)
  $wins = ($values | Where-Object { $_ -gt 1.0 }).Count
  $minimum = ($values | Measure-Object -Minimum).Minimum
  "aggregate,all,$name,$($observed.ToString('F6',[Globalization.CultureInfo]::InvariantCulture)),$($low.ToString('F6',[Globalization.CultureInfo]::InvariantCulture)),$($high.ToString('F6',[Globalization.CultureInfo]::InvariantCulture)),$wins,$($values.Count),$($minimum.ToString('F6',[Globalization.CultureInfo]::InvariantCulture))"
}
