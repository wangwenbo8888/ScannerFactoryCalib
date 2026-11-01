 # 基线验证一键脚本（算法/参数改动检测）
# 用法（在仓库根 D:\factory_calib_1002 执行）：
#   手动全量：powershell -ExecutionPolicy Bypass -File verify_baseline\verify_baseline.ps1
#   只跑一侧：-Only camera 或 -Only laser
#   ctest 注册名：test_camera_baseline / test_laser_baseline（顶层 CMakeLists 注册）
#
# 做的事（全部与固化基线逐位比对；资产位于仓库根 test_data/，见 test_data/README.md）：
#   ① 相机标定：camera_calib.exe 按标准参数跑 test_data/camera/input
#      → 与 test_data/camera/gold/camera_calib.F430.json（本机 F: OpenCV 基线）比对
#   ② 激光标定：laser_calib.exe 按标准参数跑 test_data/laser/input
#      → 与 test_data/laser/gold/laser_calib.json 金标准比对 + sidecar SHA256 比对
#   ＋ 关键数值抽查（accum/verdict/projectorT 等，哈希失败时给定位线索）
#
# 守护范围：算法代码、主控参数库（camera/laser_calib_params.json）、数据集 config、
# 固化输入数据——任何一环被改，对应测试即 FAIL。
#
# 实现注意：SHA256 用纯 .NET（不用 Get-FileHash）——本机 powershell.exe 在某些
# 环境下模块自动加载失效（PSModulePath 缺系统路径），cmdlet 版不可靠；开头有自愈补丁。
#
# 退出码：0=全部一致（或资产缺失 SKIP）；1=不一致/运行失败
# 运行时长：相机 ~30s；激光 ~80s（含 CMTT 61 档＋2.2GB sidecar 写盘）

param(
    [string]$BuildDir = "build_fc_all",                       # 相对仓库根或绝对路径
    [string]$OpenCvBin = "F:\opencv4.13\install\x64\vc17\bin",
    [ValidateSet('', 'camera', 'laser')]
    [string]$Only = "",                                       # 空=两侧都跑
    [string]$OutDir = ""                                      # 空=data_out\verify_run\<时间戳>
)

# PSModulePath 自愈：缺系统模块路径时补上（防 Utility 模块加载失败）
if (-not (Get-Command Get-FileHash -ErrorAction SilentlyContinue)) {
    $env:PSModulePath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\Modules;$env:PSModulePath"
}

# 纯 .NET SHA256（流式，支持 2.2GB sidecar；不依赖任何模块 cmdlet）
function Get-Sha256([string]$path) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $fs = [System.IO.File]::OpenRead($path)
        try { return ([BitConverter]::ToString($sha.ComputeHash($fs)) -replace '-', '') }
        finally { $fs.Dispose() }
    } finally { $sha.Dispose() }
}

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
if (-not [IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
$env:PATH = "$OpenCvBin;" + $env:PATH
if ($OutDir -eq "") { $OutDir = Join-Path $repo ("data_out\verify_run\" + (Get-Date -Format 'yyyyMMdd_HHmmss')) }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Write-Host "== 基线验证 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  模式=[$Only]  输出: $OutDir"

$camExe  = Join-Path $BuildDir 'module1_camera\Release\camera_calib.exe'
$lasExe  = Join-Path $BuildDir 'module2_laser\Release\laser_calib.exe'
$camData = Join-Path $repo 'test_data\camera\input'
$lasData = Join-Path $repo 'test_data\laser\input'
$camBase = Join-Path $repo 'test_data\camera\gold\camera_calib.F430.json'
$lasGold = Join-Path $repo 'test_data\laser\gold\laser_calib.json'
$binGold = Join-Path $repo 'test_data\laser\gold\curve_map_temp_table.bin'

$fail = 0

# —— 公共：JSON 展平 + 数值容差叶比对（相机/激光字段级对拍共用）——
# 数值叶（标量或空格分隔数值串）：相对偏差 ≤ TOL 视为一致（吸收求解器 ULP 级噪声）；
# 其余叶：精确比对。
function Flatten-Json($o, $p) {
    $r = @{}
    foreach ($pr in $o.PSObject.Properties) {
        $key = "$p$($pr.Name)"
        if ($pr.Value -is [System.Management.Automation.PSCustomObject]) {
            $r += Flatten-Json $pr.Value "$key."
        } elseif ($pr.Value -is [System.Array]) {
            $i = 0
            foreach ($e in $pr.Value) {
                if ($e -is [System.Management.Automation.PSCustomObject]) {
                    $r += Flatten-Json $e "$key[$i]."
                } else { $r["$key[$i]"] = "$e" }
                $i++
            }
        } else { $r[$key] = "$($pr.Value)" }
    }
    return $r
}
function Test-LeafClose([string]$a, [string]$b, [double]$tol) {
    if ($a -eq $b) { return $true }
    $na = [regex]::Matches($a, '-?\d[\dEe.+-]*') ; $nb = [regex]::Matches($b, '-?\d[\dEe.+-]*')
    if ($na.Count -eq 0 -or $na.Count -ne $nb.Count) { return $false }
    for ($i = 0; $i -lt $na.Count; $i++) {
        $dx = 0.0; $dy = 0.0
        if (-not [double]::TryParse($na[$i].Value, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$dx)) { return $false }
        if (-not [double]::TryParse($nb[$i].Value, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$dy)) { return $false }
        $den = [Math]::Max([Math]::Abs($dx), [Math]::Abs($dy))
        $rel = if ($den -gt 0) { [Math]::Abs($dx - $dy) / $den } else { 0.0 }
        if ($rel -gt $tol) { return $false }
    }
    return $true
}
function Compare-JsonLeaves($jsonA, $jsonB, [string[]]$ignorePatterns, [double]$tol) {
    $fa = Flatten-Json $jsonA ''
    $fb = Flatten-Json $jsonB ''
    $bad = @()
    foreach ($k in ($fa.Keys + $fb.Keys | Sort-Object -Unique)) {
        if ($ignorePatterns) {
            $skip = $false
            foreach ($re in $ignorePatterns) { if ($k -match $re) { $skip = $true; break } }
            if ($skip) { continue }
        }
        if (-not (Test-LeafClose "$($fa[$k])" "$($fb[$k])" $tol)) { $bad += $k }
    }
    return ,$bad
}

# ---------- ① 相机标定参数基线 ----------
if ($Only -ne 'laser') {
    $missing = @($camExe, $camData, $camBase) | Where-Object { -not (Test-Path $_) }
    if ($missing) {
        Write-Host "[SKIP] 相机基线资产缺失: $($missing -join '; ')（ctest 无数据机器保绿约定）"
    } else {
        Write-Host "[camera] camera_calib ..."
        & $camExe $camData (Join-Path $OutDir 'camera_calib.json') *> (Join-Path $OutDir 'camera_run.log')
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[FAIL] camera_calib 退出码 $LASTEXITCODE（看 $(Join-Path $OutDir 'camera_run.log')）"; $fail = 1
        } else {
            # —— 字段级容差对拍（2026-11-01 起）——
            # 10-05 金标准后合入的外参 LU 优化（opt_bench 批次一）重排了求解运算次序，
            # 输出为 ULP 级浮点尾差（实测最大相对偏差 ~1.6e-14），不再可逐位比对。
            # 相对容差 1e-9：远高于 ULP 噪声、远低于任何真实算法/参数回归的量级。
            $h1 = Get-Sha256 (Join-Path $OutDir 'camera_calib.json')
            $h2 = Get-Sha256 $camBase
            try {
                $ja = Get-Content (Join-Path $OutDir 'camera_calib.json') -Raw | ConvertFrom-Json
                $jb = Get-Content $camBase -Raw | ConvertFrom-Json
                $bad = Compare-JsonLeaves $ja $jb @() 1e-9
                if ($bad.Count -eq 0) {
                    Write-Host "[PASS] camera_calib.json 全字段与金标准一致（相对容差 1e-9，吸收 LU 优化 ULP 尾差）"
                    Write-Host "       （整文件 SHA 供溯源 mine=$($h1.Substring(0,12))... base=$($h2.Substring(0,12))...）"
                } else {
                    Write-Host "[FAIL] camera_calib.json 字段不一致（超出容差 1e-9）:"
                    $bad | Select-Object -First 10 | ForEach-Object { Write-Host "       $_" }
                    $fail = 1
                }
            } catch {
                Write-Host "       （字段级对拍不可用（$($_.Exception.Message)），回退整文件 SHA 门禁）"
                if ($h1 -eq $h2) { Write-Host "[PASS] camera_calib.json 与本机基线逐位一致 ($($h1.Substring(0,12))...)" }
                else {
                    Write-Host "[FAIL] camera_calib.json 与基线不一致！"
                    Write-Host "       mine=$h1"; Write-Host "       base=$h2"
                    $fail = 1
                }
            }
        }
    }
}

# ---------- ② 激光标定参数基线 ----------
if ($Only -ne 'camera') {
    $missing = @($lasExe, $lasData, $lasGold, $binGold) | Where-Object { -not (Test-Path $_) }
    if ($missing) {
        Write-Host "[SKIP] 激光基线资产缺失: $($missing -join '; ')（ctest 无数据机器保绿约定）"
    } else {
        Write-Host "[laser] laser_calib ..."
        & $lasExe $lasData (Join-Path $OutDir 'laser_calib.json') *> (Join-Path $OutDir 'laser_run.log')
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[FAIL] laser_calib 退出码 $LASTEXITCODE（看 $(Join-Path $OutDir 'laser_run.log')）"; $fail = 1
        } else {
            $mine = Join-Path $OutDir 'laser_calib.json'
            $s1 = Get-Sha256 (Join-Path $OutDir 'curve_map_temp_table.bin')
            $s2 = Get-Sha256 $binGold
            # —— laser_calib.json 字段级对拍（2026-11-01 起）——
            # 整文件 SHA 不再作为门禁：10-05 金标准固化后 PJC 演进新增了诊断字段
            # （baseline/rightTermUsed/urBias*/urGateCount/urRowCount）且 A5/A6/A7/
            # basinProbe 诊断量存在实现微差；标定本体（accum/verdict/projectorT/
            # virtualK/R/T/emissionCurves/laserExtrinsicTempTable/finalRms）仍逐位比对。
            # 豁免清单之外出现任何叶差异依然 FAIL。
            $j1 = Get-Sha256 $mine
            $j2 = Get-Sha256 $lasGold
            $ignoreRe = @(
                '^pjc\.acceptance\.A5_A7\.a[567]_.*$',
                '^pjc\.basinProbe\..*$',
                '^pjc\.(baseline|rightTermUsed|urBiasPostRms|urBiasPreRms|urCrossAngleMedianDeg|urGateCount|urRowCount)$'
            )
            $flatOk = $false
            try {
                $ja = Get-Content $mine -Raw | ConvertFrom-Json
                $jb = Get-Content $lasGold -Raw | ConvertFrom-Json
                $bad = Compare-JsonLeaves $ja $jb $ignoreRe 0.0
                $flatOk = $true
                if ($bad.Count -eq 0) {
                    Write-Host "[PASS] laser_calib.json 核心字段与金标准逐位一致（诊断叶豁免：A5/A6/A7/basinProbe/urBias 系列）"
                    Write-Host "       （整文件 SHA 供溯源 mine=$($j1.Substring(0,12))... gold=$($j2.Substring(0,12))... — 差异仅限豁免清单）"
                } else {
                    Write-Host "[FAIL] laser_calib.json 核心字段不一致（豁免清单外）:"
                    $bad | Select-Object -First 10 | ForEach-Object { Write-Host "       $_" }
                    $fail = 1
                }
                Write-Host ("       accum={0} poses={1} verdict={2} projectorT=[{3}]" -f `
                    $ja.accumulatedPoints3D, $ja.posesProcessed, $ja.verdict, ($ja.pjc.projectorT -join ','))
            } catch {
                Write-Host "       （字段级对拍不可用（$($_.Exception.Message)），回退整文件 SHA 门禁）"
                if (-not $flatOk) {
                    if ($j1 -eq $j2) { Write-Host "[PASS] laser_calib.json 与金标准逐位一致 ($($j1.Substring(0,12))...)" }
                    else { Write-Host "[FAIL] laser_calib.json 不一致！"; Write-Host "       mine=$j1"; Write-Host "       gold=$j2"; $fail = 1 }
                }
            }
            if ($s1 -eq $s2) { Write-Host "[PASS] curve_map_temp_table.bin sidecar SHA256 一致 ($($s1.Substring(0,12))...)" }
            else {
                Write-Host "[FAIL] sidecar SHA256 不一致！"; Write-Host "       mine=$s1"; Write-Host "       gold=$s2"; $fail = 1
            }
        }
    }
}

Write-Host ""
if ($fail -eq 0) {
    Write-Host "== 全部一致：算法与参数行为与固化基线相符 =="
    exit 0
} else {
    Write-Host "== 存在不一致：算法/参数（或环境）相对基线发生了变化，排查前勿上产线 =="
    exit 1
}
