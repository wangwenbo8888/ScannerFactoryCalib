 # 基线验证一键脚本（算法改动检测）
# 用法（在仓库根 D:\factory_calib_1002 执行）：
#   powershell -ExecutionPolicy Bypass -File verify_baseline\verify_baseline.ps1
# 可选参数：-BuildDir build_fc_all  -OpenCvBin F:\opencv4.13\install\x64\vc17\bin
#
# 做三件事（全部与固化基线逐位比对）：
#   ① 相机标定：camera_calib.exe 跑 module1_camera/test_data/camera
#      → 与 verify_baseline/camera_calib.F430.json（本机 F: OpenCV 基线）比对
#   ② 激光标定：laser_calib.exe 跑 module2_laser/test_data/LASER
#      → laser_calib.json 与 test_data 金标准比对 + sidecar SHA256 比对
#   ③ 关键数值抽查（accum/verdict/projectorT 等，即使哈希比对失败也打印差异线索）
#
# 退出码：0=全部一致（算法未被改动）；1=存在不一致（排查见输出）
# 运行时长：约 2 分钟（相机 ~30s + 激光 ~80s，含 CMTT 61 档）

param(
    [string]$BuildDir = "build_fc_all",
    [string]$OpenCvBin = "F:\opencv4.13\install\x64\vc17\bin"
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$env:PATH = "$OpenCvBin;" + $env:PATH

$camExe  = Join-Path $repo "$BuildDir\module1_camera\Release\camera_calib.exe"
$lasExe  = Join-Path $repo "$BuildDir\module2_laser\Release\laser_calib.exe"
$camData = Join-Path $repo "module1_camera\test_data\camera"
$lasData = Join-Path $repo "module2_laser\test_data\LASER"
$camBase = Join-Path $repo "verify_baseline\camera_calib.F430.json"
$lasGold = Join-Path $repo "module2_laser\test_data\laser_calib.json"
$binGold = Join-Path $repo "module2_laser\test_data\curve_map_temp_table.bin"

foreach ($p in @($camExe, $lasExe, $camData, $lasData, $camBase, $lasGold, $binGold)) {
    if (-not (Test-Path $p)) { Write-Host "[FAIL] 缺少必需文件: $p（先构建/核对 test_data）"; exit 1 }
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$outDir = Join-Path $repo "data_out\verify_run\$stamp"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Write-Host "== 基线验证 $stamp == 输出目录: $outDir`n"

$fail = 0

# ---------- ① 相机 ----------
Write-Host "[1/2] camera_calib ..."
& $camExe $camData (Join-Path $outDir 'camera_calib.json') *> (Join-Path $outDir 'camera_run.log')
if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] camera_calib 退出码 $LASTEXITCODE（看 camera_run.log）"; $fail = 1 }
else {
    $h1 = (Get-FileHash (Join-Path $outDir 'camera_calib.json')).Hash
    $h2 = (Get-FileHash $camBase).Hash
    if ($h1 -eq $h2) { Write-Host "[PASS] camera_calib.json 与本机基线逐位一致 ($($h1.Substring(0,12))...)" }
    else {
        Write-Host "[FAIL] camera_calib.json 与基线不一致！"
        Write-Host "       mine=$h1"; Write-Host "       base=$h2"
        Write-Host "       （先确认 OpenCV 未换构建——相机侧 SB 角点检测对 OpenCV 构建敏感，换环境须重建基线）"
        $fail = 1
    }
}

# ---------- ② 激光 ----------
Write-Host "[2/2] laser_calib ..."
& $lasExe $lasData (Join-Path $outDir 'laser_calib.json') *> (Join-Path $outDir 'laser_run.log')
if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] laser_calib 退出码 $LASTEXITCODE（看 laser_run.log）"; $fail = 1 }
else {
    $j1 = (Get-FileHash (Join-Path $outDir 'laser_calib.json')).Hash
    $j2 = (Get-FileHash $lasGold).Hash
    $s1 = (Get-FileHash (Join-Path $outDir 'curve_map_temp_table.bin')).Hash
    $s2 = (Get-FileHash $binGold).Hash
    $okJ = ($j1 -eq $j2); $okS = ($s1 -eq $s2)
    if ($okJ) { Write-Host "[PASS] laser_calib.json 与金标准逐位一致 ($($j1.Substring(0,12))...)" }
    else {
        Write-Host "[FAIL] laser_calib.json 不一致！"; Write-Host "       mine=$j1"; Write-Host "       gold=$j2"; $fail = 1
    }
    if ($okS) { Write-Host "[PASS] curve_map_temp_table.bin sidecar SHA256 一致 ($($s1.Substring(0,12))...)" }
    else {
        Write-Host "[FAIL] sidecar SHA256 不一致！"; Write-Host "       mine=$s1"; Write-Host "       gold=$s2"; $fail = 1
    }
    # 关键数值抽查（哈希败时给出定位线索）
    $a = Get-Content (Join-Path $outDir 'laser_calib.json') -Raw | ConvertFrom-Json
    Write-Host ("       accum={0} poses={1} verdict={2} projectorT=[{3}]" -f `
        $a.accumulatedPoints3D, $a.posesProcessed, $a.verdict, ($a.pjc.projectorT -join ','))
}

Write-Host ""
if ($fail -eq 0) {
    Write-Host "== 全部一致：算法行为与固化基线相符 =="
    exit 0
} else {
    Write-Host "== 存在不一致：算法（或环境）相对基线发生了变化，排查前勿上产线 =="
    exit 1
}
