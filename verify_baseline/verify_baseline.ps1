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
            $h1 = Get-Sha256 (Join-Path $OutDir 'camera_calib.json')
            $h2 = Get-Sha256 $camBase
            if ($h1 -eq $h2) { Write-Host "[PASS] camera_calib.json 与本机基线逐位一致 ($($h1.Substring(0,12))...)" }
            else {
                Write-Host "[FAIL] camera_calib.json 与基线不一致！"
                Write-Host "       mine=$h1"; Write-Host "       base=$h2"
                Write-Host "       判读：同环境不一致=算法/参数被改；换过 OpenCV=环境因素（valid 仍 49 则重建相机基线）"
                $fail = 1
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
            $j1 = Get-Sha256 (Join-Path $OutDir 'laser_calib.json')
            $j2 = Get-Sha256 $lasGold
            $s1 = Get-Sha256 (Join-Path $OutDir 'curve_map_temp_table.bin')
            $s2 = Get-Sha256 $binGold
            if ($j1 -eq $j2) { Write-Host "[PASS] laser_calib.json 与金标准逐位一致 ($($j1.Substring(0,12))...)" }
            else {
                Write-Host "[FAIL] laser_calib.json 不一致！"; Write-Host "       mine=$j1"; Write-Host "       gold=$j2"; $fail = 1
            }
            if ($s1 -eq $s2) { Write-Host "[PASS] curve_map_temp_table.bin sidecar SHA256 一致 ($($s1.Substring(0,12))...)" }
            else {
                Write-Host "[FAIL] sidecar SHA256 不一致！"; Write-Host "       mine=$s1"; Write-Host "       gold=$s2"; $fail = 1
            }
            # 关键数值抽查（诊断辅助；ConvertFrom-Json 在异常环境下可能不可用，降级跳过）
            try {
                $a = Get-Content (Join-Path $OutDir 'laser_calib.json') -Raw | ConvertFrom-Json
                Write-Host ("       accum={0} poses={1} verdict={2} projectorT=[{3}]" -f `
                    $a.accumulatedPoints3D, $a.posesProcessed, $a.verdict, ($a.pjc.projectorT -join ','))
            } catch { Write-Host "       （数值抽查跳过：$($_.Exception.Message)）" }
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
