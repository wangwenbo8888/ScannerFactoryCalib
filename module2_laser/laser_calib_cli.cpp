// laser_calib_cli.cpp — 模块2 激光标定 CLI（Task 6.2 完整实现）
//
// 当前进度: 6.3-cmtt (4-14 curve_map_temp_table + 5-3 laser_extrinsic_compensate + 写真实 JSON)
//   旧 4-13 plane_map_temp_table 已于 2026-09-03 CMTT 计划 Task 3C 退役，
//   由 4-14（三源=模块1温表+5-3表+PJC曲线，61 档 sidecar）替代。
//
// 设计依据: docs/plans/2026-07-18-factory-calib-impl.md Task 6.2 Step 0
// 算子签名以 Step 0.1 速查表为准；原 Step 1 伪代码禁止照抄。

#include "calib_io.h"
#include "laser_calib_runner_internal.h"   // fc::runLaserCalibRaw（GUI 库化共用入口）

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>   // GetModuleFileNameA（库化后无 argv[0]，以宿主 exe 目录定位主库）

#include "mask_extract_cuda.h"
#include "region_analyze_cuda.h"
#include "laser_label_cuda.h"
#include "steger_fast.h"           // 4-4 产线（2026-09-04 切换, 与原版逐位等价经全流程逐字节验证）
#include "undistort_points_cuda.h"
#include "epipolar_interp_dual_cuda.h"   // 4-6 产线（2026-09-04 切换, Labeled 路径与原版逐位等价）
#include "laser_match_cuda.h"
#include "laser_reconstruct_cuda.h"
#include "projector_joint_calib.h"   // PJC: 替代旧 4-9/4-10/4-11 端点链路
#include "laser_extrinsic_compensate_cpu.h"
#include "curve_map_temp_table.h"
#include "cmtt_container.h"      // CmttReader 复核（meta 头字段取复核值）

#include <opencv2/core/cuda.hpp>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <set>
#include <map>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <thread>

using namespace fc;
using namespace calib;

namespace {

// —— 自包含 SHA-256（紧凑实现，无第三方依赖）——
// 拷贝来源: modules/07_pipelinemgmt/pipelines/calibcompute/LaserChain.cpp
// （Sha256 类＋文件流式 sha256FileImpl，逐字一致；factory CLI 无 07 库依赖，故内嵌最小版）。
// 用途：4-14 sidecar 整文件哈希 → meta.sha256 → 标定 JSON（设计契约：整文件
// sha256 只存 JSON、不自指——CMTT 头 96B 内无 sha 字段）。07/base/core 与 09
// 均无现成哈希工具（盘点结论），故此处内嵌。
class Sha256 {
public:
    Sha256() { init(); }
    void init() {
        h_[0] = 0x6a09e667u; h_[1] = 0xbb67ae85u; h_[2] = 0x3c6ef372u; h_[3] = 0xa54ff53au;
        h_[4] = 0x510e527fu; h_[5] = 0x9b05688cu; h_[6] = 0x1f83d9abu; h_[7] = 0x5be0cd19u;
        len_ = 0; bufLen_ = 0;
    }
    void update(const uint8_t* p, size_t n) {
        len_ += n;
        while (n > 0) {
            const size_t take = std::min(n, sizeof(buf_) - bufLen_);
            std::memcpy(buf_ + bufLen_, p, take);
            bufLen_ += take; p += take; n -= take;
            if (bufLen_ == sizeof(buf_)) { block(buf_); bufLen_ = 0; }
        }
    }
    std::string hexdigest() {
        const uint64_t bitLen = len_ * 8;             // 长度先取（padding 不计入）
        const uint8_t pad = 0x80;
        update(&pad, 1);
        const uint8_t zero = 0;
        while (bufLen_ != 56) update(&zero, 1);
        uint8_t lenb[8];
        for (int i = 0; i < 8; ++i) lenb[i] = static_cast<uint8_t>(bitLen >> (56 - 8 * i));
        update(lenb, 8);
        static const char* kHex = "0123456789abcdef";
        std::string out;
        out.reserve(64);
        for (int i = 0; i < 8; ++i)
            for (int j = 28; j >= 0; j -= 4) out += kHex[(h_[i] >> j) & 0xF];
        return out;
    }
private:
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const uint8_t* p) {
        static constexpr uint32_t K[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) |
                   (uint32_t(p[4 * i + 2]) << 8) | uint32_t(p[4 * i + 3]);
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
        uint32_t e = h_[4], f = h_[5], g = h_[6], hh = h_[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
        h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += hh;
    }
    uint32_t h_[8];
    uint64_t len_ = 0;
    uint8_t buf_[64];
    size_t bufLen_ = 0;
};

// 文件流式 SHA-256（1MiB 分块）：4-14 端到端契约——hash 覆盖落盘文件字节而非
// 内存 bytes；读失败返回 false（调用方走 FAIL 清理路径，不出无哈希产物）
bool sha256FileImpl(const std::filesystem::path& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    Sha256 s;
    std::vector<uint8_t> buf(size_t{1} << 20);
    while (f) {
        f.read(reinterpret_cast<char*>(buf.data()),
               static_cast<std::streamsize>(buf.size()));
        if (f.gcount() > 0)
            s.update(buf.data(), static_cast<size_t>(f.gcount()));
    }
    if (f.bad()) return false;
    out = s.hexdigest();
    return true;
}

} // namespace

// 库化主体：CLI main 与 GUI runner（经 calib_runner.cpp）共用；
// 行为与纯 CLI 版逐语句一致（仅 argv 解析移至文件末尾的 main 包装）。
int fc::runLaserCalibRaw(const std::string& inDir, const std::string& outPath) {

    spdlog::info("=== laser_calib (build 6.3-cmtt) ===");

    // review C1/I3 防御: main 顶层 try/catch 把算子可能抛的 std::invalid_argument
    // 等异常转成 spdlog::error + exit 1, 避免进程崩溃 (Windows 退出码 0xC0000005)
    try {

    // ------------------------------------------------------------------
    // 0. 参数三层合并: 编译内置 ← laser_calib_params.json(主库) ← 数据集 config.json
    //    主库定位: exe 旁(POST_BUILD 拷贝) → cwd → 源码树兜底; 全缺＝内置(等值, warn)
    // ------------------------------------------------------------------
    LaserCalibConfig baseCfg;
    LaserOpParams    baseOps;
    std::string paramSource = "builtin defaults (master json not found)";
    {
        namespace fs = std::filesystem;
        std::vector<std::filesystem::path> cand;
        std::error_code ec;
        {
            // 库化后无 argv[0]：以宿主 exe 目录定位主库（CLI=laser_calib.exe 旁；GUI=gui exe 旁）
            char exePathBuf[MAX_PATH] = {};
            if (GetModuleFileNameA(nullptr, exePathBuf, MAX_PATH) > 0) {
                auto exeDir = std::filesystem::absolute(std::filesystem::path(exePathBuf), ec).parent_path();
                if (!ec) cand.push_back(exeDir / "laser_calib_params.json");
            }
        }
        cand.push_back(std::filesystem::path("laser_calib_params.json"));
        cand.push_back(std::filesystem::path(
            "E:/JEAMMWARE2601001/factory_calib/module2_laser/laser_calib_params.json"));
        for (const auto& p : cand) {
            std::error_code ec2;
            if (!std::filesystem::exists(p, ec2)) continue;
            std::ifstream ifs(p);
            if (!ifs.is_open()) continue;
            try {
                nlohmann::json jm = nlohmann::json::parse(ifs, nullptr, true);
                fc::applyLaserParamsJson(jm, baseCfg, baseOps);
                paramSource = p.string();
            } catch (const std::exception& e) {
                spdlog::warn("master params parse failed ({}), fallback to builtin", e.what());
            }
            break;
        }
    }
    spdlog::info("params source: {}", paramSource);

    // ------------------------------------------------------------------
    // 1. 加载输入 + 一致性校验
    // ------------------------------------------------------------------
    auto input = loadLaserInput(inDir, &baseCfg, &baseOps);
    if (!input) {
        spdlog::error("load laser input failed");
        return 1;
    }
    const auto& cfg = input->config;
    const auto& ops = input->ops;
    const auto& h = input->handoff;

    std::string why;
    if (!validateHandoffConsistency(cfg, h, why)) {
        spdlog::error("handoff inconsistent: {}", why);
        return 1;
    }

    spdlog::info("poses={}, imageSize={}x{}, referenceTemp={:.2f}",
                 input->poseFrames.size(),
                 h.imageSize.width, h.imageSize.height,
                 h.referenceTemp);

    // ------------------------------------------------------------------
    // 2. 构造 4-1/4-2/4-3 算子 (L 和 R 各一独立实例)
    //    参数: 当前用默认值或从 cfg 取 deviceId
    //    TODO 6.2-b: 从 config.json 扩展 mask threshold/erodeSize 等可配项
    // ------------------------------------------------------------------
    cv::cuda::Stream stream;

    MaskExtractParams maskParams;
    // 最优组合（2026-08-25/26 fcstepdump 六轮扫参定稿, 2026-10-04 起经主库可调）:
    // t=50/e=1/d=19/post13 + ccl minArea=0/topX27 + label tol=10
    // → 22/22 姿态满 25 线, 每帧 ~2.1 万点, PJC 多线 rms 0.44px
    maskParams.threshold       = ops.maskThreshold;
    maskParams.erodeSize       = ops.maskErodeSize;
    maskParams.laserDilateSize = ops.maskLaserDilateSize;
    maskParams.postErodeSize   = ops.maskPostErodeSize;
    MaskExtractCUDA maskL(maskParams);
    MaskExtractCUDA maskR(maskParams);
    spdlog::info("4-1 MaskExtractCUDA x2 (L/R) constructed");
    RegionAnalyzerParams cclParams;
    cclParams.deviceId = cfg.deviceId;
    cclParams.minArea   = ops.cclMinArea;
    cclParams.topXCount = ops.cclTopXCount;
    RegionAnalyzerCUDA cclL(cclParams);
    RegionAnalyzerCUDA cclR(cclParams);
    spdlog::info("4-2 RegionAnalyzerCUDA x2 (L/R) constructed");

    LaserLabelParams labelParams;
    labelParams.deviceId = cfg.deviceId;
    labelParams.scanDirection = cfg.labelScanDirection;
    labelParams.centerRowOffset = cfg.labelCenterRowOffset;
    labelParams.realLineTolerance = ops.labelRealLineTolerance;
    LaserLabelerCUDA labelL(labelParams);
    LaserLabelerCUDA labelR(labelParams);
    spdlog::info("4-3 LaserLabelerCUDA x2 (L/R) constructed "
                 "(scanDirection={}, centerRowOffset={})",
                 labelParams.scanDirection, labelParams.centerRowOffset);

    // ----- 4-4 Steger（steger_fast, 2026-09-04 起产线）-----
    // 与原版逐位等价（全 22 帧全精度对拍 n=0; 容量越界修复后两链全流程逐字节一致,
    // 见 AGENTS 已知限制 #9）; 执行层优化: 行/列扇出 kernel＋计数直写分组＋单次 D2H。
    // 参数: sigma/threshold 等经主库可调（默认＝算子定稿值）; deviceId 从 cfg
    StegerParams stegerParams;
    stegerParams.deviceId     = cfg.deviceId;
    stegerParams.sigma        = ops.stegerSigma;
    stegerParams.kernelSize   = ops.stegerKernelSize;
    stegerParams.lowThreshold = ops.stegerLowThreshold;
    stegerParams.highThreshold= ops.stegerHighThreshold;
    stegerParams.maxLabels    = ops.stegerMaxLabels;
    StegerExtractorFast stegerL(stegerParams);
    StegerExtractorFast stegerR(stegerParams);
    spdlog::info("4-4 StegerExtractorFast x2 (L/R) constructed");

    // ----- 4-5 UndistortPoints -----
    // 参数 K/D/R/P 来自 handoff（模块1 输出的内参 + 立体矫正）。
    // L 路: K=K_L, D=D_L, R=R1, P=P1
    // R 路: K=K_R, D=D_R, R=R2, P=P2
    // 修正: P 只取 3x3 内参部分（清零平移列）。undistort kernel 把 P(0,3) 直接
    // 加进矫正坐标，而 P2(0,3)=-f·Tx 是深度相关视差基准项，不应参与
    // 图像点->矫正图像点映射（否则右路全部点被平移 ~-17 万像素）。
    cv::Mat P1_3x3 = h.P1.clone(); if (P1_3x3.cols > 3) P1_3x3.at<double>(0, 3) = 0.0;
    cv::Mat P2_3x3 = h.P2.clone(); if (P2_3x3.cols > 3) P2_3x3.at<double>(0, 3) = 0.0;
    UndistortPointsParams undistL;
    undistL.cameraMatrix = h.cameraMatrixL;
    undistL.distCoeffs   = h.distCoeffsL;
    undistL.R            = h.R1;
    undistL.P            = P1_3x3;
    undistL.deviceId     = cfg.deviceId;
    undistL.validate();
    UndistortPointsParams undistR;
    undistR.cameraMatrix = h.cameraMatrixR;
    undistR.distCoeffs   = h.distCoeffsR;
    undistR.R            = h.R2;
    undistR.P            = P2_3x3;
    undistR.deviceId     = cfg.deviceId;
    undistR.validate();
    UndistortPointsCuda undistLOp(undistL);
    UndistortPointsCuda undistROp(undistR);
    spdlog::info("4-5 UndistortPointsCuda x2 (L/R) constructed (R1/P1, R2/P2 from handoff)");

    // ----- 4-6 EpipolarInterp（dual·Labeled, 2026-09-04 起产线）-----
    // Labeled 路径复刻 opt 内核、与原版 lineIdCheck=true 逐位等价（等价性证据同 4-4）;
    // mode=Labeled＝按 line_id 同线插值、输出线号透传; Scan 参数不生效。
    EpipolarInterpDualParams epipolarParams;
    epipolarParams.deviceId = cfg.deviceId;
    epipolarParams.mode     = InterpMode::Labeled;
    EpipolarInterpDualCuda epipolarL(epipolarParams);
    EpipolarInterpDualCuda epipolarR(epipolarParams);
    spdlog::info("4-6 EpipolarInterpDualCuda x2 (L/R) constructed (mode=Labeled)");

    // ----- 4-7 LaserMatch -----
    // 单实例（吃 L+R 两路）。
    LaserMatchParams matchParams;
    matchParams.deviceId = cfg.deviceId;
    // 视差上界从 Q + depthMin 推导（默认 500 拒绝 Z<342mm 近距点）:
    //   d = f·Tx / Z,  f = Q(2,3), 1/Tx = |Q(3,2)|
    {
        const double fPx = h.Q.at<double>(2, 3);
        const double invTx = std::abs(h.Q.at<double>(3, 2));
        if (invTx > 0 && cfg.depthMin > 0)
            matchParams.max_disparity =
                static_cast<float>(std::abs(fPx / invTx) / cfg.depthMin
                                   * ops.matchDisparityMarginFactor);
        spdlog::info("match max_disparity = {:.1f} (f*Tx/{:.0f}mm)",
                     matchParams.max_disparity, cfg.depthMin);
    }
    LaserMatchCuda matchOp(matchParams);
    spdlog::info("4-7 LaserMatchCuda constructed (single instance, L+R input)");

    // ----- 4-8 LaserReconstruct -----
    // 单实例，Q 矩阵按调用传入（头文件设计如此，避免跨调用累积）。
    LaserReconstructParams reconParams;
    reconParams.minDepth = cfg.depthMin;
    reconParams.maxDepth = cfg.depthMax;
    reconParams.deviceId  = cfg.deviceId;
    LaserReconstructCuda reconOp(reconParams);
    spdlog::info("4-8 LaserReconstructCuda constructed (Q per-call from handoff.Q)");

    // ----- projector_joint_calib（替代旧 4-9/4-10/4-11 端点链路, 见 docs/流水线/客户端标定流水线.md v2.2）-----
    // 模型: t(3)+发射曲线C(6)=9DOF; R=I、K=f+主点 固定（投影机与左相机绝对轴线平行）
    ProjectorJointCalibParams pjcParams;
    pjcParams.maxIterations        = ops.pjcMaxIterations;
    pjcParams.convergenceThreshold = ops.pjcConvergenceThreshold;
    pjcParams.minPoses             = ops.pjcMinPoses;
    pjcParams.minPointsPerPose     = ops.pjcMinPointsPerPose;
    pjcParams.planeFitInlierThresh = ops.pjcPlaneFitInlierThresh;
    pjcParams.enableTiming         = ops.pjcEnableTiming;
    pjcParams.lambda0              = ops.pjcLambda0;
    pjcParams.lambdaDecay          = ops.pjcLambdaDecay;
    pjcParams.huberToCauchyThresh  = ops.pjcHuberToCauchyThresh;
    pjcParams.cauchyToL2Thresh     = ops.pjcCauchyToL2Thresh;
    pjcParams.topologyEpsilon      = ops.pjcTopologyEpsilon;
    pjcParams.curveDegree          = ops.pjcCurveDegree;
    pjcParams.useCeres             = ops.pjcUseCeres;
    pjcParams.useBlockJtJ          = ops.pjcUseBlockJtJ;
    pjcParams.anomalyRmsThreshold  = ops.pjcAnomalyRmsThreshold;
    pjcParams.robustEnabled        = ops.pjcRobustEnabled;
    pjcParams.huberDelta0          = ops.pjcHuberDelta0;
    pjcParams.irlsMaxRounds        = ops.pjcIrlsMaxRounds;
    pjcParams.poseWeightEnabled    = ops.pjcPoseWeightEnabled;
    ProjectorJointCalib pjcOp(pjcParams);
    spdlog::info("PJC ProjectorJointCalib constructed (replaces 4-9/4-10/4-11)");

    // stereoK 用 StereoCalibration helper (Step 0 决定 2):
    // stereoK = P1 左上 3×3; stereoR = R (left<-right)
    calib::StereoCalibration sc;
    sc.P = h.P1;  // P1/P2 内参部分一致, stereoRectify 保证
    sc.R = h.R;
    cv::Matx33d stereoK = sc.stereoK();
    cv::Matx33d stereoR = sc.stereoR();
    spdlog::info("stereoK from handoff.P1, stereoR from handoff.R (Step 0 决定 2)");

    // ----- 5-3 LaserExtrinsicCompensate (CPU, 单实例) -----
    // 参数仅温度字段; virtual→L/R 外参按调用传入
    LaserExtrinsicCompensateCPUParams lecpParams;
    lecpParams.cte          = cfg.cte;
    lecpParams.tempStep     = cfg.tempStep;
    lecpParams.tempRangeMin = cfg.tempRangeMin;
    lecpParams.tempRangeMax = cfg.tempRangeMax;
    LaserExtrinsicCompensateCPU lecompOp(lecpParams);
    spdlog::info("5-3 LaserExtrinsicCompensateCPU constructed");

    // ------------------------------------------------------------------
    // 3. 主循环: pose × tube, 跑 4-1 ~ 4-8 + host 累积
    //    累积策略 (PJC 链路): 按姿态分组 vector<PosePointSet>（CPU, PJC 为 CPU 算子）
    // ------------------------------------------------------------------
    std::vector<calib::PosePointSet> poseSets(input->poseFrames.size());

    size_t framesOk = 0;
    size_t framesSkip = 0;

    for (size_t pi = 0; pi < input->poseFrames.size(); ++pi) {
        const auto& tubes = input->poseFrames[pi];
        for (size_t ti = 0; ti < tubes.size(); ++ti) {
            const auto& f = tubes[ti];

            // ----- 4-1 mask_extract (L + R) -----
            // Execute(const cv::Mat& gray, Stream&) → MaskExtractResult{d_grayImage, d_cleanedMask, ...}
            auto maskResL = maskL.Execute(f.leftGray, stream);
            auto maskResR = maskR.Execute(f.rightGray, stream);
            if (!maskResL.success || !maskResR.success) {
                spdlog::warn("pose {} tube {}: 4-1 mask failed (L={}, R={}), skip",
                             pi, ti, maskResL.success, maskResR.success);
                ++framesSkip;
                continue;
            }
            if (!maskResL.d_cleanedMask || !maskResR.d_cleanedMask
                || maskResL.d_cleanedMask->empty() || maskResR.d_cleanedMask->empty()) {
                spdlog::warn("pose {} tube {}: 4-1 mask empty, skip", pi, ti);
                ++framesSkip;
                continue;
            }

            // ----- 4-2 region_analyze (L + R) -----
            // Execute(const shared_ptr<GpuMat>& d_mask, Stream&) → RegionAnalysisResult{d_labeledMask CV_32SC1, components}
            auto cclResL = cclL.Execute(maskResL.d_cleanedMask, stream);
            auto cclResR = cclR.Execute(maskResR.d_cleanedMask, stream);
            if (!cclResL.success || !cclResR.success) {
                spdlog::warn("pose {} tube {}: 4-2 ccl failed (L={}, R={}), skip",
                             pi, ti, cclResL.success, cclResR.success);
                ++framesSkip;
                continue;
            }

            // ----- 4-3 laser_label (L + R) -----
            // Execute(const GpuMat& d_inputMask, Stream&) 输入 CV_32SC1 (来自 4-2)
            // → LaserLabelResult{d_labeledMask (重编号 CV_32SC1)}
            if (!cclResL.d_labeledMask || !cclResR.d_labeledMask) {
                spdlog::warn("pose {} tube {}: 4-2 d_labeledMask null, skip", pi, ti);
                ++framesSkip;
                continue;
            }
            auto labelResL = labelL.Execute(*cclResL.d_labeledMask, stream);
            auto labelResR = labelR.Execute(*cclResR.d_labeledMask, stream);
            if (!labelResL.success || !labelResR.success) {
                spdlog::warn("pose {} tube {}: 4-3 label failed (L={}, R={}), skip",
                             pi, ti, labelResL.success, labelResR.success);
                ++framesSkip;
                continue;
            }

            // ----- 4-4 steger (L + R) -----
            // Execute(d_gray, d_mask CV_32SC1, stream, GroupMode::ByLabel)
            //   输入: d_grayImage (from 4-1) + d_labeledMask (from 4-3, 重编号)
            //   输出: d_centerPoints (CV_32FC2), d_line_ids (CV_32SC1)
            if (!maskResL.d_grayImage || !maskResR.d_grayImage
                || !labelResL.d_labeledMask || !labelResR.d_labeledMask) {
                spdlog::warn("pose {} tube {}: 4-4 input null, skip", pi, ti);
                ++framesSkip;
                continue;
            }
            auto stegerResL = stegerL.Execute(*maskResL.d_grayImage,
                                              *labelResL.d_labeledMask,
                                              stream, GroupMode::ByLabel);
            auto stegerResR = stegerR.Execute(*maskResR.d_grayImage,
                                              *labelResR.d_labeledMask,
                                              stream, GroupMode::ByLabel);
            if (!stegerResL.success || !stegerResR.success) {
                spdlog::warn("pose {} tube {}: 4-4 steger failed (L={}, R={}), skip",
                             pi, ti, stegerResL.success, stegerResR.success);
                ++framesSkip;
                continue;
            }

            // ----- 4-5 undistort (L + R) -----
            // Execute(d_points, d_line_ids, stream)
            //   输入: d_centerPoints + d_line_ids (from 4-4)
            //   输出: d_rectifiedPoints + d_line_ids
            if (!stegerResL.d_centerPoints || !stegerResR.d_centerPoints
                || !stegerResL.d_line_ids    || !stegerResR.d_line_ids) {
                spdlog::warn("pose {} tube {}: 4-5 input null, skip", pi, ti);
                ++framesSkip;
                continue;
            }
            auto undistResL = undistLOp.Execute(*stegerResL.d_centerPoints,
                                                *stegerResL.d_line_ids, stream);
            auto undistResR = undistROp.Execute(*stegerResR.d_centerPoints,
                                                *stegerResR.d_line_ids, stream);
            if (!undistResL.success || !undistResR.success) {
                spdlog::warn("pose {} tube {}: 4-5 undistort failed (L={}, R={}), skip",
                             pi, ti, undistResL.success, undistResR.success);
                ++framesSkip;
                continue;
            }

            // ----- 4-6 epipolar_interp (L + R) -----
            // Execute(d_points, d_line_ids, stream)
            //   输入: d_rectifiedPoints + d_line_ids (from 4-5)
            //   输出: d_interpPoints + d_interp_line_ids
            if (!undistResL.d_rectifiedPoints || !undistResR.d_rectifiedPoints
                || !undistResL.d_line_ids      || !undistResR.d_line_ids) {
                spdlog::warn("pose {} tube {}: 4-6 input null, skip", pi, ti);
                ++framesSkip;
                continue;
            }
            auto epipolarResL = epipolarL.Execute(*undistResL.d_rectifiedPoints,
                                                  *undistResL.d_line_ids, stream);
            auto epipolarResR = epipolarR.Execute(*undistResR.d_rectifiedPoints,
                                                  *undistResR.d_line_ids, stream);
            if (!epipolarResL.success || !epipolarResR.success) {
                spdlog::warn("pose {} tube {}: 4-6 epipolar failed (L={}, R={}), skip",
                             pi, ti, epipolarResL.success, epipolarResR.success);
                ++framesSkip;
                continue;
            }

            // ----- 4-7 laser_match -----
            // Execute(d_left_pts, d_left_ids, d_right_pts, d_right_ids, stream)
            //   输入: L 路和 R 路的 d_interpPoints + d_interp_line_ids (from 4-6)
            //   输出: d_matched_left, d_matched_right, d_matched_line_ids
            if (!epipolarResL.d_interpPoints || !epipolarResR.d_interpPoints
                || !epipolarResL.d_interp_line_ids || !epipolarResR.d_interp_line_ids) {
                spdlog::warn("pose {} tube {}: 4-7 input null, skip", pi, ti);
                ++framesSkip;
                continue;
            }
            auto matchRes = matchOp.Execute(*epipolarResL.d_interpPoints,
                                            *epipolarResL.d_interp_line_ids,
                                            *epipolarResR.d_interpPoints,
                                            *epipolarResR.d_interp_line_ids,
                                            stream);
            if (!matchRes.success) {
                spdlog::warn("pose {} tube {}: 4-7 match failed ({}), skip",
                             pi, ti, matchRes.message);
                ++framesSkip;
                continue;
            }

            // ----- 4-8 laser_reconstruct -----
            // Execute(d_matched_left, d_matched_right, d_matched_line_ids, Q, stream)
            //   Q = handoff.Q (模块1 输出)
            //   输出: d_points3d (CV_32FC3), d_valid_line_ids (CV_32SC1)
            if (!matchRes.d_matched_left || !matchRes.d_matched_right
                || !matchRes.d_matched_line_ids) {
                spdlog::warn("pose {} tube {}: 4-8 input null, skip", pi, ti);
                ++framesSkip;
                continue;
            }
            auto reconRes = reconOp.Execute(*matchRes.d_matched_left,
                                            *matchRes.d_matched_right,
                                            *matchRes.d_matched_line_ids,
                                            h.Q, stream);
            if (!reconRes.success) {
                spdlog::warn("pose {} tube {}: 4-8 reconstruct failed ({}), skip",
                             pi, ti, reconRes.message);
                ++framesSkip;
                continue;
            }

            // ----- host 累积（按姿态分组, PJC 链路）-----
            if (reconRes.d_points3d && reconRes.d_valid_line_ids
                && !reconRes.d_points3d->empty()
                && !reconRes.d_valid_line_ids->empty()) {
                cv::Mat h_pts, h_ids;
                reconRes.d_points3d->download(h_pts);
                reconRes.d_valid_line_ids->download(h_ids);
                h_pts = h_pts.reshape(3, 1);   // 强制 1×N CV_32FC3
                h_ids = h_ids.reshape(1, 1);   // 强制 1×N CV_32SC1
                auto& ps = poseSets[pi];
                ps.points3d.insert(ps.points3d.end(),
                                   h_pts.begin<cv::Vec3f>(),
                                   h_pts.end<cv::Vec3f>());
                ps.lineIds.insert(ps.lineIds.end(),
                                  h_ids.begin<int>(),
                                  h_ids.end<int>());
            }

            ++framesOk;
            spdlog::info("pose {} tube {}: OK (matched={}, reconstructed={}, pose_accum={})",
                         pi, ti, matchRes.matchCount, reconRes.validCount,
                         poseSets[pi].points3d.size());
        }
    }

    spdlog::info("loop done: {} ok, {} skipped", framesOk, framesSkip);

    // —— 三维点云导出（2026-09-04 人工指令）——
    // FC_PLY_DUMP=<path.ply> 时导出 4-8 累积三维点云（左相机矫正系），
    // 每点含 pose_idx（原始姿态序）/line_id 标量；不设变量零开销。
    if (const char* plyPath = std::getenv("FC_PLY_DUMP")) {
        std::ofstream f(plyPath);
        if (f) {
            size_t total = 0;
            for (const auto& ps : poseSets) total += ps.points3d.size();
            f << "ply\nformat ascii 1.0\n"
              << "element vertex " << total << "\n"
              << "property float x\nproperty float y\nproperty float z\n"
              << "property int pose_idx\nproperty int line_id\nend_header\n";
            for (size_t pk = 0; pk < poseSets.size(); ++pk) {
                const auto& ps = poseSets[pk];
                for (size_t k = 0; k < ps.points3d.size(); ++k)
                    f << ps.points3d[k][0] << " " << ps.points3d[k][1] << " "
                      << ps.points3d[k][2] << " " << pk << " " << ps.lineIds[k] << "\n";
            }
            spdlog::info("PLY cloud dumped: {} ({} pts)", plyPath, total);
        } else {
            spdlog::warn("FC_PLY_DUMP set but cannot open: {}", plyPath);
        }
    }

    // ------------------------------------------------------------------
    // 3b. 丢弃空姿态组, 统计有效点数
    // ------------------------------------------------------------------
    {
        std::vector<calib::PosePointSet> valid;
        valid.reserve(poseSets.size());
        size_t total = 0;
        for (auto& ps : poseSets) {
            if (!ps.points3d.empty()) { total += ps.points3d.size(); valid.push_back(std::move(ps)); }
        }
        poseSets = std::move(valid);
        spdlog::info("accumulated {} poses / {} 3D points total",
                     poseSets.size(), total);
        if (poseSets.empty())
            spdlog::warn("no 3D points accumulated; downstream (PJC+) will be skipped");
    }

    // ------------------------------------------------------------------
    // 3c. projector_joint_calib 多线联合（v2: 共享 t + 每线独立曲线）
    //     模型: K=f+主点(固定, 派生自 stereoK), R=I, T=projectorT(共享优化),
    //           每条激光线独立 6 参发射曲线（CMOS 图案常数）
    //     两阶段初始化: 逐线共识 median + z-lift 盆地探测（代价面非凸,
    //                   机械初值会落劣质盆地, 实测共识+z-lift 收敛 rms 0.44px）
    //     暂存 finalVirtualK/R/T 供 5-3 与 JSON 落盘使用（对下游保持接口不变）
    // ------------------------------------------------------------------
    cv::Matx33d finalVirtualK = cv::Matx33d::eye();
    cv::Matx33d finalVirtualR = cv::Matx33d::eye();   // PJC 链路恒为 I
    cv::Vec3d   finalVirtualT(0, 0, 0);
    std::set<int> observedLineIds;                     // 运行期实际线号（诊断保留; 4-14 不消费）
    bool haveVirtualPose = false;
    nlohmann::json pjcJson;                            // F2: 曲线+验收落盘
    // mlRes 作用域不出 3c else 块——4-14（3d）消费的 PJC 发射曲线提升至此
    std::vector<calib::ImplicitCurve> pjcCurves;

    if (poseSets.empty()) {
        spdlog::error("no accumulated 3D points; skip PJC/5-3/4-14");
    } else {
        // 统计线号
        for (const auto& ps : poseSets)
            for (int lid : ps.lineIds) observedLineIds.insert(lid);

        // 重组: 按姿态分组(混线) → 按线分组（每线跨姿态）
        std::map<int, std::vector<calib::PosePointSet>> byLine;
        for (const auto& ps : poseSets) {
            // 单姿态内同线多点合并为一组
            std::map<int, calib::PosePointSet> poseByLine;
            for (size_t k = 0; k < ps.points3d.size(); ++k) {
                const int lid = ps.lineIds[k];
                auto& bucket = poseByLine[lid];
                bucket.points3d.push_back(ps.points3d[k]);
                bucket.lineIds.push_back(lid);
            }
            for (auto& kv : poseByLine)
                byLine[kv.first].push_back(std::move(kv.second));
        }

        calib::MultiLineInput mlIn;
        for (auto& kv : byLine) mlIn.lines.push_back(std::move(kv.second));
        mlIn.f = stereoK(0, 0);
        mlIn.principalPoint = cv::Point2d(stereoK(0, 2), stereoK(1, 2));
        // 点云在左相机矫正坐标系 → initialT 用 R1·(80,3,3)（R1 实测 ~10.4° yaw）
        cv::Matx33d R1m = h.R1;
        const cv::Vec3d tMech = R1m * cv::Vec3d(80.0, 3.0, 3.0);
        mlIn.initialT = tMech;

        // —— 阶段1: 逐线共识 median（rms<1 且 cond<1e15 的线）——
        std::vector<double> xs, ys, zs;
        {
            ProjectorJointCalib warmOp;
            for (const auto& line : mlIn.lines) {
                if (line.size() < 5) continue;
                ProjectorJointCalibInput si;
                si.poses = line;
                si.f = mlIn.f;
                si.principalPoint = mlIn.principalPoint;
                si.initialT = tMech;
                auto sr = warmOp.Execute(si);
                if (!sr.success) continue;
                if (sr.finalSampsonRms > 1.0) continue;
                if (sr.jacobianConditionNumber > 1e15) continue;
                xs.push_back(sr.projectorT[0]);
                ys.push_back(sr.projectorT[1]);
                zs.push_back(sr.projectorT[2]);
            }
            if (xs.size() >= 3) {
                std::sort(xs.begin(), xs.end());
                std::sort(ys.begin(), ys.end());
                std::sort(zs.begin(), zs.end());
                mlIn.initialT = cv::Vec3d(xs[xs.size() / 2], ys[ys.size() / 2],
                                          zs[zs.size() / 2]);
                spdlog::info("stage-1 per-line init: median of {} lines "
                             "t0=({:.1f},{:.1f},{:.1f})",
                             xs.size(), mlIn.initialT[0], mlIn.initialT[1],
                             mlIn.initialT[2]);
            }
        }
        // —— 阶段1.5: z-lift 盆地探测（共识点 vs z+60 短跑, 低 rms 胜）——
        // 探针决策落盘（2026-10-04）: 近平局时（差 <0.1px）盆地选择是关键诊断信息
        double basinMedianRms = -1.0, basinZLiftRms = -1.0;
        std::string basinPicked = "median";
        auto shortRun = [&](const cv::Vec3d& t0) {
            ProjectorJointCalibParams sp;
            sp.maxIterations = 15;
            ProjectorJointCalib sop(sp);
            MultiLineInput si = mlIn;
            si.initialT = t0;
            auto sr = sop.ExecuteMultiLine(si);
            return sr.success ? sr.finalSampsonRms : 1e30;
        };
        if (xs.size() >= 3) {
            const cv::Vec3d tMed = mlIn.initialT;
            const double r1 = shortRun(tMed);
            const cv::Vec3d tZlift(tMed[0], tMed[1], tMed[2] + 60.0);
            const double r2 = shortRun(tZlift);
            basinMedianRms = r1;
            basinZLiftRms = r2;
            if (r2 < r1) {
                mlIn.initialT = tZlift;
                basinPicked = "z-lift";
                spdlog::info("stage-1.5 basin probe: median rms={:.2f} z+60 rms={:.2f}"
                             " -> pick z-lift", r1, r2);
            } else {
                spdlog::info("stage-1.5 basin probe: median rms={:.2f} z+60 rms={:.2f}"
                             " -> keep median", r1, r2);
            }
        }

        // —— 阶段2: 完整多线联合 + F7 验收（自动阶梯 L0/L1）——
        // 参考精度标准 v1（docs/plans/2026-08-26-virtual-camera-calib-v2-design.md §2）
        struct AcceptanceCfg {
            // A1-A3 阈值作用在 perLine 归一化坐标残差（非像素; 与 finalSampsonRms
            // 像素口径差 ~百倍, 见 projector_joint_calib.h LineDiag 注——2026-10-04 勘正）
            double a1MaxLineRms = 0.5;      // 归一化坐标
            double a2MaxP95 = 1.5;          // 归一化坐标
            double a3MaxOutlierRatio = 0.02;
            double a5MaxLineDrift = 1.0;    // mm
            double a6MaxPoseDrift = 2.0;    // mm
            double a7MaxBasinSpread = 0.5;  // mm
        } acc;
        // 阈值自主库/数据集层注入（2026-10-04 外置；默认值同上）
        acc.a1MaxLineRms      = ops.pjcAccA1MaxLineRms;
        acc.a2MaxP95          = ops.pjcAccA2MaxP95;
        acc.a3MaxOutlierRatio = ops.pjcAccA3MaxOutlierRatio;
        acc.a5MaxLineDrift    = ops.pjcAccA5MaxLineDrift;
        acc.a6MaxPoseDrift    = ops.pjcAccA6MaxPoseDrift;
        acc.a7MaxBasinSpread  = ops.pjcAccA7MaxBasinSpread;
        auto runMultiSolve = [&](const MultiLineInput& in) {
            ProjectorJointCalib op(pjcParams);   // 主求解: ops 驱动（主库可调）
            return op.ExecuteMultiLine(in);
        };
        auto evaluate = [&](const MultiLineResult& r) {
            nlohmann::json a;
            a["a1_maxLineRms"] = 0.0;
            double worstRms = 0.0, worstP95 = 0.0, worstOut = 0.0;
            for (const auto& ld : r.perLine) {
                worstRms = std::max(worstRms, ld.rms);
                worstP95 = std::max(worstP95, ld.p95);
                worstOut = std::max(worstOut, ld.outlierRatio);
            }
            a["a1_maxLineRms"] = worstRms;
            a["a2_maxP95"] = worstP95;
            a["a3_maxOutlierRatio"] = worstOut;
            a["a1_pass"] = worstRms <= acc.a1MaxLineRms;
            a["a2_pass"] = worstP95 <= acc.a2MaxP95;
            a["a3_pass"] = worstOut <= acc.a3MaxOutlierRatio;
            return a;
        };
        auto mlRes = runMultiSolve(mlIn);
        nlohmann::json attempts = nlohmann::json::array();
        int attemptLevel = 0;
        // 自动阶梯: L0 默认 → L1 核收紧（huberDelta 0.7/0.5）
        while (mlRes.success) {
            nlohmann::json att;
            att["level"] = attemptLevel;
            att["params"] = "L" + std::to_string(attemptLevel);
            att["metrics"] = evaluate(mlRes);
            att["projectorT"] = {mlRes.projectorT[0], mlRes.projectorT[1],
                                 mlRes.projectorT[2]};
            att["finalRms"] = mlRes.finalSampsonRms;
            attempts.push_back(att);
            const auto& m = att["metrics"];
            if (m["a1_pass"].get<bool>() && m["a2_pass"].get<bool>()
                && m["a3_pass"].get<bool>())
                break;   // A1-A3 过 → 出循环
            if (attemptLevel >= 2) break;
            ++attemptLevel;
            ProjectorJointCalibParams rp = pjcParams;   // 阶梯重试继承 ops, 仅核收紧
            rp.huberDelta0 = (attemptLevel == 1) ? 0.7 : 0.5;
            ProjectorJointCalib rop(rp);
            auto rr = rop.ExecuteMultiLine(mlIn);
            if (rr.success) mlRes = std::move(rr);
            else break;
        }
        if (!mlRes.success) {
            spdlog::error("PJC multi-line failed: {}", mlRes.message);
        } else {
            finalVirtualK = cv::Matx33d::eye();
            finalVirtualK(0, 0) = mlIn.f;
            finalVirtualK(1, 1) = mlIn.f;
            finalVirtualK(0, 2) = mlIn.principalPoint.x;
            finalVirtualK(1, 2) = mlIn.principalPoint.y;
            finalVirtualR = cv::Matx33d::eye();
            finalVirtualT = mlRes.projectorT;
            haveVirtualPose = true;
            pjcCurves = mlRes.emissionCurves;   // 提升: 3d 4-14 三源之一（作用域）
            spdlog::info("PJC multi-line OK: projectorT=({:.3f},{:.3f},{:.3f}) "
                         "rms {}->{:.4f}, lines={}, poses={}, pts={}, cond={:.3g}",
                         mlRes.projectorT[0], mlRes.projectorT[1],
                         mlRes.projectorT[2],
                         mlRes.initialSampsonRms, mlRes.finalSampsonRms,
                         mlRes.lineCount, mlRes.poseCount,
                         mlRes.totalPointCount,
                         mlRes.jacobianConditionNumber);
            // F2: 曲线与诊断落盘（pjc 节）
            pjcJson["projectorT"] = {mlRes.projectorT[0], mlRes.projectorT[1],
                                     mlRes.projectorT[2]};
            pjcJson["f"] = mlIn.f;
            pjcJson["principalPoint"] = {mlIn.principalPoint.x,
                                         mlIn.principalPoint.y};
            pjcJson["finalSampsonRms"] = mlRes.finalSampsonRms;
            pjcJson["initialSampsonRms"] = mlRes.initialSampsonRms;
            pjcJson["jacobianConditionNumber"] = mlRes.jacobianConditionNumber;
            pjcJson["lineCount"] = mlRes.lineCount;
            pjcJson["poseCount"] = mlRes.poseCount;
            pjcJson["totalPointCount"] = mlRes.totalPointCount;
            pjcJson["epipolarRowStep"] = cfg.epipolarStep;   // R5: 步距随表走
            // 盆地探针决策落盘（2026-10-04；近平局时此选择是关键诊断信息）
            if (basinMedianRms >= 0.0)
                pjcJson["basinProbe"] = {{"medianRms", basinMedianRms},
                                         {"zLiftRms", basinZLiftRms},
                                         {"picked", basinPicked}};
            nlohmann::json curves = nlohmann::json::array();
            for (size_t k = 0; k < mlRes.emissionCurves.size(); ++k) {
                nlohmann::json cj;
                cj["lineId"] = (k < mlRes.usedLineIdx.size())
                    ? mlRes.usedLineIdx[static_cast<int>(k)] + 1 : k;
                const auto& c = mlRes.emissionCurves[k];
                cj["coeffs"] = {c.coeffs[0], c.coeffs[1], c.coeffs[2],
                                c.coeffs[3], c.coeffs[4], c.coeffs[5]};
                cj["discriminant"] = c.discriminant;
                cj["pointCount"] = c.pointCount;
                curves.push_back(std::move(cj));
            }
            pjcJson["emissionCurves"] = std::move(curves);

            // —— F7 深度验收: A5 剔线 / A6 剔姿态 / A7 basin（短程重优化测 t 漂移）——
            // 护栏1（假 PASS 修复 2026-09-03）: 失败哨兵 (1e9,0,0) 曾被静默跳过——失败解越多
            // 漂移统计越低越易假 PASS；现逐段计数＋逐条 warn＋solveFailures 进 verdict（任一失败强制 DEGRADED）
            int a5Fail = 0, a6Fail = 0, a7Fail = 0;
            auto driftSolve = [&](MultiLineInput in, const cv::Vec3d& tRef,
                                  int skipLineIdx, int skipPoseIdx, bool& solved) {
                solved = false;
                if (skipLineIdx >= 0) {
                    if (in.lines.size() <= 20) return cv::Vec3d(1e9, 0, 0);  // 保底 20 线
                    in.lines.erase(in.lines.begin() + skipLineIdx);
                }
                if (skipPoseIdx >= 0) {
                    for (auto& line : in.lines) {
                        if (line.size() > static_cast<size_t>(skipPoseIdx))
                            line.erase(line.begin() + skipPoseIdx);
                    }
                }
                in.initialT = tRef;                       // 从最优解出发
                ProjectorJointCalibParams sp;
                sp.maxIterations = 12;                    // 轻量: 漂移测量不需全收敛
                sp.robustEnabled = false;                 // 短程内关核加速
                sp.irlsMaxRounds = 0;
                ProjectorJointCalib sop(sp);
                auto r = sop.ExecuteMultiLine(in);
                if (!r.success) return cv::Vec3d(1e9, 0, 0);
                solved = true;
                return r.projectorT;
            };
            const cv::Vec3d tRef = mlRes.projectorT;
            double a5Worst = 0.0, a6Worst = 0.0, a7Spread = 0.0;

            // —— C1: F7 解间并行（三段任务单池动态自调度）——
            // 确定性论证（并行与串行逐位一致）:
            //   1) 每个漂移解是纯单线程确定性计算——driftSolve/A7 任务体各自构造
            //      独立 ProjectorJointCalib 实例、按值拷贝 MultiLineInput 私有副本，
            //      无共享可变状态；
            //   2) mlIn/tRef 只读共享（所有任务仅读，私有化发生在任务体内）；
            //   3) 聚合为可交换 max（逐任务 ||t-tRef|| 取 max），与完成序无关。
            //   ⇒ 任意线程数/任意调度序下，聚合结果与串行执行逐位一致。
            struct F7Task { int kind; int lineIdx; int poseIdx; cv::Vec3d t0; };
            struct F7TaskOut { bool solved = false; cv::Vec3d t{0, 0, 0}; };
            std::vector<F7Task> tasks;
            {   // A7 任务先入队（LPT: 20 迭代长任务优先，减少并行尾部拖尾）
                const double s = 3.0;
                for (int dx = -1; dx <= 1; dx += 2)
                    for (int dy = -1; dy <= 1; dy += 2)
                        for (int dz = -1; dz <= 1; dz += 2)
                            tasks.push_back({2, -1, -1,
                                             cv::Vec3d(tRef[0] + dx * s,
                                                       tRef[1] + dy * s,
                                                       tRef[2] + dz * s)});
            }
            for (size_t l = 0; l < mlIn.lines.size(); l += 3)   // A5 隔位抽样 9 线
                tasks.push_back({0, static_cast<int>(l), -1, tRef});
            const size_t nPoses = mlIn.lines.empty() ? 0 : mlIn.lines[0].size();
            for (size_t p = 0; p < nPoses; p += 2)              // A6 隔位抽样 11 姿态
                tasks.push_back({1, -1, static_cast<int>(p), tRef});
            std::vector<F7TaskOut> results(tasks.size());

            // 单任务执行体（串行/并行两路共用）
            // 护栏 2: 整个任务体包 try/catch——任何异常（含 bad_alloc）只计该任务
            // 失败＋warn（聚合时并入 solveFailures→DEGRADED），绝不外逸成 terminate
            auto runF7Task = [&](size_t idx) {
                const F7Task& tk = tasks[idx];
                F7TaskOut& out = results[idx];
                try {
                    if (tk.kind == 0) {           // A5 剔线（12 迭代短程）
                        out.t = driftSolve(mlIn, tRef, tk.lineIdx, -1, out.solved);
                    } else if (tk.kind == 1) {    // A6 剔姿态（12 迭代短程）
                        out.t = driftSolve(mlIn, tRef, -1, tk.poseIdx, out.solved);
                    } else {                      // A7 ±3mm 8 角重启（20 迭代）
                        MultiLineInput si = mlIn;
                        si.initialT = tk.t0;
                        ProjectorJointCalibParams sp;
                        sp.maxIterations = 20;
                        sp.robustEnabled = false;
                        sp.irlsMaxRounds = 0;
                        ProjectorJointCalib sop(sp);
                        auto r = sop.ExecuteMultiLine(si);
                        out.solved = r.success;
                        out.t = r.success ? r.projectorT : cv::Vec3d(1e9, 0, 0);
                    }
                } catch (const std::exception& e) {
                    out.solved = false;
                    spdlog::warn("F7 drift task exception (kind={} line={} pose={}): {}",
                                 tk.kind, tk.lineIdx, tk.poseIdx, e.what());
                } catch (...) {
                    out.solved = false;
                    spdlog::warn("F7 drift task exception (kind={} line={} pose={}): "
                                 "unknown", tk.kind, tk.lineIdx, tk.poseIdx);
                }
            };

            // 线程数（护栏 3）: 自动 T = min(hw_concurrency-2, 8)，硬顶 8。
            // 不设内存项——B（块稀疏 GN）后每解内存仅数十 MB（557MB 稠密 J 已除），
            // 8 线程峰值增量相对全链可忽略。
            // config.json 顶层键 f7Threads 覆盖: 缺省=自动; 1=串行回退; <=0 归一为 1。
            int f7T = cfg.f7Threads;   // 字段缺省 0 = 自动
            if (f7T <= 0) {
                const unsigned hw = std::thread::hardware_concurrency();
                f7T = (hw >= 3) ? static_cast<int>(hw - 2) : 1;
                if (f7T > 8) f7T = 8;
            }

            if (f7T <= 1 || tasks.size() <= 1) {
                // 纯串行路径（f7Threads=1）: 不创建线程，与改前串行行为同构
                for (size_t i = 0; i < tasks.size(); ++i) runF7Task(i);
            } else {
                const size_t T = std::min<size_t>(f7T, tasks.size());
                spdlog::info("F7 drift solve: {} tasks on {} threads (dynamic "
                             "self-scheduling)", tasks.size(), T);
                std::atomic<size_t> next{0};
                std::vector<std::thread> pool;
                pool.reserve(T);
                for (size_t w = 0; w < T; ++w)
                    pool.emplace_back([&] {
                        for (;;) {
                            const size_t i = next.fetch_add(1);
                            if (i >= tasks.size()) break;
                            runF7Task(i);
                        }
                    });
                for (auto& th : pool) th.join();
            }

            // 聚合（join 后主线程串行; max 可交换 → 序无关; 仅计 solved 任务）
            for (size_t i = 0; i < tasks.size(); ++i) {
                const F7Task& tk = tasks[i];
                const F7TaskOut& out = results[i];
                if (tk.kind == 0) {
                    if (!out.solved) {
                        ++a5Fail;
                        spdlog::warn("A5 drift solve failed: skip line {}", tk.lineIdx);
                        continue;
                    }
                    const cv::Vec3d d = out.t - tRef;
                    a5Worst = std::max(a5Worst, std::sqrt(d.dot(d)));
                } else if (tk.kind == 1) {
                    if (!out.solved) {
                        ++a6Fail;
                        spdlog::warn("A6 drift solve failed: skip pose {}", tk.poseIdx);
                        continue;
                    }
                    const cv::Vec3d d = out.t - tRef;
                    a6Worst = std::max(a6Worst, std::sqrt(d.dot(d)));
                } else {
                    if (!out.solved) {
                        ++a7Fail;
                        const cv::Vec3d off = tk.t0 - tRef;
                        spdlog::warn("A7 drift solve failed: skip restart ({:+.0f},{:+.0f},{:+.0f})mm",
                                     off[0], off[1], off[2]);
                        continue;
                    }
                    const cv::Vec3d d = out.t - tRef;
                    a7Spread = std::max(a7Spread, std::sqrt(d.dot(d)));
                }
            }
            const int solveFailures = a5Fail + a6Fail + a7Fail;
            // 汇总 acceptance
            nlohmann::json acceptance;
            {
                auto base = evaluate(mlRes);
                acceptance["A1_A3"] = base;
                nlohmann::json deep;
                deep["a5_maxLineDrift_mm"] = a5Worst;
                deep["a6_maxPoseDrift_mm"] = a6Worst;
                deep["a7_basinSpread_mm"] = a7Spread;
                deep["a5_pass"] = a5Worst <= acc.a5MaxLineDrift;
                deep["a6_pass"] = a6Worst <= acc.a6MaxPoseDrift;
                deep["a7_pass"] = a7Spread <= acc.a7MaxBasinSpread;
                acceptance["A5_A7"] = deep;
                acceptance["solveFailures"] = solveFailures;   // additive 键（护栏1）
                bool allPass = base["a1_pass"].get<bool>()
                            && base["a2_pass"].get<bool>()
                            && base["a3_pass"].get<bool>()
                            && deep["a5_pass"].get<bool>()
                            && deep["a6_pass"].get<bool>()
                            && deep["a7_pass"].get<bool>()
                            && solveFailures == 0;   // 任一 drift 解失败 → 强制 DEGRADED（即便数值全过）
                acceptance["verdict"] = allPass ? "PASS" : "DEGRADED";
                acceptance["thresholds"] = {
                    {"a1MaxLineRms", acc.a1MaxLineRms},
                    {"a2MaxP95", acc.a2MaxP95},
                    {"a3MaxOutlierRatio", acc.a3MaxOutlierRatio},
                    {"a5MaxLineDrift", acc.a5MaxLineDrift},
                    {"a6MaxPoseDrift", acc.a6MaxPoseDrift},
                    {"a7MaxBasinSpread", acc.a7MaxBasinSpread}};
                acceptance["attempts"] = attempts;
            }
            pjcJson["acceptance"] = acceptance;
            if (solveFailures > 0) {
                spdlog::warn("drift solve failures: {} (A5 line={} A6 pose={} A7 restart={})"
                             " -> verdict forced DEGRADED",
                             solveFailures, a5Fail, a6Fail, a7Fail);
            }
            spdlog::info("acceptance: {} (A5 lineDrift={:.2f}mm A6 poseDrift={:.2f}mm "
                         "A7 basin={:.2f}mm)",
                         acceptance["verdict"].get<std::string>(),
                         a5Worst, a6Worst, a7Spread);
            // perLine 诊断落盘
            nlohmann::json pl = nlohmann::json::array();
            for (const auto& ld : mlRes.perLine) {
                nlohmann::json o;
                o["lineIdx"] = ld.lineIdx;
                o["rms"] = ld.rms;
                o["p95"] = ld.p95;
                o["outlierRatio"] = ld.outlierRatio;
                o["pointCount"] = ld.pointCount;
                o["meanPoseWeight"] = ld.meanPoseWeight;
                pl.push_back(std::move(o));
            }
            pjcJson["perLine"] = std::move(pl);
        }
        pjcOp.Destroy();
    }

    // ------------------------------------------------------------------
    // 3d. 5-3 + 4-14 (PJC 成功后执行; 5-3 依赖 finalVirtualR/T, 4-14 三源=
    //     模块1 rectifyTempTable + 5-3 laserExtrinTable + PJC emissionCurves)
    // ------------------------------------------------------------------
    LaserExtrinsicCompensateCPUResult laserExtrinTable;
    calib::CurveMapTempTableResult     cmResult;
    nlohmann::json                     cmttMeta;
    bool haveLaserExtrin   = false;
    bool haveCurveMapTable = false;

    if (!haveVirtualPose) {
        spdlog::warn("no virtual pose from 4-11; skip 5-3 and 4-14");
    } else {
        // ----- 5-3 laser_extrinsic_compensate -----
        // 决定 3: virtual→R 通过链式复合
        //   R_v2r = R_stereo · R_v2l
        //   T_v2r = R_stereo · T_v2l + T_stereo
        cv::Vec3d T_stereo;
        for (int i = 0; i < 3; ++i) T_stereo(i) = h.T.at<double>(i);
        cv::Matx33d R_v2r = stereoR * finalVirtualR;
        cv::Vec3d   T_v2r = stereoR * finalVirtualT + T_stereo;

        calib::CameraExtrinsics v2l, v2r;
        for (int i = 0; i < 9; ++i) v2l.R[i] = finalVirtualR.val[i];
        for (int i = 0; i < 3; ++i) v2l.T[i] = finalVirtualT[i];
        v2l.referenceTemp = cfg.referenceTemp;
        for (int i = 0; i < 9; ++i) v2r.R[i] = R_v2r.val[i];
        for (int i = 0; i < 3; ++i) v2r.T[i] = T_v2r[i];
        v2r.referenceTemp = cfg.referenceTemp;

        spdlog::info("5-3 v2l T=({:.2f},{:.2f},{:.2f})  v2r T=({:.2f},{:.2f},{:.2f})",
                     v2l.T[0], v2l.T[1], v2l.T[2],
                     v2r.T[0], v2r.T[1], v2r.T[2]);

        laserExtrinTable = lecompOp.Execute(v2l, v2r);
        if (!laserExtrinTable.success) {
            spdlog::error("5-3 laser_extrinsic_compensate failed: {}",
                          laserExtrinTable.message);
        } else {
            haveLaserExtrin = true;
            spdlog::info("5-3 OK: virtual→L/R temp tables ({} entries each)",
                         laserExtrinTable.leftResult.table.size());
        }

        // ----- 4-14 curve_map_temp_table（61 档 sidecar; 纯 CPU; 三源=模块1温表+5-3表+PJC曲线）-----
        // 先例: modules/07_pipelinemgmt/pipelines/calibcompute/LaserChain.cpp
        //       realCurveMapTempTableOp（同流程: 生成→CmttReader 复核→落盘→文件级
        //       sha256→meta; 失败容忍语义: 前置缺失/生成失败→error log, 不出产物）
        if (!input->handoff.haveRectifyTempTable) {
            spdlog::error("4-14 skipped: camera_calib.json 缺 stereoRectifyTempTable 整表");
        } else if (pjcCurves.empty()) {
            // 07 先例语义: PJC success 但 emissionCurves 空 → 降级不出产物
            spdlog::error("4-14 skipped: PJC emissionCurves empty");
        } else if (!haveLaserExtrin) {
            spdlog::error("4-14 skipped: 5-3 laserExtrinTable unavailable");
        } else {
            calib::CurveMapTempTableGenParams cmp;
            cmp.referenceTemp = cfg.referenceTemp;
            cmp.tempHalfRange = static_cast<float>(ops.cmttTempHalfRange);  // 主库可调（源表覆盖对齐, 设计硬规则 #5）
            cmp.tempStep      = 0.5f;
            cmp.rowStep       = cfg.epipolarStep;   // 红线: rowStep 随标定参数走, 禁硬编码
            cmp.depthMin      = cfg.depthMin;       // 全档统一（设计硬规则 #2）
            cmp.depthMax      = cfg.depthMax;
            cmp.tierThreads   = cfg.cmttThreads;    // 档级并行（0=自动; 线程数无关确定性）
            try {
                calib::CurveMapTempTableGenerator cmGen(cmp);
                cmResult = cmGen.Generate(pjcCurves,
                                          input->handoff.rectifyTempTable,
                                          laserExtrinTable, h.imageSize);
                if (!cmResult.success) {
                    // 红线: 参考档 FAIL 整体不出产物（生成器已保证 sidecarBytes 空）
                    spdlog::error("4-14 curve_map_temp_table failed (ref-tier red line): {}",
                                  cmResult.message);
                } else {
                    // CmttReader 复核（落盘前: 任何复核失败不得留文件）
                    calib::CmttReader reader;
                    if (!reader.load(cmResult.sidecarBytes.data(),
                                     cmResult.sidecarBytes.size())) {
                        spdlog::error("4-14 sidecar re-verify fail (CmttReader load)");
                    } else {
                        // 落盘 {outPath 父目录}/curve_map_temp_table.bin（二进制;
                        // 目录缺失则建——失败交 ofstream 兜底; outPath 无目录前缀
                        // 时 parent_path 为空 → 文件落当前目录, 同 07 相对路径语义）
                        const std::filesystem::path outDir =
                            std::filesystem::path(outPath).parent_path();
                        std::error_code ec;
                        std::filesystem::create_directories(outDir, ec);
                        const std::filesystem::path file =
                            outDir / "curve_map_temp_table.bin";
                        bool writeOk = false;
                        {
                            std::ofstream f(file, std::ios::binary | std::ios::trunc);
                            if (f)
                                f.write(reinterpret_cast<const char*>(
                                            cmResult.sidecarBytes.data()),
                                        static_cast<std::streamsize>(
                                            cmResult.sidecarBytes.size()));
                            f.close();   // 显式 close: close 期 flush 失败经流状态可观测
                            writeOk = static_cast<bool>(f);   // open/write/close 任一失败
                        }
                        if (!writeOk) {
                            // 语义（选定）: 生成已成功仅落盘失败 → haveCurveMapTable
                            // 保持 false、meta 不写、exit 置 partial（与 07 返回 fail
                            // 同效）; 半文件 best-effort 清理（close 后再删, 避免
                            // Windows 打开句柄下 remove 必然 sharing violation）
                            std::error_code rmEc;
                            std::filesystem::remove(file, rmEc);
                            spdlog::error("4-14 sidecar write fail (partial cleanup "
                                          "best-effort): {}", file.string());
                        } else {
                            // sha256: 对落盘文件流式重读计算（端到端契约: hash 覆盖
                            // 所写文件字节而非内存 bytes——写盘/驱动层缺陷可被察觉）
                            std::string fileSha;
                            if (!sha256FileImpl(file, fileSha)) {
                                std::error_code rmEc;   // 同红线: FAIL 不留产物
                                std::filesystem::remove(file, rmEc);
                                spdlog::error("4-14 sidecar hash re-read fail "
                                              "(product removed): {}", file.string());
                            } else {
                                haveCurveMapTable = true;
                                // clamped 档数取 Reader 复核值（07 同口径）
                                int clamped = 0;
                                for (size_t i = 0; i < reader.tierCount(); ++i)
                                    if (reader.isClamped(i)) ++clamped;
                                const float clampedRatio =
                                    reader.tierCount() > 0
                                        ? static_cast<float>(clamped)
                                          / static_cast<float>(reader.tierCount())
                                        : 0.f;
                                // meta 10 键（键名照 07 CalibSerialize.cpp curveMapToJson）
                                cmttMeta = {
                                    {"success", true},
                                    {"message", "4-14 ok"},
                                    {"path", "curve_map_temp_table.bin"},
                                    {"tempBase", reader.header().tempBaseX10 / 10.0},
                                    {"tempStep", reader.header().tempStepX10 / 10.0f},
                                    {"tierCount", static_cast<int>(reader.tierCount())},
                                    {"tierCountOk", cmResult.okCount()},
                                    {"clampedRatio", clampedRatio},
                                    {"gridHash", reader.header().gridHash},
                                    {"sha256", std::move(fileSha)},
                                };
                                spdlog::info("4-14 OK: tiers={} ok={} clamped={:.4f}",
                                             static_cast<int>(reader.tierCount()),
                                             cmResult.okCount(), clampedRatio);
                                // clamped 告警阈（2026-10-04；超出仅 warn，提示档位外推）
                                if (clampedRatio > static_cast<float>(ops.cmttClampedWarnRatio))
                                    spdlog::warn("4-14 clampedRatio {:.4f} exceeds warn "
                                                 "threshold {:.4f} (extrapolated tiers)",
                                                 clampedRatio, ops.cmttClampedWarnRatio);
                            }
                        }
                    }
                }
            } catch (const std::exception& e) {
                spdlog::error("4-14 exception: {}", e.what());
            } catch (...) {
                spdlog::error("4-14 unknown exception");
            }
        }
    }

    // ------------------------------------------------------------------
    // 4. 资源销毁 (算子规范要求析构前显式 Destroy)
    // ------------------------------------------------------------------
    maskL.Destroy();    maskR.Destroy();
    cclL.Destroy();     cclR.Destroy();
    labelL.Destroy();   labelR.Destroy();
    stegerL.Destroy();  stegerR.Destroy();
    undistLOp.Destroy(); undistROp.Destroy();
    epipolarL.Destroy(); epipolarR.Destroy();
    matchOp.Destroy();
    reconOp.Destroy();
    // pjcOp.Destroy() 已在 3c 段内调用
    lecompOp.Destroy();

    // ------------------------------------------------------------------
    // 5. 写 laser_calib.json（PJC 链路版）
    // ------------------------------------------------------------------
    size_t totalAccum3D = 0;
    for (const auto& ps : poseSets) totalAccum3D += ps.points3d.size();
    nlohmann::json j;
    j["schema"]  = "factory_calib.laser_calib.v2";
    j["build"]   = "6.3-cmtt";
    j["posesProcessed"]    = input->poseFrames.size();
    j["framesOk"]          = framesOk;
    j["framesSkipped"]     = framesSkip;
    j["accumulatedPoints3D"] = totalAccum3D;
    j["haveVirtualPose"]   = haveVirtualPose;
    j["haveLaserExtrin"]   = haveLaserExtrin;
    j["haveCurveMapTable"] = haveCurveMapTable;

    if (haveVirtualPose) {
        auto matxToArray = [](const cv::Matx33d& m) {
            return std::vector<double>{m(0,0),m(0,1),m(0,2),
                                       m(1,0),m(1,1),m(1,2),
                                       m(2,0),m(2,1),m(2,2)};
        };
        j["virtualK"] = matxToArray(finalVirtualK);
        j["virtualR"] = matxToArray(finalVirtualR);
        j["virtualT"] = std::vector<double>{finalVirtualT[0],
                                            finalVirtualT[1],
                                            finalVirtualT[2]};
        if (!pjcJson.is_null())
            j["pjc"] = pjcJson;   // F2: projectorT/25 曲线/epipolarRowStep/诊断
    }
    if (haveLaserExtrin) {
        j["laserExtrinsicTempTable"] = laserExtrinTable.toJson();
    }
    if (haveCurveMapTable) {
        j["curveMapTempTable"] = cmttMeta;
    }

    // review I1: 加 status 字段 (ok|partial) 让下游消费方可识别; 不再依赖文件存在性
    int exitCode = 0;
    std::string exitStatus = "ok";
    if (!haveVirtualPose || !haveLaserExtrin || !haveCurveMapTable) {
        spdlog::warn("pipeline incomplete: virtualPose={} laserExtrin={} curveMapTable={}",
                     haveVirtualPose, haveLaserExtrin, haveCurveMapTable);
        exitCode = 1;
        exitStatus = "partial";
    }
    j["status"] = exitStatus;

    // ★ verdict 透传（2026-10-04）：status/exit 只看产物完整性，不看验收质量——
    // DEGRADED 标定此前会以 status=ok/exit=0 静默通过。新增顶层 verdict 摘要键
    // （additive-safe，不改 status 兼容语义），并在尾部醒目 warn。
    if (pjcJson.contains("acceptance") && pjcJson["acceptance"].contains("verdict")) {
        const std::string accVerdict =
            pjcJson["acceptance"]["verdict"].get<std::string>();
        j["verdict"] = accVerdict;
        if (accVerdict == "DEGRADED")
            spdlog::warn(">>> ACCEPTANCE DEGRADED: artifacts written, but quality gates "
                         "failed/uncertain -- inspect pjc.acceptance before use <<<");
    }

    if (!writeLaserJson(outPath, j)) {
        spdlog::error("cannot write output: {}", outPath);
        return 1;
    }
    spdlog::info("laser_calib (6.3-cmtt) -> {} (status={}, exit={})",
                 outPath, exitStatus, exitCode);
    return exitCode;

    } catch (const std::exception& e) {
        spdlog::error("laser_calib unhandled exception: {}", e.what());
        return 1;
    } catch (...) {
        spdlog::error("laser_calib unknown exception");
        return 1;
    }
}

#ifndef FC2_BUILDING_LIB
// CLI 入口（库化重构后抽出；GUI/exe 侧由 FC2_BUILDING_LIB 控制缺席）
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: laser_calib <input_dir> [output_json]\n"
                  << "  input_dir 含 config.json + camera_calib.json + pose_*/L_tube*.png + R_tube*.png\n";
        return 2;
    }
    std::string inDir = argv[1];
    std::string outPath = argc >= 3 ? argv[2] : "laser_calib.json";
    return fc::runLaserCalibRaw(inDir, outPath);
}
#endif  // FC2_BUILDING_LIB
