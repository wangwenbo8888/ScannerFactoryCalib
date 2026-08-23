// laser_calib_cli.cpp — 模块2 激光标定 CLI（Task 6.2 完整实现）
//
// 当前进度: 6.2-e (5-3 laser_extrinsic_compensate + 4-13 plane_map_temp_table + 写真实 JSON)
//   Task 6.2 全部 13 算子 + 2 温度表算子串通
//
// 设计依据: docs/plans/2026-07-18-factory-calib-impl.md Task 6.2 Step 0
// 算子签名以 Step 0.1 速查表为准；原 Step 1 伪代码禁止照抄。

#include "calib_io.h"
#include "laser_calib_runner_internal.h"

#include "mask_extract_cuda.h"
#include "region_analyze_cuda.h"
#include "laser_label_cuda.h"
#include "steger_extract_cuda.h"
#include "undistort_points_cuda.h"
#include "epipolar_interp_cuda.h"
#include "laser_match_cuda.h"
#include "laser_reconstruct_cuda.h"
#include "endpoint_extract_cuda.h"
#include "virtual_camera_pose_cuda.h"
#include "pose_optimize_cuda.h"
#include "projector_joint_calib.h"
#include "laser_extrinsic_compensate_cpu.h"
#include "plane_map_cuda.h"        // LineMapStats 完整定义, plane_map_temp_table.h 仅前向声明
#include "plane_map_temp_table.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <cstdio>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <string>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <cmath>
#include <exception>

using namespace fc;
using namespace calib;

namespace fc {
int runLaserCalibRaw(const std::string& inDir, const std::string& outPath) {
    spdlog::info("=== laser_calib (build 6.2-e final) ===");

    // review C1/I3 防尽: 顶层 try/catch 把算子可能抛的 std::invalid_argument
    // 等异常转成 spdlog::error + exit 1, 避免进程崩溃 (Windows 退出码 0xC0000005)
    try {

    // ------------------------------------------------------------------
    // 1. 加载输入 + 一致性校验
    // ------------------------------------------------------------------
    auto input = loadLaserInput(inDir);
    if (!input) {
        spdlog::error("load laser input failed");
        return 1;
    }
    const auto& cfg = input->config;
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
    //    mask 前端参数从 config.json 读取（TODO 6.2-b 落地）：
    //    实拍激光线宽仅 2~6px，默认 erodeSize=5 会把线整条腐蚀掉（连通域=0）
    // ------------------------------------------------------------------
    cv::cuda::Stream stream;

    MaskExtractParams maskParams;
    maskParams.threshold      = cfg.maskThreshold;
    maskParams.erodeSize      = cfg.maskErodeSize;
    maskParams.laserDilateSize = cfg.maskDilateSize;
    maskParams.minArea        = cfg.maskMinArea;
    maskParams.maxArea        = cfg.maskMaxArea;
    // config.lineIds 数量 = 产线条数：4-1 按面积留前 K 条（4-3 expectedLineCount 校验）
    if (!cfg.lineIds.empty()) maskParams.keepTopK = (int)cfg.lineIds.size();
    MaskExtractCUDA maskL(maskParams);
    MaskExtractCUDA maskR(maskParams);
    spdlog::info("4-1 MaskExtractCUDA x2 constructed (threshold={}, erode={}, dilate={}, "
                 "area=[{},{}){}, keepTopK from lineIds)",
                 maskParams.threshold, maskParams.erodeSize, maskParams.laserDilateSize,
                 maskParams.minArea, maskParams.maxArea,
                 maskParams.keepTopK > 0 ? " enabled" : "");

    RegionAnalyzerParams cclParams;
    cclParams.deviceId = cfg.deviceId;
    cclParams.minArea  = cfg.maskMinArea;
    cclParams.maxArea  = cfg.maskMaxArea;
    RegionAnalyzerCUDA cclL(cclParams);
    RegionAnalyzerCUDA cclR(cclParams);
    spdlog::info("4-2 RegionAnalyzerCUDA x2 constructed");

    LaserLabelParams labelParams;
    labelParams.deviceId = cfg.deviceId;
    // config.lineIds 数量 = 产线规格条数，喂 4-3 做编号后校验（不符仅告警不拒帧）
    if (!cfg.lineIds.empty()) labelParams.expectedLineCount = (int)cfg.lineIds.size();
    LaserLabelerCUDA labelL(labelParams);
    LaserLabelerCUDA labelR(labelParams);
    spdlog::info("4-3 LaserLabelerCUDA x2 constructed{}",
                 labelParams.expectedLineCount > 0
                     ? " (expectedLineCount=" + std::to_string(labelParams.expectedLineCount) + ")"
                     : "");

    // ----- 4-4 Steger -----
    // 参数: sigma/threshold 用默认; deviceId 从 cfg
    StegerParams stegerParams;
    stegerParams.deviceId = cfg.deviceId;
    StegerExtractorCUDA stegerL(stegerParams);
    StegerExtractorCUDA stegerR(stegerParams);
    spdlog::info("4-4 StegerExtractorCUDA x2 (L/R) constructed");

    // ----- 4-5 UndistortPoints -----
    // 参数 K/D/R/P 来自 handoff（模块1 输出的内参 + 立体矫正）。
    // L 路: K=K_L, D=D_L, R=R1, P=P1
    // R 路: K=K_R, D=D_R, R=R2, P=P2
    // 注意: 算子把 P[0][3] 当作加性常量并入 x（把归一化点当 Z=1），而 P2[0][3]=-fx·B
    // 是基线项——直接传会把 R 路整体平移 -53142px（视差虚增至 54000+，match 全拒）。
    // 正确的矫正坐标只需 P 的 K 部分（与 cv::undistortPoints 行为一致），
    // 物理视差由左右光线 B/Z 自然形成 → 这里把 P 的平移列清零。
    cv::Mat P1k = h.P1.clone(), P2k = h.P2.clone();
    if (P1k.cols > 3) P1k.at<double>(0, 3) = 0.0;
    if (P2k.cols > 3) P2k.at<double>(0, 3) = 0.0;
    UndistortPointsParams undistL;
    undistL.cameraMatrix = h.cameraMatrixL;
    undistL.distCoeffs   = h.distCoeffsL;
    undistL.R            = h.R1;
    undistL.P            = P1k;
    undistL.deviceId     = cfg.deviceId;
    undistL.validate();
    UndistortPointsParams undistR;
    undistR.cameraMatrix = h.cameraMatrixR;
    undistR.distCoeffs   = h.distCoeffsR;
    undistR.R            = h.R2;
    undistR.P            = P2k;
    undistR.deviceId     = cfg.deviceId;
    undistR.validate();
    UndistortPointsCuda undistLOp(undistL);
    UndistortPointsCuda undistROp(undistR);
    spdlog::info("4-5 UndistortPointsCuda x2 (R1/P1, R2/P2 from handoff, baseline column zeroed)");

    // ----- 4-6 EpipolarInterp -----
    // lineIdCheck=true 标定模式（按 line_id 同线插值；扫描模式才用 false）
    EpipolarInterpParams epipolarParams;
    epipolarParams.deviceId   = cfg.deviceId;
    epipolarParams.lineIdCheck = true;
    epipolarParams.epipolar_row_step = cfg.interpStep;
    epipolarParams.max_x_diff       = cfg.interpMaxXDiff;
    epipolarParams.max_y_span       = cfg.interpMaxYSpan;
    EpipolarInterpCuda epipolarL(epipolarParams);
    EpipolarInterpCuda epipolarR(epipolarParams);
    spdlog::info("4-6 EpipolarInterpCuda x2 (lineIdCheck=true, step={}, max_x_diff={}, max_y_span={})",
                 cfg.interpStep, cfg.interpMaxXDiff, cfg.interpMaxYSpan);

    // ----- 4-7 LaserMatch -----
    // 单实例吃 L+R 两路输入
    LaserMatchParams matchParams;
    matchParams.deviceId = cfg.deviceId;
    matchParams.min_disparity = cfg.matchMinDisparity;
    matchParams.max_disparity = cfg.matchMaxDisparity;
    matchParams.epipolar_row_step = cfg.interpStep;
    LaserMatchCuda matchOp(matchParams);
    spdlog::info("4-7 LaserMatchCuda constructed (single instance, L+R input, "
                 "disparity=[%.0f,%.0f])",
                 matchParams.min_disparity, matchParams.max_disparity);

    // ----- 4-8 LaserReconstruct -----
    // 单实例，Q 矩阵按调用传入（头文件设计如此，避免跨调用累积）。
    LaserReconstructParams reconParams;
    reconParams.minDepth = cfg.depthMin;
    reconParams.maxDepth = cfg.depthMax;
    reconParams.deviceId  = cfg.deviceId;
    LaserReconstructCuda reconOp(reconParams);
    spdlog::info("4-8 LaserReconstructCuda constructed (Q per-call from handoff.Q)");

    // ----- 4-9 EndpointExtract -----
    EndpointExtractParams epParams;
    epParams.deviceId = cfg.deviceId;
    EndpointExtractCuda endpointOp(epParams);
    spdlog::info("4-9 EndpointExtractCuda constructed");

    // ----- 4-10 VirtualCameraPose -----
    VirtualCameraPoseParams vcpParams;
    vcpParams.deviceId = cfg.deviceId;
    // 每 (pose,line) 线段只有 2 个端点（两点定线），默认 minPointsPerLine=3 会全拒
    vcpParams.minPointsPerLine = 2;
    VirtualCameraPoseCuda vcpOp(vcpParams);
    spdlog::info("4-10 VirtualCameraPoseCuda constructed");

    // ----- 4-11 ProjectorJointCalib（新算法，取代 PoseOptimize）-----
    // 投影仪光心 t + CMOS 发射曲线联合标定；K/R 不由此算子优化，
    // 由 4-10 VirtualCameraPose 提供，喂给下游 5-3/4-13。
    ProjectorJointCalibParams pjcParams;
    ProjectorJointCalib projectorOp(pjcParams);
    spdlog::info("4-11 ProjectorJointCalib constructed (replaces PoseOptimize)");

    // stereoK / stereoR 用 StereoCalibration helper (Step 0 决定 2):
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
    //    6.2-c: 跑到 reconstruct 并累积 d_points3d 到 host vector
    //    6.2-d/e: 循环结束后用累积结果跑 4-9~4-13
    //    累积策略 (Step 0 决定 1): host 端 vector, 循环末尾统一 upload
    // ------------------------------------------------------------------
    std::vector<cv::Vec3f> host_points3d;
    std::vector<int>       host_line_ids;
    // 4-9 逐 pose 执行后的端点累积（线号已按 pose 偏移），喂 4-10
    std::vector<cv::Vec3f> hostEndpoints;
    std::vector<int>       hostEndpointIds;
    int totalEndpoints = 0, totalEpLines = 0;
    // per-pose 累积：ProjectorJointCalib 需按姿态分组（每姿态一块平板）
    std::vector<std::vector<cv::Vec3f>> posePoints(input->poseFrames.size());
    std::vector<std::vector<int>>       poseLineIds(input->poseFrames.size());

    size_t framesOk = 0;
    size_t framesSkip = 0;

    for (size_t pi = 0; pi < input->poseFrames.size(); ++pi) {
        const auto& tubes = input->poseFrames[pi];
        for (size_t ti = 0; ti < tubes.size(); ++ti) {
          try {
            const auto& f = tubes[ti];

            // ----- 4-1 mask_extract (L + R) -----
            // Execute(const cv::Mat& gray, Stream&) → MaskExtractResult{d_grayImage, d_cleanedMask, ...}
            auto maskResL = maskL.Execute(f.leftGray, stream);
            auto maskResR = maskR.Execute(f.rightGray, stream);
            if (!maskResL.success || !maskResR.success) {
                cudaError_t sticky = cudaGetLastError();   // [dbg] 诊断残留 CUDA 错误
                spdlog::warn("pose {} tube {}: 4-1 mask failed (L={}, R={}) msg='{}' sticky_cuda={}",
                             pi, ti, maskResL.success, maskResR.success,
                             maskResL.success ? maskResR.message : maskResL.message,
                             cudaGetErrorString(sticky));
                ++framesSkip;
                continue;
            }
            if (!maskResL.d_cleanedMask || !maskResR.d_cleanedMask
                || maskResL.d_cleanedMask->empty() || maskResR.d_cleanedMask->empty()) {
                spdlog::warn("pose {} tube {}: 4-1 mask empty, skip", pi, ti);
                ++framesSkip;
                continue;
            }

            // [debug] 逐 pose 导出 4-1 清洗后的二值掩膜（L/R 各一张，0/255）
            {
                std::error_code ec;
                std::filesystem::path dbgDir =
                    std::filesystem::path(outPath).parent_path() / "debug_masks";
                std::filesystem::create_directories(dbgDir, ec);
                auto saveMask = [&](const cv::cuda::GpuMat& dm, const char* side) {
                    cv::Mat hm;
                    dm.download(hm, stream);
                    cudaStreamSynchronize(cv::cuda::StreamAccessor::getStream(stream));
                    if (hm.empty()) return;
                    cv::Mat vis;
                    hm.convertTo(vis, CV_8UC1, 255.0);   // 0/1 → 0/255
                    char name[64];
                    std::snprintf(name, sizeof(name), "pose_%02llu_t%llu_%s_mask.png",
                                  (unsigned long long)pi, (unsigned long long)ti, side);
                    cv::imwrite((dbgDir / name).string(), vis);
                };
                saveMask(*maskResL.d_cleanedMask, "L");
                saveMask(*maskResR.d_cleanedMask, "R");
            }

            // ----- 4-2 region_analyze (L + R) -----
            // Execute(const shared_ptr<GpuMat>& d_mask, Stream&) → RegionAnalysisResult{d_labeledMask CV_32SC1, components}
            auto cclResL = cclL.Execute(maskResL.d_cleanedMask, stream);
            // if (pi == 0) { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) spdlog::error("[probe] after 4-2 L: {}", cudaGetErrorString(e_)); }
            auto cclResR = cclR.Execute(maskResR.d_cleanedMask, stream);
            // if (pi == 0) { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) spdlog::error("[probe] after 4-2 R: {}", cudaGetErrorString(e_)); }
            if (!cclResL.success || !cclResR.success) {
                spdlog::warn("pose {} tube {}: 4-2 ccl failed (L={}, R={}), skip",
                             pi, ti, cclResL.success, cclResR.success);
                ++framesSkip;
                continue;
            }

            // [debug] 逐 pose 导出 4-2 连通域标记图（每连通域一种伪彩色）
            if (cclResL.d_labeledMask && cclResR.d_labeledMask) {
                std::error_code ec;
                std::filesystem::path dbgDir =
                    std::filesystem::path(outPath).parent_path() / "debug_ccl";
                std::filesystem::create_directories(dbgDir, ec);
                auto saveCcl = [&](const cv::cuda::GpuMat& dm, const char* side) {
                    cv::Mat hm;
                    dm.download(hm, stream);
                    cudaStreamSynchronize(cv::cuda::StreamAccessor::getStream(stream));
                    if (hm.empty()) return;
                    cv::Mat labels;
                    hm.reshape(1, hm.rows).convertTo(labels, CV_32SC1);
                    double dmax;
                    cv::minMaxIdx(labels, nullptr, &dmax);
                    const int nlab = (int)dmax + 1;
                    // 8 位伪彩色调色板（label 0=背景黑，其余循环取色）
                    cv::Mat vis(hm.rows, hm.cols, CV_8UC3, cv::Scalar(0, 0, 0));
                    for (int lb = 1; lb < nlab; ++lb) {
                        cv::Scalar color((lb * 61) & 255, (lb * 127 + 40) & 255,
                                         (lb * 251 + 80) & 255);
                        cv::Mat bin = (labels == lb);
                        vis.setTo(color, bin);
                    }
                    char name[64];
                    std::snprintf(name, sizeof(name), "pose_%02llu_t%llu_%s_ccl.png",
                                  (unsigned long long)pi, (unsigned long long)ti, side);
                    cv::imwrite((dbgDir / name).string(), vis);
                    spdlog::info("[debug] pose {} tube {} {}: {} components",
                                 pi, ti, side, nlab - 1);
                };
                saveCcl(*cclResL.d_labeledMask, "L");
                saveCcl(*cclResR.d_labeledMask, "R");
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
            // if (pi == 0) { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) spdlog::error("[probe] after 4-3 L: {}", cudaGetErrorString(e_)); }
            auto labelResR = labelR.Execute(*cclResR.d_labeledMask, stream);
            // if (pi == 0) { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) spdlog::error("[probe] after 4-3 R: {}", cudaGetErrorString(e_)); }
            if (!labelResL.success || !labelResR.success) {
                spdlog::warn("pose {} tube {}: 4-3 label failed (L={}, R={}), skip",
                             pi, ti, labelResL.success, labelResR.success);
                ++framesSkip;
                continue;
            }

            // [debug] 逐 pose 导出 4-3 重编号后的激光线图
            // （伪彩色区分线号 + 顶端数字标注，与 4-2 的 CCL 原始编号对照）
            if (labelResL.d_labeledMask && labelResR.d_labeledMask) {
                std::error_code ec;
                std::filesystem::path dbgDir =
                    std::filesystem::path(outPath).parent_path() / "debug_labels";
                std::filesystem::create_directories(dbgDir, ec);
                auto saveLabels = [&](const cv::cuda::GpuMat& dm,
                                      const cv::cuda::GpuMat& dgray,
                                      const char* side) {
                    cv::Mat hm, hgray;
                    dm.download(hm, stream);
                    dgray.download(hgray, stream);
                    cudaStreamSynchronize(cv::cuda::StreamAccessor::getStream(stream));
                    if (hm.empty()) return;
                    cv::Mat labels;
                    hm.reshape(1, hm.rows).convertTo(labels, CV_32SC1);
                    double dmax;
                    cv::minMaxIdx(labels, nullptr, &dmax);
                    const int nlab = (int)dmax + 1;

                    // 灰度底图提亮 + 每条线伪彩色 + 线号标注
                    cv::Mat vis;
                    cv::cvtColor(hgray, vis, cv::COLOR_GRAY2BGR);
                    double mn, mx;
                    cv::minMaxIdx(hgray, &mn, &mx);
                    vis.convertTo(vis, -1, 255.0 / (mx - mn + 1), -mn * 255.0 / (mx - mn + 1));
                    int maxId = 0;
                    for (int lb = 1; lb < nlab; ++lb) {
                        cv::Mat bin = (labels == lb);
                        if (cv::countNonZero(bin) == 0) continue;
                        cv::Scalar color((lb * 61) & 255, (lb * 127 + 40) & 255,
                                         (lb * 251 + 80) & 255);
                        vis.setTo(color, bin);
                        // 标注位置：该线最上像素处
                        cv::Mat points;
                        cv::findNonZero(bin, points);
                        int topY = points.rows, topX = 0;
                        for (int k = 0; k < points.rows; ++k) {
                            if (points.at<cv::Point>(k).y < topY) {
                                topY = points.at<cv::Point>(k).y;
                                topX = points.at<cv::Point>(k).x;
                            }
                        }
                        char idTxt[8];
                        std::snprintf(idTxt, sizeof(idTxt), "%d", lb);
                        cv::putText(vis, idTxt, cv::Point(topX, std::max(18, topY - 4)),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                        cv::putText(vis, idTxt, cv::Point(topX - 1, std::max(19, topY - 5)),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
                        maxId = std::max(maxId, lb);
                    }
                    char name[64];
                    std::snprintf(name, sizeof(name), "pose_%02llu_t%llu_%s_label.png",
                                  (unsigned long long)pi, (unsigned long long)ti, side);
                    cv::imwrite((dbgDir / name).string(), vis);
                    spdlog::info("[debug] pose {} tube {} {}: max line id = {}", pi, ti, side, maxId);
                };
                saveLabels(*labelResL.d_labeledMask, *maskResL.d_grayImage, "L");
                saveLabels(*labelResR.d_labeledMask, *maskResR.d_grayImage, "R");
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
            // if (pi == 0) { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) spdlog::error("[probe] after 4-4 R: {}", cudaGetErrorString(e_)); }
            if (!stegerResL.success || !stegerResR.success) {
                spdlog::warn("pose {} tube {}: 4-4 steger failed (L={}, R={}), skip",
                             pi, ti, stegerResL.success, stegerResR.success);
                ++framesSkip;
                continue;
            }

            // [debug] 逐 pose 导出 4-4 steger 亚像素中心点图
            // （灰度底 + 按线号伪彩画亚像素点；点跨行画 1px 方块，放大可见连续折线）
            {
                std::error_code ec;
                std::filesystem::path dbgDir =
                    std::filesystem::path(outPath).parent_path() / "debug_steger";
                std::filesystem::create_directories(dbgDir, ec);
                auto saveSteger = [&](const cv::cuda::GpuMat& dpts,
                                      const cv::cuda::GpuMat& dids,
                                      const cv::cuda::GpuMat& dgray,
                                      const char* side) {
                    cv::Mat pts, ids, hgray;
                    dpts.download(pts, stream);
                    dids.download(ids, stream);
                    dgray.download(hgray, stream);
                    cudaStreamSynchronize(cv::cuda::StreamAccessor::getStream(stream));
                    if (pts.empty() || hgray.empty()) return;
                    // 灰度底图压暗（÷3）：彩点夹在亮线里不可见，压暗后对比立现
                    cv::Mat vis;
                    cv::cvtColor(hgray, vis, cv::COLOR_GRAY2BGR);
                    vis.convertTo(vis, -1, 1.0 / 3.0);
                    const cv::Vec2f* p = pts.ptr<cv::Vec2f>();
                    const int* lid = ids.ptr<int>();
                    const int n = (int)pts.total();
                    // steger 逐像素响应：每行多个候选都落在中心附近（线宽 4~6px），
                    // 全画会覆盖整条线带。按 (line, row) 聚合取 x 均值，
                    // 还原单像素中心轨迹（下游 4-6 极线重采样同样按行归一）。
                    std::unordered_map<long long, std::pair<double, int>> rowAcc;
                    for (int k = 0; k < n; ++k) {
                        int ry = (int)std::lround(p[k][1]);
                        long long key = (static_cast<long long>(lid[k]) << 32)
                                      | (unsigned int)ry;
                        auto& acc = rowAcc[key];
                        acc.first += p[k][0];
                        acc.second += 1;
                    }
                    int drawn = 0;
                    for (const auto& [key, acc] : rowAcc) {
                        int lb = (int)(key >> 32);
                        int ry = (int)(key & 0xFFFFFFFF);
                        double fx = acc.first / acc.second;
                        cv::Scalar color((lb * 61) & 255, (lb * 127 + 40) & 255,
                                         (lb * 251 + 80) & 255);
                        cv::Point c((int)std::lround(fx), ry);
                        if (c.x < 0 || c.x >= vis.cols) continue;
                        cv::rectangle(vis, c, c, color);   // 1px 亚像素中心点
                        if (ry % 25 == 0 && c.x >= 2 && c.x + 2 < vis.cols) {
                            cv::line(vis, cv::Point(c.x - 2, ry), cv::Point(c.x + 2, ry), color);
                            cv::line(vis, cv::Point(c.x, ry - 2), cv::Point(c.x, ry + 2), color);
                        }
                        ++drawn;
                    }
                    char name[64];
                    std::snprintf(name, sizeof(name), "pose_%02llu_t%llu_%s_steger.png",
                                  (unsigned long long)pi, (unsigned long long)ti, side);
                    cv::imwrite((dbgDir / name).string(), vis);
                    spdlog::info("[debug] pose {} tube {} {}: {} subpixel points",
                                 pi, ti, side, n);
                };
                if (stegerResL.d_centerPoints && stegerResL.d_line_ids)
                    saveSteger(*stegerResL.d_centerPoints, *stegerResL.d_line_ids,
                               *maskResL.d_grayImage, "L");
                if (stegerResR.d_centerPoints && stegerResR.d_line_ids)
                    saveSteger(*stegerResR.d_centerPoints, *stegerResR.d_line_ids,
                               *maskResR.d_grayImage, "R");
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
            // if (pi == 0) { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) spdlog::error("[probe] after 4-5 R: {{}}", cudaGetErrorString(e_)); }

            // [verify] OpenCV cv::undistortPoints 参照：逐点比对 kernel 输出
            static int n0dump_ = 0;
            {
                auto verify = [&](const cv::cuda::GpuMat& dsrc,
                                  const cv::cuda::GpuMat& dkernelOut,
                                  const cv::Mat& K, const cv::Mat& D,
                                  const cv::Mat& Rm, const cv::Mat& Pk,
                                  const char* side) {
                    if (!undistResL.success) return;
                    cv::Mat src, kern;
                    dsrc.download(src, stream);
                    dkernelOut.download(kern, stream);
                    cudaStreamSynchronize(cv::cuda::StreamAccessor::getStream(stream));
                    if (src.empty() || kern.empty()) return;
                    cv::Mat ref;
                    cv::undistortPoints(src, ref, K, D, Rm, Pk);   // OpenCV 全参数形态
                    if (pi == 0 && n0dump_ < 3) {
                        const cv::Vec2f* ps = src.ptr<cv::Vec2f>();
                        const cv::Vec2f* pr2 = ref.ptr<cv::Vec2f>();
                        const cv::Vec2f* pk2 = kern.ptr<cv::Vec2f>();
                        spdlog::info("[verify 4-5 dump] in=({:.3f},{:.3f}) ref=({:.3f},{:.3f}) kern=({:.3f},{:.3f})",
                                     ps[0][0], ps[0][1], pr2[0][0], pr2[0][1], pk2[0][0], pk2[0][1]);
                        ++n0dump_;
                    }
                    std::vector<double> dd;
                    dd.reserve(ref.total());
                    int n = (int)ref.total();
                    const cv::Vec2f* pr = ref.ptr<cv::Vec2f>();
                    const cv::Vec2f* pk = kern.ptr<cv::Vec2f>();
                    int bad = 0;
                    for (int k = 0; k < n; ++k) {
                        double d = std::hypot((double)pr[k][0] - pk[k][0],
                                              (double)pr[k][1] - pk[k][1]);
                        dd.push_back(d);
                        if (d > 0.01) ++bad;
                    }
                    std::sort(dd.begin(), dd.end());
                    auto q = [&](double p) { return dd.empty() ? 0.0 : dd[(size_t)(p * (dd.size() - 1))]; };
                    spdlog::info("[verify 4-5] pose {} {} {}: n={} median={:.4f} p99={:.4f} "
                                 "p99.99={:.4f} max={:.4f}px  (>0.01px: {} 个 = {:.4f}%)",
                                 pi, ti, side, n, q(0.5), q(0.99), q(0.9999), dd.back(),
                                 bad, 100.0 * bad / std::max(n, 1));
                };
                verify(*stegerResL.d_centerPoints, *undistResL.d_rectifiedPoints,
                       h.cameraMatrixL, h.distCoeffsL, h.R1, P1k, "L");
                verify(*stegerResR.d_centerPoints, *undistResR.d_rectifiedPoints,
                       h.cameraMatrixR, h.distCoeffsR, h.R2, P2k, "R");
            }
            if (!undistResL.success || !undistResR.success) {
                spdlog::warn("pose {} tube {}: 4-5 undistort failed (L={}, R={}), skip",
                             pi, ti, undistResL.success, undistResR.success);
                ++framesSkip;
                continue;
            }

            // [debug] 逐 pose 导出 4-5 去畸变+矫正后的点（画在立体矫正图上）
            if (undistResL.d_rectifiedPoints && undistResR.d_rectifiedPoints) {
                std::error_code ec;
                auto dbgDir = std::filesystem::path(outPath).parent_path() / "debug_undist";
                std::filesystem::create_directories(dbgDir, ec);
                // 立体矫正底图（handoff K/D/R1|2/P1|2；map 与 P 平移列无关）
                cv::Mat mapXL, mapYL, mapXR, mapYR, rectL, rectR;
                cv::initUndistortRectifyMap(h.cameraMatrixL, h.distCoeffsL, h.R1, h.P1,
                                            h.imageSize, CV_32FC1, mapXL, mapYL);
                cv::initUndistortRectifyMap(h.cameraMatrixR, h.distCoeffsR, h.R2, h.P2,
                                            h.imageSize, CV_32FC1, mapXR, mapYR);
                cv::remap(f.leftGray, rectL, mapXL, mapYL, cv::INTER_LINEAR);
                cv::remap(f.rightGray, rectR, mapXR, mapYR, cv::INTER_LINEAR);
                auto savePts = [&](const cv::cuda::GpuMat& dpts, const cv::cuda::GpuMat& dids,
                                   const cv::Mat& rect, const char* side, const char* tag) {
                    cv::Mat pts, ids;
                    dpts.download(pts, stream);
                    dids.download(ids, stream);
                    cudaStreamSynchronize(cv::cuda::StreamAccessor::getStream(stream));
                    if (pts.empty()) return;
                    // 与 debug_steger 同款呈现：底图压暗 + (线号,行) 聚合取 x 均值
                    // + 每 25 行十字刻度（全点直画会覆盖整条线带）
                    cv::Mat vis;
                    cv::cvtColor(rect, vis, cv::COLOR_GRAY2BGR);
                    vis.convertTo(vis, -1, 1.0 / 3.0);
                    const cv::Vec2f* p = pts.ptr<cv::Vec2f>();
                    const int* lid = ids.ptr<int>();
                    std::unordered_map<long long, std::pair<double, int>> acc2;
                    for (size_t k = 0; k < pts.total(); ++k) {
                        int ry = (int)std::lround(p[k][1]);
                        if (ry < 0) continue;
                        long long key = (static_cast<long long>(lid[k]) << 32)
                                      | (unsigned int)ry;
                        auto& a = acc2[key];
                        a.first += p[k][0];
                        a.second += 1;
                    }
                    for (const auto& [key, a] : acc2) {
                        int lb = (int)(key >> 32);
                        int ry = (int)(key & 0xFFFFFFFF);
                        cv::Point q((int)std::lround(a.first / a.second), ry);
                        if (q.x < 0 || q.x >= vis.cols || ry >= vis.rows) continue;
                        cv::Scalar c((lb * 61) & 255, (lb * 127 + 40) & 255,
                                     (lb * 251 + 80) & 255);
                        cv::rectangle(vis, q, q, c);
                        if (ry % 25 == 0 && q.x >= 2 && q.x + 2 < vis.cols) {
                            cv::line(vis, cv::Point(q.x - 2, ry), cv::Point(q.x + 2, ry), c);
                            cv::line(vis, cv::Point(q.x, ry - 2), cv::Point(q.x, ry + 2), c);
                        }
                    }
                    char name[64];
                    std::snprintf(name, sizeof(name), "pose_%02llu_t%llu_%s_%s.png",
                                  (unsigned long long)pi, (unsigned long long)ti, side, tag);
                    cv::imwrite((dbgDir / name).string(), vis);
                };
                if (undistResL.d_line_ids)
                    savePts(*undistResL.d_rectifiedPoints, *undistResL.d_line_ids, rectL, "L", "undist");
                if (undistResR.d_line_ids)
                    savePts(*undistResR.d_rectifiedPoints, *undistResR.d_line_ids, rectR, "R", "undist");
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
            // if (pi == 0) { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) spdlog::error("[probe] after 4-6 R: {{}}", cudaGetErrorString(e_)); }
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
            // if (pi == 0) { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) spdlog::error("[probe] after 4-7: {}", cudaGetErrorString(e_)); }
            if (!matchRes.success) {
                spdlog::warn("pose {} tube {}: 4-7 match failed ({}), skip",
                             pi, ti, matchRes.message);
                ++framesSkip;
                continue;
            }

            // [debug] 逐 pose 导出 4-7 匹配点对（L/R 矫正图同色同线号，视差肉眼可见）
            if (matchRes.d_matched_left && matchRes.d_matched_right && matchRes.d_matched_line_ids) {
                std::error_code ec;
                auto dbgDir = std::filesystem::path(outPath).parent_path() / "debug_match";
                std::filesystem::create_directories(dbgDir, ec);
                cv::Mat mapXL, mapYL, mapXR, mapYR, rectL, rectR;
                cv::initUndistortRectifyMap(h.cameraMatrixL, h.distCoeffsL, h.R1, h.P1,
                                            h.imageSize, CV_32FC1, mapXL, mapYL);
                cv::initUndistortRectifyMap(h.cameraMatrixR, h.distCoeffsR, h.R2, h.P2,
                                            h.imageSize, CV_32FC1, mapXR, mapYR);
                cv::remap(f.leftGray, rectL, mapXL, mapYL, cv::INTER_LINEAR);
                cv::remap(f.rightGray, rectR, mapXR, mapYR, cv::INTER_LINEAR);
                cv::Mat ptsL, ptsR, mids;
                matchRes.d_matched_left->download(ptsL, stream);
                matchRes.d_matched_right->download(ptsR, stream);
                matchRes.d_matched_line_ids->download(mids, stream);
                cudaStreamSynchronize(cv::cuda::StreamAccessor::getStream(stream));
                cv::Mat visL, visR;
                cv::cvtColor(rectL, visL, cv::COLOR_GRAY2BGR);
                cv::cvtColor(rectR, visR, cv::COLOR_GRAY2BGR);
                const cv::Vec2f* pl = ptsL.ptr<cv::Vec2f>();
                const cv::Vec2f* pr = ptsR.ptr<cv::Vec2f>();
                const int* lid = mids.ptr<int>();
                // 逐线号统计匹配数 ＋ 记录每线在 L/R 图的代表点（用于标注编号）
                std::map<int, int> lineCount;                    // lineId → 匹配点数
                std::map<int, cv::Point> lineTopL, lineTopR;     // lineId → 标注位置
                for (size_t k = 0; k < mids.total(); ++k) {
                    cv::Scalar c((lid[k] * 61) & 255, (lid[k] * 127 + 40) & 255,
                                 (lid[k] * 251 + 80) & 255);
                    cv::Point ql((int)std::lround(pl[k][0]), (int)std::lround(pl[k][1]));
                    cv::Point qr((int)std::lround(pr[k][0]), (int)std::lround(pr[k][1]));
                    ++lineCount[lid[k]];
                    if (ql.x >= 0 && ql.x < visL.cols && ql.y >= 0 && ql.y < visL.rows) {
                        cv::rectangle(visL, ql, ql, c);
                        auto it = lineTopL.find(lid[k]);
                        if (it == lineTopL.end() || ql.y < it->second.y)
                            lineTopL[lid[k]] = ql;
                    }
                    if (qr.x >= 0 && qr.x < visR.cols && qr.y >= 0 && qr.y < visR.rows) {
                        cv::rectangle(visR, qr, qr, c);
                        auto it = lineTopR.find(lid[k]);
                        if (it == lineTopR.end() || qr.y < it->second.y)
                            lineTopR[lid[k]] = qr;
                    }
                }
                // L/R 图上标注线号数字（黑白双描边保证可读）
                auto tagLines = [](cv::Mat& vis, const std::map<int, cv::Point>& tops) {
                    for (const auto& [lb, p] : tops) {
                        cv::Point tp(std::max(2, p.x - 8), std::max(18, p.y - 6));
                        char t[8];
                        std::snprintf(t, sizeof(t), "%d", lb);
                        cv::putText(vis, t, tp + cv::Point(-1, -1), cv::FONT_HERSHEY_SIMPLEX,
                                    0.6, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
                        cv::putText(vis, t, tp, cv::FONT_HERSHEY_SIMPLEX,
                                    0.6, cv::Scalar(80, 255, 255), 1, cv::LINE_AA);
                    }
                };
                tagLines(visL, lineTopL);
                tagLines(visR, lineTopR);
                // 日志输出逐线匹配计数（L/R 对应关系一眼可见）
                {
                    std::string tbl = "[debug] pose " + std::to_string(pi) + " tube "
                                    + std::to_string(ti) + " matched lines: ";
                    for (const auto& [lb, cnt] : lineCount)
                        tbl += std::to_string(lb) + ":" + std::to_string(cnt) + " ";
                    spdlog::info("{}  (total {} lines / {} pairs)",
                                 tbl, lineCount.size(), mids.total());
                }
                char nl[64], nr[64];
                std::snprintf(nl, sizeof(nl), "pose_%02llu_t%llu_L_match.png",
                              (unsigned long long)pi, (unsigned long long)ti);
                std::snprintf(nr, sizeof(nr), "pose_%02llu_t%llu_R_match.png",
                              (unsigned long long)pi, (unsigned long long)ti);
                cv::imwrite((dbgDir / nl).string(), visL);
                cv::imwrite((dbgDir / nr).string(), visR);
                spdlog::info("[debug] pose {} tube {}: {} matched pairs -> debug_match",
                             pi, ti, mids.total());

                // [debug] 导出匹配对视差 CSV（4-7 之后、4-8 之前）
                // 列: pose,side 即 L 坐标系, index, xL, yL, xR, yR, disparity, lineId
                {
                    std::error_code ec;
                    auto dbgDir2 = std::filesystem::path(outPath).parent_path() / "debug_match";
                    std::filesystem::create_directories(dbgDir2, ec);
                    static std::ofstream dcsv(dbgDir2 / "disparity_all.csv", std::ios::trunc);
                    if (pi == 0 && ti == 0)
                        dcsv << "pose,xL,yL,xR,yR,disparity,lineId\n";
                    for (size_t k = 0; k < mids.total(); ++k) {
                        dcsv << std::fixed << std::setprecision(3)
                             << pi << ','
                             << pl[k][0] << ',' << pl[k][1] << ','
                             << pr[k][0] << ',' << pr[k][1] << ','
                             << (pl[k][0] - pr[k][0]) << ','
                             << lid[k] << '\n';
                    }
                }

                // 极线匹配总览图：L/R 上下拼接，匹配对同色连线＋y 差标注
                // （矫正正确时连线水平；y 差≠0 的连线直接暴露极线误差）
                {
                    const int H = visL.rows, W = visL.cols;
                    cv::Mat canvas(2 * H + 8, W, CV_8UC3, cv::Scalar(30, 30, 30));
                    visL.copyTo(canvas(cv::Rect(0, 0, W, H)));
                    visR.copyTo(canvas(cv::Rect(0, H + 8, W, H)));
                    // 抽稀连线（全画会糊）：每隔 stride 取一对，共 ~800 对
                    const size_t n = mids.total();
                    const size_t stride = std::max<size_t>(1, n / 800);
                    double dySum = 0; size_t dyN = 0;
                    for (size_t k = 0; k < n; k += stride) {
                        cv::Scalar c((lid[k] * 61) & 255, (lid[k] * 127 + 40) & 255,
                                     (lid[k] * 251 + 80) & 255);
                        cv::Point ql((int)std::lround(pl[k][0]), (int)std::lround(pl[k][1]));
                        cv::Point qr((int)std::lround(pr[k][0]), (int)std::lround(pr[k][1]));
                        if (qr.y >= 0 && qr.y < H) {
                            double dy = double(ql.y) - qr.y;
                            dySum += dy; ++dyN;
                            cv::line(canvas, ql, qr + cv::Point(0, H + 8), c, 1, cv::LINE_AA);
                        }
                    }
                    char txt[96];
                    std::snprintf(txt, sizeof(txt), "pairs=%zu  mean|dy|=%.2f px (epipolar residual)",
                                  n, dyN ? dySum / dyN : 0.0);
                    cv::putText(canvas, txt, cv::Point(20, H + 8 - 6),
                                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                                cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
                    char nm[64];
                    std::snprintf(nm, sizeof(nm), "pose_%02llu_t%llu_match_pairs.png",
                                  (unsigned long long)pi, (unsigned long long)ti);
                    cv::imwrite((dbgDir / nm).string(), canvas);
                }
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
            // if (pi == 0) { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) spdlog::error("[probe] after 4-8: {}", cudaGetErrorString(e_)); }
            if (!reconRes.success) {
                spdlog::warn("pose {} tube {}: 4-8 reconstruct failed ({}), skip",
                             pi, ti, reconRes.message);
                ++framesSkip;
                continue;
            }

            // ----- host 累积 (决定 1) -----
            if (reconRes.d_points3d && reconRes.d_valid_line_ids
                && !reconRes.d_points3d->empty()
                && !reconRes.d_valid_line_ids->empty()) {
                cv::Mat h_pts, h_ids;
                reconRes.d_points3d->download(h_pts);
                reconRes.d_valid_line_ids->download(h_ids);
                h_pts = h_pts.reshape(3, 1);   // 强制 1×N CV_32FC3
                h_ids = h_ids.reshape(1, 1);   // 强制 1×N CV_32SC1
                {   // 线号净化：LaserLabelerCUDA 的物理线号必在 [0, maxLabels=256)。
                    // 实测序列运行中 match/reconstruct 的 fid 通道偶发读到
                    // 未初始化显存（float 位型，见 开发记录 2026-08-23）——
                    // 越界 id 一律丢弃，对应 3D 点不同步累积。
                    const int kMaxPhysLineId = 255;
                    int dropped = 0;
                    size_t w = 0;
                    for (size_t r = 0; r < h_ids.total(); ++r) {
                        int id = h_ids.ptr<int>()[r];
                        if (id < 0 || id > kMaxPhysLineId) { ++dropped; continue; }
                        if (w != r) {
                            h_ids.ptr<int>()[w] = id;
                            h_pts.ptr<cv::Vec3f>()[w] = h_pts.ptr<cv::Vec3f>()[r];
                        }
                        ++w;
                    }
                    if (dropped > 0) {
                        spdlog::warn("pose {} tube {}: dropped {} pts with corrupt line ids "
                                     "(kept {}/{})", pi, ti, dropped, w, h_ids.total());
                        h_ids = h_ids.colRange(0, (int)w);
                        h_pts = h_pts.colRange(0, (int)w);
                    }
                }
                host_points3d.insert(host_points3d.end(),
                                     h_pts.begin<cv::Vec3f>(),
                                     h_pts.end<cv::Vec3f>());
                host_line_ids.insert(host_line_ids.end(),
                                     h_ids.begin<int>(),
                                     h_ids.end<int>());

                // [debug] 逐 pose 导出 4-8 重建 3D 点（ASC，X Y Z lineId）
                {
                    std::error_code ec;
                    auto dbgDir = std::filesystem::path(outPath).parent_path() / "debug_recon";
                    std::filesystem::create_directories(dbgDir, ec);
                    char name[64];
                    std::snprintf(name, sizeof(name), "pose_%02llu_t%llu_points3d.asc",
                                  (unsigned long long)pi, (unsigned long long)ti);
                    std::ofstream asc(dbgDir / name);
                    if (asc.is_open()) {
                        asc << "X Y Z lineId\n";
                        for (int k = 0; k < (int)h_pts.total(); ++k) {
                            const cv::Vec3f& p = h_pts.ptr<cv::Vec3f>()[k];
                            asc << std::fixed << std::setprecision(4)
                                << p[0] << ' ' << p[1] << ' ' << p[2] << ' '
                                << h_ids.ptr<int>()[k] << '\n';
                        }
                    }
                }
                // per-pose 累积（按当前 pi 分组，供 ProjectorJointCalib）
                posePoints[pi].insert(posePoints[pi].end(),
                                      h_pts.begin<cv::Vec3f>(),
                                      h_pts.end<cv::Vec3f>());
                poseLineIds[pi].insert(poseLineIds[pi].end(),
                                       h_ids.begin<int>(),
                                       h_ids.end<int>());

                // 导出首 pose 重建点云（ASC：X Y Z 每行一点，CloudCompare 可直接打开）
                if (pi == 0 && h_pts.total() > 0) {
                    std::error_code ec;
                    std::filesystem::path ascPath = std::filesystem::path(outPath);
                    ascPath.replace_filename(
                        ascPath.stem().string() + "_pose0_points3d.asc");
                    std::ofstream asc(ascPath);
                    if (asc.is_open()) {
                        asc << "X Y Z\n";
                        for (int k = 0; k < (int)h_pts.total(); ++k) {
                            const cv::Vec3f& p = h_pts.ptr<cv::Vec3f>()[k];
                            asc << std::fixed << std::setprecision(4)
                                << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
                        }
                        spdlog::info("pose 0 point cloud -> {} ({} pts)",
                                     ascPath.string(), h_pts.total());
                    } else {
                        spdlog::warn("cannot write pose0 ASC: {}", ascPath.string());
                    }
                }

                // ----- 4-9 endpoint_extract（逐 pose 执行）-----
                // 同一物理线号在不同 pose 是不同 3D 线段（板位姿不同）；
                // 扁平累积后统一提端点会把多段散点并成一条"线"，RANSAC 拟合必败
                //（实测 Insufficient valid lines: 0）。逐 pose 提端点，
                // 线号按 pose 偏移防跨 pose 碰撞，4-10 按 (pose,line) 拟合。
                cv::cuda::GpuMat d_pts3d, d_lids;
                cv::Mat m3d(1, (int)h_pts.total(), CV_32FC3, h_pts.ptr<cv::Vec3f>());
                cv::Mat mid(1, (int)h_ids.total(), CV_32SC1, h_ids.ptr<int>());
                d_pts3d.upload(m3d);
                d_lids.upload(mid);
                auto epRes = endpointOp.Execute(d_pts3d, d_lids, stream);
                if (!epRes.success) {
                    spdlog::warn("pose {} tube {}: 4-9 endpoint_extract failed ({}), "
                                 "该 pose 不参与虚拟光心求解",
                                 pi, ti, epRes.message);
                } else if (epRes.d_endpoints && epRes.d_endpoint_ids) {
                    cv::Mat he, hid;
                    epRes.d_endpoints->download(he);
                    epRes.d_endpoint_ids->download(hid);
                    if (!he.empty() && !hid.empty()) {
                        // [debug] 逐 pose 导出 4-9 端点（ASC，X Y Z lineId）
                        {
                            std::error_code ec;
                            auto dbgDir = std::filesystem::path(outPath).parent_path() / "debug_ep";
                            std::filesystem::create_directories(dbgDir, ec);
                            char name[64];
                            std::snprintf(name, sizeof(name), "pose_%02llu_t%llu_endpoints.asc",
                                          (unsigned long long)pi, (unsigned long long)ti);
                            std::ofstream asc(dbgDir / name);
                            if (asc.is_open()) {
                                asc << "X Y Z lineId\n";
                                for (int k = 0; k < (int)he.total(); ++k) {
                                    const cv::Vec3f& p = he.ptr<cv::Vec3f>()[k];
                                    asc << std::fixed << std::setprecision(4)
                                        << p[0] << ' ' << p[1] << ' ' << p[2] << ' '
                                        << hid.ptr<int>()[k] << '\n';
                                }
                                spdlog::info("[debug] pose {} tube {}: {} endpoints -> debug_ep",
                                             pi, ti, he.total());
                            }
                        }
                        const int idOffset = static_cast<int>(pi) * 256;  // maxLabels=256
                        hostEndpoints.insert(hostEndpoints.end(),
                                             he.begin<cv::Vec3f>(),
                                             he.end<cv::Vec3f>());
                        const int* p = hid.ptr<int>();
                        for (size_t k = 0; k < hid.total(); ++k)
                            hostEndpointIds.push_back(p[k] + idOffset);
                        totalEndpoints += (int)he.total();
                        totalEpLines += epRes.numLines;
                    }
                }
            }

            ++framesOk;
            spdlog::info("pose {} tube {}: OK (matched={}, reconstructed={}, total_accum={})",
                         pi, ti, matchRes.matchCount, reconRes.validCount,
                         host_points3d.size());
          } catch (const std::exception& e) {
              // 单 pose 异常不再带崩全流程（GUI 内曾因栈展开中二次抛出 terminate）
              spdlog::error("pose {} tube {}: EXCEPTION {} — skip this pose", pi, ti, e.what());
              ++framesSkip;
          } catch (...) {
              spdlog::error("pose {} tube {}: UNKNOWN EXCEPTION — skip this pose", pi, ti);
              ++framesSkip;
          }
        }
    }

    // ------------------------------------------------------------------
    // 3b. 循环结束: 汇报累积（3D 点供 JSON 诊断/4-11；端点喂 4-10）
    // ------------------------------------------------------------------
    spdlog::info("loop done: {} ok, {} skipped | accumulated {} 3D pts, "
                 "{} endpoints / {} lines (per-pose 4-9)",
                 framesOk, framesSkip, host_points3d.size(),
                 totalEndpoints, totalEpLines);

    // ------------------------------------------------------------------
    // 3c. 4-10 / 4-11 一次性执行（4-9 已在循环内逐 pose 完成）
    //     暂存 finalVirtualK/R/T 供 6.2-e 的 5-3 / 4-13 使用
    // ------------------------------------------------------------------
    cv::Matx33d finalVirtualK = cv::Matx33d::eye();   // 来自 4-10（新算法不优化 K）
    cv::Matx33d finalVirtualR = cv::Matx33d::eye();   // 来自 4-10（新算法不优化 R）
    cv::Vec3d   finalVirtualT(0, 0, 0);               // 来自 ProjectorJointCalib
    calib::ProjectorJointCalibResult projectorRes;    // 4-11 新算法结果
    bool haveVirtualPose = false;

    if (hostEndpoints.empty()) {
        spdlog::error("no endpoints accumulated; skip 4-10~4-13");
    } else {
        // 上传逐 pose 累积的端点（线号已按 pose 偏移）
        cv::cuda::GpuMat d_allEndpoints, d_allEndpointIds;
        cv::Mat me(1, (int)hostEndpoints.size(), CV_32FC3, hostEndpoints.data());
        cv::Mat mei(1, (int)hostEndpointIds.size(), CV_32SC1, hostEndpointIds.data());
        d_allEndpoints.upload(me);
        d_allEndpointIds.upload(mei);
        {
            // ----- 4-10 virtual_camera_pose -----
            // Execute(d_endpoints, d_endpoint_ids, Matx33d& stereoK, Matx33d& stereoR, stream)
            // 注意: d_endpoints 是 2N（每线两端点），配对的是 d_endpoint_ids（2N），
            // 不是 d_line_ids（N）——4-10 要求两者元素数一致。
            auto vcpRes = vcpOp.Execute(d_allEndpoints, d_allEndpointIds,
                                        stereoK, stereoR, stream);
            if (!vcpRes.success) {
                spdlog::error("4-10 virtual_camera_pose failed: {}", vcpRes.message);
            } else {
                    spdlog::info("4-10 OK: virtualT=({:.3f},{:.3f},{:.3f}), "
                                 "{} lines, fit_err={:.4f}",
                                 vcpRes.virtualT[0], vcpRes.virtualT[1], vcpRes.virtualT[2],
                                 vcpRes.numLines, vcpRes.avgLineFittingError);

                    // ----- 4-11 ProjectorJointCalib（新算法，取代 PoseOptimize）-----
                    // 输入: per-pose 点云 + f/主点(从 virtualK) + initialT(从 virtualT)
                    // 输出: projectorT(投影仪光心) + emissionCurve(CMOS 发射曲线)
                    // K/R 不由此算子优化 → finalVirtualK/R 沿用 4-10 结果
                    calib::ProjectorJointCalibInput pjcInput;
                    for (size_t ppi = 0; ppi < posePoints.size(); ++ppi) {
                        if (posePoints[ppi].size() >= 10) {
                            calib::PosePointSet pset;
                            pset.points3d = posePoints[ppi];
                            pset.lineIds  = poseLineIds[ppi];
                            pjcInput.poses.push_back(std::move(pset));
                        }
                    }
                    pjcInput.f              = vcpRes.virtualK(0, 0);
                    pjcInput.principalPoint = cv::Point2d(vcpRes.virtualK(0, 2),
                                                          vcpRes.virtualK(1, 2));
                    pjcInput.initialT       = vcpRes.virtualT;
                    projectorRes = projectorOp.Execute(pjcInput);
                    if (!projectorRes.success) {
                        spdlog::error("4-11 ProjectorJointCalib failed: {}",
                                      projectorRes.message);
                    } else {
                        spdlog::info("4-11 OK: projectorT=({:.3f},{:.3f},{:.3f}) "
                                     "sampsonRms={:.4f}(init={:.4f}) cond={:.3e} "
                                     "poses={}/{}pts flag={}",
                                     projectorRes.projectorT[0],
                                     projectorRes.projectorT[1],
                                     projectorRes.projectorT[2],
                                     projectorRes.finalSampsonRms,
                                     projectorRes.initialSampsonRms,
                                     projectorRes.jacobianConditionNumber,
                                     projectorRes.poseCount,
                                     projectorRes.totalPointCount,
                                     static_cast<int>(projectorRes.qualityFlag));
                        finalVirtualK = vcpRes.virtualK;         // K 沿用 4-10
                        finalVirtualR = vcpRes.virtualR;         // R 沿用 4-10
                        finalVirtualT = projectorRes.projectorT; // T 用新算法
                        haveVirtualPose = true;
                    }
                }
            }
        }

    // ------------------------------------------------------------------
    // 3d. 5-3 + 4-13 (4-11 成功后执行; 都依赖 finalVirtualK/R/T)
    // ------------------------------------------------------------------
    LaserExtrinsicCompensateCPUResult laserExtrinTable;
    PlaneMapTempTableResult           planeTable;
    bool haveLaserExtrin = false;
    bool havePlaneTable  = false;

    if (!haveVirtualPose) {
        spdlog::warn("no virtual pose from 4-11; skip 5-3 and 4-13");
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

        // ----- 4-13 plane_map_temp_table -----
        // 决定 5: 内部已含 4-12 + virtual_pixel_gen; Execute() 无参, 参数全在构造期填
        PlaneMapTempTableParams pmtt;
        pmtt.cameraMatrixL = h.cameraMatrixL; pmtt.distCoeffsL = h.distCoeffsL;
        pmtt.cameraMatrixR = h.cameraMatrixR; pmtt.distCoeffsR = h.distCoeffsR;
        pmtt.imageSize = h.imageSize;
        pmtt.R = h.R; pmtt.T = h.T;
        pmtt.virtualK = finalVirtualK;
        pmtt.virtualR = finalVirtualR;
        pmtt.virtualT = finalVirtualT;

        // lineIds 来源优先级 (review C1):
        //   1. cfg.lineIds (config.json 显式)
        //   2. 从 4-11 finalLineCurves 反推 (运行期实际出现的线号)
        //   3. 都没有 → 跳过 4-13 (避免 PlaneMapTempTable 构造抛异常崩溃)
        std::vector<int> effectiveLineIds = cfg.lineIds;
        if (effectiveLineIds.empty()) {
            // 新算法不输出 lineCurves，从累积的实际线号推断
            std::set<int> seen(host_line_ids.begin(), host_line_ids.end());
            effectiveLineIds.assign(seen.begin(), seen.end());
            if (!effectiveLineIds.empty()) {
                spdlog::info("4-13 lineIds inferred from accumulated line_ids: {} lines",
                             effectiveLineIds.size());
            }
        }
        if (effectiveLineIds.empty()) {
            spdlog::error("4-13 lineIds empty: config 不提供且 4-11 无 lineCurves, "
                          "跳过 plane_map_temp_table");
        } else {
            pmtt.lineIds = effectiveLineIds;
            pmtt.referenceTemp = cfg.referenceTemp;
            pmtt.cte           = cfg.cte;
            pmtt.tempStep      = cfg.tempStep;
            pmtt.tempRangeMin  = cfg.tempRangeMin;
            pmtt.tempRangeMax  = cfg.tempRangeMax;
            pmtt.alpha         = cfg.rectifyAlpha;
            pmtt.flags         = cfg.rectifyFlags;
            pmtt.deviceId      = cfg.deviceId;
            pmtt.gridStep      = cfg.gridStep;
            pmtt.depthMin      = cfg.depthMin;
            pmtt.depthMax      = cfg.depthMax;
            pmtt.depthSamples  = cfg.depthSamples;
            pmtt.epipolarStep  = cfg.epipolarStep;

            // review C1 防尽: 算子构造/执行可能抛 std::invalid_argument 等
            try {
                PlaneMapTempTable pmttOp(pmtt);
                planeTable = pmttOp.Execute();
                if (!planeTable.success) {
                    spdlog::error("4-13 plane_map_temp_table failed: {}",
                                  planeTable.message);
                } else {
                    havePlaneTable = true;
                    spdlog::info("4-13 OK: {} temp entries", planeTable.table.size());
                }
                pmttOp.Destroy();
            } catch (const std::exception& e) {
                spdlog::error("4-13 plane_map_temp_table exception: {}", e.what());
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
    endpointOp.Destroy();
    vcpOp.Destroy();
    // ProjectorJointCalib 为纯 CPU，Destroy() 空实现，无需调用
    lecompOp.Destroy();

    // ------------------------------------------------------------------
    // 5. 写 laser_calib.json（6.2-e 完整版）
    // ------------------------------------------------------------------
    nlohmann::json j;
    j["schema"]  = "factory_calib.laser_calib.v1";
    j["build"]   = "6.2-e";
    j["posesProcessed"]    = input->poseFrames.size();
    j["framesOk"]          = framesOk;
    j["framesSkipped"]     = framesSkip;
    j["accumulatedPoints3D"] = host_points3d.size();
    j["haveVirtualPose"]   = haveVirtualPose;
    j["haveLaserExtrin"]   = haveLaserExtrin;
    j["havePlaneTable"]    = havePlaneTable;

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
        // 新算法输出：投影仪光心 + CMOS 发射曲线
        j["projectorT"] = std::vector<double>{projectorRes.projectorT[0],
                                              projectorRes.projectorT[1],
                                              projectorRes.projectorT[2]};
        j["emissionCurve"] = {
            {"coeffs", std::vector<double>{
                projectorRes.emissionCurve.coeffs[0],
                projectorRes.emissionCurve.coeffs[1],
                projectorRes.emissionCurve.coeffs[2],
                projectorRes.emissionCurve.coeffs[3],
                projectorRes.emissionCurve.coeffs[4],
                projectorRes.emissionCurve.coeffs[5]}},
            {"discriminant", projectorRes.emissionCurve.discriminant},
            {"sampsonRms", projectorRes.emissionCurve.sampsonRms},
            {"pointCount", projectorRes.emissionCurve.pointCount}
        };
        j["projectorQualityFlag"] = static_cast<int>(projectorRes.qualityFlag);
        j["projectorCondNumber"]  = projectorRes.jacobianConditionNumber;
        j["projectorSampsonRms"]  = projectorRes.finalSampsonRms;
    }
    if (haveLaserExtrin) {
        j["laserExtrinsicTempTable"] = laserExtrinTable.toJson();
    }
    if (havePlaneTable) {
        j["planeMapTempTable"] = planeTable.toJson();
    }

    // review I1: 加 status 字段 (ok|partial) 让下游消费方可识别, 不再依赖文件存在与否
    int exitCode = 0;
    std::string exitStatus = "ok";
    if (!haveVirtualPose || !haveLaserExtrin || !havePlaneTable) {
        spdlog::warn("pipeline incomplete: virtualPose={} laserExtrin={} planeTable={}",
                     haveVirtualPose, haveLaserExtrin, havePlaneTable);
        exitCode = 1;
        exitStatus = "partial";
    }
    j["status"] = exitStatus;

    if (!writeLaserJson(outPath, j)) {
        spdlog::error("cannot write output: {}", outPath);
        return 1;
    }
    spdlog::info("laser_calib (6.2-e) -> {} (status={}, exit={})",
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
}  // namespace fc

// =============================================================================
// CLI 入口：薄壳，调用 runLaserCalibRaw（库构建时跳过 main，避免与 GUI/exe 重定义）
// =============================================================================
#ifndef FC2_BUILDING_LIB
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
