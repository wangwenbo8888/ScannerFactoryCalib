/**
 * @file laser_match_scan_v2_cuda.h
 * @brief 激光线匹配扫描CUDA算子 V2（优化版）— 公开头文件
 *
 * 与 v1 (laser_match_scan_cuda.h) 的 API 对齐，性能优化差异：
 *   - 32bit 排序键（row 13bit + x 19bit@1/256px 量化）
 *   - kernelMatch 固定网格 + warp ballot 并行（外层贪心序保持串行，语义等价）
 *   - 全程单次主机同步（末尾 pinned 16B 批量回传）
 *   - 结果缓冲持久化 + ROI 视图返回（有效期至下一次 Execute/Destroy/SetParams）
 *   - status 双缓冲 ping-pong（无 clone）
 *
 * ⚠ 生命周期契约：result 中所有 GpuMat 为内部持久缓冲的视图，
 *    有效期至该实例下一次 Execute() / Destroy() / setParams()。
 * v1 语义保真清单（逐字复刻）：
 *   贪心 xL 升序 + 先占 / ri==hitSm 仅对上次命中去重 / rowKey>=8192 丢弃 /
 *   y<0 钳 row0 / 候选 64 截断 + dir0→dir1 序 + firstLid / effRight=min(count,cap)
 */

#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
#include <stdexcept>
#include "common/calib_types.h"
#include "common/calib_result_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {

struct WarmupConfig;

// ============================================================================
// LaserMatchScanParamsV2
// ============================================================================

struct LaserMatchScanParamsV2 {
    float match_threshold = 1.0f;
    float epipolar_row_step = 0.5f;
    int max_right_per_row = 1024;   // 语义上限保留；V2 kernelMatch smem 容量另见 SMEM_RIGHT_CAP
    float vL_tolerance = 0.01f;
    int deviceId = 0;

    void validate() const {
        if (match_threshold <= 0.0f)
            throw std::invalid_argument("LaserMatchScanParamsV2::match_threshold must be > 0");
        if (epipolar_row_step <= 0.0f)
            throw std::invalid_argument("LaserMatchScanParamsV2::epipolar_row_step must be > 0");
        if (max_right_per_row <= 0)
            throw std::invalid_argument("LaserMatchScanParamsV2::max_right_per_row must be > 0");
        if (vL_tolerance < 0.0f)
            throw std::invalid_argument("LaserMatchScanParamsV2::vL_tolerance must be >= 0");
        if (deviceId < 0)
            throw std::invalid_argument("LaserMatchScanParamsV2::deviceId must be >= 0");
    }

    nlohmann::json toJson() const {
        return {
            {"match_threshold", match_threshold},
            {"epipolar_row_step", epipolar_row_step},
            {"max_right_per_row", max_right_per_row},
            {"vL_tolerance", vL_tolerance},
            {"deviceId", deviceId}
        };
    }

    static LaserMatchScanParamsV2 fromJson(const nlohmann::json& j) {
        LaserMatchScanParamsV2 p;
        if (j.contains("match_threshold"))
            p.match_threshold = j.at("match_threshold").get<float>();
        if (j.contains("epipolar_row_step"))
            p.epipolar_row_step = j.at("epipolar_row_step").get<float>();
        if (j.contains("max_right_per_row"))
            p.max_right_per_row = j.at("max_right_per_row").get<int>();
        if (j.contains("vL_tolerance"))
            p.vL_tolerance = j.at("vL_tolerance").get<float>();
        if (j.contains("deviceId"))
            p.deviceId = j.at("deviceId").get<int>();
        p.validate();
        return p;
    }
};

// ============================================================================
// LaserMatchScanResultV2
// ============================================================================

struct LaserMatchScanResultV2 {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    // ⚠ 内部持久缓冲的 ROI 视图，有效期至下一次 Execute/Destroy/setParams
    std::shared_ptr<cv::cuda::GpuMat> d_matched_left;
    std::shared_ptr<cv::cuda::GpuMat> d_matched_right;
    std::shared_ptr<cv::cuda::GpuMat> d_matched_line_ids;
    int matchedCount = 0;

    // status 为 ping-pong 双缓冲当前帧视图（有效期同上）
    std::shared_ptr<cv::cuda::GpuMat> d_left_status;
    std::shared_ptr<cv::cuda::GpuMat> d_right_status;

    int totalLeftPoints = 0;
    int totalRightPoints = 0;
    int excludedLeftCount = 0;
    int excludedRightCount = 0;

    LaserMatchScanResultV2() = default;
    ~LaserMatchScanResultV2() = default;

    LaserMatchScanResultV2(LaserMatchScanResultV2&&) = default;
    LaserMatchScanResultV2& operator=(LaserMatchScanResultV2&&) = default;

    LaserMatchScanResultV2(const LaserMatchScanResultV2&) = delete;
    LaserMatchScanResultV2& operator=(const LaserMatchScanResultV2&) = delete;
};

// ============================================================================
// LaserMatchScanCudaV2
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 调用方持有
// 说明: 温度补偿映射表与当前温度由调用方经 SetTempTable/SetCurrentTemperature 注入；
//       结果缓冲为容量式持久缓冲，每次 Execute 覆写并返回视图。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================

class SCANNER_API LaserMatchScanCudaV2 {
public:
    static constexpr const char* kLogTag = "07-LaserMatchScanCudaV2";

    explicit LaserMatchScanCudaV2(const LaserMatchScanParamsV2& params = {});
    ~LaserMatchScanCudaV2();

    LaserMatchScanCudaV2(const LaserMatchScanCudaV2&) = delete;
    LaserMatchScanCudaV2& operator=(const LaserMatchScanCudaV2&) = delete;

    bool LoadTempTable(const std::string& jsonPath);

    // 算子规范 §3.6：只读共享资源（温度补偿映射表）经注入传入，算子内部不写入。
    bool SetTempTable(std::shared_ptr<const LaserPlaneMapTempTable> table);

    void SetCurrentTemperature(double temperature);

    LaserMatchScanResultV2 Execute(
        const cv::cuda::GpuMat& d_left_points,
        const cv::cuda::GpuMat& d_left_line_ids,
        const cv::cuda::GpuMat& d_right_points,
        const cv::cuda::GpuMat& d_right_line_ids,
        cv::cuda::Stream& stream);

    LaserMatchScanResultV2 Execute(
        const cv::cuda::GpuMat& d_left_points,
        const cv::cuda::GpuMat& d_left_line_ids,
        const cv::cuda::GpuMat& d_right_points,
        const cv::cuda::GpuMat& d_right_line_ids);

    void Destroy();

    void Warmup(int maxLeftPoints, int maxRightPoints);
    void Warmup(const WarmupConfig& config);
    void SetParams(const LaserMatchScanParamsV2& params);
    const LaserMatchScanParamsV2& GetParams() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getLaserMatchScanCudaV2Info();

} // namespace calib
