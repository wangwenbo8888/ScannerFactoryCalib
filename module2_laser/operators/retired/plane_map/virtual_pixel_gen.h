#pragma once


#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include "common/quality_flag.h"

namespace calib {

struct VirtualPixelGenParams {
    float gridStep = 0.5f;
    int deviceId = 0;

    void validate() const {
        if (deviceId < 0)
            throw std::invalid_argument("VirtualPixelGenParams::deviceId must be >= 0");
        if (gridStep <= 0.0f)
            throw std::invalid_argument("VirtualPixelGenParams::gridStep must be > 0");
    }
};

// ============================================================
// 结果（仅移动）
// ============================================================
struct VirtualPixelGenResult {
    bool success = false;
    QualityFlag qualityFlag = QualityFlag::Normal;
    std::string message;
    cv::cuda::GpuMat d_virtualPixels;   // generated virtual pixels (empty on failure)

    VirtualPixelGenResult() = default;
    VirtualPixelGenResult(const VirtualPixelGenResult&) = delete;
    VirtualPixelGenResult& operator=(const VirtualPixelGenResult&) = delete;
    VirtualPixelGenResult(VirtualPixelGenResult&&) noexcept = default;
    VirtualPixelGenResult& operator=(VirtualPixelGenResult&&) noexcept = default;
};

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: virtualK/图像尺寸/线号按调用传入；Impl 仅持有每调用重置的 GPU 暂存缓冲，SetParams 缓存 gridStep 等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class VirtualPixelGenerator {
public:
    static constexpr const char* kLogTag = "12-VirtualPixelGen";

    explicit VirtualPixelGenerator(const VirtualPixelGenParams& params = {});
    ~VirtualPixelGenerator();

    VirtualPixelGenerator(const VirtualPixelGenerator&) = delete;
    VirtualPixelGenerator& operator=(const VirtualPixelGenerator&) = delete;

    VirtualPixelGenResult Execute(const cv::Matx33d& virtualK,
                                const cv::Size& imageSize,
                                const std::vector<int>& lineIds,
                                cv::cuda::Stream& stream);

    VirtualPixelGenResult Execute(const cv::Matx33d& virtualK,
                                const cv::Size& imageSize,
                                const std::vector<int>& lineIds);

    void Destroy();

    void Warmup(int maxPixels);
    void SetParams(const VirtualPixelGenParams& params);
    const VirtualPixelGenParams& GetParams() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace calib