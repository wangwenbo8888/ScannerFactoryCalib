/**
 * @file extrinsic_compensate_cpu.cpp
 * @brief 相机外参补偿表CPU算子 - 实现文件
 *
 * 算法流程：
 *   1. 验证输入外参和算子参数
 *   2. 计算温度范围：T_start = refTemp + rangeMin, T_end = refTemp + rangeMax
 *   3. 对每个温度步距：
 *      a. 计算 ΔT = T - refTemp
 *      b. 计算 scale = 1 + cte × ΔT
 *      c. R 不变（原样拷贝）
 *      d. T_new = T₀ × scale
 *      e. deltaT_vec = T₀ × cte × ΔT
 *   4. 返回补偿表结果
 */

#include "extrinsic_compensate_cpu.h"
#include "common/calib_logging.h"
#include <cmath>
#include <algorithm>
#include <spdlog/spdlog.h>

using namespace calib;

OperatorInfo getExtrinsicCompensateCPUInfo() {
    return OperatorInfo{"ExtrinsicCompensateCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

CALIB_DEFINE_LOG_TAG(02, ExtrinsicCompensateCPU);

// ============================================================
// 构造 / 析构
// ============================================================

ExtrinsicCompensateCPU::ExtrinsicCompensateCPU(const ExtrinsicCompensateCPUParams& params)
    : params_(params) {
    params_.validate();
    spdlog::info("[{}] 初始化完成, CTE={:.2e} /°C, 步距={:.2f}°C, 范围=[{:.1f}, {:.1f}]°C",
                 kLogTag, params_.cte, params_.tempStep,
                 params_.tempRangeMin, params_.tempRangeMax);
}

ExtrinsicCompensateCPU::~ExtrinsicCompensateCPU() = default;

// ============================================================
// 核心计算
// ============================================================

ExtrinsicCompensateCPUResult ExtrinsicCompensateCPU::Execute(const CameraExtrinsics& extrinsics) {
    ExtrinsicCompensateCPUResult result;

    // 1. 验证输入
    extrinsics.validate();
    params_.validate();

    // 2. 计算温度范围
    double tStart = extrinsics.referenceTemp + params_.tempRangeMin;
    double tEnd   = extrinsics.referenceTemp + params_.tempRangeMax;
    double step   = params_.tempStep;

    // 3. 计算参考基线长度
    double baseline = std::sqrt(
        extrinsics.T[0] * extrinsics.T[0] +
        extrinsics.T[1] * extrinsics.T[1] +
        extrinsics.T[2] * extrinsics.T[2]
    );

    // 4. 遍历每个温度步距
    double epsilon = step * 1e-6;  // 浮点容差，确保包含端点
    for (double t = tStart; t <= tEnd + epsilon; t += step) {
        double deltaT = t - extrinsics.referenceTemp;
        double scale = 1.0 + params_.cte * deltaT;

        ExtrinsicCompensatedEntry entry;
        entry.temperature = t;
        entry.deltaT = deltaT;

        // R 不变，原样拷贝
        std::copy(std::begin(extrinsics.R), std::end(extrinsics.R), std::begin(entry.R));

        // T 缩放
        for (int i = 0; i < 3; ++i) {
            entry.T[i] = extrinsics.T[i] * scale;
            entry.deltaT_vec[i] = extrinsics.T[i] * params_.cte * deltaT;
        }

        result.table.push_back(std::move(entry));
    }

    // 5. 填充结果
    result.success = true;
    result.message = "";
    result.referenceTemp = extrinsics.referenceTemp;
    result.cte = params_.cte;
    result.baselineRef = baseline;
    result.referenceExtrinsics = extrinsics;

    spdlog::info("[{}] 计算完成, 表大小={}, 温度范围=[{:.2f}, {:.2f}]°C, 参考基线={:.4f}",
                 kLogTag, result.table.size(), tStart, tEnd, baseline);

    return result;
}

// ============================================================
// 参数管理
// ============================================================

void ExtrinsicCompensateCPU::SetParams(const ExtrinsicCompensateCPUParams& params) {
    params.validate();
    params_ = params;
    spdlog::info("[{}] 参数已更新, CTE={:.2e} /°C, 步距={:.2f}°C",
                 kLogTag, params_.cte, params_.tempStep);
}

const ExtrinsicCompensateCPUParams& ExtrinsicCompensateCPU::GetParams() const {
    return params_;
}

void ExtrinsicCompensateCPU::Destroy() {
}
