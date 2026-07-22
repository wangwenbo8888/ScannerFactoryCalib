/**
 * @file intrinsic_compensate_cpu.cpp
 * @brief 相机内参补偿表CPU算子 - 实现文件
 *
 * 算法流程：
 *   1. 验证输入内参和算子参数
 *   2. 计算温度范围：T_start = refTemp + rangeMin, T_end = refTemp + rangeMax
 *   3. 对每个温度步距：
 *      a. 计算 ΔT = T - refTemp
 *      b. 计算 scale = 1 + cte × ΔT
 *      c. 补偿内参 = 原始内参 × scale
 *      d. 记录偏移量 delta = 原始内参 × cte × ΔT
 *   4. 返回补偿表结果
 */

#include "intrinsic_compensate_cpu.h"
#include "common/calib_logging.h"
#include <cmath>
#include <spdlog/spdlog.h>

using namespace calib;

OperatorInfo getIntrinsicCompensateCPUInfo() {
    return OperatorInfo{"IntrinsicCompensateCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

CALIB_DEFINE_LOG_TAG(01, IntrinsicCompensateCPU);

// ============================================================
// 构造 / 析构
// ============================================================

IntrinsicCompensateCPU::IntrinsicCompensateCPU(const IntrinsicCompensateCPUParams& params)
    : params_(params) {
    params_.validate();
    spdlog::info("[{}] 初始化完成, CTE={:.2e} /°C, 步距={:.2f}°C, 范围=[{:.1f}, {:.1f}]°C",
                 kLogTag, params_.cte, params_.tempStep,
                 params_.tempRangeMin, params_.tempRangeMax);
}

IntrinsicCompensateCPU::~IntrinsicCompensateCPU() = default;

// ============================================================
// 核心计算
// ============================================================

IntrinsicCompensateCPUResult IntrinsicCompensateCPU::Execute(const CameraIntrinsics& intrinsics) {
    IntrinsicCompensateCPUResult result;

    // 1. 验证输入
    intrinsics.validate();
    params_.validate();

    // 2. 计算温度范围
    double tStart = intrinsics.referenceTemp + params_.tempRangeMin;
    double tEnd   = intrinsics.referenceTemp + params_.tempRangeMax;
    double step   = params_.tempStep;

    // 3. 遍历每个温度步距
    double epsilon = step * 1e-6;  // 浮点容差，确保包含端点
    for (double t = tStart; t <= tEnd + epsilon; t += step) {
        double deltaT = t - intrinsics.referenceTemp;
        double scale = 1.0 + params_.cte * deltaT;

        CompensatedEntry entry;
        entry.temperature = t;
        entry.deltaT = deltaT;
        entry.fx = intrinsics.fx * scale;
        entry.fy = intrinsics.fy * scale;
        entry.cx = intrinsics.cx * scale;
        entry.cy = intrinsics.cy * scale;
        entry.deltaFx = intrinsics.fx * params_.cte * deltaT;
        entry.deltaFy = intrinsics.fy * params_.cte * deltaT;
        entry.deltaCx = intrinsics.cx * params_.cte * deltaT;
        entry.deltaCy = intrinsics.cy * params_.cte * deltaT;

        result.table.push_back(std::move(entry));
    }

    // 4. 填充结果
    result.success = true;
    result.message = "";
    result.referenceTemp = intrinsics.referenceTemp;
    result.cte = params_.cte;
    result.referenceIntrinsics = intrinsics;

    spdlog::info("[{}] 计算完成, 表大小={}, 温度范围=[{:.2f}, {:.2f}]°C",
                 kLogTag, result.table.size(), tStart, tEnd);

    return result;
}

// ============================================================
// 参数管理
// ============================================================

void IntrinsicCompensateCPU::SetParams(const IntrinsicCompensateCPUParams& params) {
    params.validate();
    params_ = params;
    spdlog::info("[{}] 参数已更新, CTE={:.2e} /°C, 步距={:.2f}°C",
                 kLogTag, params_.cte, params_.tempStep);
}

const IntrinsicCompensateCPUParams& IntrinsicCompensateCPU::GetParams() const {
    return params_;
}

void IntrinsicCompensateCPU::Destroy() {
}
