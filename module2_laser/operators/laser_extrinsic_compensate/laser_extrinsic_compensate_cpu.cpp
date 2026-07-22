/**
 * @file laser_extrinsic_compensate_cpu.cpp
 * @brief 激光器虚拟相机外参补偿表CPU算子 - 实现文件
 *
 * 算法流程：
 *   1. 验证输入外参和算子参数
 *   2. 将 LaserExtrinsicCompensateCPUParams 转换为 ExtrinsicCompensateCPUParams
 *   3. 调用 ExtrinsicCompensateCPU::compute() 计算 virtual-to-left 补偿表
 *   4. 调用 ExtrinsicCompensateCPU::compute() 计算 virtual-to-right 补偿表
 *   5. 组装结果返回
 */

#include "laser_extrinsic_compensate_cpu.h"
#include "common/calib_logging.h"
#include <spdlog/spdlog.h>

using namespace calib;

OperatorInfo getLaserExtrinsicCompensateCPUInfo() {
    return OperatorInfo{"LaserExtrinsicCompensateCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

CALIB_DEFINE_LOG_TAG(03, LaserExtrinsicCompensateCPU);

// ============================================================
// 构造 / 析构
// ============================================================

LaserExtrinsicCompensateCPU::LaserExtrinsicCompensateCPU(const LaserExtrinsicCompensateCPUParams& params)
    : params_(params)
    , compensator_(ExtrinsicCompensateCPUParams{
          params.cte, params.tempStep, params.tempRangeMin, params.tempRangeMax
      }) {
    params_.validate();
    spdlog::info("[{}] 初始化完成, CTE={:.2e} /°C, 步距={:.2f}°C, 范围=[{:.1f}, {:.1f}]°C",
                 kLogTag, params_.cte, params_.tempStep,
                 params_.tempRangeMin, params_.tempRangeMax);
}

LaserExtrinsicCompensateCPU::~LaserExtrinsicCompensateCPU() = default;

// ============================================================
// 核心计算
// ============================================================

LaserExtrinsicCompensateCPUResult LaserExtrinsicCompensateCPU::Execute(
    const CameraExtrinsics& virtualToLeft,
    const CameraExtrinsics& virtualToRight) {
    LaserExtrinsicCompensateCPUResult result;

    // 1. 验证输入
    virtualToLeft.validate();
    virtualToRight.validate();
    params_.validate();

    spdlog::info("[{}] 开始计算, virtual-to-left 基线={:.4f} mm, virtual-to-right 基线={:.4f} mm",
                 kLogTag,
                 std::sqrt(virtualToLeft.T[0] * virtualToLeft.T[0] +
                           virtualToLeft.T[1] * virtualToLeft.T[1] +
                           virtualToLeft.T[2] * virtualToLeft.T[2]),
                 std::sqrt(virtualToRight.T[0] * virtualToRight.T[0] +
                           virtualToRight.T[1] * virtualToRight.T[1] +
                           virtualToRight.T[2] * virtualToRight.T[2]));

    // 2. 同步参数到内部补偿器
    ExtrinsicCompensateCPUParams innerParams;
    innerParams.cte = params_.cte;
    innerParams.tempStep = params_.tempStep;
    innerParams.tempRangeMin = params_.tempRangeMin;
    innerParams.tempRangeMax = params_.tempRangeMax;
    compensator_.SetParams(innerParams);

    // 3. 计算 virtual-to-left 补偿表
    try {
        result.leftResult = compensator_.Execute(virtualToLeft);
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("virtual-to-left 补偿计算失败: ") + e.what();
        spdlog::error("[{}] {}", kLogTag, result.message);
        return result;
    }

    // 4. 计算 virtual-to-right 补偿表
    try {
        result.rightResult = compensator_.Execute(virtualToRight);
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("virtual-to-right 补偿计算失败: ") + e.what();
        spdlog::error("[{}] {}", kLogTag, result.message);
        return result;
    }

    // 5. 填充结果
    result.success = true;
    result.message = "";
    result.referenceTemp = virtualToLeft.referenceTemp;
    result.cte = params_.cte;

    spdlog::info("[{}] 计算完成, left 表大小={}, right 表大小={}",
                 kLogTag, result.leftResult.table.size(), result.rightResult.table.size());

    return result;
}

// ============================================================
// 参数管理
// ============================================================

void LaserExtrinsicCompensateCPU::SetParams(const LaserExtrinsicCompensateCPUParams& params) {
    params.validate();
    params_ = params;

    ExtrinsicCompensateCPUParams innerParams;
    innerParams.cte = params.cte;
    innerParams.tempStep = params.tempStep;
    innerParams.tempRangeMin = params.tempRangeMin;
    innerParams.tempRangeMax = params.tempRangeMax;
    compensator_.SetParams(innerParams);

    spdlog::info("[{}] 参数已更新, CTE={:.2e} /°C, 步距={:.2f}°C",
                 kLogTag, params_.cte, params_.tempStep);
}

const LaserExtrinsicCompensateCPUParams& LaserExtrinsicCompensateCPU::GetParams() const {
    return params_;
}

void LaserExtrinsicCompensateCPU::Destroy() {
}
