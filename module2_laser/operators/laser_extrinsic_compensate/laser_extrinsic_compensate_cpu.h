/**
 * @file laser_extrinsic_compensate_cpu.h
 * @brief 激光器虚拟相机外参补偿表CPU算子 - 公开头文件（�?C++，CPU 实现�? *
 * 所属流程：温度补偿 �? * 平台：CPU
 *
 * 功能：双目相机通过6061-T6铝合金骨架连接，骨架上安装激光器�? *       激光器被虚拟为相机，已知虚拟相机与�?右相机的外参�? *       温度变化导致虚拟相机与左/右相机的外参变化�? *       根据温度变化值批量计算各温度点下虚拟相机相对�?右相机的补偿外参�? *
 * 物理模型：复�?02extrinsic_compensate_cpu 的基线缩放模�? *   scale = 1 + α × ΔT
 *   T(T) = T₀ × scale
 *   R(T) = R₀（不变）
 *
 * 分别�?virtual-to-left �?virtual-to-right 两组外参调用 02 算子�? *
 * 输入：虚拟相机→左相机外�?+ 虚拟相机→右相机外参 + 标定温度
 * 输出：两组温度补偿表 (JSON)
 */

#pragma once


#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include "common/calib_types.h"
#include "extrinsic_compensate_cpu.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

// ============================================================
// LaserExtrinsicCompensateCPUParams �?算子参数
// ============================================================

/**
 * @brief 激光器虚拟相机外参温度补偿算子参数
 *
 * 参数复用 02 算子�?ExtrinsicCompensateCPUParams�? * 同时用于 virtual-to-left �?virtual-to-right 两组补偿计算�? */
struct LaserExtrinsicCompensateCPUParams {
    double cte = 23.6e-6;         ///< 材料线膨胀系数 (/°C)
    double tempStep = 0.2;        ///< 温度步距 (°C)
    double tempRangeMin = -10.0;  ///< 参考温度下方范�?(°C)
    double tempRangeMax = 10.0;   ///< 参考温度上方范�?(°C)

    void validate() const {
        if (cte <= 0.0)
            throw std::invalid_argument("LaserExtrinsicCompensateCPUParams::cte must be > 0");
        if (tempStep <= 0.0)
            throw std::invalid_argument("LaserExtrinsicCompensateCPUParams::tempStep must be > 0");
        if (tempRangeMin > tempRangeMax)
            throw std::invalid_argument("LaserExtrinsicCompensateCPUParams::tempRangeMin must be <= tempRangeMax");
    }

    nlohmann::json toJson() const {
        return {
            {"cte", cte},
            {"tempStep", tempStep},
            {"tempRangeMin", tempRangeMin},
            {"tempRangeMax", tempRangeMax}
        };
    }

    static LaserExtrinsicCompensateCPUParams fromJson(const nlohmann::json& j) {
        LaserExtrinsicCompensateCPUParams p;
        if (j.contains("cte"))          p.cte = j.at("cte").get<double>();
        if (j.contains("tempStep"))     p.tempStep = j.at("tempStep").get<double>();
        if (j.contains("tempRangeMin")) p.tempRangeMin = j.at("tempRangeMin").get<double>();
        if (j.contains("tempRangeMax")) p.tempRangeMax = j.at("tempRangeMax").get<double>();
        return p;
    }
};

// ============================================================
// LaserExtrinsicCompensateCPUResult �?补偿表结�?// ============================================================

/**
 * @brief 激光器虚拟相机外参温度补偿表计算结�? *
 * 包含虚拟相机→左相机 �?虚拟相机→右相机 两组补偿表�? */
struct LaserExtrinsicCompensateCPUResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    double referenceTemp = 0.0;
    double cte = 0.0;

    ///< 虚拟相机→左相机 的补偿表
    ExtrinsicCompensateCPUResult leftResult;

    ///< 虚拟相机→右相机 的补偿表
    ExtrinsicCompensateCPUResult rightResult;

    LaserExtrinsicCompensateCPUResult() = default;
    ~LaserExtrinsicCompensateCPUResult() = default;

    LaserExtrinsicCompensateCPUResult(LaserExtrinsicCompensateCPUResult&&) = default;
    LaserExtrinsicCompensateCPUResult& operator=(LaserExtrinsicCompensateCPUResult&&) = default;

    LaserExtrinsicCompensateCPUResult(const LaserExtrinsicCompensateCPUResult&) = delete;
    LaserExtrinsicCompensateCPUResult& operator=(const LaserExtrinsicCompensateCPUResult&) = delete;

    nlohmann::json toJson() const {
        return {
            {"success", success},
            {"message", message},
            {"referenceTemperature", referenceTemp},
            {"cte", cte},
            {"virtualToLeft", leftResult.toJson()},
            {"virtualToRight", rightResult.toJson()},
            {"tableSize", leftResult.table.size()}
        };
    }
};

// ============================================================
// LaserExtrinsicCompensateCPU �?核心算子�?// ============================================================

/**
 * @brief 激光器虚拟相机外参温度补偿�?CPU 算子
 *
 * 对虚拟相机→左相�?�?虚拟相机→右相机 两组外参�? * 分别调用 ExtrinsicCompensateCPU 进行温度补偿计算�? *
 * 线程安全约束�? * - 非线程安全：不同线程不得同时调用 Execute() �?setParams()
 * - 多实例并发场景各实例独立持有参数无需加锁
 */
class SCANNER_API LaserExtrinsicCompensateCPU {
public:
    static constexpr const char* kLogTag = "03-LaserExtrinsicCompensateCPU";

    explicit LaserExtrinsicCompensateCPU(const LaserExtrinsicCompensateCPUParams& params = {});
    ~LaserExtrinsicCompensateCPU();

    LaserExtrinsicCompensateCPU(const LaserExtrinsicCompensateCPU&) = delete;
    LaserExtrinsicCompensateCPU& operator=(const LaserExtrinsicCompensateCPU&) = delete;

    /**
    * @brief 计算激光器虚拟相机外参温度补偿�? 
    * @param virtualToLeft  标定温度下虚拟相机→左相机的外参
     * @param virtualToRight 标定温度下虚拟相机→右相机的外参
     * @return 包含两组补偿表的结果
     */
    LaserExtrinsicCompensateCPUResult Execute(
        const CameraExtrinsics& virtualToLeft,
        const CameraExtrinsics& virtualToRight
    );

    void SetParams(const LaserExtrinsicCompensateCPUParams& params);
    const LaserExtrinsicCompensateCPUParams& GetParams() const;

    void Warmup() { }

    void Destroy();

private:
    LaserExtrinsicCompensateCPUParams params_;
    ExtrinsicCompensateCPU compensator_;
};

OperatorInfo getLaserExtrinsicCompensateCPUInfo();

} // namespace calib