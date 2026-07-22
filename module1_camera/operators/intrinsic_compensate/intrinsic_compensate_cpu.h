/**
 * @file intrinsic_compensate_cpu.h
 * @brief 相机内参补偿表CPU算子 - 公开头文件（�?C++，CPU 实现�? *
 * 所属流程：温度补偿 �? * 平台：CPU
 *
 * 功能：基于镜头筒材料�?061-T6铝合金）的热膨胀效应�? *       按温度步距批量计算每个温度点对应的补偿内参，
 *       输出 JSON 补偿表�? *
 * 物理模型：线性热膨胀
 *   scale = 1 + α × ΔT
 *   fx(T) = fx₀ × scale, fy(T) = fy₀ × scale
 *   cx(T) = cx₀ × scale, cy(T) = cy₀ × scale
 *
 * 输入：标定温度下的相机内�?(fx, fy, cx, cy) + 标定温度
 * 输出：温度补偿表 (JSON)
 */

#pragma once


#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <stdexcept>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

// ============================================================
// CameraIntrinsics �?标定温度下的相机内参
// ============================================================

/**
 * @brief 相机内参（标定温度下�? */
struct CameraIntrinsics {
    double fx = 0.0;              ///< 焦距 x (像素)
    double fy = 0.0;              ///< 焦距 y (像素)
    double cx = 0.0;              ///< 主点 x (像素)
    double cy = 0.0;              ///< 主点 y (像素)
    double referenceTemp = 25.0;  ///< 标定时的温度 (°C)

    void validate() const {
        if (fx <= 0.0)
            throw std::invalid_argument("CameraIntrinsics::fx must be > 0");
        if (fy <= 0.0)
            throw std::invalid_argument("CameraIntrinsics::fy must be > 0");
        if (cx <= 0.0)
            throw std::invalid_argument("CameraIntrinsics::cx must be > 0");
        if (cy <= 0.0)
            throw std::invalid_argument("CameraIntrinsics::cy must be > 0");
    }

    nlohmann::json toJson() const {
        return {
            {"fx", fx}, {"fy", fy},
            {"cx", cx}, {"cy", cy},
            {"referenceTemp", referenceTemp}
        };
    }

    static CameraIntrinsics fromJson(const nlohmann::json& j) {
        CameraIntrinsics ci;
        if (j.contains("fx")) ci.fx = j.at("fx").get<double>();
        if (j.contains("fy")) ci.fy = j.at("fy").get<double>();
        if (j.contains("cx")) ci.cx = j.at("cx").get<double>();
        if (j.contains("cy")) ci.cy = j.at("cy").get<double>();
        if (j.contains("referenceTemp")) ci.referenceTemp = j.at("referenceTemp").get<double>();
        return ci;
    }
};

// ============================================================
// IntrinsicCompensateCPUParams �?算子参数
// ============================================================

/**
 * @brief 温度补偿算子参数
 */
struct IntrinsicCompensateCPUParams {
    double cte = 23.6e-6;         ///< 材料线膨胀系数 (/°C)
    double tempStep = 0.2;        ///< 温度步距 (°C)
    double tempRangeMin = -10.0;  ///< 参考温度下方范�?(°C)
    double tempRangeMax = 10.0;   ///< 参考温度上方范�?(°C)

    void validate() const {
        if (cte <= 0.0)
            throw std::invalid_argument("IntrinsicCompensateCPUParams::cte must be > 0");
        if (tempStep <= 0.0)
            throw std::invalid_argument("IntrinsicCompensateCPUParams::tempStep must be > 0");
        if (tempRangeMin > tempRangeMax)
            throw std::invalid_argument("IntrinsicCompensateCPUParams::tempRangeMin must be <= tempRangeMax");
    }

    nlohmann::json toJson() const {
        return {
            {"cte", cte},
            {"tempStep", tempStep},
            {"tempRangeMin", tempRangeMin},
            {"tempRangeMax", tempRangeMax}
        };
    }

    static IntrinsicCompensateCPUParams fromJson(const nlohmann::json& j) {
        IntrinsicCompensateCPUParams p;
        if (j.contains("cte"))         p.cte = j.at("cte").get<double>();
        if (j.contains("tempStep"))    p.tempStep = j.at("tempStep").get<double>();
        if (j.contains("tempRangeMin")) p.tempRangeMin = j.at("tempRangeMin").get<double>();
        if (j.contains("tempRangeMax")) p.tempRangeMax = j.at("tempRangeMax").get<double>();
        return p;
    }
};

// ============================================================
// CompensatedEntry �?单个温度步距的补偿结�?// ============================================================

/**
 * @brief 单个温度步距的补偿结�? */
struct CompensatedEntry {
    double temperature = 0.0;  ///< 当前温度 (°C)
    double deltaT = 0.0;       ///< 与参考温度差 (°C)
    double fx = 0.0;           ///< 补偿后焦�?x (像素)
    double fy = 0.0;           ///< 补偿后焦�?y (像素)
    double cx = 0.0;           ///< 补偿后主�?x (像素)
    double cy = 0.0;           ///< 补偿后主�?y (像素)
    double deltaFx = 0.0;      ///< 焦距x偏移�?(像素)
    double deltaFy = 0.0;      ///< 焦距y偏移�?(像素)
    double deltaCx = 0.0;      ///< 主点x偏移�?(像素)
    double deltaCy = 0.0;      ///< 主点y偏移�?(像素)

    nlohmann::json toJson() const {
        return {
            {"temperature", temperature},
            {"deltaT", deltaT},
            {"fx", fx}, {"fy", fy},
            {"cx", cx}, {"cy", cy},
            {"deltaFx", deltaFx}, {"deltaFy", deltaFy},
            {"deltaCx", deltaCx}, {"deltaCy", deltaCy}
        };
    }
};

// ============================================================
// IntrinsicCompensateCPUResult �?补偿表结�?// ============================================================

/**
 * @brief 温度补偿表计算结�? */
struct IntrinsicCompensateCPUResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    double referenceTemp = 0.0;
    double cte = 0.0;
    CameraIntrinsics referenceIntrinsics;
    std::vector<CompensatedEntry> table;

    IntrinsicCompensateCPUResult() = default;
    ~IntrinsicCompensateCPUResult() = default;

    IntrinsicCompensateCPUResult(IntrinsicCompensateCPUResult&&) = default;
    IntrinsicCompensateCPUResult& operator=(IntrinsicCompensateCPUResult&&) = default;

    IntrinsicCompensateCPUResult(const IntrinsicCompensateCPUResult&) = delete;
    IntrinsicCompensateCPUResult& operator=(const IntrinsicCompensateCPUResult&) = delete;

    nlohmann::json toJson() const {
        nlohmann::json tableArr = nlohmann::json::array();
        for (const auto& entry : table) {
            tableArr.push_back(entry.toJson());
        }
        return {
            {"success", success},
            {"message", message},
            {"referenceTemperature", referenceTemp},
            {"cte", cte},
            {"referenceIntrinsics", referenceIntrinsics.toJson()},
            {"table", tableArr},
            {"tableSize", table.size()}
        };
    }
};

// ============================================================
// IntrinsicCompensateCPU �?核心算子�?// ============================================================

/**
 * @brief 相机内参温度补偿�?CPU 算子
 *
 * 线程安全约束�? * - 非线程安全：不同线程不得同时调用 Execute() �?setParams()
 * - 多实例并发场景各实例独立持有参数无需加锁
 */
class SCANNER_API IntrinsicCompensateCPU {
public:
    static constexpr const char* kLogTag = "01-IntrinsicCompensateCPU";

    explicit IntrinsicCompensateCPU(const IntrinsicCompensateCPUParams& params = {});
    ~IntrinsicCompensateCPU();

    IntrinsicCompensateCPU(const IntrinsicCompensateCPU&) = delete;
    IntrinsicCompensateCPU& operator=(const IntrinsicCompensateCPU&) = delete;

    /**
    * @brief 计算温度补偿�? 
    * @param intrinsics 标定温度下的相机内参
     * @return 补偿表结果（包含每个温度步距的补偿内参）
     */
    IntrinsicCompensateCPUResult Execute(const CameraIntrinsics& intrinsics);

    void SetParams(const IntrinsicCompensateCPUParams& params);
    const IntrinsicCompensateCPUParams& GetParams() const;

    void Warmup() { }

    void Destroy();

private:
    IntrinsicCompensateCPUParams params_;
};

OperatorInfo getIntrinsicCompensateCPUInfo();

} // namespace calib