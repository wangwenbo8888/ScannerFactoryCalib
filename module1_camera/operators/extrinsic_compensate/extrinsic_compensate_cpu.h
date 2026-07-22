/**
 * @file extrinsic_compensate_cpu.h
 * @brief 相机外参补偿表CPU算子 - 公开头文件（�?C++，CPU 实现�? *
 * 所属流程：温度补偿 �? * 平台：CPU
 *
 * 功能：基于骨架材料（6061-T6铝合金）的热膨胀效应�? *       按温度步距批量计算每个温度点对应的补偿外参（R, T），
 *       输出 JSON 补偿表�? *
 * 物理模型：仅基线缩放
 *   scale = 1 + α × ΔT
 *   T(T) = T₀ × scale
 *   R(T) = R₀（不变）
 *
 * 输入：标定温度下的双目外�?(R, T) + 标定温度
 * 输出：温度补偿表 (JSON)
 */

#pragma once


#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

// ============================================================
// CameraExtrinsics �?标定温度下的双目外参
// ============================================================

/**
 * @brief 双目相机外参（标定温度下�? */
struct CameraExtrinsics {
    double R[9] = {};            ///< 旋转矩阵 3×3 行主序展开
    double T[3] = {};            ///< 平移向量 (tx, ty, tz)
    double referenceTemp = 25.0; ///< 标定时的温度 (°C)

    void validate() const {
        double tNorm = std::sqrt(T[0]*T[0] + T[1]*T[1] + T[2]*T[2]);
        if (tNorm <= 0.0)
            throw std::invalid_argument("CameraExtrinsics::T must be non-zero (baseline required)");
    }

    nlohmann::json toJson() const {
        auto rArr = nlohmann::json::array();
        for (int i = 0; i < 9; ++i) rArr.push_back(R[i]);
        auto tArr = nlohmann::json::array();
        for (int i = 0; i < 3; ++i) tArr.push_back(T[i]);
        return {
            {"R", rArr},
            {"T", tArr},
            {"referenceTemp", referenceTemp}
        };
    }

    static CameraExtrinsics fromJson(const nlohmann::json& j) {
        CameraExtrinsics ce;
        if (j.contains("R") && j["R"].is_array()) {
            for (int i = 0; i < 9 && i < static_cast<int>(j["R"].size()); ++i)
                ce.R[i] = j["R"][i].get<double>();
        }
        if (j.contains("T") && j["T"].is_array()) {
            for (int i = 0; i < 3 && i < static_cast<int>(j["T"].size()); ++i)
                ce.T[i] = j["T"][i].get<double>();
        }
        if (j.contains("referenceTemp")) ce.referenceTemp = j.at("referenceTemp").get<double>();
        return ce;
    }
};

// ============================================================
// ExtrinsicCompensateCPUParams �?算子参数
// ============================================================

/**
 * @brief 温度补偿算子参数
 */
struct ExtrinsicCompensateCPUParams {
    double cte = 23.6e-6;         ///< 材料线膨胀系数 (/°C)
    double tempStep = 0.2;        ///< 温度步距 (°C)
    double tempRangeMin = -10.0;  ///< 参考温度下方范�?(°C)
    double tempRangeMax = 10.0;   ///< 参考温度上方范�?(°C)

    void validate() const {
        if (cte <= 0.0)
            throw std::invalid_argument("ExtrinsicCompensateCPUParams::cte must be > 0");
        if (tempStep <= 0.0)
            throw std::invalid_argument("ExtrinsicCompensateCPUParams::tempStep must be > 0");
        if (tempRangeMin > tempRangeMax)
            throw std::invalid_argument("ExtrinsicCompensateCPUParams::tempRangeMin must be <= tempRangeMax");
    }

    nlohmann::json toJson() const {
        return {
            {"cte", cte},
            {"tempStep", tempStep},
            {"tempRangeMin", tempRangeMin},
            {"tempRangeMax", tempRangeMax}
        };
    }

    static ExtrinsicCompensateCPUParams fromJson(const nlohmann::json& j) {
        ExtrinsicCompensateCPUParams p;
        if (j.contains("cte"))          p.cte = j.at("cte").get<double>();
        if (j.contains("tempStep"))     p.tempStep = j.at("tempStep").get<double>();
        if (j.contains("tempRangeMin")) p.tempRangeMin = j.at("tempRangeMin").get<double>();
        if (j.contains("tempRangeMax")) p.tempRangeMax = j.at("tempRangeMax").get<double>();
        return p;
    }
};

// ============================================================
// ExtrinsicCompensatedEntry �?单个温度步距的补偿结�?// ============================================================

/**
 * @brief 单个温度步距的补偿结�? */
struct ExtrinsicCompensatedEntry {
    double temperature = 0.0;     ///< 当前温度 (°C)
    double deltaT = 0.0;          ///< 与参考温度差 (°C)
    double R[9] = {};             ///< 补偿后旋�?(= R₀，不�?
    double T[3] = {};             ///< 补偿后平移
    double deltaT_vec[3] = {};    ///< T 偏移量 = T₀ × α × ΔT

    nlohmann::json toJson() const {
        auto rArr = nlohmann::json::array();
        for (int i = 0; i < 9; ++i) rArr.push_back(R[i]);
        auto tArr = nlohmann::json::array();
        for (int i = 0; i < 3; ++i) tArr.push_back(T[i]);
        auto dArr = nlohmann::json::array();
        for (int i = 0; i < 3; ++i) dArr.push_back(deltaT_vec[i]);
        return {
            {"temperature", temperature},
            {"deltaT", deltaT},
            {"R", rArr},
            {"T", tArr},
            {"deltaT_vec", dArr}
        };
    }
};

// ============================================================
// ExtrinsicCompensateCPUResult �?补偿表结�?// ============================================================

/**
 * @brief 温度补偿表计算结�? */
struct ExtrinsicCompensateCPUResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    double referenceTemp = 0.0;
    double cte = 0.0;
    double baselineRef = 0.0;
    CameraExtrinsics referenceExtrinsics;
    std::vector<ExtrinsicCompensatedEntry> table;

    ExtrinsicCompensateCPUResult() = default;
    ~ExtrinsicCompensateCPUResult() = default;

    ExtrinsicCompensateCPUResult(ExtrinsicCompensateCPUResult&&) = default;
    ExtrinsicCompensateCPUResult& operator=(ExtrinsicCompensateCPUResult&&) = default;

    ExtrinsicCompensateCPUResult(const ExtrinsicCompensateCPUResult&) = delete;
    ExtrinsicCompensateCPUResult& operator=(const ExtrinsicCompensateCPUResult&) = delete;

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
            {"baselineRef", baselineRef},
            {"referenceExtrinsics", referenceExtrinsics.toJson()},
            {"table", tableArr},
            {"tableSize", table.size()}
        };
    }
};

// ============================================================
// ExtrinsicCompensateCPU �?核心算子�?// ============================================================

/**
 * @brief 相机外参温度补偿�?CPU 算子
 *
 * 线程安全约束�? * - 非线程安全：不同线程不得同时调用 Execute() �?setParams()
 * - 多实例并发场景各实例独立持有参数无需加锁
 */
class SCANNER_API ExtrinsicCompensateCPU {
public:
    static constexpr const char* kLogTag = "02-ExtrinsicCompensateCPU";

    explicit ExtrinsicCompensateCPU(const ExtrinsicCompensateCPUParams& params = {});
    ~ExtrinsicCompensateCPU();

    ExtrinsicCompensateCPU(const ExtrinsicCompensateCPU&) = delete;
    ExtrinsicCompensateCPU& operator=(const ExtrinsicCompensateCPU&) = delete;

    /**
    * @brief 计算温度补偿�? 
    * @param extrinsics 标定温度下的双目外参
     * @return 补偿表结果（包含每个温度步距的补偿外参）
     */
    ExtrinsicCompensateCPUResult Execute(const CameraExtrinsics& extrinsics);

    void SetParams(const ExtrinsicCompensateCPUParams& params);
    const ExtrinsicCompensateCPUParams& GetParams() const;

    void Warmup() { }

    void Destroy();

private:
    ExtrinsicCompensateCPUParams params_;
};

OperatorInfo getExtrinsicCompensateCPUInfo();

} // namespace calib