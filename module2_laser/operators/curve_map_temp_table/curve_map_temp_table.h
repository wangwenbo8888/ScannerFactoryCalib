// curve_map_temp_table.h — curve_map 温度表批量生成器（61 档 sidecar 生成链 Task 4）
//
// 设计: docs/plans/2026-09-01-curve-map-temp-table-design.md（硬规则 #1/#2/#3/#4/#6/#8）
// 流程: 参考档先行（atReference 零插值＋原始曲线）→ blob 头 52B 直解取冻结基准
//       （rowStep/roi/rowCount/rowMin/depth, 不做全量 Load）→ 温度网格逐档
//       （at() 插值＋曲线重标＋冻结 ROI）→ CmttBuilder 拼装 sidecar。
// 红线: 参考档 FAIL 整体不出产物（#3）; rowStep/深度窗由调用方注入禁硬编码（#2/#6）;
//       重标按 Task 0 像素坐标契约（q′ = s·q + m, s = f_T/f_ref, m = pp_T − s·pp_ref）。
// 内存契约: 档 blob 只驻 sidecarBytes 一份（生成期 move-through, 不在 tiers 保留）——
//       Task 5 金标准对拍取参考档: CmttReader::load(sidecarBytes) →
//       tierBytes(referenceIndex())。

#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <cstdint>
#include <string>
#include <vector>
#include "common/scanner_api.h"
#include "curve_map.h"
#include "cmtt_container.h"
#include "temp_param_interpolator.h"

namespace calib {

struct CurveMapTempTableGenParams {
    double referenceTemp = 0.0;    // 参考温度（须与两源表 referenceTemp 一致）
    float tempHalfRange = 15.f;    // ±15°C（须为 tempStep 整数倍）
    float tempStep = 0.5f;         // 档距
    float rowStep = 0.0f;          // 红线: 调用方从标定 JSON 继承（无猜默认）
    float depthMin = 100.0f;       // 全档统一（硬规则 #2）
    float depthMax = 700.0f;
    int tierThreads = 0;           // 档级并行线程数: 0=自动 min(硬件线程-2, 16);
                                   //   >0=精确值; <=-1 视为 1（线程数无关确定性）

    void validate() const;
    nlohmann::json toJson() const;
    static CurveMapTempTableGenParams fromJson(const nlohmann::json& j);
};

struct CurveMapTempTierResult {
    double temperature = 0.0;
    bool ok = false;               // Generate＋冻结网格断言全通过
    uint32_t entryCount = 0;
    uint8_t flags = 0;             // kCmttFlag* 位组合（bit0 参考/bit1 missing/bit2 clamped）
    uint32_t rowCountWithData = 0;
    // 注: 档 blob 不在此保留（内存契约, 见文件头）——载荷只驻 sidecarBytes
};

struct CurveMapTempTableResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;
    std::vector<CurveMapTempTierResult> tiers;   // 温度升序
    float clampedRatio = 0.f;      // clamped 档数/总档数
    uint64_t gridHash = 0;         // 冻结栅格身份哈希（FNV-1a 逐字段）
    std::vector<uint8_t> sidecarBytes;   // CmttBuilder 产物（硬规则 #3: 参考档失败为空）

    int okCount() const;
    /// 参考档在 tiers 内下标; 取参考档 blob: CmttReader::load(sidecarBytes) →
    /// tierBytes(referenceIndex())（Task 5 消费契约）
    size_t referenceIndex() const;

    CurveMapTempTableResult() = default;
    ~CurveMapTempTableResult() = default;
    CurveMapTempTableResult(CurveMapTempTableResult&&) = default;
    CurveMapTempTableResult& operator=(CurveMapTempTableResult&&) = default;
    CurveMapTempTableResult(const CurveMapTempTableResult&) = delete;
    CurveMapTempTableResult& operator=(const CurveMapTempTableResult&) = delete;
};

/// 曲线重标（Task 0 像素坐标契约一般式）: 温档 T 像素映射 q′ = s·q + (mx,my) 下的
/// 隐式二次曲线系数变换。m=0 时自动退化为纯缩放（d,e ×s; f ×s²）。
/// @pre s > 0 且有限（s = f_T/f_ref, 两侧 f 逐档为正）; 违例抛 std::invalid_argument
SCANNER_API ImplicitCurve rescaleCurve(const ImplicitCurve& curve,
                                       double s, double mx, double my);

/// 冻结栅格基准（参考档 blob 头解析值, 六字段）
struct CmttFrozenGrid {
    float rowStep = 0.f;
    cv::Rect roi{0, 0, 0, 0};
    uint32_t rowCount = 0;
    uint32_t rowMin = 0;
    float depthMin = 0.f;
    float depthMax = 0.f;
};

/// 冻结网格断言（硬规则 #1 防线, header-only）: 只解析 blob 前 52B（HeaderV1）比对
/// rowStep/roi/rowCount/rowMin/depthMin/depthMax, 不做 CurveMapTable::Load 全量
/// 走读（61 档 × 31MB 的解析成本只付一次给消费侧）。
/// 自洽防线: |rowMin − floor(roi.y/rowStep + 0.5)| ≤ 1（金标准实例 roi.y=286,
/// rowStep=0.7 → 408.57 → rowMin=408）——同尺寸字段重排（rowMin↔dataRows）会被
/// 此关系大声拒绝, 这是对"复刻布局漂移"的防线; 根治方案为 factory_calib 侧人工
/// 加访问器（设计文档遗留项）。
SCANNER_API bool cmttTierGridMatches(const uint8_t* blob, size_t size,
                                     const CmttFrozenGrid& frozen, std::string* why);

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态（params 构造期定; Generate 一次性生成, 无跨调用累积）
// 并发策略: Generate 内档级 omp 并行（嵌套禁用→行级自动串行）; 线程数无关确定性;
//           每实例非线程安全
// ==============================

class SCANNER_API CurveMapTempTableGenerator {
public:
    static constexpr const char* kLogTag = "12d-CurveMapTempTable";

    /// @throws std::invalid_argument params 非法（rowStep≤0/深度窗/档距/整除性/NaN）
    explicit CurveMapTempTableGenerator(const CurveMapTempTableGenParams& params);
    CurveMapTempTableGenerator(const CurveMapTempTableGenerator&) = delete;
    CurveMapTempTableGenerator& operator=(const CurveMapTempTableGenerator&) = delete;

    /// 两源表构造期校验异常（TempParamInterpolator）与参考温一致性违约均穿透抛出。
    /// @return success=false 当且仅当参考档失败（硬规则 #3, sidecarBytes 空, message 透传）
    CurveMapTempTableResult Generate(
        const std::vector<ImplicitCurve>& curves,
        const StereoRectifyTempTableResult& rectifyTable,
        const LaserExtrinsicCompensateCPUResult& laserExtrinTable,
        const cv::Size& imageSize);

private:
    CurveMapTempTableGenParams params_;
};

} // namespace calib
