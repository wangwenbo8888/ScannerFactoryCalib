// curve_map.h — 极线切曲线映射表算子（激光标定-12b）
//
// 设计: docs/plans/2026-08-26-curve-map-design.md（五轮审查定稿）
// 模型: 矫正系直连——三台相机（矫正左/矫正右/虚拟=R:I 同左）零翻译;
//       左相机 ROI 逐行采样 → 左↔虚拟极线（过像外极点的线束）×25 条曲线解析求交
//       → 交点深度 → uR = uL − f·B/Z（视差恒正）→ 深度窗过滤 → 0.7 栅格量化。
// 输出: CSR 行索引＋线分组＋[ΔuL:6|Δd:10zigzag] 位打包压缩表。
// 红线: 禁 R1/R2（矫正系直连）; rowStep 从标定 JSON 继承禁硬编码; FAIL 不出产物。

#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"
#include "projector_joint_calib.h"

namespace calib {

struct WarmupConfig;

// ============================================================================
// CurveMapParams
// ============================================================================

struct CurveMapParams {
    /// 极线行距（像素, y_k = k*rowStep）。必须与上下游一致（0.7）。
    /// 红线: 由调用方从标定 JSON 继承, 本算子不提供"猜默认"路径之外的来源。
    float epipolarRowStep = 0.7f;
    /// 深度收录窗（mm, 左相机深度）
    float depthMin = 100.0f;
    float depthMax = 700.0f;
    /// 行内 uL 采样步（像素）; 默认与行距一致
    float uSampleStep = -1.0f;    // <0 → 取 epipolarRowStep
    /// 覆盖统计空洞告警阈值（uL 格数）
    int   maxGap = 8;
    /// OpenMP 线程数（0=自动）
    int   threads = 0;

    float effectiveUSampleStep() const {
        return uSampleStep > 0.0f ? uSampleStep : epipolarRowStep;
    }

    void validate() const {
        if (epipolarRowStep <= 0.0f)
            throw std::invalid_argument("CurveMapParams::epipolarRowStep must be > 0");
        if (uSampleStep <= 0.0f && uSampleStep != -1.0f)
            throw std::invalid_argument("CurveMapParams::uSampleStep must be > 0 or -1 (auto)");
        if (depthMin <= 0.0f || depthMax <= depthMin)
            throw std::invalid_argument("CurveMapParams::depth window invalid (need 0 < min < max)");
        if (maxGap < 1)
            throw std::invalid_argument("CurveMapParams::maxGap must be >= 1");
        if (threads < 0)
            throw std::invalid_argument("CurveMapParams::threads must be >= 0");
    }

    nlohmann::json toJson() const {
        return {
            {"epipolarRowStep", epipolarRowStep},
            {"depthMin", depthMin},
            {"depthMax", depthMax},
            {"uSampleStep", effectiveUSampleStep()},
            {"maxGap", maxGap},
            {"threads", threads}
        };
    }
};

// ============================================================================
// 输入 / 输出
// ============================================================================

struct CurveMapInput {
    /// 25 条发射曲线（←PJC / laser_calib.json pjc.emissionCurves）
    std::vector<ImplicitCurve> curves;
    /// 虚拟相机（矫正系直连: R≡I 不入参, 只需 K/T）
    cv::Matx33d virtualK = cv::Matx33d::eye();
    cv::Vec3d   virtualT = cv::Vec3d(0, 0, 0);
    /// 左相机（矫正）: f/主点（P1 3x3）; 右基线 B = |P2(0,3)|/f（mm）
    double f = 0.0;
    cv::Point2d principalPoint = cv::Point2d(0, 0);
    double baseline = 0.0;
    /// 像面尺寸（矫正后）
    cv::Size imageSize = cv::Size(0, 0);
    /// 有效区（←标定 JSON validRoi 实读）
    cv::Rect roi = cv::Rect(0, 0, 0, 0);
};

/// 单条表记录的展开形态（生成期中间态/查询返回）
struct CurveMapEntry {
    uint16_t row = 0;     // 绝对行号（k, y = k*rowStep）
    uint8_t  lid = 0;     // 激光线号（1..25; 0 仅占位）
    uint16_t uL = 0;      // 左 u 栅格号
    uint16_t uR = 0;      // 右 u 栅格号（量化后; 恒 uR < uL）
};

/// 每行×每线覆盖统计（诊断）
struct CurveMapCover {
    uint16_t row = 0;
    uint8_t  lid = 0;
    uint16_t uLmin = 0;
    uint16_t uLmax = 0;
    uint32_t count = 0;
    uint16_t maxGapSeen = 0;   // 该(行,线)内最大 uL 断档（格）
};

struct CurveMapResult {
    bool        success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    /// 压缩表原始字节（文件头+行索引+行块; 格式见设计 E.2）
    std::vector<uint8_t> tableBytes;
    /// 表统计
    uint32_t entryCount = 0;
    uint32_t rowCountWithData = 0;
    uint64_t tableBytesSize = 0;
    /// 覆盖诊断（仅含有数据的 行×线）
    std::vector<CurveMapCover> covers;
    /// 生效参数回显（含 rowStep——供落盘元数据）
    CurveMapParams effectiveParams;

    CurveMapResult() = default;
    ~CurveMapResult() = default;
    CurveMapResult(CurveMapResult&&) = default;
    CurveMapResult& operator=(CurveMapResult&&) = default;
    CurveMapResult(const CurveMapResult&) = delete;
    CurveMapResult& operator=(const CurveMapResult&) = delete;
};

// ============================================================================
// 压缩表（运行期持有 + 查询）
// ============================================================================

/// 查询候选（解码后的展开形态）
struct CurveMapCand {
    uint8_t  lid = 0;
    uint16_t uR = 0;
};

class SCANNER_API CurveMapTable {
public:
    static constexpr uint32_t kMagic = 0x434D5055;   // "CMPU"

    CurveMapTable() = default;

    /// 从字节流载入（CurveMapResult.tableBytes 或文件内容）
    bool Load(const uint8_t* data, size_t size, std::string& err);
    bool LoadFromFile(const std::string& path, std::string& err);
    bool SaveToFile(const std::string& path, std::string& err) const;

    /// 行内查询: 返回该 (row, uL) 的全部候选（每线至多 2, 双根）。
    /// cands 为输出缓冲（调用方给 ≥64）; 返回候选数。
    int Lookup(int row, int uL, CurveMapCand* cands, int candsCap) const;

    /// 整行枚举（e2e 适配用）: 解码该行全部条目升序返回; cap 不够返回 -1。
    int EnumerateRow(int row, CurveMapEntry* out, int cap) const;

    // ---- 元数据 ----
    float    rowStep() const { return rowStep_; }
    float    depthMin() const { return depthMin_; }
    float    depthMax() const { return depthMax_; }
    cv::Rect roi() const { return roi_; }
    uint32_t entryCount() const { return entryCount_; }
    uint32_t rowCount() const { return rowCount_; }        // 绝对行数（含空行）
    bool     valid() const { return !rows_.empty() || entryCount_ > 0; }

private:
    friend class CurveMapGenerator;
    struct LineGroup {
        uint8_t  lid = 0;
        uint16_t anchorUL = 0;   // 组内首条目绝对 uL（栅格）
        std::vector<uint16_t> words;  // 位打包条目流（含转义）
    };
    struct RowData {
        bool valid = false;
        std::vector<LineGroup> groups;
    };
    std::vector<RowData> rows_;      // 按绝对行号索引
    std::vector<uint64_t> rowByteOffsets_;  // 原始字节偏移（序列化用）

    float rowStep_ = 0.7f;
    float depthMin_ = 100.0f;
    float depthMax_ = 700.0f;
    cv::Rect roi_{0, 0, 0, 0};
    uint32_t entryCount_ = 0;
    uint32_t rowCount_ = 0;
    std::vector<uint8_t> raw_;   // 载入的原始字节（查询期引用）
};

// ============================================================================
// 生成器
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 曲线/外参/参数按调用传入; Generate 一次性生成, 无跨调用累积。
// 并发策略: 每实例非线程安全（§1.4）; 内部 OpenMP 行级并行安全。
// ==============================

class SCANNER_API CurveMapGenerator {
public:
    static constexpr const char* kLogTag = "12b-CurveMap";

    explicit CurveMapGenerator(const CurveMapParams& params = {});
    ~CurveMapGenerator() = default;
    CurveMapGenerator(const CurveMapGenerator&) = delete;
    CurveMapGenerator& operator=(const CurveMapGenerator&) = delete;

    CurveMapResult Generate(const CurveMapInput& input);

    void SetParams(const CurveMapParams& params);
    const CurveMapParams& GetParams() const;

    void Warmup(int /*rows*/, int /*cols*/) {}
    void Destroy() {}

private:
    CurveMapParams params_;
};

OperatorInfo getCurveMapInfo();

} // namespace calib
