// curve_map_temp_table.cpp — curve_map 温度表批量生成器实现
//
// 设计: docs/plans/2026-09-01-curve-map-temp-table-design.md（Task 4; 硬规则 #1/#2/#3/#4/#6/#8）
// 重标: Task 0 像素坐标契约——q′ = s·q + m（s = f_T/f_ref, m = pp_T − s·pp_ref）,
//       一般式见 rescaleCurve（m=0 自动退化为纯缩放, 鲁棒于主点不严格同比例）。
// 内存: 档 blob 生成期 move-through 至 CmttBuilder → sidecarBytes（全程只一份）。

#include "curve_map_temp_table.h"
#include "common/calib_logging.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace calib {

CALIB_DEFINE_LOG_TAG(12d, CurveMapTempTable);

namespace {

// ---- FNV-1a 64（冻结栅格身份哈希; 自包含 ~15 行） ----
constexpr uint64_t kFnvOffset64 = 14695981039346656037ull;
constexpr uint64_t kFnvPrime64 = 1099511628211ull;

void fnvMix(uint64_t& h, const void* p, size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= kFnvPrime64;
    }
}

// 与 curve_map.cpp 序列化 HeaderV1 同布局（该处文件私有无法 include, 按契约复刻）
struct BlobHeaderV1 {
    uint32_t magic;
    uint32_t version;
    float    rowStep;
    float    depthMin;
    float    depthMax;
    int32_t  roiX, roiY, roiW, roiH;
    uint32_t rowCount;
    uint32_t entryCount;
    uint32_t rowMin;
    uint32_t dataRows;
};
static_assert(sizeof(BlobHeaderV1) == 52, "CMPU HeaderV1 layout drifted");

/// blob 头自洽解析（header-only——不做 CurveMapTable::Load 全量走读）。
/// 防线: ①截断/magic/version/rowStep>0; ②派生一致性
/// |rowMin − floor(roi.y/rowStep + 0.5)| ≤ 1（金标准实例: roi.y=286, rowStep=0.7
/// → 408.57 → rowMin=408）。同尺寸字段重排（rowMin↔dataRows）会被此关系大声
/// 拒绝——这是对"复刻布局漂移"的防线; 根治方案为 factory_calib 侧人工加
/// rowMin()/depth 访问器（设计文档遗留项）。
bool parseBlobHeader(const uint8_t* blob, size_t size, BlobHeaderV1& h,
                     std::string* why) {
    if (size < sizeof(BlobHeaderV1)) {
        if (why) *why = "blob truncated (header)";
        return false;
    }
    std::memcpy(&h, blob, sizeof(h));
    if (h.magic != CurveMapTable::kMagic) {
        if (why) *why = "bad magic";
        return false;
    }
    if (h.version != 1u) {
        if (why) *why = "unsupported version";
        return false;
    }
    if (!(h.rowStep > 0.0f)) {
        if (why) *why = "bad rowStep";
        return false;
    }
    const double expect = std::floor(
        static_cast<double>(h.roiY) / static_cast<double>(h.rowStep) + 0.5);
    if (std::fabs(static_cast<double>(h.rowMin) - expect) > 1.0) {
        if (why)
            *why = fmt::format("rowMin {} inconsistent with roi.y {}/rowStep {} (derived {})",
                               h.rowMin, h.roiY, h.rowStep, expect);
        return false;
    }
    return true;
}

/// 单档 CurveMapInput 组装（f/pp/B/virtualT/roi/virtualK 均温档参数）
CurveMapInput makeTierInput(const std::vector<ImplicitCurve>& curves,
                            const InterpParams& prm, const cv::Size& imageSize,
                            const cv::Rect& roi) {
    CurveMapInput in;
    in.curves = curves;
    in.f = prm.f;
    in.principalPoint = prm.principalPoint;
    in.baseline = prm.baseline;
    in.imageSize = imageSize;
    in.roi = roi;
    in.virtualT = prm.virtualT;
    in.virtualK = cv::Matx33d(prm.f, 0.0, prm.principalPoint.x,
                              0.0, prm.f, prm.principalPoint.y,
                              0.0, 0.0, 1.0);
    return in;
}

/// 档级并行线程数解析: >0=精确值; 0=自动 min(硬件线程-2, 16)（下限 1）; <=-1 视为 1。
/// 无 OpenMP 构建退化为 1（pragma 缺席, 纯串行）。
int resolveTierLanes(int tierThreads) {
    if (tierThreads > 0) return tierThreads;
#if defined(_OPENMP)
    if (tierThreads == 0) {
        int hw = omp_get_num_procs();
        if (hw < 1) hw = 1;
        int t = hw - 2;
        if (t < 1) t = 1;
        return t > 16 ? 16 : t;
    }
#endif
    return 1;
}

} // namespace

// ============================================================================
// 参数校验
// ============================================================================

void CurveMapTempTableGenParams::validate() const {
    if (std::isnan(referenceTemp))
        throw std::invalid_argument(
            "CurveMapTempTableGenParams::referenceTemp must not be NaN");
    if (!(rowStep > 0.0f))
        throw std::invalid_argument(
            "CurveMapTempTableGenParams::rowStep must be > 0 (inherit from calib JSON)");
    if (!(tempStep > 0.0f))
        throw std::invalid_argument("CurveMapTempTableGenParams::tempStep must be > 0");
    if (!(tempHalfRange >= 0.0f))
        throw std::invalid_argument(
            "CurveMapTempTableGenParams::tempHalfRange must be >= 0");
    if (!(depthMin > 0.0f) || !(depthMax > depthMin))
        throw std::invalid_argument(
            "CurveMapTempTableGenParams: invalid depth window (need 0 < min < max)");
    // 温网整除性: 端点含入依赖 halfRange 为 step 整数倍（不可整除 → 端点漂移, 拒绝）
    const double nGrid =
        static_cast<double>(tempHalfRange) / static_cast<double>(tempStep);
    if (std::fabs(nGrid - std::floor(nGrid + 0.5)) > 1e-6)
        throw std::invalid_argument(
            "CurveMapTempTableGenParams: tempHalfRange " +
            std::to_string(tempHalfRange) + " is not an integer multiple of tempStep " +
            std::to_string(tempStep));
}

nlohmann::json CurveMapTempTableGenParams::toJson() const {
    return {
        {"referenceTemp", referenceTemp},
        {"tempHalfRange", tempHalfRange},
        {"tempStep", tempStep},
        {"rowStep", rowStep},
        {"depthMin", depthMin},
        {"depthMax", depthMax},
        {"tierThreads", tierThreads}
    };
}

CurveMapTempTableGenParams CurveMapTempTableGenParams::fromJson(const nlohmann::json& j) {
    CurveMapTempTableGenParams p;
    if (j.contains("referenceTemp")) p.referenceTemp = j.at("referenceTemp").get<double>();
    if (j.contains("tempHalfRange")) p.tempHalfRange = j.at("tempHalfRange").get<float>();
    if (j.contains("tempStep"))      p.tempStep      = j.at("tempStep").get<float>();
    if (j.contains("rowStep"))       p.rowStep       = j.at("rowStep").get<float>();
    if (j.contains("depthMin"))      p.depthMin      = j.at("depthMin").get<float>();
    if (j.contains("depthMax"))      p.depthMax      = j.at("depthMax").get<float>();
    if (j.contains("tierThreads"))   p.tierThreads   = j.at("tierThreads").get<int>();
    return p;
}

// ============================================================================
// 曲线重标（Task 0 契约）
// ============================================================================

ImplicitCurve rescaleCurve(const ImplicitCurve& curve, double s, double mx, double my) {
    // 前置条件: s>0 且有限（s = f_T/f_ref, 两侧 f 逐档为正 → 违例即上游参数已坏）
    if (!(s > 0.0) || !std::isfinite(s))
        throw std::invalid_argument("rescaleCurve: scale s must be > 0 and finite");
    // 参考温像素曲线 F(q)=0, 温档映射 q′ = s·q + m ⇒ q = (q′−m)/s 代入后乘 s²:
    //   a′=a  b′=b  c′=c
    //   d′ = s·d − (2a·mx + b·my)
    //   e′ = s·e − (b·mx + 2c·my)
    //   f′ = s²·f − s·(d·mx + e·my) + (a·mx² + b·mx·my + c·my²)
    const double a = curve.coeffs[0], b = curve.coeffs[1], c = curve.coeffs[2];
    const double d = curve.coeffs[3], e = curve.coeffs[4], f0 = curve.coeffs[5];
    ImplicitCurve r = curve;
    r.coeffs[3] = s * d - (2.0 * a * mx + b * my);
    r.coeffs[4] = s * e - (b * mx + 2.0 * c * my);
    r.coeffs[5] = s * s * f0 - s * (d * mx + e * my)
                + (a * mx * mx + b * mx * my + c * my * my);
    return r;
}

// ============================================================================
// 冻结网格断言（硬规则 #1 防线, header-only）
// ============================================================================

bool cmttTierGridMatches(const uint8_t* blob, size_t size,
                         const CmttFrozenGrid& frozen, std::string* why) {
    BlobHeaderV1 h{};
    if (!parseBlobHeader(blob, size, h, why))
        return false;
    if (h.rowStep != frozen.rowStep) {
        if (why)
            *why = fmt::format("rowStep {} != frozen {}", h.rowStep, frozen.rowStep);
        return false;
    }
    if (h.roiX != frozen.roi.x || h.roiY != frozen.roi.y ||
        h.roiW != frozen.roi.width || h.roiH != frozen.roi.height) {
        if (why)
            *why = fmt::format("roi ({},{},{},{}) != frozen ({},{},{},{})",
                               h.roiX, h.roiY, h.roiW, h.roiH,
                               frozen.roi.x, frozen.roi.y,
                               frozen.roi.width, frozen.roi.height);
        return false;
    }
    if (h.rowCount != frozen.rowCount) {
        if (why)
            *why = fmt::format("rowCount {} != frozen {}", h.rowCount, frozen.rowCount);
        return false;
    }
    if (h.rowMin != frozen.rowMin) {
        if (why)
            *why = fmt::format("rowMin {} != frozen {}", h.rowMin, frozen.rowMin);
        return false;
    }
    if (h.depthMin != frozen.depthMin || h.depthMax != frozen.depthMax) {
        if (why)
            *why = fmt::format("depth [{},{}] != frozen [{},{}]",
                               h.depthMin, h.depthMax, frozen.depthMin, frozen.depthMax);
        return false;
    }
    return true;
}

// ============================================================================
// Generator
// ============================================================================

CurveMapTempTableGenerator::CurveMapTempTableGenerator(
    const CurveMapTempTableGenParams& params)
    : params_(params)
{
    params_.validate();
}

CurveMapTempTableResult CurveMapTempTableGenerator::Generate(
    const std::vector<ImplicitCurve>& curves,
    const StereoRectifyTempTableResult& rectifyTable,
    const LaserExtrinsicCompensateCPUResult& laserExtrinTable,
    const cv::Size& imageSize)
{
    CurveMapTempTableResult result;

    // ---- 1. 插值器（构造校验异常穿透） ----
    TempParamInterpolator interp(rectifyTable, laserExtrinTable);
    if (interp.referenceTemp() != params_.referenceTemp) {
        CALIB_LOG_ERROR("reference temp mismatch: params {}C vs tables {}C",
                        params_.referenceTemp, interp.referenceTemp());
        throw std::invalid_argument(
            "CurveMapTempTableGenerator: params referenceTemp (" +
            std::to_string(params_.referenceTemp) +
            "C) != source tables referenceTemp (" +
            std::to_string(interp.referenceTemp()) + "C)");
    }

    // ---- 2. 参考档先行（硬规则 #4: 零插值＋原始曲线不重标） ----
    // CurveMapGenerator 每档独立实例（§1.4 契约）: 档级并行下不跨档复用实例。
    const InterpParams refP = interp.atReference();
    CurveMapResult refR;
    {
        CurveMapParams cmp;
        cmp.epipolarRowStep = params_.rowStep;
        cmp.depthMin = params_.depthMin;
        cmp.depthMax = params_.depthMax;
        CurveMapGenerator gen(cmp);
        refR = gen.Generate(makeTierInput(curves, refP, imageSize, refP.roi));
    }
    if (!refR.success) {
        result.success = false;
        result.message = refR.message;   // 透传（硬规则 #3）
        CALIB_LOG_ERROR("reference tier failed, no sidecar: {}", refR.message);
        return result;
    }

    // ---- 3. 冻结基准（参考档 blob 头 52B 直解, 不做全量 Load） ----
    BlobHeaderV1 hdr{};
    {
        std::string why;
        if (!parseBlobHeader(refR.tableBytes.data(), refR.tableBytes.size(), hdr, &why)) {
            result.success = false;
            result.message = "reference tier blob header invalid: " + why;
            CALIB_LOG_ERROR("{}", result.message);
            return result;
        }
    }
    CmttFrozenGrid frozen;
    frozen.rowStep = hdr.rowStep;
    frozen.roi = cv::Rect(hdr.roiX, hdr.roiY, hdr.roiW, hdr.roiH);
    frozen.rowCount = hdr.rowCount;
    frozen.rowMin = hdr.rowMin;
    frozen.depthMin = hdr.depthMin;
    frozen.depthMax = hdr.depthMax;

    // gridHash: FNV-1a 逐字段（rowStep/roi 四元/rowMin/rowCount/depthMin/depthMax;
    // 深度取 blob 头解析值——与 blob 实际承载口径一致, 不随 params 记忆漂移）
    uint64_t gridHash = kFnvOffset64;
    fnvMix(gridHash, &hdr.rowStep, sizeof(hdr.rowStep));
    const int32_t roi4[4] = {hdr.roiX, hdr.roiY, hdr.roiW, hdr.roiH};
    fnvMix(gridHash, roi4, sizeof(roi4));
    fnvMix(gridHash, &hdr.rowMin, sizeof(hdr.rowMin));
    fnvMix(gridHash, &hdr.rowCount, sizeof(hdr.rowCount));
    fnvMix(gridHash, &hdr.depthMin, sizeof(hdr.depthMin));
    fnvMix(gridHash, &hdr.depthMax, sizeof(hdr.depthMax));

    // ---- 4. 温度网格（含端点, 参考档在内; temp = ref + k·step 逐位同算式） ----
    // 档级 omp 并行: 每档独占下标写（tiers/tierBlobs 预 resize）; 嵌套并行默认禁用
    // （MSVC OMP 2.0）→ 内层 CurveMapGenerator 行级 omp 自动单线程串行, 行计算序
    // 与纯串行一致 → 线程数无关确定性（硬规则 #8 维持; 勿开 OMP_NESTED）。
    const int nHalf = static_cast<int>(std::lround(
        static_cast<double>(params_.tempHalfRange) / static_cast<double>(params_.tempStep)));
    const size_t total = static_cast<size_t>(2 * nHalf) + 1;
    result.tiers.resize(total);                   // 下标写（档序 = 温度升序）
    std::vector<std::vector<uint8_t>> tierBlobs;  // 档载荷: move-through 至 sidecar
    tierBlobs.resize(total);

    const int lanes = resolveTierLanes(params_.tierThreads);
    CALIB_LOG_INFO("tier-parallel: tiers={} lanes={}", static_cast<int>(total), lanes);

#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic) num_threads(lanes)
#endif
    for (int k = -nHalf; k <= nHalf; ++k) {
        const size_t idx = static_cast<size_t>(k + nHalf);
        const double temp = params_.referenceTemp + k * static_cast<double>(params_.tempStep);
        const bool isRef = (k == 0);

        CurveMapTempTierResult tier;
        tier.temperature = temp;
        uint8_t flags = isRef ? kCmttFlagReference : uint8_t{0};

        std::vector<uint8_t> blob;                 // ok 档载荷; missing 档留空
        bool ok = false;
        try {
            const InterpParams prm = isRef ? refP : interp.at(temp);
            if (prm.clamped) flags |= kCmttFlagClamped;

            if (isRef) {
                // 复用步骤 2 产物（零插值路径）; 冻结基准即由它解析, 断言对其平凡成立
                ok = true;
                tier.entryCount = refR.entryCount;
                tier.rowCountWithData = refR.rowCountWithData;
                blob = std::move(refR.tableBytes);
            } else {
                // 曲线重标（Task 0 契约）: s = f′/f_ref, m = pp′ − s·pp_ref
                const double s = prm.f / refP.f;
                const double mx = prm.principalPoint.x - s * refP.principalPoint.x;
                const double my = prm.principalPoint.y - s * refP.principalPoint.y;
                std::vector<ImplicitCurve> rescaled;
                rescaled.reserve(curves.size());
                for (const auto& c : curves)
                    rescaled.push_back(rescaleCurve(c, s, mx, my));

                CurveMapParams cmp;                // 每档独立实例（§1.4 契约）
                cmp.epipolarRowStep = params_.rowStep;
                cmp.depthMin = params_.depthMin;
                cmp.depthMax = params_.depthMax;
                CurveMapGenerator gen(cmp);
                CurveMapResult r = gen.Generate(
                    makeTierInput(rescaled, prm, imageSize, frozen.roi));   // roi 冻结基准
                if (!r.success) {
                    CALIB_LOG_WARN("tier {}C generate failed: {}", temp, r.message);
                } else {
                    std::string why;
                    if (!cmttTierGridMatches(r.tableBytes.data(), r.tableBytes.size(),
                                             frozen, &why)) {
                        CALIB_LOG_WARN("tier {}C frozen-grid violation, degrade to missing: {}",
                                       temp, why);
                    } else {
                        ok = true;
                        tier.entryCount = r.entryCount;
                        tier.rowCountWithData = r.rowCountWithData;
                        blob = std::move(r.tableBytes);
                    }
                }
            }
        } catch (const std::exception& e) {
            // omp 区内绝不让异常逃逸: 该档降级 missing（与现失败宽容语义一致）
            CALIB_LOG_WARN("tier {}C exception, degrade to missing: {}", temp, e.what());
            ok = false;
            blob.clear();
            tier.entryCount = 0;
            tier.rowCountWithData = 0;
        } catch (...) {
            CALIB_LOG_WARN("tier {}C unknown exception, degrade to missing", temp);
            ok = false;
            blob.clear();
            tier.entryCount = 0;
            tier.rowCountWithData = 0;
        }

        if (!ok) flags |= kCmttFlagMissing;   // fail → bit1, blob 空
        tier.ok = ok;
        tier.flags = flags;
        result.tiers[idx] = std::move(tier);
        tierBlobs[idx] = std::move(blob);
    }

    // ---- 5. sidecar 拼装（blob move-through: 数据只驻 sidecarBytes 一份） ----
    int missing = 0, clamped = 0;
    for (const auto& t : result.tiers) {
        if (!t.ok) ++missing;
        if (t.flags & kCmttFlagClamped) ++clamped;
    }

    CmttBuilder builder;
    builder.setHeader(static_cast<int32_t>(std::lround(params_.referenceTemp * 10.0)),
                      static_cast<uint16_t>(std::lround(params_.tempStep * 10.0)),
                      frozen.rowStep, frozen.depthMin, frozen.depthMax,
                      frozen.roi, frozen.rowMin, frozen.rowCount, gridHash);
    for (size_t i = 0; i < result.tiers.size(); ++i)
        builder.addTier(static_cast<int32_t>(std::lround(result.tiers[i].temperature * 10.0)),
                        std::move(tierBlobs[i]), result.tiers[i].entryCount,
                        result.tiers[i].flags);
    result.sidecarBytes = builder.build();

    // ---- 6/7. 质量与统计 ----
    result.success = true;
    result.gridHash = gridHash;
    result.clampedRatio =
        total > 0 ? static_cast<float>(clamped) / static_cast<float>(total) : 0.f;
    result.qualityFlag =
        (missing == 0 && clamped == 0) ? QualityFlag::Normal : QualityFlag::Warning;
    result.message = fmt::format("{} tiers: {} ok, {} missing, {} clamped",
                                 total, total - static_cast<size_t>(missing), missing, clamped);
    CALIB_LOG_INFO("Generate: {}", result.message);

    return result;
}

// ============================================================================
// Result 便捷访问
// ============================================================================

int CurveMapTempTableResult::okCount() const {
    int n = 0;
    for (const auto& t : tiers)
        if (t.ok) ++n;
    return n;
}

size_t CurveMapTempTableResult::referenceIndex() const {
    for (size_t i = 0; i < tiers.size(); ++i)
        if (tiers[i].flags & kCmttFlagReference) return i;
    return std::numeric_limits<size_t>::max();
}

} // namespace calib
