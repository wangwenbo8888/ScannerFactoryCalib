// cmtt_cuda.cpp — CMTT CUDA 试验版编排实现
//
// 流程复刻 operators/curve_map_temp_table/curve_map_temp_table.cpp（未改原文件）:
//   参考档(CUDA, 零插值路径) → blob 头直解冻结基准 → FNV gridHash →
//   温度网格逐档(at()插值 → rescaleCurve 重标 → CUDA 生成 → cmttTierGridMatches
//   冻结断言) → CmttBuilder 拼装 sidecar。
// 档级串行（原版 omp 档级并行；GPU 单档内部已并行，串行编排足够吃满 GPU）。

#include "cmtt_cuda.h"
#include "common/calib_logging.h"

#include <cmath>
#include <cstring>

using namespace calib;

namespace {

// FNV-1a 逐字段（与原版同式）
constexpr uint64_t kFnvOffset64 = 14695981039346656037ULL;
inline void fnvMix(uint64_t& h, const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
}

/// 参考档 blob 前 52B 直解（复刻原 parseBlobHeader 布局）
struct BlobHeaderV1 {
    uint32_t magic, version;
    float rowStep, depthMin, depthMax;
    int32_t roiX, roiY, roiW, roiH;
    uint32_t rowCount, entryCount, rowMin, dataRows;
};
bool parseBlobHeader(const uint8_t* p, size_t n, BlobHeaderV1& out) {
    if (n < sizeof(BlobHeaderV1)) return false;
    std::memcpy(&out, p, sizeof(BlobHeaderV1));
    return out.magic == CurveMapTable::kMagic && out.version == 1;
}

/// 单档 CurveMapInput 组装（复刻原 makeTierInput 口径）
CurveMapInput makeTierInput(const std::vector<ImplicitCurve>& cs,
                            const InterpParams& prm,
                            const cv::Size& imageSize,
                            const cv::Rect& roi) {
    CurveMapInput in;
    in.curves = cs;
    in.f = prm.f;
    in.principalPoint = prm.principalPoint;
    in.baseline = prm.baseline;
    in.virtualT = prm.virtualT;
    in.imageSize = imageSize;
    in.roi = roi;
    return in;
}

} // namespace

CALIB_DEFINE_LOG_TAG(XD, CmttCuda);

CmttCudaGenerator::CmttCudaGenerator(const CurveMapTempTableGenParams& params)
    : params_(params)
{
    params_.validate();
}

CurveMapTempTableResult CmttCudaGenerator::Generate(
    const std::vector<ImplicitCurve>& curves,
    const StereoRectifyTempTableResult& rectifyTable,
    const LaserExtrinsicCompensateCPUResult& laserExtrinTable,
    const cv::Size& imageSize)
{
    CurveMapTempTableResult result;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        TempParamInterpolator interp(rectifyTable, laserExtrinTable);

        CurveMapParams cmp;
        cmp.epipolarRowStep = params_.rowStep;
        CurveMapCuda gen(cmp);
        // 基线景深窗显式接线（2026-10-09）：语义别名——与直写 depthMin/Max 同一
        // 真相源，行为逐位不变；概念定义见 curve_map_cuda.h / 设计文档 §2
        gen.SetBaselineDepthWindow(params_.depthMin, params_.depthMax);

        // ---- 参考档（零插值路径） ----
        const InterpParams refP = interp.atReference();
        CurveMapResult refR = gen.Generate(
            makeTierInput(curves, refP, imageSize, refP.roi));
        if (!refR.success) {
            result.success = false;
            result.message = refR.message;
            CALIB_LOG_ERROR("reference tier failed, no sidecar: {}", refR.message);
            return result;
        }

        // ---- 冻结基准 + gridHash ----
        BlobHeaderV1 hdr{};
        if (!parseBlobHeader(refR.tableBytes.data(), refR.tableBytes.size(), hdr)) {
            result.success = false;
            result.message = "reference tier blob header invalid";
            return result;
        }
        CmttFrozenGrid frozen;
        frozen.rowStep = hdr.rowStep;
        frozen.roi = cv::Rect(hdr.roiX, hdr.roiY, hdr.roiW, hdr.roiH);
        frozen.rowCount = hdr.rowCount;
        frozen.rowMin = hdr.rowMin;
        frozen.depthMin = hdr.depthMin;
        frozen.depthMax = hdr.depthMax;

        uint64_t gridHash = kFnvOffset64;
        fnvMix(gridHash, &hdr.rowStep, sizeof(hdr.rowStep));
        const int32_t roi4[4] = {hdr.roiX, hdr.roiY, hdr.roiW, hdr.roiH};
        fnvMix(gridHash, roi4, sizeof(roi4));
        fnvMix(gridHash, &hdr.rowMin, sizeof(hdr.rowMin));
        fnvMix(gridHash, &hdr.rowCount, sizeof(hdr.rowCount));
        fnvMix(gridHash, &hdr.depthMin, sizeof(hdr.depthMin));
        fnvMix(gridHash, &hdr.depthMax, sizeof(hdr.depthMax));

        // ---- 温度网格逐档 ----
        const int nHalf = static_cast<int>(std::lround(
            static_cast<double>(params_.tempHalfRange) / static_cast<double>(params_.tempStep)));
        // 非对称档位（2026-10-13）: tiersBelow/tiersAbove 双正时覆盖对称半范围路径
        const int kLow  = (params_.tiersBelow > 0) ? -params_.tiersBelow : -nHalf;
        const int kHigh = (params_.tiersAbove > 0) ?  params_.tiersAbove :  nHalf;
        const size_t total = static_cast<size_t>(kHigh - kLow) + 1;
        result.tiers.resize(total);
        std::vector<std::vector<uint8_t>> tierBlobs(total);

        for (int k = kLow; k <= kHigh; ++k) {
            const size_t idx = static_cast<size_t>(k - kLow);
            const double temp = params_.referenceTemp + k * static_cast<double>(params_.tempStep);
            const bool isRef = (k == 0);

            CurveMapTempTierResult tier;
            tier.temperature = temp;
            uint8_t flags = isRef ? kCmttFlagReference : uint8_t{0};

            std::vector<uint8_t> blob;
            bool ok = false;
            try {
                const InterpParams prm = isRef ? interp.atReference() : interp.at(temp);
                if (prm.clamped) flags |= kCmttFlagClamped;

                std::vector<ImplicitCurve> useCurves;
                if (isRef) {
                    useCurves = curves;               // 参考档原始曲线
                } else {
                    // Task 0 像素坐标契约: s = f'/f_ref, m = pp' − s·pp_ref
                    const double s = prm.f / refP.f;
                    const double mx = prm.principalPoint.x - s * refP.principalPoint.x;
                    const double my = prm.principalPoint.y - s * refP.principalPoint.y;
                    useCurves.reserve(curves.size());
                    for (const auto& c : curves)
                        useCurves.push_back(rescaleCurve(c, s, mx, my));
                }
                CurveMapResult r = gen.Generate(
                    makeTierInput(useCurves, prm, imageSize, frozen.roi));
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
            } catch (const std::exception& e) {
                CALIB_LOG_WARN("tier {}C exception, degrade to missing: {}", temp, e.what());
                ok = false; blob.clear();
                tier.entryCount = 0; tier.rowCountWithData = 0;
            }
            if (!ok) flags |= kCmttFlagMissing;
            tier.ok = ok;
            tier.flags = flags;
            result.tiers[idx] = std::move(tier);
            tierBlobs[idx] = std::move(blob);
        }

        // ---- 拼装 sidecar ----
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

        result.success = true;
        result.gridHash = gridHash;
        result.clampedRatio = total ? static_cast<float>(clamped) / static_cast<float>(total) : 0.f;

        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        CALIB_LOG_INFO("CMTT(CUDA) done: tiers={} ok={} clamped={:.4f} total={}ms",
                       static_cast<int>(total),
                       static_cast<int>(total) - missing, result.clampedRatio, ms);
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Exception: ") + e.what();
        CALIB_LOG_ERROR("CMTT(CUDA) exception: {}", e.what());
    }
    return result;
}
