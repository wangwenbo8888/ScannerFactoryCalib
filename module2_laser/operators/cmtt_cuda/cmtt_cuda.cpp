// cmtt_cuda.cpp — CMTT CUDA 试验版编排实现
//
// 流程复刻 operators/curve_map_temp_table/curve_map_temp_table.cpp（未改原文件）:
//   参考档(CUDA, 零插值路径) → blob 头直解冻结基准 → FNV gridHash →
//   温度网格逐档(at()插值 → rescaleCurve 重标 → CUDA 生成 → cmttTierGridMatches
//   冻结断言) → CmttBuilder 拼装 sidecar。
// 档级双执行流流水线（2026-11-14，原串行循环）: 单档 ~340ms = GPU 求交/排序
// ~200ms + CPU 打包/序列化 ~140ms, 串行时 GPU/CPU 交替空转；原子领档双流
// （一线程 CPU 打包时另一线程喂 GPU）总时间趋近 GPU-only。逐档独立实例 +
// 默认流全局串行 + 档间零依赖 → sidecar 与串行版逐字节一致。

#include "cmtt_cuda.h"
#include "common/calib_logging.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <thread>

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
        CurveMapCuda gen(cmp);   // 参考档专用（非参考档见 runTier 内每档独立实例）
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

        // ---- 温度网格逐档（2026-11-14: 档级双执行流流水线, 原子领档动态自调度）----
        // 单档 ~340ms = GPU 求交/排序 ~200ms + CPU 打包/序列化 ~140ms, 原串行循环
        // 中 GPU 与 CPU 交替空转。双执行流: 一线程做某档 CPU 打包时另一线程提交
        // 下一档 GPU 工作 → 总时间趋近 GPU-only。安全性:
        //   ① 每档独立 CurveMapCuda 实例（GPU 资源零共享, 见 runTier）;
        //   ② CUDA 默认流全局串行——多线程提交自动排队, 无需显式同步;
        //   ③ 档间零数据依赖, 各档只写自己的 tiers[idx]/tierBlobs[idx] 槽位
        //      → 结果与完成序无关, sidecar 与串行版逐字节一致。
        const int nHalf = static_cast<int>(std::lround(
            static_cast<double>(params_.tempHalfRange) / static_cast<double>(params_.tempStep)));
        // 非对称档位（2026-10-13）: tiersBelow/tiersAbove 双正时覆盖对称半范围路径
        const int kLow  = (params_.tiersBelow > 0) ? -params_.tiersBelow : -nHalf;
        const int kHigh = (params_.tiersAbove > 0) ?  params_.tiersAbove :  nHalf;
        const size_t total = static_cast<size_t>(kHigh - kLow) + 1;
        result.tiers.resize(total);
        std::vector<std::vector<uint8_t>> tierBlobs(total);

        // 参考档（k==0）主线程先行直填: 复用循环前参考档产物（与 CPU 版
        // curve_map_temp_table.cpp 同型——冻结基准即由 refR 解析而来, 网格断言
        // 对其平凡成立 → 省一次冗余生成）; 线程体跳过该档。
        const size_t refIdx = static_cast<size_t>(-kLow);
        {
            const InterpParams prm = interp.atReference();
            CurveMapTempTierResult tier;
            tier.temperature = params_.referenceTemp;
            uint8_t flags = kCmttFlagReference;
            if (prm.clamped) flags |= kCmttFlagClamped;
            tier.entryCount = refR.entryCount;
            tier.rowCountWithData = refR.rowCountWithData;
            tier.ok = true;
            tier.flags = flags;
            result.tiers[refIdx] = std::move(tier);
            tierBlobs[refIdx] = std::move(refR.tableBytes);
        }

        // 单档生成体（原串行循环体提取, 仅非参考档; 各档独立 idx 槽位写入）
        auto runTier = [&](size_t idx) {
            const double temp = params_.referenceTemp
                              + static_cast<double>(static_cast<int>(idx) + kLow)
                                * static_cast<double>(params_.tempStep);
            CurveMapTempTierResult tier;
            tier.temperature = temp;
            uint8_t flags = uint8_t{0};

            std::vector<uint8_t> blob;
            bool ok = false;
            try {
                const InterpParams prm = interp.at(temp);   // 查询纯 const, 线程安全
                if (prm.clamped) flags |= kCmttFlagClamped;

                // Task 0 像素坐标契约: s = f'/f_ref, m = pp' − s·pp_ref
                const double s = prm.f / refP.f;
                const double mx = prm.principalPoint.x - s * refP.principalPoint.x;
                const double my = prm.principalPoint.y - s * refP.principalPoint.y;
                std::vector<ImplicitCurve> useCurves;
                useCurves.reserve(curves.size());
                for (const auto& c : curves)
                    useCurves.push_back(rescaleCurve(c, s, mx, my));
                // 每档独立 CurveMapCuda 实例（GPU 资源零共享; 与参考档实例及
                // 其他档实例互不可见 → 并发安全性仅依赖 CUDA 运行时与默认流）
                CurveMapParams cmpT;
                cmpT.epipolarRowStep = params_.rowStep;
                CurveMapCuda genT(cmpT);
                // 基线景深窗接线口径同参考档（语义别名, 行为逐位不变）
                genT.SetBaselineDepthWindow(params_.depthMin, params_.depthMax);
                CurveMapResult r = genT.Generate(
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
            } catch (...) {   // 线程体兜底: 非标准异常不外逸（计该档缺失, 防 terminate）
                CALIB_LOG_WARN("tier {}C unknown exception, degrade to missing", temp);
                ok = false; blob.clear();
                tier.entryCount = 0; tier.rowCountWithData = 0;
            }
            if (!ok) flags |= kCmttFlagMissing;
            tier.ok = ok;
            tier.flags = flags;
            result.tiers[idx] = std::move(tier);
            tierBlobs[idx] = std::move(blob);
        };

        // 线程数: tierThreads 1=串行回退; 0/负=自动→2（默认流全局串行天花板下
        // 2 执行流即饱和: 一流喂 GPU、一流 CPU 打包, 更多只增默认流争用——与
        // CPU 版"0=min(硬件线程-2,16)"语义不同, GPU 编排按物理上限取值）;
        // >=2=精确值（仍受 total 封顶）。主线程参与领档（主+worker 双执行流）。
        int wantT = params_.tierThreads;
        if (wantT <= 0) wantT = 2;
        const size_t T = std::min<size_t>(static_cast<size_t>(wantT), total);
        if (T <= 1) {
            // 纯串行路径（tierThreads=1）: 与改前串行循环同构
            for (size_t idx = 0; idx < total; ++idx) {
                if (idx == refIdx) continue;
                runTier(idx);
            }
        } else {
            CALIB_LOG_INFO("CMTT(CUDA) tier pipeline: {} tiers on {} streams "
                           "(dynamic self-scheduling)",
                           static_cast<int>(total), static_cast<int>(T));
            std::atomic<size_t> next{0};
            auto worker = [&] {
                for (;;) {
                    const size_t idx = next.fetch_add(1);
                    if (idx >= total) break;
                    if (idx == refIdx) continue;
                    runTier(idx);
                }
            };
            std::vector<std::thread> pool;
            pool.reserve(T - 1);
            for (size_t w = 0; w + 1 < T; ++w)
                pool.emplace_back(worker);
            worker();   // 主线程也是执行流之一
            for (auto& th : pool) th.join();
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
