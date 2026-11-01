// curve_map_cuda.cu — curve_map 单档生成器 CUDA 试验实现
//
// 移植自 operators/curve_map/curve_map.cpp（未改动原文件）:
//   - C.1/C.2 几何内核逐式复刻（双精度; uL 采样值由 CPU 按原循环累加产生后上传,
//     保证量化输入与 CPU 逐位一致）
//   - GPU: (行×线×uL) 并行求交 → 原子池收集 → thrust 键排序(分组+组内 (uL,d) 全序)
//   - CPU: 排序结果线性扫描 → 差分+位打包+序列化（复刻原 assembly 逻辑, 格式 E.2）
// 编译: -fmad=false（抑制 FMA 缩并, 对齐 MSVC CPU 舍入行为, 提高等价性）

#include "curve_map_cuda.h"
#include "common/calib_logging.h"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace calib;

// ============================================================================
// GPU 侧
// ============================================================================

namespace {

constexpr uint16_t kEscapeCode = 0x3F;
constexpr int kAnchorInterval = 32;

/// GPU 收集的原始条目（12B）
struct GpuRaw {
    uint16_t uLq;
    uint16_t uRq;
    int32_t  dInt;
    uint16_t rowAbs;   // 0..nRows-1（数据行局部号）
    uint16_t line;     // 0..nLines-1
};

/// 键: bucket(=rowAbs*nLines+line, ≤16b) << 48 | uLq(12b) << 36 | (dInt+4096, 13b)
__device__ __forceinline__ unsigned long long makeKey(
    int rowAbs, int line, int nLines, uint16_t uLq, int32_t dInt) {
    const unsigned int bucket =
        static_cast<unsigned int>(rowAbs) * static_cast<unsigned int>(nLines)
        + static_cast<unsigned int>(line);
    return (static_cast<unsigned long long>(bucket) << 48)
         | (static_cast<unsigned long long>(uLq) << 36)
         | (static_cast<unsigned long long>(dInt + 4096) & 0x1FFFULL);
}

/// 求交内核: 一线程一 (行×线×uL 样本)。逐式复刻 intersectLineConic + 过滤链。
__global__ void intersectKernel(
    const double* __restrict__ d_coeffs,   // nLines*6
    const double* __restrict__ d_usamples, // nU（CPU 累加循环产物, 与原实现逐位同）
    int nLines, int nU,
    int rowMin, int rowBase, int nRowsTotal, int rowsInBatch,
    double rowStepD, double invRowStepD,
    double f, double cx, double cy,
    double tx, double ty, double tz, double B,
    int W, int H,
    double depthMin, double depthMax,
    unsigned long long* __restrict__ keys,
    GpuRaw* __restrict__ vals,
    int cap, int* __restrict__ count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rowsInBatch * nLines * nU;
    if (idx >= total) return;

    const int iU = idx % nU;
    const int tmp = idx / nU;
    const int iL = tmp % nLines;
    const int iR = tmp / nLines;
    const int rowAbs = rowBase + iR;
    if (rowAbs >= nRowsTotal) return;

    const double uLpx = d_usamples[iU];
    // CPU 口径: v = (rowMin + r) * rowStep（绝对行号; rowAbs 仅作分组桶键）
    const double v = (rowMin + rowAbs) * rowStepD;

    const double* c = d_coeffs + (size_t)iL * 6;
    const double A = c[0], Bc = c[1], C = c[2], D = c[3], E = c[4], F0 = c[5];

    // 极点（本行所有极线共过）
    const double Ex = f * tx / tz + cx;
    const double Ey = f * ty / tz + cy;

    const double du = Ex - uLpx;
    const double dv = Ey - v;

    // F(u0+t·du, v0+t·dv) 展开（与 CPU 同式同序）
    const double a = A*du*du + Bc*du*dv + C*dv*dv;
    const double b = 2*A*uLpx*du + Bc*(uLpx*dv + v*du) + 2*C*v*dv + D*du + E*dv;
    const double cc = A*uLpx*uLpx + Bc*uLpx*v + C*v*v + D*uLpx + E*v + F0;

    double hits[2][2];
    int nh = 0;
    if (fabs(a) < 1e-18) {
        if (fabs(b) > 1e-18) {
            const double t = -cc / b;
            hits[0][0] = uLpx + t*du; hits[0][1] = v + t*dv; nh = 1;
        }
    } else {
        const double disc = b*b - 4*a*cc;
        if (disc >= 0.0) {
            const double sq = sqrt(disc);
            const double t1 = (-b + sq) / (2*a);
            hits[0][0] = uLpx + t1*du; hits[0][1] = v + t1*dv; nh = 1;
            if (sq > 1e-15) {
                const double t2 = (-b - sq) / (2*a);
                hits[1][0] = uLpx + t2*du; hits[1][1] = v + t2*dv; nh = 2;
            }
        }
    }

    for (int h = 0; h < nh; ++h) {
        const double q_u = hits[h][0], q_v = hits[h][1];
        if (q_u < 0 || q_u >= (double)W || q_v < 0 || q_v >= (double)H) continue;
        const double qc = q_u - cx;
        const double denom = qc - (uLpx - cx);
        if (fabs(denom) < 1e-9) continue;
        const double Z = (qc * tz - f * tx) / denom;
        if (Z < depthMin || Z > depthMax) continue;
        const double uRpx = uLpx - f * B / Z;
        if (uRpx < 0 || uRpx >= (double)W) continue;
        const uint16_t uLq = (uint16_t)llround(uLpx * invRowStepD);
        const uint16_t uRq = (uint16_t)llround(uRpx * invRowStepD);
        const int32_t dInt = (int32_t)llround((uLpx - uRpx) * invRowStepD);
        if (uRq >= uLq) continue;
        const int pos = atomicAdd(count, 1);
        if (pos < cap) {
            keys[pos] = makeKey(rowAbs, iL, nLines, uLq, dInt);
            GpuRaw r;
            r.uLq = uLq; r.uRq = uRq; r.dInt = dInt;
            r.rowAbs = (uint16_t)rowAbs; r.line = (uint16_t)iL;
            vals[pos] = r;
        }
    }
}

// ---- 位打包（复刻 curve_map.cpp 匿名域, 原样保留语义） ----
inline uint32_t zigzagEnc(int32_t v) {
    return static_cast<uint32_t>((v << 1) ^ (v >> 31));
}
inline uint16_t packEntry(uint16_t duL, int32_t dd) {
    const uint32_t z = zigzagEnc(dd);
    const uint32_t d10 = std::min(z, 0x3FFu);
    return static_cast<uint16_t>(((duL & 0x3F) << 10) | (d10 & 0x3FF));
}

template <typename T>
void writePOD(std::vector<uint8_t>& out, const T& v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

} // namespace

// ============================================================================
// Host 侧
// ============================================================================

CALIB_DEFINE_LOG_TAG(XC, CurveMapCuda);

CurveMapCuda::CurveMapCuda(const CurveMapParams& params) : params_(params) {
    params_.validate();
}

void CurveMapCuda::SetParams(const CurveMapParams& params) {
    params_ = params;
    params_.validate();
}

// ---- 基线景深窗（语义别名；设计文档 2026-10-09-curve-map-cuda-baseline-depth §3）----
// isfinite 前置堵 NaN 静默放行；临时对象"先窄化→先校验→后提交"，违例时 params_ 原值不动。
void CurveMapCuda::SetBaselineDepthWindow(double d1, double d2) {
    if (!std::isfinite(d1) || !std::isfinite(d2))
        throw std::invalid_argument(
            "SetBaselineDepthWindow: values must be finite (NaN/Inf rejected)");
    CurveMapParams tmp = params_;
    tmp.depthMin = static_cast<float>(d1);
    tmp.depthMax = static_cast<float>(d2);
    tmp.validate();   // 按窄化后 float 值校验 0 < min < max
    params_ = tmp;    // 校验通过才提交
}

std::pair<double, double> CurveMapCuda::GetBaselineDepthWindow() const {
    return { static_cast<double>(params_.depthMin),
             static_cast<double>(params_.depthMax) };
}

CurveMapResult CurveMapCuda::Generate(const CurveMapInput& in) {
    CurveMapResult result;
    result.effectiveParams = params_;
    try {
        // ---- 校验（复刻原 Generate 前置检查） ----
        if (in.curves.empty()) { result.success=false; result.message="no input curves"; return result; }
        if (in.f <= 0 || in.baseline <= 0) { result.success=false; result.message="invalid f/baseline"; return result; }
        if (in.imageSize.width <= 0 || in.imageSize.height <= 0) { result.success=false; result.message="invalid imageSize"; return result; }
        if (in.roi.width <= 0 || in.roi.height <= 0 || in.roi.x < 0 || in.roi.y < 0 ||
            in.roi.x + in.roi.width > in.imageSize.width ||
            in.roi.y + in.roi.height > in.imageSize.height) {
            result.success=false; result.message="invalid roi"; return result;
        }
        if (std::fabs(in.virtualT(2)) < 1e-6) {
            result.success=false; result.message="virtualT.z ~= 0: epipole at infinity unsupported"; return result;
        }

        const double invRowStep = 1.0 / static_cast<double>(params_.epipolarRowStep);
        const double uStep = static_cast<double>(params_.effectiveUSampleStep());
        const int rowCount = static_cast<int>(std::round(in.imageSize.height * invRowStep));
        const int rowMin = std::max(0, static_cast<int>(std::floor(in.roi.y * invRowStep)));
        const int rowMax = std::min(rowCount - 1,
            static_cast<int>(std::ceil((in.roi.y + in.roi.height) * invRowStep)));
        const double uMin = static_cast<double>(in.roi.x);
        const double uMax = static_cast<double>(in.roi.x + in.roi.width) - 1.0;

        const int nRows = rowMax - rowMin + 1;
        const int nLines = static_cast<int>(in.curves.size());

        // uL 采样序列: 逐位复刻 CPU 的累加循环（uLpx += uStep）
        std::vector<double> uSamples;
        for (double u = uMin; u <= uMax + 1e-9; u += uStep) uSamples.push_back(u);
        const int nU = static_cast<int>(uSamples.size());
        if (nU == 0 || nRows <= 0) {
            result.success = false;
            result.message = "empty sampling grid";
            return result;
        }

        // ---- 上传常量 ----
        const double f = in.f, cx = in.principalPoint.x, cy = in.principalPoint.y;
        const double tx = in.virtualT(0), ty = in.virtualT(1), tz = in.virtualT(2);
        const double B = in.baseline;

        std::vector<double> coeffs((size_t)nLines * 6);
        for (int l = 0; l < nLines; ++l)
            std::memcpy(&coeffs[(size_t)l * 6], in.curves[(size_t)l].coeffs, 6 * sizeof(double));

        double *d_coeffs = nullptr, *d_usamples = nullptr;
        unsigned long long* d_keys = nullptr;
        GpuRaw* d_vals = nullptr;
        int* d_count = nullptr;
        cudaMalloc(&d_coeffs, coeffs.size() * sizeof(double));
        cudaMalloc(&d_usamples, nU * sizeof(double));
        cudaMemcpy(d_coeffs, coeffs.data(), coeffs.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_usamples, uSamples.data(), nU * sizeof(double), cudaMemcpyHostToDevice);

        // ---- 分批求交 + 收集 ----
        thrust::device_vector<unsigned long long> allKeys;
        thrust::device_vector<GpuRaw> allVals;
        const int kBatchRows = 128;
        const size_t batchCap = (size_t)kBatchRows * nLines * nU * 2;
        cudaMalloc(&d_keys, batchCap * sizeof(unsigned long long));
        cudaMalloc(&d_vals, batchCap * sizeof(GpuRaw));
        cudaMalloc(&d_count, sizeof(int));
        int h_count = 0;

        for (int rowBase = 0; rowBase < nRows; rowBase += kBatchRows) {
            const int rowsInBatch = std::min(kBatchRows, nRows - rowBase);
            cudaMemset(d_count, 0, sizeof(int));
            // 纯 1D grid: idx 唯一映射 (行×线×uL)（旧版误用 gridDim.y 致重复执行）
            const int totalTasks = rowsInBatch * nLines * nU;
            const dim3 grid((unsigned)((totalTasks + 255) / 256));
            intersectKernel<<<grid, 256, 0>>>(
                d_coeffs, d_usamples, nLines, nU, rowMin, rowBase, nRows, rowsInBatch,
                (double)params_.epipolarRowStep, invRowStep,
                f, cx, cy, tx, ty, tz, B,
                in.imageSize.width, in.imageSize.height,
                (double)params_.depthMin, (double)params_.depthMax,
                d_keys, d_vals, (int)batchCap, d_count);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                cudaFree(d_coeffs); cudaFree(d_usamples);
                cudaFree(d_keys); cudaFree(d_vals); cudaFree(d_count);
                result.success = false;
                result.message = std::string("CUDA kernel error: ") + cudaGetErrorString(err);
                return result;
            }
            cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
            const int kept = std::min(h_count, (int)batchCap);
            if (kept > 0) {
                const size_t old = allKeys.size();
                allKeys.resize(old + (size_t)kept);
                allVals.resize(old + (size_t)kept);
                thrust::copy(thrust::device,
                             d_keys, d_keys + kept, allKeys.begin() + old);
                thrust::copy(thrust::device,
                             d_vals, d_vals + kept, allVals.begin() + old);
            }
        }
        cudaFree(d_coeffs); cudaFree(d_usamples);
        cudaFree(d_keys); cudaFree(d_vals); cudaFree(d_count);

        // ---- 排序（键全序 ⇒ 行×线分组 + 组内 (uL, d) 升序） ----
        thrust::sort_by_key(thrust::device,
                            allKeys.begin(), allKeys.end(), allVals.begin());

        const size_t nEntries = allVals.size();
        std::vector<GpuRaw> ents(nEntries);
        if (nEntries > 0)
            cudaMemcpy(ents.data(), thrust::raw_pointer_cast(allVals.data()),
                       nEntries * sizeof(GpuRaw), cudaMemcpyDeviceToHost);
        allKeys.clear(); allKeys.shrink_to_fit();
        allVals.clear(); allVals.shrink_to_fit();

        // ---- 打包 + 序列化（复刻原 assembly, 线性扫描已排序条目） ----
        struct GroupBuild {
            uint8_t lid; uint16_t anchor;
            std::vector<uint16_t> words; uint32_t count;
        };
        std::vector<std::vector<GroupBuild>> rowGroups((size_t)nRows);
        uint32_t totalEntries = 0;
        uint32_t rowsWithData = 0;

        size_t i = 0;
        while (i < nEntries) {
            const int r = ents[i].rowAbs;
            const int l = ents[i].line;
            size_t j = i;
            GroupBuild g;
            g.lid = static_cast<uint8_t>(l + 1);
            g.anchor = ents[i].uLq;
            uint16_t prevUL = 0;
            int32_t prevDint = 0;
            bool first = true;
            int sinceAnchor = 0;
            while (j < nEntries && ents[j].rowAbs == r && ents[j].line == l) {
                const uint16_t uLq = ents[j].uLq;
                const int32_t dInt = ents[j].dInt;
                const int32_t dd = dInt - prevDint;
                const int32_t du = first ? 0 : (int32_t)(uLq - prevUL);
                const bool overZigzag = zigzagEnc(dd) > 0x3FF;
                const bool needEscape = (du >= (int32_t)kEscapeCode) || overZigzag
                                        || (sinceAnchor >= kAnchorInterval);
                if (needEscape) {
                    g.words.push_back(packEntry(kEscapeCode, 0));
                    g.words.push_back(uLq);
                    g.words.push_back(static_cast<uint16_t>(dInt));
                } else {
                    g.words.push_back(packEntry(static_cast<uint16_t>(du), dd));
                }
                prevUL = uLq; prevDint = dInt;
                first = false;
                sinceAnchor = needEscape ? 0 : sinceAnchor + 1;
                ++j;
            }
            g.count = static_cast<uint32_t>(j - i);
            rowGroups[(size_t)r].push_back(std::move(g));
            totalEntries += static_cast<uint32_t>(j - i);
            // 覆盖统计
            CurveMapCover cv;
            cv.row = static_cast<uint16_t>(rowMin + r);
            cv.lid = static_cast<uint8_t>(l + 1);
            cv.uLmin = ents[i].uLq;
            cv.uLmax = ents[j - 1].uLq;
            cv.count = static_cast<uint32_t>(j - i);
            uint16_t maxGap = 0;
            for (size_t k = i + 1; k < j; ++k)
                maxGap = std::max(maxGap, static_cast<uint16_t>(ents[k].uLq - ents[k-1].uLq));
            cv.maxGapSeen = maxGap;
            result.covers.push_back(cv);
            i = j;
        }
        for (int r = 0; r < nRows; ++r)
            if (!rowGroups[(size_t)r].empty()) ++rowsWithData;
        result.entryCount = totalEntries;
        result.rowCountWithData = rowsWithData;

        // 序列化（E.2: 52B 头 + 绝对行索引 + 行块; 与原实现逐字段同布局）
        std::vector<uint8_t>& buf = result.tableBytes;
        struct HeaderV1 {
            uint32_t magic; uint32_t version;
            float rowStep, depthMin, depthMax;
            int32_t roiX, roiY, roiW, roiH;
            uint32_t rowCount, entryCount, rowMin, dataRows;
        };
        HeaderV1 h{};
        h.magic = CurveMapTable::kMagic;
        h.version = 1;
        h.rowStep = params_.epipolarRowStep;
        h.depthMin = params_.depthMin;
        h.depthMax = params_.depthMax;
        h.roiX = in.roi.x; h.roiY = in.roi.y;
        h.roiW = in.roi.width; h.roiH = in.roi.height;
        h.rowCount = static_cast<uint32_t>(rowCount);
        h.entryCount = totalEntries;
        h.rowMin = static_cast<uint32_t>(rowMin);
        h.dataRows = static_cast<uint32_t>(nRows);
        writePOD(buf, h);
        const size_t indexPos = buf.size();
        std::vector<uint64_t> offsets((size_t)nRows + 1, 0);
        buf.resize(indexPos + offsets.size() * sizeof(uint64_t));
        for (int r = 0; r < nRows; ++r) {
            offsets[(size_t)r] = buf.size();
            for (const auto& g : rowGroups[(size_t)r]) {
                writePOD(buf, g.lid);
                writePOD(buf, g.count);
                writePOD(buf, g.anchor);
                const uint32_t wn = static_cast<uint32_t>(g.words.size());
                writePOD(buf, wn);
                for (uint16_t w : g.words) writePOD(buf, w);
            }
        }
        offsets[(size_t)nRows] = buf.size();
        std::memcpy(buf.data() + indexPos, offsets.data(),
                    offsets.size() * sizeof(uint64_t));
        result.tableBytesSize = buf.size();

        // 质量判定（同原实现）
        result.success = true;
        result.message = "OK";
        uint16_t worstGap = 0;
        for (const auto& cv : result.covers)
            worstGap = std::max(worstGap, cv.maxGapSeen);
        if (totalEntries == 0) {
            result.qualityFlag = QualityFlag::Warning;
            result.message = "no entries in depth window";
        } else if (worstGap > static_cast<uint16_t>(params_.maxGap)) {
            result.qualityFlag = QualityFlag::Degraded;
            result.message = "coverage gap " + std::to_string(worstGap) +
                             " grids > maxGap " + std::to_string(params_.maxGap);
        } else {
            result.qualityFlag = QualityFlag::Normal;
        }

        CALIB_LOG_INFO("Generate(CUDA): rows={}/{} entries={} bytes={} worstGap={} flag={}",
                       rowsWithData, nRows, totalEntries, buf.size(), worstGap,
                       static_cast<int>(result.qualityFlag));
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Exception: ") + e.what();
    } catch (...) {
        result.success = false;
        result.message = "Unknown exception";
    }
    return result;
}
