// curve_map.cpp — 极线切曲线映射表实现
//
// C.1 几何: 极线 q(Z) = p_L − [tz·(p_L−c) − f·t_xy]/(Z−tz)（过极点 E 的线束）
// C.2 求交: 直线参数化代入 F(u,v)=0 → 二次方程（双根均验, 退化防护）
// C.3 生成: 逐行 OpenMP 并行 → 线分组排序 → 差分+位打包
// E.2 格式: [ΔuL:6bit|Δd:10bit zigzag]; ΔuL≥0x3F 转义绝对 uL; 每 32 条强制转义（轴标）

#include "curve_map.h"
#include "common/calib_logging.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <optional>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace calib;

namespace {

constexpr uint16_t kEscapeCode = 0x3F;      // ΔuL 转义标记（正常域 [0,62]）
constexpr int kAnchorInterval = 32;         // 每 32 条强制转义（二分轴标）
constexpr int kMaxCandsPerLookup = 64;      // 行内候选缓冲上界（25 线 × 双根）

// ---- zigzag: 有符号 → 无符号编码（Δd 10bit） ----
inline uint32_t zigzagEnc(int32_t v) {
    return static_cast<uint32_t>((v << 1) ^ (v >> 31));
}
inline int32_t zigzagDec(uint32_t u) {
    return static_cast<int32_t>(u >> 1) ^ -static_cast<int32_t>(u & 1);
}

// ---- 位打包: [ΔuL:6bit | Δd:10bit] → uint16 ----
inline uint16_t packEntry(uint16_t duL, int32_t dd) {
    const uint32_t z = zigzagEnc(dd);
    // dd 需在 ±511 内（10bit zigzag 上限）; 调用侧保证（超限走强制转义+拆分, 见设计）
    const uint32_t d10 = std::min(z, 0x3FFu);
    return static_cast<uint16_t>(((duL & 0x3F) << 10) | (d10 & 0x3FF));
}
inline void unpackEntry(uint16_t w, uint16_t& duL, int32_t& dd) {
    duL = (w >> 10) & 0x3F;
    dd = zigzagDec(w & 0x3FF);
}

// ---- 小端读写（序列化） ----
template <typename T>
void writePOD(std::vector<uint8_t>& out, const T& v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}
struct Reader {
    const uint8_t* p;
    size_t n;
    size_t off = 0;
    bool ok = true;
    template <typename T>
    T read() {
        T v{};
        if (off + sizeof(T) > n) { ok = false; return v; }
        std::memcpy(&v, p + off, sizeof(T));
        off += sizeof(T);
        return v;
    }
};

// ============================================================================
// 几何内核（C.1/C.2）
// ============================================================================

/// 直线×隐式二次曲线求交。直线 q(t) = (u0 + t·du, v0 + t·dv)。
/// 返回 0..2 个交点（像素坐标）。退化（a≈0）退一次方程。
int intersectLineConic(const ImplicitCurve& c,
                       double u0, double v0, double du, double dv,
                       double out[2][2]) {
    const double A = c.coeffs[0], B = c.coeffs[1], C = c.coeffs[2];
    const double D = c.coeffs[3], E = c.coeffs[4], F0 = c.coeffs[5];
    // F(u0+t·du, v0+t·dv) 展开
    const double a = A*du*du + B*du*dv + C*dv*dv;
    const double b = 2*A*u0*du + B*(u0*dv + v0*du) + 2*C*v0*dv + D*du + E*dv;
    const double cc = A*u0*u0 + B*u0*v0 + C*v0*v0 + D*u0 + E*v0 + F0;

    int n = 0;
    if (std::fabs(a) < 1e-18) {
        // 退化: 一次方程 b·t + cc = 0
        if (std::fabs(b) > 1e-18) {
            const double t = -cc / b;
            out[n][0] = u0 + t*du; out[n][1] = v0 + t*dv; ++n;
        }
        return n;
    }
    const double disc = b*b - 4*a*cc;
    if (disc < 0) return 0;
    const double sq = std::sqrt(disc);
    const double t1 = (-b + sq) / (2*a);
    const double t2 = (-b - sq) / (2*a);
    out[n][0] = u0 + t1*du; out[n][1] = v0 + t1*dv; ++n;
    if (sq > 1e-15) {   // 重根只记一次
        out[n][0] = u0 + t2*du; out[n][1] = v0 + t2*dv; ++n;
    }
    return n;
}

/// 单行求交结果（生成期中间态, 栅格坐标）
struct RawEntry {
    uint16_t uL;
    uint16_t uR;
    double   d;    // 视差（栅格, 浮点; 排序次级键）
};

/// 处理一行: 对行内每个 uL 格 × 每条曲线求交+过滤。
/// 返回按 (lid, uL, d) 排序的条目集。
std::vector<RawEntry> processRowForLine(const ImplicitCurve& curve, uint8_t lid,
                                        double v, double uStart, double uEnd, double uStep,
                                        const CurveMapInput& in,
                                        const CurveMapParams& prm,
                                        double invRowStep) {
    std::vector<RawEntry> out;
    const double f = in.f;
    const double cx = in.principalPoint.x, cy = in.principalPoint.y;
    const double tx = in.virtualT(0), ty = in.virtualT(1), tz = in.virtualT(2);
    const double B = in.baseline;
    const int W = in.imageSize.width;

    // 极点 E（本行所有极线共过）
    const double Ex = f * tx / tz + cx;
    const double Ey = f * ty / tz + cy;

    for (double uLpx = uStart; uLpx <= uEnd + 1e-9; uLpx += uStep) {
        // 极线: 过 p=(uLpx, v) 与 E。方向取 (E − p) 归一化不必要, 直接用 (du,dv)。
        const double du = Ex - uLpx;
        const double dv = Ey - v;

        double hits[2][2];
        const int nh = intersectLineConic(curve, uLpx, v, du, dv, hits);
        for (int h = 0; h < nh; ++h) {
            const double q_u = hits[h][0], q_v = hits[h][1];
            // 交点须在虚拟像面内（曲线物理存在于 CMOS 像面）
            if (q_u < 0 || q_u >= static_cast<double>(W) ||
                q_v < 0 || q_v >= static_cast<double>(in.imageSize.height))
                continue;
            // 深度: 由极线参数 t 反解 Z（q = p − N/(Z−tz), N = tz·(p−c) − f·t_xy）
            // 反解更稳妥的路径: 联立左射线与虚拟反投影。
            // 左射线: X_L = Z·(uL−cx)/f, Y_L = Z·(v−cy)/f
            // 虚拟投影: q_u = f·(X_L − tx)/(Z − tz) + cx
            //   → (q_u − cx)·(Z − tz) = f·(Z·(uL−cx)/f − tx)
            //   → Z·((q_u−cx) − (uL−cx)) = (q_u−cx)·tz − f·tx
            const double qc = q_u - cx;
            const double denom = qc - (uLpx - cx);
            if (std::fabs(denom) < 1e-9) continue;
            const double Z = (qc * tz - f * tx) / denom;
            if (Z < prm.depthMin || Z > prm.depthMax) continue;
            // 右对应（视差恒正: uR = uL − f·B/Z）
            const double uRpx = uLpx - f * B / Z;
            if (uRpx < 0 || uRpx >= static_cast<double>(W)) continue;
            RawEntry e;
            e.uL = static_cast<uint16_t>(std::lround(uLpx * invRowStep));
            e.uR = static_cast<uint16_t>(std::lround(uRpx * invRowStep));
            e.d = (uLpx - uRpx) * invRowStep;
            if (e.uR >= e.uL) continue;   // 量化后视差≥1 格保护
            out.push_back(e);
        }
    }
    std::sort(out.begin(), out.end(), [](const RawEntry& a, const RawEntry& b) {
        if (a.uL != b.uL) return a.uL < b.uL;
        return a.d < b.d;   // 同 uL 双根: 按 d 次级排序（设计 E.2）
    });
    // 同 (uL,d) 完全重合去重（重根阈值内）
    out.erase(std::unique(out.begin(), out.end(), [](const RawEntry& a, const RawEntry& b) {
        return a.uL == b.uL && a.d == b.d;
    }), out.end());
    return out;
}

} // namespace

// ============================================================================
// Generator
// ============================================================================

CALIB_DEFINE_LOG_TAG(12b, CurveMap);

CurveMapGenerator::CurveMapGenerator(const CurveMapParams& params)
    : params_(params)
{
    params_.validate();
}

void CurveMapGenerator::SetParams(const CurveMapParams& params) {
    params_ = params;
    params_.validate();
}

const CurveMapParams& CurveMapGenerator::GetParams() const { return params_; }

CurveMapResult CurveMapGenerator::Generate(const CurveMapInput& in) {
    CurveMapResult result;
    result.effectiveParams = params_;
    try {
        if (in.curves.empty()) {
            result.success = false;
            result.message = "no input curves";
            return result;
        }
        if (in.f <= 0 || in.baseline <= 0) {
            result.success = false;
            result.message = "invalid f/baseline";
            return result;
        }
        if (in.imageSize.width <= 0 || in.imageSize.height <= 0) {
            result.success = false;
            result.message = "invalid imageSize";
            return result;
        }
        if (in.roi.width <= 0 || in.roi.height <= 0 ||
            in.roi.x < 0 || in.roi.y < 0 ||
            in.roi.x + in.roi.width > in.imageSize.width ||
            in.roi.y + in.roi.height > in.imageSize.height) {
            result.success = false;
            result.message = "invalid roi";
            return result;
        }
        if (std::fabs(in.virtualT(2)) < 1e-6) {
            result.success = false;
            result.message = "virtualT.z ~= 0: epipole at infinity unsupported";
            return result;
        }

        const double invRowStep = 1.0 / static_cast<double>(params_.epipolarRowStep);
        const double uStep = static_cast<double>(params_.effectiveUSampleStep());
        const int rowCount = static_cast<int>(std::round(in.imageSize.height * invRowStep));

        // ROI 行/列范围（栅格）
        const int rowMin = std::max(0, static_cast<int>(std::floor(in.roi.y * invRowStep)));
        const int rowMax = std::min(rowCount - 1,
                                    static_cast<int>(std::ceil((in.roi.y + in.roi.height) * invRowStep)));
        const double uMin = static_cast<double>(in.roi.x);
        const double uMax = static_cast<double>(in.roi.x + in.roi.width) - 1.0;

        // ---- 逐行（并行）: 每 (行,线) 求 RawEntry ----
        const int nRows = rowMax - rowMin + 1;
        const int nLines = static_cast<int>(in.curves.size());
        // rows[lid][rowLocal]
        std::vector<std::vector<std::vector<RawEntry>>> perLineRows(
            nLines, std::vector<std::vector<RawEntry>>(nRows));

        int nThreads = params_.threads;
#if defined(_OPENMP)
        if (nThreads <= 0) nThreads = omp_get_max_threads();
        if (nThreads > nRows) nThreads = nRows;
        if (nThreads < 1) nThreads = 1;
        omp_set_num_threads(nThreads);
#endif
        (void)nThreads;

#if defined(_OPENMP)
        #pragma omp parallel for schedule(dynamic, 16)
#endif
        for (int r = 0; r < nRows; ++r) {
            const int row = rowMin + r;
            const double v = row * static_cast<double>(params_.epipolarRowStep);
            for (int l = 0; l < nLines; ++l) {
                perLineRows[static_cast<size_t>(l)][static_cast<size_t>(r)] =
                    processRowForLine(in.curves[static_cast<size_t>(l)],
                                      static_cast<uint8_t>(l + 1),
                                      v, uMin, uMax, uStep, in, params_, invRowStep);
            }
        }

        // ---- 组装行块 + 压缩 + 覆盖统计（串行, 快） ----
        struct GroupBuild {
            uint8_t lid;
            uint16_t anchor;
            std::vector<uint16_t> words;
            uint32_t count;
        };
        std::vector<std::vector<GroupBuild>> rowGroups(nRows);
        uint32_t totalEntries = 0;
        uint32_t rowsWithData = 0;

        for (int r = 0; r < nRows; ++r) {
            for (int l = 0; l < nLines; ++l) {
                auto& ents = perLineRows[static_cast<size_t>(l)][static_cast<size_t>(r)];
                if (ents.empty()) continue;
                GroupBuild g;
                g.lid = static_cast<uint8_t>(l + 1);
                g.anchor = ents.front().uL;
                g.count = static_cast<uint32_t>(ents.size());
                // 差分编码
                // ⚠ dd = 取整绝对值之差（lround(e.d) − prevDint）——若取"浮点差再取整",
                //   真值 Δ≈0.557/条目 每次舍入 +0.44 误差持续累计（实测漂至 +14 格）。
                //   取整差使解码 d ≡ lround(真值 d) 逐条目成立, 误差有界 ±0.5 格。
                uint16_t prevUL = 0;
                int32_t prevDint = 0;
                bool first = true;
                int sinceAnchor = 0;
                for (const auto& e : ents) {
                    const int32_t dInt = static_cast<int32_t>(std::lround(e.d));
                    const int32_t dd = dInt - prevDint;
                    int32_t du = first ? 0 : static_cast<int32_t>(e.uL - prevUL);
                    const bool overZigzag = zigzagEnc(dd) > 0x3FF;
                    const bool needEscape = (du >= kEscapeCode) || overZigzag
                                            || (sinceAnchor >= kAnchorInterval);
                    if (needEscape) {
                        // 转义条目（3 word）: [0x3F|0] + uint16 绝对 uL + uint16 绝对 d
                        const auto absD = static_cast<uint16_t>(dInt);
                        g.words.push_back(packEntry(kEscapeCode, 0));
                        g.words.push_back(e.uL);
                        g.words.push_back(absD);
                    } else {
                        g.words.push_back(packEntry(static_cast<uint16_t>(du), dd));
                    }
                    prevUL = e.uL;
                    prevDint = dInt;
                    first = false;
                    sinceAnchor = needEscape ? 0 : sinceAnchor + 1;
                }
                rowGroups[static_cast<size_t>(r)].push_back(std::move(g));
                totalEntries += g.count;
                // 覆盖统计
                CurveMapCover cv;
                cv.row = static_cast<uint16_t>(rowMin + r);
                cv.lid = static_cast<uint8_t>(l + 1);
                cv.uLmin = ents.front().uL;
                cv.uLmax = ents.back().uL;
                cv.count = g.count;
                uint16_t maxGap = 0;
                for (size_t k = 1; k < ents.size(); ++k)
                    maxGap = std::max(maxGap,
                        static_cast<uint16_t>(ents[k].uL - ents[k-1].uL));
                cv.maxGapSeen = maxGap;
                result.covers.push_back(cv);
            }
            if (!rowGroups[static_cast<size_t>(r)].empty()) ++rowsWithData;
        }
        result.entryCount = totalEntries;
        result.rowCountWithData = rowsWithData;

        // ---- 序列化（文件头 + 绝对行索引 + 行块） ----
        std::vector<uint8_t>& buf = result.tableBytes;
        struct HeaderV1 {
            uint32_t magic;
            uint32_t version;
            float    rowStep;
            float    depthMin;
            float    depthMax;
            int32_t  roiX, roiY, roiW, roiH;
            uint32_t rowCount;      // 绝对行数
            uint32_t entryCount;
            uint32_t rowMin;        // 数据行起始（绝对行号）
            uint32_t dataRows;      // 数据行数
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
        std::vector<uint64_t> offsets(static_cast<size_t>(nRows) + 1, 0);
        buf.resize(indexPos + offsets.size() * sizeof(uint64_t));
        for (int r = 0; r < nRows; ++r) {
            offsets[static_cast<size_t>(r)] = buf.size();
            for (const auto& g : rowGroups[static_cast<size_t>(r)]) {
                writePOD(buf, g.lid);
                writePOD(buf, g.count);
                writePOD(buf, g.anchor);
                const uint32_t wn = static_cast<uint32_t>(g.words.size());
                writePOD(buf, wn);
                for (uint16_t w : g.words) writePOD(buf, w);
            }
        }
        offsets[static_cast<size_t>(nRows)] = buf.size();
        std::memcpy(buf.data() + indexPos, offsets.data(),
                    offsets.size() * sizeof(uint64_t));
        result.tableBytesSize = buf.size();

        // ---- 质量判定 ----
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

        CALIB_LOG_INFO("Generate: rows={}/{} entries={} bytes={} worstGap={} flag={}",
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

// ============================================================================
// Table: 序列化/查询
// ============================================================================

bool CurveMapTable::Load(const uint8_t* data, size_t size, std::string& err) {
    struct HeaderV1 {
        uint32_t magic, version;
        float rowStep, depthMin, depthMax;
        int32_t roiX, roiY, roiW, roiH;
        uint32_t rowCount, entryCount, rowMin, dataRows;
    };
    Reader rd{data, size, 0, true};
    const HeaderV1 h = rd.read<HeaderV1>();
    if (!rd.ok) { err = "truncated header"; return false; }
    if (h.magic != kMagic) { err = "bad magic"; return false; }
    if (h.version != 1) { err = "unsupported version"; return false; }
    if (h.rowStep <= 0) { err = "bad rowStep"; return false; }

    rowStep_ = h.rowStep;
    depthMin_ = h.depthMin;
    depthMax_ = h.depthMax;
    roi_ = cv::Rect(h.roiX, h.roiY, h.roiW, h.roiH);
    rowCount_ = h.rowCount;
    entryCount_ = h.entryCount;

    const size_t nIdx = static_cast<size_t>(h.dataRows) + 1;
    if (rd.off + nIdx * sizeof(uint64_t) > size) { err = "truncated index"; return false; }
    std::vector<uint64_t> offsets(nIdx);
    std::memcpy(offsets.data(), data + rd.off, nIdx * sizeof(uint64_t));
    rd.off += nIdx * sizeof(uint64_t);

    raw_.assign(data, data + size);
    rows_.assign(h.rowCount, RowData{});
    for (uint32_t r = 0; r < h.dataRows; ++r) {
        const uint64_t beg = offsets[r];
        const uint64_t end = offsets[r + 1];
        if (beg == end) continue;
        RowData rdRow;
        rdRow.valid = true;
        size_t p = static_cast<size_t>(beg);
        while (p < static_cast<size_t>(end)) {
            LineGroup g;
            std::memcpy(&g.lid, raw_.data() + p, sizeof(uint8_t)); p += 1;
            uint32_t cnt = 0;
            std::memcpy(&cnt, raw_.data() + p, sizeof(uint32_t)); p += 4;
            std::memcpy(&g.anchorUL, raw_.data() + p, sizeof(uint16_t)); p += 2;
            uint32_t wn = 0;
            std::memcpy(&wn, raw_.data() + p, sizeof(uint32_t)); p += 4;
            if (p + static_cast<size_t>(wn) * 2 > raw_.size()) { err = "truncated group"; return false; }
            g.words.resize(wn);
            std::memcpy(g.words.data(), raw_.data() + p, static_cast<size_t>(wn) * 2);
            p += static_cast<size_t>(wn) * 2;
            (void)cnt;
            rdRow.groups.push_back(std::move(g));
        }
        const uint32_t absRow = h.rowMin + r;
        if (absRow < h.rowCount)
            rows_[static_cast<size_t>(absRow)] = std::move(rdRow);
    }
    return true;
}

bool CurveMapTable::LoadFromFile(const std::string& path, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)),
                               std::istreambuf_iterator<char>());
    return Load(data.data(), data.size(), err);
}

bool CurveMapTable::SaveToFile(const std::string& path, std::string& err) const {
    if (raw_.empty()) { err = "table not loaded/built"; return false; }
    std::ofstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    f.write(reinterpret_cast<const char*>(raw_.data()),
            static_cast<std::streamsize>(raw_.size()));
    return static_cast<bool>(f);
}

int CurveMapTable::Lookup(int row, int uL, CurveMapCand* cands, int candsCap) const {
    if (row < 0 || static_cast<uint32_t>(row) >= rows_.size()) return 0;
    const RowData& rd = rows_[static_cast<size_t>(row)];
    if (!rd.valid) return 0;
    int n = 0;
    for (const auto& g : rd.groups) {
        // 组内顺扫（组内条目按 uL 升序; 数据量~百级, GPU 侧换并行/轴标二分）
        uint16_t curUL = g.anchorUL;
        double curD = 0;
        uint32_t pos = 0;
        const uint32_t wn = static_cast<uint32_t>(g.words.size());
        while (pos < wn && n < candsCap) {
            const uint16_t word = g.words[pos];
            uint16_t duL; int32_t dd;
            unpackEntry(word, duL, dd);
            if (duL == kEscapeCode) {
                if (pos + 2 >= wn) break;   // payload 越界防护（需 pos+1, pos+2 有效）
                const uint16_t absUL = g.words[pos + 1];
                const uint16_t absD = g.words[pos + 2];
                pos += 3;   // 跳过 转义word+absUL+absD
                curUL = absUL;
                curD = absD;    // 双绝对: 直接置位（非累加）
            } else {
                curUL = static_cast<uint16_t>(curUL + duL);
                curD += dd;
                ++pos;
            }
            if (curUL == static_cast<uint16_t>(uL)) {
                // uR = uL − d（符号红线: 视差恒正, 单测钉死）
                const int32_t d = static_cast<int32_t>(std::lround(curD));
                const int32_t uR = static_cast<int32_t>(uL) - d;
                if (uR >= 0) {
                    cands[n].lid = g.lid;
                    cands[n].uR = static_cast<uint16_t>(uR);
                    ++n;
                }
                // 继续——双根可能有第二条同 uL
            } else if (curUL > static_cast<uint16_t>(uL)) {
                break;   // 组内升序, 越过即无
            }
        }
    }
    return n;
}

int CurveMapTable::EnumerateRow(int row, CurveMapEntry* out, int cap) const {
    if (row < 0 || static_cast<uint32_t>(row) >= rows_.size()) return 0;
    const RowData& rd = rows_[static_cast<size_t>(row)];
    if (!rd.valid) return 0;
    int n = 0;
    for (const auto& g : rd.groups) {
        uint16_t curUL = g.anchorUL;
        double curD = 0;
        uint32_t pos = 0;
        const uint32_t wn = static_cast<uint32_t>(g.words.size());
        while (pos < wn) {
            const uint16_t word = g.words[pos];
            uint16_t duL; int32_t dd;
            unpackEntry(word, duL, dd);
            if (duL == kEscapeCode) {
                if (pos + 2 >= wn) break;
                curUL = g.words[pos + 1];
                curD = g.words[pos + 2];
                pos += 3;
            } else {
                curUL = static_cast<uint16_t>(curUL + duL);
                curD += dd;
                ++pos;
            }
            if (n >= cap) return -1;
            out[n].row = static_cast<uint16_t>(row);
            out[n].lid = g.lid;
            out[n].uL = curUL;
            const int32_t d = static_cast<int32_t>(std::lround(curD));
            out[n].uR = static_cast<uint16_t>(static_cast<int32_t>(curUL) - d);
            ++n;
        }
    }
    return n;
}

OperatorInfo getCurveMapInfo() {
    return {"CurveMap", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}
