/**
 * @file laser_match_scan_v3_cuda_impl.cu
 * @brief 激光线匹配扫描CUDA算子 V2 — CUDA实现（struct Impl 方法）
 *
 * 相对 v1 的优化（语义等价，验证以 v1 输出为基准）：
 *   1. 32bit 排序键 (row 13b << 19 | xq 19b@1/256px)：CUB pass 8→4；keygen 顺带写 idx 序列
 *   2. kernelMatchV3：固定 grid=8192（免 max_row 回传同步/免 atomicMax 单点争用）；
 *      smem 16KB→4KB（占用 4→16 块/SM）；二分改 warp ballot 线性判定
 *      （外层贪心 li 序保持串行、全部 lane 冗余执行同一路径 → 语义与 v1 一致）
 *   3. kernelLookupTableMultiV3：候选去重走寄存器局部数组（免 stride-256B 全局重读+逐次除法）；
 *      kernelScatterLineIds 折叠进查表 kernel（省一次 launch）
 *   4. 候选数组 uR/lid 打包为 8B CandPackV3（kernelMatch 每候选 1 次读）
 *   5. 全程单次主机同步：末尾 pinned 16B 批量回传 [match,exclL,exclR,overflow]
 *   6. 结果/status 持久缓冲 + ROI/双缓冲视图（免每帧 7 组 malloc + clone）
 *   7. 无逐帧计时日志（v1 的 LM_ENABLE_TIMING 常开为已知开销）
 */

#include "laser_match_scan_v3_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include "common/json_utils.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <numeric>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <climits>

using namespace calib;

CALIB_DEFINE_LOG_TAG(07, LaserMatchScanCudaV3);

// ============================================================================
// Configuration Constants
// ============================================================================

static constexpr int BLOCK_SIZE = 256;
static constexpr int V3_MAX_CARDS = 32;   // per-point card cap (nearest card per lid)
static constexpr int V3_MAX_LIDS = 32;    // lid bucket stride (lids are 1..25)
static constexpr int MATCH_BLOCK = 32;          // kernelMatchV3: 1 warp per row
static constexpr int MAX_LOOKUP_CANDS = 64;     // 与 Impl::MAX_LOOKUP_CANDS 同值
static constexpr int SMEM_RIGHT_CAP = 256;      // 与 Impl::SMEM_RIGHT_CAP 同值
static constexpr unsigned int KEY_SENTINEL = 0xFFFFFFFFu;
static constexpr unsigned int XQ_ONE = 256u;    // x 量化 1/256 px
static constexpr unsigned int XQ_MASK = (1u << 19) - 1u;

// ============================================================================
// CUDA Error Handling
// ============================================================================

template<typename T>
static inline void safeCudaFree(T*& ptr) {
    if (ptr != nullptr) {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            CALIB_LOG_ERROR("cudaFree failed: {}", cudaGetErrorString(err));
        }
        ptr = nullptr;
    }
}

#ifndef NDEBUG
class ScopedFlagV2 {
public:
    explicit ScopedFlagV2(std::atomic<bool>* flag) : flag_(flag) {
        flag_->store(true);
    }
    ~ScopedFlagV2() { flag_->store(false); }
    ScopedFlagV2(const ScopedFlagV2&) = delete;
    ScopedFlagV2& operator=(const ScopedFlagV2&) = delete;
private:
    std::atomic<bool>* flag_;
};
#endif

// ============================================================================
// Impl ctor/dtor
// ============================================================================

LaserMatchScanCudaV3::Impl::Impl(const LaserMatchScanParamsV3& params)
    : params_(params)
{
    params_.validate();

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("[07-LaserMatchScanCudaV3] No CUDA devices found");
    }
    if (params_.deviceId >= deviceCount) {
        throw std::invalid_argument("[07-LaserMatchScanCudaV3] deviceId >= device count");
    }

    old_device_id_ = 0;
    cudaGetDevice(&old_device_id_);
    if (params_.deviceId != old_device_id_) {
        cudaSetDevice(params_.deviceId);
    }

    if (cudaMallocHost((void**)&h_counts_, 4 * sizeof(int)) != cudaSuccess) {
        h_counts_ = nullptr;
        CALIB_LOG_WARN("pinned counts alloc failed; fallback to staged copy");
    }
}

LaserMatchScanCudaV3::Impl::~Impl() {
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        CALIB_LOG_ERROR("cudaDeviceSynchronize in destructor failed: {}",
                        cudaGetErrorString(sync_err));
    }
    safeCudaFree(d_cub_temp_);
    if (h_counts_) { cudaFreeHost(h_counts_); h_counts_ = nullptr; }
}

// ============================================================================
// allocateBuffers —— 容量式 grow-only（同尺寸 create 为 no-op）
// ============================================================================

bool LaserMatchScanCudaV3::Impl::allocateBuffers(int leftCount, int rightCount) {
    if (leftCount > 0) {
        d_left_keys_in_.create(1, leftCount, CV_32SC1);
        d_left_keys_out_.create(1, leftCount, CV_32SC1);
        d_left_idx_in_.create(1, leftCount, CV_32SC1);
        d_left_sorted_idx_.create(1, leftCount, CV_32SC1);
        d_left_sorted_pts_.create(1, leftCount, CV_32FC2);
        d_left_sorted_lids_.create(1, leftCount, CV_32SC1);
        d_card_lid_.create(1, leftCount * V3_MAX_CARDS, CV_32SC1);
        d_card_hitRx_.create(1, leftCount * V3_MAX_CARDS, CV_32FC1);
        d_card_count_.create(1, leftCount, CV_32SC1);
        d_out_flags_.create(1, leftCount, CV_32SC1);
        d_out_left_.create(1, leftCount, CV_32FC2);
        d_out_right_.create(1, leftCount, CV_32FC2);
        d_out_lid_.create(1, leftCount, CV_32SC1);
        d_out_track_.create(1, leftCount, CV_32SC1);
        d_cell_seg_.create(1, MAX_EPIPOLAR_ROWS * V3_MAX_LIDS, CV_32SC1);
        d_matched_left_cap_.create(1, leftCount, CV_32FC2);
        d_matched_right_cap_.create(1, leftCount, CV_32FC2);
        d_matched_lids_cap_.create(1, leftCount, CV_32SC1);
        d_matched_tracks_cap_.create(1, leftCount, CV_32SC1);
        d_left_status_[0].create(1, leftCount, CV_32SC1);
        d_left_status_[1].create(1, leftCount, CV_32SC1);
    }
    if (rightCount > 0) {
        d_right_keys_in_.create(1, rightCount, CV_32SC1);
        d_right_keys_out_.create(1, rightCount, CV_32SC1);
        d_right_idx_in_.create(1, rightCount, CV_32SC1);
        d_right_sorted_idx_.create(1, rightCount, CV_32SC1);
        d_right_sorted_pts_.create(1, rightCount, CV_32FC2);
        d_right_sorted_lids_.create(1, rightCount, CV_32SC1);
        d_right_status_[0].create(1, rightCount, CV_32SC1);
        d_right_status_[1].create(1, rightCount, CV_32SC1);
    }
    d_counts_.create(1, 4, CV_32SC1);

    d_left_row_start_.create(1, MAX_EPIPOLAR_ROWS, CV_32SC1);
    d_left_row_count_.create(1, MAX_EPIPOLAR_ROWS, CV_32SC1);
    d_right_row_start_.create(1, MAX_EPIPOLAR_ROWS, CV_32SC1);
    d_right_row_count_.create(1, MAX_EPIPOLAR_ROWS, CV_32SC1);

    int maxCount = leftCount > rightCount ? leftCount : rightCount;
    if (maxCount > 0) {
        size_t sort_temp = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr, sort_temp,
            (unsigned int*)nullptr, (unsigned int*)nullptr,
            (int*)nullptr, (int*)nullptr,
            maxCount, 0, sizeof(unsigned int) * 8, 0);
        if (cub_temp_size_ < sort_temp) {
            safeCudaFree(d_cub_temp_);
            cudaMalloc(&d_cub_temp_, sort_temp);
            cub_temp_size_ = sort_temp;
        }
    }

    return true;
}

// ============================================================================
// warmup
// ============================================================================

void LaserMatchScanCudaV3::Impl::warmup(int maxLeftPoints, int maxRightPoints) {
    if (maxLeftPoints <= 0 || maxRightPoints <= 0) {
        CALIB_LOG_WARN("warmup(): invalid sizes (left={}, right={}), skipping",
                       maxLeftPoints, maxRightPoints);
        return;
    }
    allocateBuffers(maxLeftPoints, maxRightPoints);
    warmed_up_ = true;
    warmup_left_ = maxLeftPoints;
    warmup_right_ = maxRightPoints;
    CALIB_LOG_INFO("warmup(): allocated buffers for {} left, {} right points",
                   maxLeftPoints, maxRightPoints);
}

// ============================================================================
// loadTempTable —— JSON 便捷注入（同 v1）
// ============================================================================

bool LaserMatchScanCudaV3::Impl::LoadTempTable(const std::string& jsonPath) {
    try {
        std::ifstream ifs(jsonPath);
        if (!ifs.is_open()) {
            CALIB_LOG_ERROR("loadTempTable(): cannot open file: {}", jsonPath);
            return false;
        }

        nlohmann::json j;
        ifs >> j;

        auto table = std::make_shared<LaserPlaneMapTempTable>();
        const auto& arr = j.at("table");
        for (const auto& entry : arr) {
            const double temp = entry.at("temperature").get<double>();
            const auto& mapData = entry.at("mapData");
            const int n = static_cast<int>(mapData.size());

            LaserPlaneMap planeMap;
            planeMap.temperature = temp;
            planeMap.totalPairs = n;
            cv::Mat m(n, 4, CV_32FC1);
            for (int i = 0; i < n; ++i) {
                const auto& row = mapData[i];
                m.at<float>(i, 0) = row[0].get<float>();
                m.at<float>(i, 1) = row[1].get<float>();
                m.at<float>(i, 2) = row[2].get<float>();
                m.at<float>(i, 3) = row[3].get<float>();
            }
            planeMap.leftToRightMap = std::move(m);
            table->table[temp] = std::move(planeMap);
        }

        if (!SetTempTable(table)) {
            return false;
        }
        CALIB_LOG_INFO("loadTempTable(): loaded {} temperature entries from {}",
                       tempTable_.size(), jsonPath);
        return true;

    } catch (const std::exception& e) {
        CALIB_LOG_ERROR("loadTempTable(): exception: {}", e.what());
        return false;
    }
}

// ============================================================================
// SetTempTable —— 公共类型 → 内部扁平表示（同 v1）
// ============================================================================

bool LaserMatchScanCudaV3::Impl::SetTempTable(
    std::shared_ptr<const LaserPlaneMapTempTable> table)
{
    if (!table || table->empty()) {
        CALIB_LOG_ERROR("SetTempTable(): null or empty temp table");
        return false;
    }

    tempTable_.clear();

    for (const auto& kv : table->table) {
        const double temp = kv.first;
        const LaserPlaneMap& planeMap = kv.second;
        const cv::Mat& m = planeMap.leftToRightMap;
        if (m.empty()) {
            CALIB_LOG_WARN("SetTempTable(): skip temp={} (empty leftToRightMap)", temp);
            continue;
        }

        TempTableEntry te;
        te.temperature = temp;
        te.entryCount = m.rows;
        te.mapData.resize(te.entryCount * 4);

        int maxLid = -1;
        for (int i = 0; i < te.entryCount; ++i) {
            const float* r = m.ptr<float>(i);
            te.mapData[i * 4 + 0] = r[0];
            te.mapData[i * 4 + 1] = r[1];
            te.mapData[i * 4 + 2] = r[2];
            te.mapData[i * 4 + 3] = r[3];
            const int lid = static_cast<int>(r[3]);
            if (lid > maxLid) maxLid = lid;
        }
        te.numLines = maxLid + 1;
        tempTable_.push_back(std::move(te));
    }

    if (tempTable_.empty()) {
        CALIB_LOG_ERROR("SetTempTable(): no valid entries after conversion");
        tableLoaded_ = false;
        return false;
    }

    std::sort(tempTable_.begin(), tempTable_.end(),
              [](const TempTableEntry& a, const TempTableEntry& b) {
                  return a.temperature < b.temperature;
              });

    tableLoaded_ = true;
    return true;
}

// ============================================================================
// setCurrentTemperature（同 v1）
// ============================================================================

void LaserMatchScanCudaV3::Impl::SetCurrentTemperature(double temperature) {
    if (!tableLoaded_ || tempTable_.empty()) {
        throw std::runtime_error("[07-LaserMatchScanCudaV3] temp table not loaded");
    }

    if (temperature < tempTable_.front().temperature ||
        temperature > tempTable_.back().temperature) {
        CALIB_LOG_WARN("setCurrentTemperature(): temp={} out of range [{}, {}], clamping",
                       temperature,
                       tempTable_.front().temperature,
                       tempTable_.back().temperature);
    }

    auto it = std::lower_bound(tempTable_.begin(), tempTable_.end(), temperature,
        [](const TempTableEntry& e, double val) { return e.temperature < val; });

    if (it == tempTable_.end()) --it;
    if (it != tempTable_.begin()) {
        auto prev = std::prev(it);
        if (std::abs(prev->temperature - temperature) <
            std::abs(it->temperature - temperature)) {
            it = prev;
        }
    }

    currentTemp_ = it->temperature;
    tempSet_ = true;

    cv::cuda::Stream stream;
    uploadMapTable(*it, stream);
    stream.waitForCompletion();

    CALIB_LOG_INFO("setCurrentTemperature(): temp={} -> selected entry temp={}, {} map entries, {} lines",
                   temperature, currentTemp_, it->entryCount, it->numLines);
}

// ============================================================================
// uploadMapTable（同 v1）
// ============================================================================

void LaserMatchScanCudaV3::Impl::uploadMapTable(
    const TempTableEntry& entry, cv::cuda::Stream& stream)
{
    activeMapCount_ = entry.entryCount;
    activeNumLines_ = entry.numLines;

    cv::Mat h_map(activeMapCount_, 1, CV_32FC4,
                  const_cast<float*>(entry.mapData.data()));
    d_map_table_.upload(h_map, stream);

    std::vector<int> h_line_start(activeNumLines_, 0);
    std::vector<int> h_line_count(activeNumLines_, 0);

    for (int i = 0; i < activeMapCount_; ++i) {
        int lid = static_cast<int>(entry.mapData[i * 4 + 3]);
        if (lid >= 0 && lid < activeNumLines_) {
            if (h_line_count[lid] == 0) {
                h_line_start[lid] = i;
            }
            h_line_count[lid]++;
        }
    }

    cv::Mat h_ls(1, activeNumLines_, CV_32SC1, h_line_start.data());
    cv::Mat h_lc(1, activeNumLines_, CV_32SC1, h_line_count.data());
    d_map_line_start_.upload(h_ls, stream);
    d_map_line_count_.upload(h_lc, stream);

    // Build by-row CSR view for global (scan-mode) lookup
    {
        int N = activeMapCount_;
        std::vector<std::vector<int>> buckets(MAX_EPIPOLAR_ROWS);
        for (int i = 0; i < N; ++i) {
            float yL = entry.mapData[i * 4 + 1];
            int row = (int)std::round(yL / params_.epipolar_row_step);
            if (row < 0) row = 0;
            if (row >= MAX_EPIPOLAR_ROWS) row = MAX_EPIPOLAR_ROWS - 1;
            buckets[row].push_back(i);
        }
        std::vector<float> h_byrow;
        std::vector<int> h_row_start(MAX_EPIPOLAR_ROWS, 0);
        std::vector<int> h_row_count(MAX_EPIPOLAR_ROWS, 0);
        int offset = 0;
        for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r) {
            h_row_start[r] = offset;
            h_row_count[r] = (int)buckets[r].size();
            std::sort(buckets[r].begin(), buckets[r].end(), [&](int a, int b) {
                return entry.mapData[a * 4] < entry.mapData[b * 4];
            });
            for (int idx : buckets[r]) {
                h_byrow.push_back(entry.mapData[idx * 4]);
                h_byrow.push_back(entry.mapData[idx * 4 + 2]);
                h_byrow.push_back(entry.mapData[idx * 4 + 3]);
                h_byrow.push_back(entry.mapData[idx * 4 + 1]);
            }
            offset += h_row_count[r];
        }
        cv::Mat h_byrow_mat(N, 1, CV_32FC4, h_byrow.data());
        d_map_byrow_.upload(h_byrow_mat, stream);
        cv::Mat h_rs(1, MAX_EPIPOLAR_ROWS, CV_32SC1, h_row_start.data());
        cv::Mat h_rc(1, MAX_EPIPOLAR_ROWS, CV_32SC1, h_row_count.data());
        d_map_row_start_.upload(h_rs, stream);
        d_map_row_count_.upload(h_rc, stream);
    }
}

// ============================================================================
// setParams
// ============================================================================

void LaserMatchScanCudaV3::Impl::setParams(const LaserMatchScanParamsV3& params) {
#ifndef NDEBUG
    if (inProcess_.load()) {
        CALIB_LOG_ERROR("setParams(): called while process() is running");
        throw std::runtime_error("[07-LaserMatchScanCudaV3] setParams() called during process()");
    }
#endif
    params.validate();
    bool deviceChanged = (params.deviceId != params_.deviceId);
    params_ = params;
    if (deviceChanged) {
        cudaSetDevice(params_.deviceId);
    }
    warmed_up_ = false;
    warmup_left_ = 0;
    warmup_right_ = 0;
    CALIB_LOG_INFO("setParams(): params updated, warmup reset");
}

// ============================================================================
// CUDA Kernels
// ============================================================================

struct MatchPairV3 {
    int leftIdx;
    int rightIdx;
};

struct CandPackV3 {
    float uR;
    int lid;
};

// K1: 32bit 复合排序键 + 顺带 idx 序列（省 thrust::sequence）
//     key = (rowKey << 19) | xq,  xq = x*256 取整 [0, 2^19)
//     row >= 8192 → KEY_SENTINEL（排序沉底, CSR build 丢弃, 与 v1 一致）
__global__ void kernelGenKeysIdxV3(
    const float2* __restrict__ d_points,
    int count,
    float epipolar_row_step,
    unsigned int* __restrict__ d_keys,
    int* __restrict__ d_idx_in)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float2 pt = d_points[idx];
    int rowKey = static_cast<int>(roundf(pt.y / epipolar_row_step));
    if (rowKey < 0) rowKey = 0;

    unsigned int key;
    if (rowKey >= 8192) {
        key = KEY_SENTINEL;
    } else {
        float x = (pt.x >= 0.0f) ? pt.x : 0.0f;
        int xq = __float2int_rn(x * static_cast<float>(XQ_ONE));
        if (xq < 0) xq = 0;
        if (xq > static_cast<int>(XQ_MASK)) xq = static_cast<int>(XQ_MASK);
        key = (static_cast<unsigned int>(rowKey) << 19) | static_cast<unsigned int>(xq);
    }

    d_keys[idx] = key;
    d_idx_in[idx] = idx;
}

// K2: scatter sorted（同 v1）
__global__ void kernelScatterSortedV3(
    const float2* __restrict__ d_src_pts,
    const int*    __restrict__ d_src_lids,
    const int*    __restrict__ d_sorted_idx,
    int count,
    float2* __restrict__ d_dst_pts,
    int*    __restrict__ d_dst_lids)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int src = d_sorted_idx[idx];
    d_dst_pts[idx] = d_src_pts[src];
    d_dst_lids[idx] = d_src_lids[src];
}

// K3: CSR 构建（读 32bit 键, sentinel 丢弃; 同 v1 语义）
__global__ void kernelBuildCSRV3(
    const unsigned int* __restrict__ d_sort_keys,
    int count,
    int numRows,
    int* __restrict__ d_row_start,
    int* __restrict__ d_row_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    unsigned int key = d_sort_keys[idx];
    if (key == KEY_SENTINEL) return;
    unsigned int rowKey = key >> 19;
    if (static_cast<int>(rowKey) >= numRows) return;

    atomicMin(&d_row_start[rowKey], idx);
    atomicAdd(reinterpret_cast<unsigned int*>(&d_row_count[rowKey]), 1u);
}

// ============================================================================
// V3 kernels: nearest-card issue + segment voting + conflict drop-both + gather
// ============================================================================

// V3-K1: issue cards (nearest entry per lid) + HIT check against right row CSR.
// Runs in ORIGINAL index space (no left sort needed). Card = (lid, hitRx).
__global__ void kernelIssueCardsV3(
    const float2* __restrict__ d_left_pts,        // orig order
    int leftCount,
    const float4* __restrict__ d_map_byrow,       // (xL, uR, lineId, yL) sorted by xL in row
    const int*    __restrict__ d_map_row_start,
    const int*    __restrict__ d_map_row_count,
    const float2* __restrict__ d_right_sorted_pts,
    const int*    __restrict__ d_right_row_start,
    const int*    __restrict__ d_right_row_count,
    float epipolar_row_step,
    float vL_tolerance,
    float match_window,
    int*   __restrict__ d_card_lid,               // [i*V3_MAX_CARDS + c]
    float* __restrict__ d_card_hitRx,
    int*   __restrict__ d_card_count,
    int*   __restrict__ d_left_status)            // orig: >=1 card / 0 none / -1 no-map
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= leftCount) return;

    d_card_count[i] = 0;
    d_left_status[i] = -1;

    const float2 p = d_left_pts[i];
    const int row = (int)roundf(p.y / epipolar_row_step);
    if (row < 0 || row >= 8192) return;
    const int mapCnt = d_map_row_count[row];
    if (mapCnt == 0) return;

    const float xL = p.x;
    const int start = d_map_row_start[row];

    // binary search first map entry with x >= xL
    int lo = start, hi = start + mapCnt - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (d_map_byrow[mid].x < xL) lo = mid + 1; else hi = mid;
    }

    // nearest entry per lid within vL_tolerance (expand from lo in both dirs)
    int nc = 0;
    int locLid[V3_MAX_CARDS];
    float locU[V3_MAX_CARDS];
    float locDx[V3_MAX_CARDS];
    for (int dir = 0; dir < 2; ++dir) {
        int q = (dir == 0) ? lo : lo - 1;
        const int step = (dir == 0) ? 1 : -1;
        while (q >= start && q < start + mapCnt) {
            const float dx = d_map_byrow[q].x - xL;
            if (fabsf(dx) > vL_tolerance) break;
            const int lid = (int)d_map_byrow[q].z;
            const float uR = d_map_byrow[q].y;
            const float adx = fabsf(dx);
            int found = -1;
            for (int c = 0; c < nc; ++c)
                if (locLid[c] == lid) { found = c; break; }
            if (found < 0) {
                if (nc < V3_MAX_CARDS) {
                    locLid[nc] = lid; locU[nc] = uR; locDx[nc] = adx;
                    ++nc;
                }
            } else if (adx < locDx[found]) {
                locDx[found] = adx;
                locU[found] = uR;
            }
            q += step;
        }
    }

    // HIT check each card: nearest right point within +-match_window
    const int rStart = d_right_row_start[row];
    const int rCnt   = d_right_row_count[row];
    int out = 0;
    for (int c = 0; c < nc; ++c) {
        const float uR = locU[c];
        float hitRx = -1.0f;
        if (rCnt > 0 && rStart >= 0) {
            int rlo = rStart, rhi = rStart + rCnt - 1;
            while (rlo < rhi) {
                int mid = (rlo + rhi) / 2;
                if (d_right_sorted_pts[mid].x < uR - match_window) rlo = mid + 1; else rhi = mid;
            }
            if (rlo < rStart + rCnt &&
                fabsf(d_right_sorted_pts[rlo].x - uR) <= match_window)
                hitRx = d_right_sorted_pts[rlo].x;
        }
        if (hitRx < 0.0f) continue;                // card fails HIT
        d_card_lid[i * V3_MAX_CARDS + out] = locLid[c];
        d_card_hitRx[i * V3_MAX_CARDS + out] = hitRx;
        ++out;
    }
    d_card_count[i] = out;
    d_left_status[i] = out > 0 ? 1 : 0;
}

// V3-K2: segment vote histogram (orig space; seg map from host)
__global__ void kernelSegVoteV3(
    const int* __restrict__ d_seg_map,             // orig -> segIdx
    const int* __restrict__ d_card_lid,
    const int* __restrict__ d_card_count,
    int leftCount,
    int* __restrict__ d_seg_votes)                 // [segIdx * V3_MAX_LIDS + (lid-1)]
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= leftCount) return;
    const int seg = d_seg_map[i];
    if (seg < 0) return;
    const int nc = d_card_count[i];
    for (int c = 0; c < nc; ++c) {
        const int lid = d_card_lid[i * V3_MAX_CARDS + c];
        if (lid >= 1 && lid <= V3_MAX_LIDS)
            atomicAdd(&d_seg_votes[seg * V3_MAX_LIDS + (lid - 1)], 1);
    }
}

// V3-K3: per-segment winner (argmax votes)
__global__ void kernelSegWinnerV3(
    const int* __restrict__ d_seg_votes,
    int segCount,
    int* __restrict__ d_seg_winner)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= segCount) return;
    int best = 0, bestLid = -1;
    for (int l = 0; l < V3_MAX_LIDS; ++l) {
        const int v = d_seg_votes[s * V3_MAX_LIDS + l];
        if (v > best) { best = v; bestLid = l + 1; }
    }
    d_seg_winner[s] = bestLid;                     // -1 = no card at all
}

// V3-K4: conflict detection on (row, winner-lid) + drop-both marking.
__global__ void kernelConflictMarkV3(
    const float2* __restrict__ d_left_pts,
    const int*    __restrict__ d_seg_map,
    int leftCount,
    const int*    __restrict__ d_seg_winner,
    float epipolar_row_step,
    int* __restrict__ d_cell_seg,                  // [row * V3_MAX_LIDS + (lid-1)], init -1
    int* __restrict__ d_seg_drop)                  // init 0
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= leftCount) return;
    const int seg = d_seg_map[i];
    if (seg < 0) return;
    const int winner = d_seg_winner[seg];
    if (winner < 1) return;
    const int row = (int)roundf(d_left_pts[i].y / epipolar_row_step);
    if (row < 0 || row >= 8192) return;
    const int cell = row * V3_MAX_LIDS + (winner - 1);
    const int prev = atomicCAS(&d_cell_seg[cell], -1, seg);
    if (prev != -1 && prev != seg) {
        atomicExch(&d_seg_drop[seg], 1);
        atomicExch(&d_seg_drop[prev], 1);
    }
}

// V3-K5: gather flags + temp output (winner card of kept segments)
__global__ void kernelGatherV3(
    const float2* __restrict__ d_left_pts,
    const int*    __restrict__ d_left_tracks,
    const int*    __restrict__ d_seg_map,
    const int*    __restrict__ d_seg_winner,
    const int*    __restrict__ d_seg_drop,
    const int*    __restrict__ d_card_lid,
    const float*  __restrict__ d_card_hitRx,
    const int*    __restrict__ d_card_count,
    int leftCount,
    int* __restrict__ d_out_flags,                 // [leftCount]
    float2* __restrict__ d_out_left,               // temp (uncompacted)
    float2* __restrict__ d_out_right,
    int*   __restrict__ d_out_lid,
    int*   __restrict__ d_out_track)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= leftCount) return;
    d_out_flags[i] = 0;
    const int seg = d_seg_map[i];
    if (seg < 0) return;
    if (d_seg_drop[seg] != 0) return;              // dropped segment
    const int winner = d_seg_winner[seg];
    if (winner < 1) return;
    const int nc = d_card_count[i];
    for (int c = 0; c < nc; ++c) {
        if (d_card_lid[i * V3_MAX_CARDS + c] == winner) {
            d_out_flags[i] = 1;
            d_out_left[i] = d_left_pts[i];
            d_out_right[i] = make_float2(d_card_hitRx[i * V3_MAX_CARDS + c],
                                         d_left_pts[i].y);
            d_out_lid[i] = winner;
            d_out_track[i] = d_left_tracks[i];
            return;
        }
    }
}
// K7: status 计数（输出到 d_counts_[slot]）
__global__ void kernelCountStatusV2(
    const int* __restrict__ d_status,
    int count,
    int targetValue,
    int* __restrict__ d_result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    if (d_status[idx] == targetValue) {
        atomicAdd(d_result, 1);
    }
}

// ============================================================================
// process — 优化版流水线：仅末尾一次主机同步
// ============================================================================

LaserMatchScanResultV3 LaserMatchScanCudaV3::Impl::process(
    const cv::cuda::GpuMat& d_left_points,
    const cv::cuda::GpuMat& d_left_line_ids,
    const cv::cuda::GpuMat& d_right_points,
    const cv::cuda::GpuMat& d_right_line_ids,
    cv::cuda::Stream& stream)
{
#ifndef NDEBUG
    ScopedFlagV2 guard(&inProcess_);
#endif

    LaserMatchScanResultV3 result;

    try {
        if (!tempSet_) {
            result.success = false;
            result.message = "Temperature not set: call setCurrentTemperature() first";
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        const int leftCount = d_left_points.rows * d_left_points.cols;
        const int rightCount = d_right_points.rows * d_right_points.cols;

        result.totalLeftPoints = leftCount;
        result.totalRightPoints = rightCount;

        if (leftCount == 0 || rightCount == 0) {
            result.success = true;
            result.message = "No points to match";
            result.d_matched_left = std::make_shared<cv::cuda::GpuMat>();
            result.d_matched_right = std::make_shared<cv::cuda::GpuMat>();
            result.d_matched_line_ids = std::make_shared<cv::cuda::GpuMat>();
            result.d_left_status = std::make_shared<cv::cuda::GpuMat>();
            result.d_right_status = std::make_shared<cv::cuda::GpuMat>();
            result.matchedCount = 0;
            return result;
        }

        if (!allocateBuffers(leftCount, rightCount)) {
            result.success = false;
            result.message = "GPU buffer allocation failed";
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
        cudaError_t kernel_err;

        // ── Step 0: 计数清零 [match, exclL, exclR, overflow] ──
        cudaMemsetAsync(d_counts_.ptr<int>(), 0, 4 * sizeof(int), cuda_stream);

        // ---- V3 Step 1-4 (right side only): keys/sort/scatter/CSR ----
        {
            int grid = (rightCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelGenKeysIdxV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_right_points.ptr<float2>(),
                rightCount,
                params_.epipolar_row_step,
                reinterpret_cast<unsigned int*>(d_right_keys_in_.ptr<int>()),
                d_right_idx_in_.ptr<int>());
        }

        cub::DeviceRadixSort::SortPairs(
            d_cub_temp_, cub_temp_size_,
            reinterpret_cast<unsigned int*>(d_right_keys_in_.ptr<int>()),
            reinterpret_cast<unsigned int*>(d_right_keys_out_.ptr<int>()),
            d_right_idx_in_.ptr<int>(),
            d_right_sorted_idx_.ptr<int>(),
            rightCount, 0, sizeof(unsigned int) * 8, cuda_stream);

        {
            int grid = (rightCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelScatterSortedV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_right_points.ptr<float2>(),
                d_right_line_ids.ptr<int>(),
                d_right_sorted_idx_.ptr<int>(),
                rightCount,
                d_right_sorted_pts_.ptr<float2>(),
                d_right_sorted_lids_.ptr<int>());
        }

        cudaMemsetAsync(d_right_row_start_.ptr<int>(), 0x7F,
                        MAX_EPIPOLAR_ROWS * sizeof(int), cuda_stream);
        cudaMemsetAsync(d_right_row_count_.ptr<int>(), 0,
                        MAX_EPIPOLAR_ROWS * sizeof(int), cuda_stream);
        {
            int grid = (rightCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelBuildCSRV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<const unsigned int*>(d_right_keys_out_.ptr<int>()),
                rightCount, MAX_EPIPOLAR_ROWS,
                d_right_row_start_.ptr<int>(),
                d_right_row_count_.ptr<int>());
        }

        if ((kernel_err = cudaGetLastError()) != cudaSuccess) {
            result.success = false;
            result.message = std::string("sort/CSR stage failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        // ---- V3 Step 5: issue cards (original index space) ----
        const int gen = status_gen_ & 1;
        cudaMemsetAsync(d_left_status_[gen].ptr<int>(), 0, leftCount * sizeof(int), cuda_stream);
        cudaMemsetAsync(d_right_status_[gen].ptr<int>(), 0, rightCount * sizeof(int), cuda_stream);
        {
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelIssueCardsV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_left_points.ptr<float2>(),
                leftCount,
                d_map_byrow_.ptr<float4>(),
                d_map_row_start_.ptr<int>(),
                d_map_row_count_.ptr<int>(),
                d_right_sorted_pts_.ptr<float2>(),
                d_right_row_start_.ptr<int>(),
                d_right_row_count_.ptr<int>(),
                params_.epipolar_row_step,
                params_.vL_tolerance,
                params_.match_threshold,
                d_card_lid_.ptr<int>(),
                d_card_hitRx_.ptr<float>(),
                d_card_count_.ptr<int>(),
                d_left_status_[gen].ptr<int>());
        }

        if ((kernel_err = cudaGetLastError()) != cudaSuccess) {
            result.success = false;
            result.message = std::string("kernelIssueCardsV3 failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        // ---- V3 Step 6: segment table (host unique of track ids) ----
        std::vector<int> h_tracks(static_cast<size_t>(leftCount));
        cudaMemcpyAsync(h_tracks.data(), d_left_line_ids.ptr<int>(),
                        leftCount * sizeof(int), cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);
        std::vector<int> uniqIds(h_tracks);
        std::sort(uniqIds.begin(), uniqIds.end());
        uniqIds.erase(std::unique(uniqIds.begin(), uniqIds.end()), uniqIds.end());
        const int segCount = static_cast<int>(uniqIds.size());
        std::vector<int> h_segmap(static_cast<size_t>(leftCount), -1);
        for (int i = 0; i < leftCount; ++i)
            h_segmap[static_cast<size_t>(i)] = static_cast<int>(
                std::lower_bound(uniqIds.begin(), uniqIds.end(), h_tracks[static_cast<size_t>(i)]) - uniqIds.begin());
        {
            cv::Mat h_segmap_m(1, leftCount, CV_32SC1, h_segmap.data());
            d_seg_map_.upload(h_segmap_m, stream);
        }
        d_seg_votes_.create(1, segCount * V3_MAX_LIDS, CV_32SC1);
        d_seg_winner_.create(1, segCount, CV_32SC1);
        d_seg_drop_.create(1, segCount, CV_32SC1);
        cudaMemsetAsync(d_seg_votes_.ptr<int>(), 0, segCount * V3_MAX_LIDS * sizeof(int), cuda_stream);
        cudaMemsetAsync(d_seg_winner_.ptr<int>(), -1, segCount * sizeof(int), cuda_stream);
        cudaMemsetAsync(d_seg_drop_.ptr<int>(), 0, segCount * sizeof(int), cuda_stream);
        cudaMemsetAsync(d_cell_seg_.ptr<int>(), -1, MAX_EPIPOLAR_ROWS * V3_MAX_LIDS * sizeof(int), cuda_stream);

        // ---- V3 Step 7: segment vote -> winner -> conflict drop-both ----
        {
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelSegVoteV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_seg_map_.ptr<int>(),
                d_card_lid_.ptr<int>(),
                d_card_count_.ptr<int>(),
                leftCount,
                d_seg_votes_.ptr<int>());
        }
        {
            int grid = (segCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelSegWinnerV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_seg_votes_.ptr<int>(),
                segCount,
                d_seg_winner_.ptr<int>());
        }
        {
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelConflictMarkV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_left_points.ptr<float2>(),
                d_seg_map_.ptr<int>(),
                leftCount,
                d_seg_winner_.ptr<int>(),
                params_.epipolar_row_step,
                d_cell_seg_.ptr<int>(),
                d_seg_drop_.ptr<int>());
        }

        if ((kernel_err = cudaGetLastError()) != cudaSuccess) {
            result.success = false;
            result.message = std::string("segment stage failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        // ---- V3 Step 8: gather winner cards of kept segments ----
        {
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelGatherV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_left_points.ptr<float2>(),
                d_left_line_ids.ptr<int>(),
                d_seg_map_.ptr<int>(),
                d_seg_winner_.ptr<int>(),
                d_seg_drop_.ptr<int>(),
                d_card_lid_.ptr<int>(),
                d_card_hitRx_.ptr<float>(),
                d_card_count_.ptr<int>(),
                leftCount,
                d_out_flags_.ptr<int>(),
                d_out_left_.ptr<float2>(),
                d_out_right_.ptr<float2>(),
                d_out_lid_.ptr<int>(),
                d_out_track_.ptr<int>());
        }

        if ((kernel_err = cudaGetLastError()) != cudaSuccess) {
            result.success = false;
            result.message = std::string("kernelGatherV3 failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        // ---- V3 Step 9: compact flagged outputs (CUB DeviceSelect x4) ----
        {
            size_t t1 = 0;
            cub::DeviceSelect::Flagged(nullptr, t1,
                (const float2*)nullptr, (const int*)nullptr,
                (float2*)nullptr, (int*)nullptr, leftCount, cuda_stream);
            if (select_temp_cap_ < t1 || d_select_temp_ == nullptr) {
                safeCudaFree(d_select_temp_);
                cudaError_t err = cudaMalloc(&d_select_temp_, t1);
                if (err != cudaSuccess) {
                    result.success = false;
                    result.message = "select temp alloc failed";
                    CALIB_LOG_ERROR("process(): {}", result.message);
                    return result;
                }
                select_temp_size_ = t1;
                select_temp_cap_ = t1;
            }
            cub::DeviceSelect::Flagged(d_select_temp_, select_temp_size_,
                (const float2*)d_out_left_.ptr<float2>(), d_out_flags_.ptr<int>(),
                d_matched_left_cap_.ptr<float2>(), d_counts_.ptr<int>(), leftCount, cuda_stream);
            cub::DeviceSelect::Flagged(d_select_temp_, select_temp_size_,
                (const float2*)d_out_right_.ptr<float2>(), d_out_flags_.ptr<int>(),
                d_matched_right_cap_.ptr<float2>(), d_counts_.ptr<int>(), leftCount, cuda_stream);
            cub::DeviceSelect::Flagged(d_select_temp_, select_temp_size_,
                (const int*)d_out_lid_.ptr<int>(), d_out_flags_.ptr<int>(),
                d_matched_lids_cap_.ptr<int>(), d_counts_.ptr<int>(), leftCount, cuda_stream);
            cub::DeviceSelect::Flagged(d_select_temp_, select_temp_size_,
                (const int*)d_out_track_.ptr<int>(), d_out_flags_.ptr<int>(),
                d_matched_tracks_cap_.ptr<int>(), d_counts_.ptr<int>(), leftCount, cuda_stream);
        }

        // ---- V3 Step 10: single host sync (count + diagnostics) ----
        int h_final = 0;
        cudaMemcpyAsync(&h_final, d_counts_.ptr<int>(), sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        std::vector<int> h_drop(static_cast<size_t>(segCount));
        cudaMemcpyAsync(h_drop.data(), d_seg_drop_.ptr<int>(),
                        segCount * sizeof(int), cudaMemcpyDeviceToHost, cuda_stream);
        std::vector<int> h_leftstat(static_cast<size_t>(leftCount));
        cudaMemcpyAsync(h_leftstat.data(), d_left_status_[gen].ptr<int>(),
                        leftCount * sizeof(int), cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);

        int droppedSegs = 0;
        for (int s = 0; s < segCount; ++s) droppedSegs += h_drop[static_cast<size_t>(s)] ? 1 : 0;
        int noCard = 0;
        for (int i = 0; i < leftCount; ++i) if (h_leftstat[static_cast<size_t>(i)] == 0) ++noCard;

        result.matchedCount = h_final;
        result.excludedLeftCount = noCard;
        result.excludedRightCount = 0;
        result.trackTotal = segCount;
        result.trackDropped = droppedSegs;

        if (h_final > 0) {
            result.d_matched_left = std::make_shared<cv::cuda::GpuMat>(
                d_matched_left_cap_, cv::Rect(0, 0, h_final, 1));
            result.d_matched_right = std::make_shared<cv::cuda::GpuMat>(
                d_matched_right_cap_, cv::Rect(0, 0, h_final, 1));
            result.d_matched_line_ids = std::make_shared<cv::cuda::GpuMat>(
                d_matched_lids_cap_, cv::Rect(0, 0, h_final, 1));
            result.d_matched_track_ids = std::make_shared<cv::cuda::GpuMat>(
                d_matched_tracks_cap_, cv::Rect(0, 0, h_final, 1));
        } else {
            result.d_matched_left = std::make_shared<cv::cuda::GpuMat>();
            result.d_matched_right = std::make_shared<cv::cuda::GpuMat>();
            result.d_matched_line_ids = std::make_shared<cv::cuda::GpuMat>();
            result.d_matched_track_ids = std::make_shared<cv::cuda::GpuMat>();
        }
        result.d_left_status = std::make_shared<cv::cuda::GpuMat>(d_left_status_[gen]);
        result.d_right_status = std::make_shared<cv::cuda::GpuMat>(d_right_status_[gen]);

        status_gen_++;

        result.success = true;
        result.message = fmt::format("OK: {} matched, {}/{} tracks kept, {} no-card left",
                                     h_final, segCount - droppedSegs, segCount, noCard);

        CALIB_LOG_DEBUG("process(): {}", result.message);
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Exception: ") + e.what();
        CALIB_LOG_ERROR("process(): {}", result.message);
    } catch (...) {
        result.success = false;
        result.message = "Unknown exception";
        CALIB_LOG_ERROR("process(): {}", result.message);
    }

    return result;
}
