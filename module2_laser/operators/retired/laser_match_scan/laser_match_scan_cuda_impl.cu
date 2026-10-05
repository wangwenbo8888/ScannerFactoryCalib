/**
 * @file laser_match_scan_cuda_impl.cu
 * @brief 激光线匹配扫描CUDA算子 - CUDA实现（struct Impl 方法）
 *
 * 本文件在 Task 3 中实现所有基础设施方法。
 * process() 方法为占位实现，将在 Task 5 中完整实现。
 */

#include "laser_match_scan_cuda_pimpl.h"
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
#include <chrono>
#include <climits>

using namespace calib;


CALIB_DEFINE_LOG_TAG(07, LaserMatchScanCuda);

// ============================================================================
// Configuration Constants
// ============================================================================

static constexpr int BLOCK_SIZE = 256;
static constexpr int MAX_LOOKUP_CANDS = 64;   // 与 Impl::MAX_LOOKUP_CANDS 同值（kernel 作用域）

#ifdef LM_ENABLE_TIMING
#define LM_TIMER_DECL() \
    auto _t0 = std::chrono::high_resolution_clock::now(); \
    auto _t1 = _t0; \
    (void)_t0; (void)_t1

#define LM_TIMER_MARK(name) \
    _t1 = std::chrono::high_resolution_clock::now(); \
    CALIB_LOG_INFO("  [TIMING] {:<30s} {:>10.4f} ms", #name, \
        std::chrono::duration<double, std::milli>(_t1 - _t0).count()); \
    _t0 = _t1
#endif

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

// ============================================================================
// ScopedFlag (Debug-only thread safety)
// ============================================================================

#ifndef NDEBUG
class ScopedFlag {
public:
    explicit ScopedFlag(std::atomic<bool>* flag) : flag_(flag) {
        flag_->store(true);
    }
    ~ScopedFlag() { flag_->store(false); }
    ScopedFlag(const ScopedFlag&) = delete;
    ScopedFlag& operator=(const ScopedFlag&) = delete;
private:
    std::atomic<bool>* flag_;
};
#endif

// ============================================================================
// Impl ctor/dtor
// ============================================================================

LaserMatchScanCuda::Impl::Impl(const LaserMatchScanParams& params)
    : params_(params)
{
    params_.validate();

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("[07-LaserMatchScanCuda] No CUDA devices found");
    }
    if (params_.deviceId >= deviceCount) {
        throw std::invalid_argument("[07-LaserMatchScanCuda] deviceId >= device count");
    }

    old_device_id_ = 0;
    cudaGetDevice(&old_device_id_);
    if (params_.deviceId != old_device_id_) {
        cudaSetDevice(params_.deviceId);
    }
}

LaserMatchScanCuda::Impl::~Impl() {
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        CALIB_LOG_ERROR("cudaDeviceSynchronize in destructor failed: {}",
                        cudaGetErrorString(sync_err));
    }
    safeCudaFree(d_cub_temp_);
}

// ============================================================================
// allocateBuffers
// ============================================================================

bool LaserMatchScanCuda::Impl::allocateBuffers(int leftCount, int rightCount) {
    if (leftCount > 0) {
        d_left_sorted_pts_.create(1, leftCount, CV_32FC2);
        d_left_sorted_lids_.create(1, leftCount, CV_32SC1);
        d_left_sorted_idx_.create(1, leftCount, CV_32SC1);
    d_left_cand_uR_.create(1, leftCount * MAX_LOOKUP_CANDS, CV_32FC1);
    d_left_cand_lid_.create(1, leftCount * MAX_LOOKUP_CANDS, CV_32SC1);
    d_left_cand_count_.create(1, leftCount, CV_32SC1);
    d_left_status_.create(1, leftCount, CV_32SC1);
        d_match_left_idx_.create(1, leftCount * 2, CV_32SC1);  // MatchPair=2 ints; leftCount pairs max
        d_left_keys_in_.create(1, leftCount, CV_64FC1);
        d_left_keys_out_.create(1, leftCount, CV_64FC1);
        d_left_idx_in_.create(1, leftCount, CV_32SC1);
        d_out_line_ids_sorted_.create(1, leftCount, CV_32SC1);
        d_out_line_ids_orig_.create(1, leftCount, CV_32SC1);
    }
    if (rightCount > 0) {
        d_right_sorted_pts_.create(1, rightCount, CV_32FC2);
        d_right_sorted_lids_.create(1, rightCount, CV_32SC1);
        d_right_sorted_idx_.create(1, rightCount, CV_32SC1);
        d_right_status_.create(1, rightCount, CV_32SC1);
        d_right_keys_in_.create(1, rightCount, CV_64FC1);
        d_right_keys_out_.create(1, rightCount, CV_64FC1);
        d_right_idx_in_.create(1, rightCount, CV_32SC1);
    }
    d_match_count_.create(1, 1, CV_32SC1);
    d_max_row_key_.create(1, 2, CV_32SC1);

    d_left_row_start_.create(1, MAX_EPIPOLAR_ROWS, CV_32SC1);
    d_left_row_count_.create(1, MAX_EPIPOLAR_ROWS, CV_32SC1);
    d_right_row_start_.create(1, MAX_EPIPOLAR_ROWS, CV_32SC1);
    d_right_row_count_.create(1, MAX_EPIPOLAR_ROWS, CV_32SC1);

    d_map_row_start_.create(1, MAX_EPIPOLAR_ROWS, CV_32SC1);
    d_map_row_count_.create(1, MAX_EPIPOLAR_ROWS, CV_32SC1);

    int maxCount = leftCount > rightCount ? leftCount : rightCount;
    if (maxCount > 0) {
        size_t sort_temp = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr, sort_temp,
            (unsigned long long*)nullptr, (unsigned long long*)nullptr,
            (int*)nullptr, (int*)nullptr,
            maxCount, 0, sizeof(unsigned long long) * 8, 0);
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

void LaserMatchScanCuda::Impl::warmup(int maxLeftPoints, int maxRightPoints) {
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
// loadTempTable
// ============================================================================

bool LaserMatchScanCuda::Impl::LoadTempTable(const std::string& jsonPath) {
    try {
        std::ifstream ifs(jsonPath);
        if (!ifs.is_open()) {
            CALIB_LOG_ERROR("loadTempTable(): cannot open file: {}", jsonPath);
            return false;
        }

        nlohmann::json j;
        ifs >> j;

        // JSON → 公共类型 LaserPlaneMapTempTable，再经 SetTempTable 统一注入转换。
        // JSON 格式：{"table":[{"temperature":..,"mapData":[[xL,yL,uR,lineId],...]}]}
        auto table = std::make_shared<LaserPlaneMapTempTable>();
        const auto& arr = j.at("table");
        for (const auto& entry : arr) {
            const double temp = entry.at("temperature").get<double>();
            const auto& mapData = entry.at("mapData");
            const int n = static_cast<int>(mapData.size());

            LaserPlaneMap planeMap;
            planeMap.temperature = temp;
            planeMap.totalPairs = n;
            cv::Mat m(n, 4, CV_32FC1);   // leftToRightMap: 每行 [xL, yL, uR, lineId]
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
// SetTempTable —— 公共类型 LaserPlaneMapTempTable → 内部扁平查表表示
// 算子规范 §3.6：只读映射表经注入传入，转换结果缓存于实例（不修改源表）。
// ============================================================================

bool LaserMatchScanCuda::Impl::SetTempTable(
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
// setCurrentTemperature
// ============================================================================

void LaserMatchScanCuda::Impl::SetCurrentTemperature(double temperature) {
    if (!tableLoaded_ || tempTable_.empty()) {
        throw std::runtime_error("[07-LaserMatchScanCuda] temp table not loaded");
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
// uploadMapTable
// ============================================================================

void LaserMatchScanCuda::Impl::uploadMapTable(
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
                h_byrow.push_back(entry.mapData[idx * 4]);       // xL
                h_byrow.push_back(entry.mapData[idx * 4 + 2]);   // uR
                h_byrow.push_back(entry.mapData[idx * 4 + 3]);   // lineId
                h_byrow.push_back(entry.mapData[idx * 4 + 1]);   // yL
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

void LaserMatchScanCuda::Impl::setParams(const LaserMatchScanParams& params) {
#ifndef NDEBUG
    if (inProcess_.load()) {
        CALIB_LOG_ERROR("setParams(): called while process() is running");
        throw std::runtime_error("[07-LaserMatchScanCuda] setParams() called during process()");
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

struct MatchPair {
    int leftIdx;
    int rightIdx;
};

// K1a: kernelGenerateSortKeys — generates composite sort key + finds max row key
//      sortKey = (rowKey << 32) | floatBits(x)   [ensures sort by row, then by u within row]
__global__ void kernelGenerateSortKeys(
    const float2* __restrict__ d_points,
    int count,
    float epipolar_row_step,
    unsigned long long* __restrict__ d_sort_keys,
    unsigned int* __restrict__ d_max_row_key)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float2 pt = d_points[idx];
    int rowKeySigned = static_cast<int>(roundf(pt.y / epipolar_row_step));
    unsigned int rowKey = rowKeySigned > 0 ? static_cast<unsigned int>(rowKeySigned) : 0u;

    unsigned int xBits = (pt.x >= 0.0f) ? __float_as_uint(pt.x) : 0u;

    unsigned long long sortKey =
        (static_cast<unsigned long long>(rowKey) << 32) |
        static_cast<unsigned long long>(xBits);

    d_sort_keys[idx] = sortKey;
    atomicMax(d_max_row_key, rowKey);
}

// K1b: kernelScatterSorted — gather sorted data using sorted indices from CUB
__global__ void kernelScatterSorted(
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

// K2: kernelBuildCSRGPU — builds CSR row_start/row_count from sorted 64-bit keys
//     Pre-condition: d_row_start initialized to large value, d_row_count to 0
__global__ void kernelBuildCSRGPU(
    const unsigned long long* __restrict__ d_sort_keys,
    int count,
    int numRows,
    int* __restrict__ d_row_start,
    int* __restrict__ d_row_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    unsigned int rowKey = static_cast<unsigned int>(d_sort_keys[idx] >> 32);
    if (static_cast<int>(rowKey) >= numRows) return;

    atomicMin(&d_row_start[rowKey], idx);
    atomicAdd(reinterpret_cast<unsigned int*>(&d_row_count[rowKey]), 1u);
}

// K2: kernelLookupTable — looks up uR_expected for each left point
__global__ void kernelLookupTable(
    const float2* __restrict__ d_left_points,
    const int*    __restrict__ d_left_line_ids,
    int leftCount,
    const float4* __restrict__ d_map_table,
    const int*    __restrict__ d_map_line_start,
    const int*    __restrict__ d_map_line_count,
    int numLines,
    float vL_tolerance,
    float* __restrict__ d_uR_expected)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= leftCount) return;

    float xL = d_left_points[idx].x;
    float yL = d_left_points[idx].y;
    int lid  = d_left_line_ids[idx];

    if (lid < 0 || lid >= numLines || d_map_line_count[lid] == 0) {
        d_uR_expected[idx] = NAN;
        return;
    }

    int start = d_map_line_start[lid];
    int cnt   = d_map_line_count[lid];

    int lo = start, hi = start + cnt - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (d_map_table[mid].y < yL) lo = mid + 1;
        else hi = mid;
    }

    int best = lo;
    if (lo > start && fabsf(d_map_table[lo-1].y - yL) < fabsf(d_map_table[lo].y - yL))
        best = lo - 1;

    if (fabsf(d_map_table[best].y - yL) > vL_tolerance) {
        d_uR_expected[idx] = NAN;
        return;
    }

    float bestDist = fabsf(d_map_table[best].x - xL);
    int bestU = best;
    for (int i = best - 1; i >= start; i--) {
        if (fabsf(d_map_table[i].y - yL) > vL_tolerance) break;
        float d = fabsf(d_map_table[i].x - xL);
        if (d < bestDist) { bestDist = d; bestU = i; }
    }
    for (int i = best + 1; i < start + cnt; i++) {
        if (fabsf(d_map_table[i].y - yL) > vL_tolerance) break;
        float d = fabsf(d_map_table[i].x - xL);
        if (d < bestDist) { bestDist = d; bestU = i; }
    }

    d_uR_expected[idx] = d_map_table[bestU].z;
}

// K2b: kernelLookupTableScan — global by-row search (scan mode).
//      Searches ALL table entries at the left point's epipolar row (sorted by xL)
//      to find the closest xL match; outputs uR_expected AND the calibration line_id.
__global__ void kernelLookupTableMulti(
    const float2* __restrict__ d_left_sorted_pts,
    int leftCount,
    const float4* __restrict__ d_map_byrow,       // (xL, uR, lineId, yL)
    const int*    __restrict__ d_map_row_start,
    const int*    __restrict__ d_map_row_count,
    float epipolar_row_step,
    float vL_tolerance,
    float* __restrict__ d_cand_uR,
    int*   __restrict__ d_cand_lid,
    int*   __restrict__ d_cand_count,
    int*   __restrict__ d_out_line_ids_sorted)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= leftCount) return;

    float xL = d_left_sorted_pts[idx].x;
    float yL = d_left_sorted_pts[idx].y;
    int row = (int)roundf(yL / epipolar_row_step);
    d_cand_count[idx] = 0;
    d_out_line_ids_sorted[idx] = -1;
    if (row < 0 || row >= 8192 || d_map_row_count[row] == 0) return;

    const int start = d_map_row_start[row];
    const int cnt   = d_map_row_count[row];
    int lo = start, hi = start + cnt - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (d_map_byrow[mid].x < xL) lo = mid + 1; else hi = mid;
    }
    // 从插入点向两侧展开收集全部 |x - xL| <= vL_tolerance 的条目
    // 去重: 同 (lineId, uR栅格) 只留一个（容差窗会捞进同线相邻栅格的重复假设）
    int n = 0;
    int firstLid = -1;
    for (int dir = 0; dir < 2; ++dir) {
        int p = (dir == 0) ? lo : lo - 1;
        const int step = (dir == 0) ? 1 : -1;
        while (p >= start && p < start + cnt && n < MAX_LOOKUP_CANDS) {
            if (fabsf(d_map_byrow[p].x - xL) > vL_tolerance) break;
            const float uR = d_map_byrow[p].y;
            const int lid = (int)d_map_byrow[p].z;
            const int uRg = __float2int_rn(uR / epipolar_row_step);
            bool dup = false;
            for (int q = 0; q < n; ++q) {
                if (d_cand_lid[idx * MAX_LOOKUP_CANDS + q] == lid &&
                    __float2int_rn(d_cand_uR[idx * MAX_LOOKUP_CANDS + q] /
                                   epipolar_row_step) == uRg) {
                    dup = true; break;
                }
            }
            if (!dup) {
                d_cand_uR[idx * MAX_LOOKUP_CANDS + n] = uR;
                d_cand_lid[idx * MAX_LOOKUP_CANDS + n] = lid;
                if (firstLid < 0) firstLid = lid;
                ++n;
            }
            p += step;
        }
    }
    d_cand_count[idx] = n;
    d_out_line_ids_sorted[idx] = firstLid;
}

// K2c: kernelScatterLineIds — scatter looked-up line_id from sortedIdx to originalIdx
__global__ void kernelScatterLineIds(
    const int* __restrict__ d_sorted_idx,
    const int* __restrict__ d_line_ids_sorted,
    int count,
    int* __restrict__ d_line_ids_orig)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    d_line_ids_orig[d_sorted_idx[i]] = d_line_ids_sorted[i];
}

// K3: kernelMatch — per-epipolar-row block, sequential match within row
__global__ void kernelMatch(
    const float2* __restrict__ d_left_pts,
    const int*    __restrict__ d_left_lids,
    const int*    __restrict__ d_left_orig_idx,
    const float*  __restrict__ d_cand_uR,
    const int*    __restrict__ d_cand_lid,
    const int*    __restrict__ d_cand_count,
    const int*    __restrict__ d_left_row_start,
    const int*    __restrict__ d_left_row_count,
    const float2* __restrict__ d_right_pts,
    const int*    __restrict__ d_right_lids,
    const int*    __restrict__ d_right_orig_idx,
    const int*    __restrict__ d_right_row_start,
    const int*    __restrict__ d_right_row_count,
    int maxRightPerRow,
    float threshold,
    int maxMatchPairs,
    int*    __restrict__ d_line_ids_out,   // [origIdx] 命中候选的线号（供 extract）
    MatchPair* __restrict__ d_match_pairs,
    int*       __restrict__ d_left_status,
    int*       __restrict__ d_right_status,
    int*       __restrict__ d_match_count)
{
    int row = blockIdx.x;

    int leftCount  = d_left_row_count[row];
    int leftStart  = d_left_row_start[row];
    int rightCount = d_right_row_count[row];
    int rightStart = d_right_row_start[row];

    if (leftCount == 0) return;

    int effRight = min(rightCount, maxRightPerRow);

    extern __shared__ char smem_raw[];
    float* sm_uR       = (float*)smem_raw;
    int*   sm_lid      = (int*)(sm_uR + maxRightPerRow);
    int*   sm_occupied = sm_lid + maxRightPerRow;
    int*   sm_origIdx  = sm_occupied + maxRightPerRow;

    for (int i = threadIdx.x; i < effRight; i += blockDim.x) {
        int gi = rightStart + i;
        sm_uR[i]       = d_right_pts[gi].x;
        sm_lid[i]      = d_right_lids[gi];
        sm_occupied[i] = 0;
        sm_origIdx[i]  = d_right_orig_idx[gi];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int li = 0; li < leftCount; li++) {
            int gi = leftStart + li;
            const int nc = d_cand_count[gi];
            int lOrig    = d_left_orig_idx[gi];

            if (nc == 0) {
                d_left_status[lOrig] = -1;
                continue;
            }

            // 多候选判决: 遍历全部 (uR_c, lid_c) 假设, 收集范围内未占用右点
            // 恰好 1 个总命中 → 匹配（线号=命中假设的 lid）; 0/≥2 → 排除
            int hitSm = -1, hitCand = -1, totalHits = 0;
            for (int c = 0; c < nc && totalHits < 2; ++c) {
                const float uR_exp = d_cand_uR[gi * MAX_LOOKUP_CANDS + c];
                const int   candLid = d_cand_lid[gi * MAX_LOOKUP_CANDS + c];
                const float lo_val = uR_exp - threshold;
                const float hi_val = uR_exp + threshold;

                int bs_lo = 0, bs_hi = effRight;
                while (bs_lo < bs_hi) {
                    int mid = (bs_lo + bs_hi) >> 1;
                    if (sm_uR[mid] < lo_val) bs_lo = mid + 1;
                    else bs_hi = mid;
                }
                int range_start = bs_lo;
                bs_lo = range_start; bs_hi = effRight;
                while (bs_lo < bs_hi) {
                    int mid = (bs_lo + bs_hi) >> 1;
                    if (sm_uR[mid] <= hi_val) bs_lo = mid + 1;
                    else bs_hi = mid;
                }
                int range_end = bs_lo;

                for (int ri = range_start; ri < range_end; ri++) {
                    if (sm_occupied[ri]) continue;
                    // 线号一致性: 右点带真实线号时(≠0), 假设线号须一致才计命中
                    // (消除扫描模式多线假设的歧义误杀; 0=未知线号, 不过滤, 保持旧行为)
                    if (sm_lid[ri] != 0 && sm_lid[ri] != candLid) continue;
                    // 去重: 同一右点已被先前假设命中则不重复计数
                    // (同线相邻采样格/双根的重复假设窗口重叠同一右点, 非真歧义)
                    if (ri == hitSm) continue;
                    totalHits++;
                    hitSm = ri;
                    hitCand = c;
                    if (totalHits >= 2) break;
                }
            }

            if (totalHits == 1) {
                sm_occupied[hitSm] = 1;
                int rOrig = sm_origIdx[hitSm];

                int pos = atomicAdd(d_match_count, 1);
                if (pos < maxMatchPairs) {
                    d_match_pairs[pos].leftIdx  = lOrig;
                    d_match_pairs[pos].rightIdx = rOrig;
                    d_left_status[lOrig]  = 1;
                    d_right_status[rOrig] = 1;
                    d_line_ids_out[lOrig] =
                        d_cand_lid[gi * MAX_LOOKUP_CANDS + hitCand];
                }
            } else {
                d_left_status[lOrig] = -1;
            }
        }
    }
    __syncthreads();
}

// K4: kernelExtractMatchedCoords — extract compact coordinate arrays
__global__ void kernelExtractMatchedCoords(
    const float2* __restrict__ d_left_pts,
    const float2* __restrict__ d_right_pts,
    const int*    __restrict__ d_left_lids,
    const MatchPair* __restrict__ d_pairs,
    int matchCount,
    float2* __restrict__ d_out_left,
    float2* __restrict__ d_out_right,
    int*    __restrict__ d_out_lids)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= matchCount) return;

    int li = d_pairs[idx].leftIdx;
    int ri = d_pairs[idx].rightIdx;

    d_out_left[idx]  = d_left_pts[li];
    d_out_right[idx] = d_right_pts[ri];
    d_out_lids[idx]  = d_left_lids[li];
}

// Helper: count status values
__global__ void kernelCountStatus(
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
// process — full GPU matching pipeline
// ============================================================================

LaserMatchScanResult LaserMatchScanCuda::Impl::process(
    const cv::cuda::GpuMat& d_left_points,
    const cv::cuda::GpuMat& d_left_line_ids,
    const cv::cuda::GpuMat& d_right_points,
    const cv::cuda::GpuMat& d_right_line_ids,
    cv::cuda::Stream& stream)
{
#ifndef NDEBUG
    ScopedFlag guard(&inProcess_);
#endif

    LaserMatchScanResult result;

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

#ifdef LM_ENABLE_TIMING
        LM_TIMER_DECL();
        CALIB_LOG_INFO("process() TIMING: leftCount={}, rightCount={}", leftCount, rightCount);
#endif

        // ── Step 1a: Generate composite sort keys on GPU + find max row key ──

        cudaMemsetAsync(d_max_row_key_.ptr<int>(), 0, 2 * sizeof(int), cuda_stream);

        {
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelGenerateSortKeys<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_left_points.ptr<float2>(),
                leftCount,
                params_.epipolar_row_step,
                reinterpret_cast<unsigned long long*>(d_left_keys_in_.ptr<double>()),
                reinterpret_cast<unsigned int*>(d_max_row_key_.ptr<int>()));
        }
        {
            int grid = (rightCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelGenerateSortKeys<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_right_points.ptr<float2>(),
                rightCount,
                params_.epipolar_row_step,
                reinterpret_cast<unsigned long long*>(d_right_keys_in_.ptr<double>()),
                reinterpret_cast<unsigned int*>(d_max_row_key_.ptr<int>()) + 1);
        }

#ifdef LM_ENABLE_TIMING
        LM_TIMER_MARK(step1a_keygen);
#endif

        // ── Step 1b: CUB Radix Sort (sort by row, then by u within row) ──

        {
            thrust::device_ptr<int> lptr(d_left_idx_in_.ptr<int>());
            thrust::sequence(thrust::cuda::par.on(cuda_stream), lptr, lptr + leftCount);
        }
        {
            thrust::device_ptr<int> rptr(d_right_idx_in_.ptr<int>());
            thrust::sequence(thrust::cuda::par.on(cuda_stream), rptr, rptr + rightCount);
        }

        cub::DeviceRadixSort::SortPairs(
            d_cub_temp_, cub_temp_size_,
            reinterpret_cast<unsigned long long*>(d_left_keys_in_.ptr<double>()),
            reinterpret_cast<unsigned long long*>(d_left_keys_out_.ptr<double>()),
            d_left_idx_in_.ptr<int>(),
            d_left_sorted_idx_.ptr<int>(),
            leftCount, 0, sizeof(unsigned long long) * 8, cuda_stream);

        cub::DeviceRadixSort::SortPairs(
            d_cub_temp_, cub_temp_size_,
            reinterpret_cast<unsigned long long*>(d_right_keys_in_.ptr<double>()),
            reinterpret_cast<unsigned long long*>(d_right_keys_out_.ptr<double>()),
            d_right_idx_in_.ptr<int>(),
            d_right_sorted_idx_.ptr<int>(),
            rightCount, 0, sizeof(unsigned long long) * 8, cuda_stream);

#ifdef LM_ENABLE_TIMING
        LM_TIMER_MARK(step1b_sort);
#endif

        // ── Step 1c: Scatter sorted point data ──

        {
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelScatterSorted<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_left_points.ptr<float2>(),
                d_left_line_ids.ptr<int>(),
                d_left_sorted_idx_.ptr<int>(),
                leftCount,
                d_left_sorted_pts_.ptr<float2>(),
                d_left_sorted_lids_.ptr<int>());
        }
        {
            int grid = (rightCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelScatterSorted<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_right_points.ptr<float2>(),
                d_right_line_ids.ptr<int>(),
                d_right_sorted_idx_.ptr<int>(),
                rightCount,
                d_right_sorted_pts_.ptr<float2>(),
                d_right_sorted_lids_.ptr<int>());
        }

#ifdef LM_ENABLE_TIMING
        LM_TIMER_MARK(step1c_scatter);
#endif

        // ── Step 2: Build CSR row indices on GPU (pre-allocated MAX_EPIPOLAR_ROWS) ──

        cudaMemsetAsync(d_left_row_start_.ptr<int>(), 0x7F,
                        MAX_EPIPOLAR_ROWS * sizeof(int), cuda_stream);
        cudaMemsetAsync(d_left_row_count_.ptr<int>(), 0,
                        MAX_EPIPOLAR_ROWS * sizeof(int), cuda_stream);
        cudaMemsetAsync(d_right_row_start_.ptr<int>(), 0x7F,
                        MAX_EPIPOLAR_ROWS * sizeof(int), cuda_stream);
        cudaMemsetAsync(d_right_row_count_.ptr<int>(), 0,
                        MAX_EPIPOLAR_ROWS * sizeof(int), cuda_stream);

        {
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelBuildCSRGPU<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<const unsigned long long*>(d_left_keys_out_.ptr<double>()),
                leftCount, MAX_EPIPOLAR_ROWS,
                d_left_row_start_.ptr<int>(),
                d_left_row_count_.ptr<int>());
        }
        {
            int grid = (rightCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelBuildCSRGPU<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<const unsigned long long*>(d_right_keys_out_.ptr<double>()),
                rightCount, MAX_EPIPOLAR_ROWS,
                d_right_row_start_.ptr<int>(),
                d_right_row_count_.ptr<int>());
        }

#ifdef LM_ENABLE_TIMING
        LM_TIMER_MARK(step2_csr);
#endif

        // ── Step 3: Initialize status and match count ──

        cudaMemsetAsync(d_left_status_.ptr<int>(), 0, leftCount * sizeof(int), cuda_stream);
        cudaMemsetAsync(d_right_status_.ptr<int>(), 0, rightCount * sizeof(int), cuda_stream);
        cudaMemsetAsync(d_match_count_.ptr<int>(), 0, sizeof(int), cuda_stream);

#ifdef LM_ENABLE_TIMING
        LM_TIMER_MARK(step3_init);
#endif

        // ── Step 4: Lookup uR_expected for each left point ──

        {
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelLookupTableMulti<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_left_sorted_pts_.ptr<float2>(),
                leftCount,
                d_map_byrow_.ptr<float4>(),
                d_map_row_start_.ptr<int>(),
                d_map_row_count_.ptr<int>(),
                params_.epipolar_row_step,
                params_.vL_tolerance,
                d_left_cand_uR_.ptr<float>(),
                d_left_cand_lid_.ptr<int>(),
                d_left_cand_count_.ptr<int>(),
                d_out_line_ids_sorted_.ptr<int>());
            kernelScatterLineIds<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_left_sorted_idx_.ptr<int>(),
                d_out_line_ids_sorted_.ptr<int>(),
                leftCount,
                d_out_line_ids_orig_.ptr<int>());
        }

#ifdef LM_ENABLE_TIMING
        LM_TIMER_MARK(step4_lookup);
#endif

        // ── Step 4b: Single sync point — download max_row_key to get numEpipolarRows_ ──

        int h_max_keys[2] = {0, 0};
        cudaMemcpyAsync(h_max_keys, d_max_row_key_.ptr<int>(), 2 * sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);

        numEpipolarRows_ = (h_max_keys[0] > h_max_keys[1] ? h_max_keys[0] : h_max_keys[1]) + 1;

        cudaError_t kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("GPU pipeline failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

#ifdef LM_ENABLE_TIMING
        LM_TIMER_MARK(step4b_sync);
#endif

        // ── Step 5: Match — one block per epipolar row ──

        size_t smem_size = static_cast<size_t>(params_.max_right_per_row) *
                           (sizeof(float) + 3 * sizeof(int));

        kernelMatch<<<numEpipolarRows_, 32, smem_size, cuda_stream>>>(
            d_left_sorted_pts_.ptr<float2>(),
            d_left_sorted_lids_.ptr<int>(),
            d_left_sorted_idx_.ptr<int>(),
            d_left_cand_uR_.ptr<float>(),
            d_left_cand_lid_.ptr<int>(),
            d_left_cand_count_.ptr<int>(),
            d_left_row_start_.ptr<int>(),
            d_left_row_count_.ptr<int>(),
            d_right_sorted_pts_.ptr<float2>(),
            d_right_sorted_lids_.ptr<int>(),
            d_right_sorted_idx_.ptr<int>(),
            d_right_row_start_.ptr<int>(),
            d_right_row_count_.ptr<int>(),
            params_.max_right_per_row,
            params_.match_threshold,
            leftCount,
            d_out_line_ids_orig_.ptr<int>(),
            reinterpret_cast<MatchPair*>(d_match_left_idx_.ptr<int>()),
            d_left_status_.ptr<int>(),
            d_right_status_.ptr<int>(),
            d_match_count_.ptr<int>());

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("kernelMatch failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        cudaStreamSynchronize(cuda_stream);

#ifdef LM_ENABLE_TIMING
        LM_TIMER_MARK(step5_match);
#endif

        // ── Step 6: Download match count ──

        int h_match_count = 0;
        cudaMemcpyAsync(&h_match_count, d_match_count_.ptr<int>(), sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);

        result.matchedCount = h_match_count;

#ifdef LM_ENABLE_TIMING
        LM_TIMER_MARK(step6_download);
#endif

        // ── Step 7: Extract matched coordinates ──

        result.d_matched_left = std::make_shared<cv::cuda::GpuMat>();
        result.d_matched_right = std::make_shared<cv::cuda::GpuMat>();
        result.d_matched_line_ids = std::make_shared<cv::cuda::GpuMat>();

        if (h_match_count > 0) {
            result.d_matched_left->create(1, h_match_count, CV_32FC2);
            result.d_matched_right->create(1, h_match_count, CV_32FC2);
            result.d_matched_line_ids->create(1, h_match_count, CV_32SC1);

            int grid = (h_match_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelExtractMatchedCoords<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_left_points.ptr<float2>(),
                d_right_points.ptr<float2>(),
                d_out_line_ids_orig_.ptr<int>(),
                reinterpret_cast<MatchPair*>(d_match_left_idx_.ptr<int>()),
                h_match_count,
                result.d_matched_left->ptr<float2>(),
                result.d_matched_right->ptr<float2>(),
                result.d_matched_line_ids->ptr<int>());

            kernel_err = cudaGetLastError();
            if (kernel_err != cudaSuccess) {
                result.success = false;
                result.message = std::string("kernelExtractMatchedCoords failed: ") + cudaGetErrorString(kernel_err);
                CALIB_LOG_ERROR("process(): {}", result.message);
                return result;
            }
        }

        cudaStreamSynchronize(cuda_stream);

#ifdef LM_ENABLE_TIMING
        LM_TIMER_MARK(step7_extract);
#endif

        // ── Step 8: Count excluded points ──

        cv::cuda::GpuMat d_excl_left(1, 1, CV_32SC1);
        cv::cuda::GpuMat d_excl_right(1, 1, CV_32SC1);
        cudaMemsetAsync(d_excl_left.ptr<int>(), 0, sizeof(int), cuda_stream);
        cudaMemsetAsync(d_excl_right.ptr<int>(), 0, sizeof(int), cuda_stream);

        {
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelCountStatus<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_left_status_.ptr<int>(), leftCount, -1, d_excl_left.ptr<int>());
        }
        {
            int grid = (rightCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelCountStatus<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_right_status_.ptr<int>(), rightCount, -1, d_excl_right.ptr<int>());
        }

        int h_excl_left = 0, h_excl_right = 0;
        cudaMemcpyAsync(&h_excl_left, d_excl_left.ptr<int>(), sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaMemcpyAsync(&h_excl_right, d_excl_right.ptr<int>(), sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);

        result.excludedLeftCount = h_excl_left;
        result.excludedRightCount = h_excl_right;

#ifdef LM_ENABLE_TIMING
        LM_TIMER_MARK(step8_excl);
#endif

        // ── Copy status arrays to result ──

        result.d_left_status = std::make_shared<cv::cuda::GpuMat>(d_left_status_.clone());
        result.d_right_status = std::make_shared<cv::cuda::GpuMat>(d_right_status_.clone());

        result.success = true;
        result.message = fmt::format("OK: {} matched, {} left excluded, {} right excluded",
                                     h_match_count, h_excl_left, h_excl_right);

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
