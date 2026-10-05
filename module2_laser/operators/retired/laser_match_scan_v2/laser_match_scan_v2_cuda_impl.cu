/**
 * @file laser_match_scan_v2_cuda_impl.cu
 * @brief 激光线匹配扫描CUDA算子 V2 — CUDA实现（struct Impl 方法）
 *
 * 相对 v1 的优化（语义等价，验证以 v1 输出为基准）：
 *   1. 32bit 排序键 (row 13b << 19 | xq 19b@1/256px)：CUB pass 8→4；keygen 顺带写 idx 序列
 *   2. kernelMatchV2：固定 grid=8192（免 max_row 回传同步/免 atomicMax 单点争用）；
 *      smem 16KB→4KB（占用 4→16 块/SM）；二分改 warp ballot 线性判定
 *      （外层贪心 li 序保持串行、全部 lane 冗余执行同一路径 → 语义与 v1 一致）
 *   3. kernelLookupTableMultiV2：候选去重走寄存器局部数组（免 stride-256B 全局重读+逐次除法）；
 *      kernelScatterLineIds 折叠进查表 kernel（省一次 launch）
 *   4. 候选数组 uR/lid 打包为 8B CandPackV2（kernelMatch 每候选 1 次读）
 *   5. 全程单次主机同步：末尾 pinned 16B 批量回传 [match,exclL,exclR,overflow]
 *   6. 结果/status 持久缓冲 + ROI/双缓冲视图（免每帧 7 组 malloc + clone）
 *   7. 无逐帧计时日志（v1 的 LM_ENABLE_TIMING 常开为已知开销）
 */

#include "laser_match_scan_v2_cuda_pimpl.h"
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

CALIB_DEFINE_LOG_TAG(07, LaserMatchScanCudaV2);

// ============================================================================
// Configuration Constants
// ============================================================================

static constexpr int BLOCK_SIZE = 256;
static constexpr int MATCH_BLOCK = 32;          // kernelMatchV2: 1 warp per row
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

LaserMatchScanCudaV2::Impl::Impl(const LaserMatchScanParamsV2& params)
    : params_(params)
{
    params_.validate();

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("[07-LaserMatchScanCudaV2] No CUDA devices found");
    }
    if (params_.deviceId >= deviceCount) {
        throw std::invalid_argument("[07-LaserMatchScanCudaV2] deviceId >= device count");
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

LaserMatchScanCudaV2::Impl::~Impl() {
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

bool LaserMatchScanCudaV2::Impl::allocateBuffers(int leftCount, int rightCount) {
    if (leftCount > 0) {
        d_left_keys_in_.create(1, leftCount, CV_32SC1);
        d_left_keys_out_.create(1, leftCount, CV_32SC1);
        d_left_idx_in_.create(1, leftCount, CV_32SC1);
        d_left_sorted_idx_.create(1, leftCount, CV_32SC1);
        d_left_sorted_pts_.create(1, leftCount, CV_32FC2);
        d_left_sorted_lids_.create(1, leftCount, CV_32SC1);
        d_cand_pack_.create(1, leftCount * MAX_LOOKUP_CANDS * 8, CV_8UC1);
        d_cand_count_.create(1, leftCount, CV_32SC1);
        d_out_line_ids_orig_.create(1, leftCount, CV_32SC1);
        d_match_pairs_.create(1, leftCount * 2, CV_32SC1);       // MatchPairV2 = 2 int
        d_matched_left_cap_.create(1, leftCount, CV_32FC2);
        d_matched_right_cap_.create(1, leftCount, CV_32FC2);
        d_matched_lids_cap_.create(1, leftCount, CV_32SC1);
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

void LaserMatchScanCudaV2::Impl::warmup(int maxLeftPoints, int maxRightPoints) {
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

bool LaserMatchScanCudaV2::Impl::LoadTempTable(const std::string& jsonPath) {
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

bool LaserMatchScanCudaV2::Impl::SetTempTable(
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

void LaserMatchScanCudaV2::Impl::SetCurrentTemperature(double temperature) {
    if (!tableLoaded_ || tempTable_.empty()) {
        throw std::runtime_error("[07-LaserMatchScanCudaV2] temp table not loaded");
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

void LaserMatchScanCudaV2::Impl::uploadMapTable(
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

void LaserMatchScanCudaV2::Impl::setParams(const LaserMatchScanParamsV2& params) {
#ifndef NDEBUG
    if (inProcess_.load()) {
        CALIB_LOG_ERROR("setParams(): called while process() is running");
        throw std::runtime_error("[07-LaserMatchScanCudaV2] setParams() called during process()");
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

struct MatchPairV2 {
    int leftIdx;
    int rightIdx;
};

struct CandPackV2 {
    float uR;
    int lid;
};

// K1: 32bit 复合排序键 + 顺带 idx 序列（省 thrust::sequence）
//     key = (rowKey << 19) | xq,  xq = x*256 取整 [0, 2^19)
//     row >= 8192 → KEY_SENTINEL（排序沉底, CSR build 丢弃, 与 v1 一致）
__global__ void kernelGenKeysIdxV2(
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
__global__ void kernelScatterSortedV2(
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
__global__ void kernelBuildCSRV2(
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

// K4: 查表（全局按行搜索, 扫描模式）—— v2 改动：
//     ① 去重走寄存器局部数组（uRg 接收时算一次; 免 stride-256B 全局重读与逐次除法）
//     ② 候选写为 8B 打包 CandPackV2
//     ③ kernelScatterLineIds 折叠进来（查到 firstLid 直接散回原始下标）
__global__ void kernelLookupTableMultiV2(
    const float2* __restrict__ d_left_sorted_pts,
    int leftCount,
    const float4* __restrict__ d_map_byrow,       // (xL, uR, lineId, yL)
    const int*    __restrict__ d_map_row_start,
    const int*    __restrict__ d_map_row_count,
    float epipolar_row_step,
    float vL_tolerance,
    CandPackV2*   __restrict__ d_cand,            // [idx*MAX_LOOKUP_CANDS + n]
    int*          __restrict__ d_cand_count,
    int*          __restrict__ d_out_line_ids_orig,
    const int*    __restrict__ d_sorted_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= leftCount) return;

    float xL = d_left_sorted_pts[idx].x;
    float yL = d_left_sorted_pts[idx].y;

    int orig = d_sorted_idx[idx];
    d_out_line_ids_orig[orig] = -1;      // 折叠 scatter: 先写 -1（早退路径同 v1）
    d_cand_count[idx] = 0;

    int row = (int)roundf(yL / epipolar_row_step);
    if (row < 0 || row >= 8192 || d_map_row_count[row] == 0) return;

    // 局部去重缓存
    int loc_lid[MAX_LOOKUP_CANDS];
    int loc_uRg[MAX_LOOKUP_CANDS];

    const int start = d_map_row_start[row];
    const int cnt   = d_map_row_count[row];
    int lo = start, hi = start + cnt - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (d_map_byrow[mid].x < xL) lo = mid + 1; else hi = mid;
    }
    // 从插入点向两侧展开收集全部 |x - xL| <= vL_tolerance 的条目（dir0 向上, dir1 向下; 同 v1）
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
                if (loc_lid[q] == lid && loc_uRg[q] == uRg) {
                    dup = true; break;
                }
            }
            if (!dup) {
                loc_lid[n] = lid;
                loc_uRg[n] = uRg;
                d_cand[idx * MAX_LOOKUP_CANDS + n].uR = uR;
                d_cand[idx * MAX_LOOKUP_CANDS + n].lid = lid;
                if (firstLid < 0) firstLid = lid;
                ++n;
            }
            p += step;
        }
    }
    d_cand_count[idx] = n;
    d_out_line_ids_orig[orig] = firstLid;
}

// K5: 匹配 —— 一行一 warp。外层贪心 li 序串行（全 lane 冗余执行同一数据 → 判定序列与 v1 一致），
//     每候选的右点窗口判定改 ballot 并行（升序取位 = v1 的升序线性扫描）。
//     语义复刻点：先占 occupied / ri==hitSm 仅对上次命中去重 / lid 一致性过滤 / totalHits 恰一命中。
__global__ void kernelMatchV2(
    const float2* __restrict__ d_left_pts,
    const int*    __restrict__ d_left_lids,
    const int*    __restrict__ d_left_orig_idx,
    const CandPackV2* __restrict__ d_cands,
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
    int*    __restrict__ d_line_ids_out,   // [origIdx] 命中候选线号
    MatchPairV2* __restrict__ d_match_pairs,
    int*    __restrict__ d_left_status,
    int*    __restrict__ d_right_status,
    int*    __restrict__ d_match_count,
    int*    __restrict__ d_overflow_flag)
{
    const int row = blockIdx.x;

    const int leftCount  = d_left_row_count[row];
    if (leftCount == 0) return;
    const int leftStart  = d_left_row_start[row];
    const int rightCount = d_right_row_count[row];
    const int rightStart = d_right_row_start[row];

    const int lane = threadIdx.x;
    const int cap = min(maxRightPerRow, SMEM_RIGHT_CAP);
    const int effRight = min(rightCount, maxRightPerRow);

    if (rightCount > SMEM_RIGHT_CAP && lane == 0) {
        *d_overflow_flag = 1;   // 行右点数超 smem 容量（v1 上限 1024, v2 256）
    }

    extern __shared__ char smem_raw[];
    float* sm_uR       = reinterpret_cast<float*>(smem_raw);
    int*   sm_lid      = reinterpret_cast<int*>(sm_uR + cap);
    int*   sm_occupied = sm_lid + cap;
    int*   sm_origIdx  = sm_occupied + cap;

    for (int i = lane; i < effRight; i += blockDim.x) {
        int gi = rightStart + i;
        sm_uR[i]       = d_right_pts[gi].x;
        sm_lid[i]      = d_right_lids[gi];
        sm_occupied[i] = 0;
        sm_origIdx[i]  = d_right_orig_idx[gi];
    }
    __syncthreads();

    // 全 lane 冗余执行同一 li 循环（数据一致 → 无分歧）；每候选 ballot 并行窗口判定
    for (int li = 0; li < leftCount; li++) {
        int gi = leftStart + li;
        const int nc = d_cand_count[gi];
        int lOrig    = d_left_orig_idx[gi];

        if (nc == 0) {
            if (lane == 0) d_left_status[lOrig] = -1;
            continue;
        }

        int hitSm = -1, hitCand = -1, totalHits = 0;
        for (int c = 0; c < nc && totalHits < 2; ++c) {
            const CandPackV2 cp = d_cands[gi * MAX_LOOKUP_CANDS + c];
            const float lo_val = cp.uR - threshold;
            const float hi_val = cp.uR + threshold;

            // 右点升序分块 ballot（块内 lane=ri, 块序即 ri 升序 → 命中序列与 v1 线性扫描一致）
            for (int base = 0; base < effRight && totalHits < 2; base += 32) {
                const int ri = base + lane;
                bool pred = false;
                if (ri < effRight) {
                    pred = (sm_uR[ri] >= lo_val) && (sm_uR[ri] <= hi_val)
                        && (sm_occupied[ri] == 0)
                        && !(sm_lid[ri] != 0 && sm_lid[ri] != cp.lid)
                        && (ri != hitSm);
                }
                unsigned int mask = __ballot_sync(0xffffffffu, pred);
                while (mask != 0u && totalHits < 2) {
                    const int b = __ffs(mask) - 1;
                    mask &= ~(1u << b);
                    totalHits++;
                    hitSm = base + b;
                    hitCand = c;
                }
            }
        }

        if (totalHits == 1) {
            if (lane == 0) {
                sm_occupied[hitSm] = 1;
                int rOrig = sm_origIdx[hitSm];

                int pos = atomicAdd(d_match_count, 1);
                if (pos < maxMatchPairs) {
                    d_match_pairs[pos].leftIdx  = lOrig;
                    d_match_pairs[pos].rightIdx = rOrig;
                    d_left_status[lOrig]  = 1;
                    d_right_status[rOrig] = 1;
                    d_line_ids_out[lOrig] = d_cands[gi * MAX_LOOKUP_CANDS + hitCand].lid;
                }
            }
        } else {
            if (lane == 0) d_left_status[lOrig] = -1;
        }
        __syncwarp();
    }
}

// K6: 提取紧凑坐标（count 设备端读取, 免主机回传定 grid）
__global__ void kernelExtractMatchedCoordsV2(
    const float2* __restrict__ d_left_pts,
    const float2* __restrict__ d_right_pts,
    const int*    __restrict__ d_left_lids,
    const MatchPairV2* __restrict__ d_pairs,
    const int*    __restrict__ d_match_count,
    int maxIdx,
    float2* __restrict__ d_out_left,
    float2* __restrict__ d_out_right,
    int*    __restrict__ d_out_lids)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= maxIdx) return;
    if (idx >= *d_match_count) return;

    int li = d_pairs[idx].leftIdx;
    int ri = d_pairs[idx].rightIdx;

    d_out_left[idx]  = d_left_pts[li];
    d_out_right[idx] = d_right_pts[ri];
    d_out_lids[idx]  = d_left_lids[li];
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

LaserMatchScanResultV2 LaserMatchScanCudaV2::Impl::process(
    const cv::cuda::GpuMat& d_left_points,
    const cv::cuda::GpuMat& d_left_line_ids,
    const cv::cuda::GpuMat& d_right_points,
    const cv::cuda::GpuMat& d_right_line_ids,
    cv::cuda::Stream& stream)
{
#ifndef NDEBUG
    ScopedFlagV2 guard(&inProcess_);
#endif

    LaserMatchScanResultV2 result;

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

        // ── Step 1: 32bit 键 + idx 序列（左右各一 kernel, 免 thrust::sequence/免 atomicMax）──
        {
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelGenKeysIdxV2<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_left_points.ptr<float2>(),
                leftCount,
                params_.epipolar_row_step,
                reinterpret_cast<unsigned int*>(d_left_keys_in_.ptr<int>()),
                d_left_idx_in_.ptr<int>());
        }
        {
            int grid = (rightCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelGenKeysIdxV2<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_right_points.ptr<float2>(),
                rightCount,
                params_.epipolar_row_step,
                reinterpret_cast<unsigned int*>(d_right_keys_in_.ptr<int>()),
                d_right_idx_in_.ptr<int>());
        }

        // ── Step 2: CUB 32bit RadixSort（pass 数为 64bit 的一半）──
        cub::DeviceRadixSort::SortPairs(
            d_cub_temp_, cub_temp_size_,
            reinterpret_cast<unsigned int*>(d_left_keys_in_.ptr<int>()),
            reinterpret_cast<unsigned int*>(d_left_keys_out_.ptr<int>()),
            d_left_idx_in_.ptr<int>(),
            d_left_sorted_idx_.ptr<int>(),
            leftCount, 0, sizeof(unsigned int) * 8, cuda_stream);

        cub::DeviceRadixSort::SortPairs(
            d_cub_temp_, cub_temp_size_,
            reinterpret_cast<unsigned int*>(d_right_keys_in_.ptr<int>()),
            reinterpret_cast<unsigned int*>(d_right_keys_out_.ptr<int>()),
            d_right_idx_in_.ptr<int>(),
            d_right_sorted_idx_.ptr<int>(),
            rightCount, 0, sizeof(unsigned int) * 8, cuda_stream);

        // ── Step 3: scatter sorted ──
        {
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelScatterSortedV2<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_left_points.ptr<float2>(),
                d_left_line_ids.ptr<int>(),
                d_left_sorted_idx_.ptr<int>(),
                leftCount,
                d_left_sorted_pts_.ptr<float2>(),
                d_left_sorted_lids_.ptr<int>());
        }
        {
            int grid = (rightCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelScatterSortedV2<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_right_points.ptr<float2>(),
                d_right_line_ids.ptr<int>(),
                d_right_sorted_idx_.ptr<int>(),
                rightCount,
                d_right_sorted_pts_.ptr<float2>(),
                d_right_sorted_lids_.ptr<int>());
        }

        // ── Step 4: CSR（memset + atomic 构建; sentinel 丢弃）──
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
            kernelBuildCSRV2<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<const unsigned int*>(d_left_keys_out_.ptr<int>()),
                leftCount, MAX_EPIPOLAR_ROWS,
                d_left_row_start_.ptr<int>(),
                d_left_row_count_.ptr<int>());
        }
        {
            int grid = (rightCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelBuildCSRV2<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
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

        // ── Step 5: status memset（ping-pong 当前代）──
        const int gen = status_gen_ & 1;
        cudaMemsetAsync(d_left_status_[gen].ptr<int>(), 0, leftCount * sizeof(int), cuda_stream);
        cudaMemsetAsync(d_right_status_[gen].ptr<int>(), 0, rightCount * sizeof(int), cuda_stream);

        // ── Step 6: 查表（含 lineIds scatter 折叠）──
        {
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelLookupTableMultiV2<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_left_sorted_pts_.ptr<float2>(),
                leftCount,
                d_map_byrow_.ptr<float4>(),
                d_map_row_start_.ptr<int>(),
                d_map_row_count_.ptr<int>(),
                params_.epipolar_row_step,
                params_.vL_tolerance,
                reinterpret_cast<CandPackV2*>(d_cand_pack_.ptr<unsigned char>()),
                d_cand_count_.ptr<int>(),
                d_out_line_ids_orig_.ptr<int>(),
                d_left_sorted_idx_.ptr<int>());
        }

        if ((kernel_err = cudaGetLastError()) != cudaSuccess) {
            result.success = false;
            result.message = std::string("lookup stage failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        // ── Step 7: 匹配（固定 grid=MAX_EPIPOLAR_ROWS, 免行数回传同步）──
        {
            size_t smem_size = static_cast<size_t>(SMEM_RIGHT_CAP) *
                               (sizeof(float) + 3 * sizeof(int));
            kernelMatchV2<<<MAX_EPIPOLAR_ROWS, MATCH_BLOCK, smem_size, cuda_stream>>>(
                d_left_sorted_pts_.ptr<float2>(),
                d_left_sorted_lids_.ptr<int>(),
                d_left_sorted_idx_.ptr<int>(),
                reinterpret_cast<const CandPackV2*>(d_cand_pack_.ptr<unsigned char>()),
                d_cand_count_.ptr<int>(),
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
                reinterpret_cast<MatchPairV2*>(d_match_pairs_.ptr<int>()),
                d_left_status_[gen].ptr<int>(),
                d_right_status_[gen].ptr<int>(),
                d_counts_.ptr<int>(),
                d_counts_.ptr<int>() + 3);

            if ((kernel_err = cudaGetLastError()) != cudaSuccess) {
                result.success = false;
                result.message = std::string("kernelMatchV2 failed: ") + cudaGetErrorString(kernel_err);
                CALIB_LOG_ERROR("process(): {}", result.message);
                return result;
            }
        }

        // ── Step 8: 提取（grid 上界 leftCount, kernel 内按设备端 count 早退）──
        {
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelExtractMatchedCoordsV2<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_left_points.ptr<float2>(),
                d_right_points.ptr<float2>(),
                d_out_line_ids_orig_.ptr<int>(),
                reinterpret_cast<const MatchPairV2*>(d_match_pairs_.ptr<int>()),
                d_counts_.ptr<int>(),
                leftCount,
                d_matched_left_cap_.ptr<float2>(),
                d_matched_right_cap_.ptr<float2>(),
                d_matched_lids_cap_.ptr<int>());
        }

        // ── Step 9: status 计数 ×2 → d_counts_[1..2] ──
        {
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelCountStatusV2<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_left_status_[gen].ptr<int>(), leftCount, -1, d_counts_.ptr<int>() + 1);
        }
        {
            int grid = (rightCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelCountStatusV2<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_right_status_[gen].ptr<int>(), rightCount, -1, d_counts_.ptr<int>() + 2);
        }

        if ((kernel_err = cudaGetLastError()) != cudaSuccess) {
            result.success = false;
            result.message = std::string("extract/count stage failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        // ── Step 10: 唯一一次主机同步 —— pinned 16B 批量回传 ──
        int* h_dst = h_counts_ ? h_counts_ : (int*)alloca(4 * sizeof(int));
        if (h_counts_) {
            cudaMemcpyAsync(h_counts_, d_counts_.ptr<int>(), 4 * sizeof(int),
                            cudaMemcpyDeviceToHost, cuda_stream);
        } else {
            cudaMemcpy(h_dst, d_counts_.ptr<int>(), 4 * sizeof(int),
                       cudaMemcpyDeviceToHost);
        }
        cudaStreamSynchronize(cuda_stream);

        if (h_counts_) {
            h_dst = h_counts_;
        }
        const int h_match_count = h_dst[0];
        const int h_excl_left   = h_dst[1];
        const int h_excl_right  = h_dst[2];

        if (h_dst[3] != 0 && !overflow_warned_) {
            overflow_warned_ = true;
            CALIB_LOG_WARN("kernelMatchV2: row right-count exceeded SMEM_RIGHT_CAP={}, "
                           "results diverge from v1 (cap=1024) on such rows", SMEM_RIGHT_CAP);
        }

        // ── 结果组装（容量缓冲 ROI 视图; 生命周期至下次 Execute）──
        result.matchedCount = h_match_count;
        result.excludedLeftCount = h_excl_left;
        result.excludedRightCount = h_excl_right;

        if (h_match_count > 0) {
            result.d_matched_left = std::make_shared<cv::cuda::GpuMat>(
                d_matched_left_cap_, cv::Rect(0, 0, h_match_count, 1));
            result.d_matched_right = std::make_shared<cv::cuda::GpuMat>(
                d_matched_right_cap_, cv::Rect(0, 0, h_match_count, 1));
            result.d_matched_line_ids = std::make_shared<cv::cuda::GpuMat>(
                d_matched_lids_cap_, cv::Rect(0, 0, h_match_count, 1));
        } else {
            result.d_matched_left = std::make_shared<cv::cuda::GpuMat>();
            result.d_matched_right = std::make_shared<cv::cuda::GpuMat>();
            result.d_matched_line_ids = std::make_shared<cv::cuda::GpuMat>();
        }
        result.d_left_status = std::make_shared<cv::cuda::GpuMat>(d_left_status_[gen]);
        result.d_right_status = std::make_shared<cv::cuda::GpuMat>(d_right_status_[gen]);

        status_gen_++;

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
