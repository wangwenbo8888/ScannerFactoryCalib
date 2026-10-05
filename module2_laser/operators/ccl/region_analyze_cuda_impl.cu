/**
 * @file region_analyze_cuda_impl.cu
 * @brief 激光连通域分析算子 CUDA 实现 — 全 GPU 精简管线（仅包围盒）
 *
 * 算法步骤（全部在 GPU 上完成）:
 *   Step 1:  GPU CCL               cv::cuda::connectedComponents (conn=8)
 *   Step 2:  GPU initStatsKernel    初始化统计缓冲 (5 数组)
 *   Step 3:  GPU computeStatsKernel 逐像素 atomic 统计 (5 原子, 无质心)
 *   Step 4:  GPU buildRemapKernel   面积过滤 + 稠密重编号
 *   Step 5:  GPU relabelKernel      生成重编号掩膜
 *   Step 6:  GPU compactStatsKernel 压缩包围盒结果
 *   Step 7:  同步 + D2H (pinned memory, 小传输)
 *
 * 性能要点（2026-08 优化）:
 *   - topX 面积收集 GPU 化：collectValidAreasKernel 把有效域面积压缩到紧凑
 *     缓冲（实测 ~30-100 个 int）再小传输 D2H，替代旧版整张 12.6MB 面积表
 *     pageable D2H + CPU 扫 315 万标签（旧版隐藏开销 ~10ms/帧）。
 *   - Execute 首次调用自动完成缓冲分配（自预热），同尺寸后续调用不再重复进入。
 *   - 注意：BKE 输出标签稀疏（真实 2048x1536 掩膜 max_label 实测达 ~3.1M，
 *     接近像素总数），统计数组必须按像素总数分配，不能用 max_label 缩小扫描。
 *   - 计时口径：事件区间 1~7 为 GPU 步骤分解；WALL 为整函数墙钟（含 topX/copyTo）。
 *
 * 对比旧版优化点:
 *   - 删除 minMaxLoc (~2.3ms → 0ms): 不需要提前知道 max_label
 *   - 删除质心 atomicAdd unsigned long long (~0.3ms): 5 原子代替 7 原子
 *   - pinned memory D2H (~1.0ms → ~0.3ms): 降低传输延迟
 */

#include "region_analyze_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/cudaimgproc.hpp>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

using namespace calib;


CALIB_DEFINE_LOG_TAG(07, RegionAnalyzerCUDA);

// ============================================================
// CUDA Kernels
// ============================================================

/// topX 前置收集：把有效域面积压缩到紧凑缓冲（替代旧版 12.6MB 面积表整表 D2H + CPU 全量扫描）
/// 注意：BKE 标签空间稀疏（max_label 可达像素总数），无法用 max_label 缩小扫描范围，
///       只能把"传输与 CPU 计算"压缩到有效域数量级（实测 ~30-100 个）。
__global__ void collectValidAreasKernel(
    const int* __restrict__ areas,
    int scan_range,
    int min_area, int max_area,
    int* __restrict__ out_areas,
    int* __restrict__ out_count,
    int out_cap)
{
    int label = blockIdx.x * blockDim.x + threadIdx.x + 1;  // skip background=0
    if (label >= scan_range) return;

    int a = areas[label];
    if (a >= min_area && a <= max_area && a > 0) {
        int slot = atomicAdd(out_count, 1);
        if (slot < out_cap) out_areas[slot] = a;
    }
}

/// 初始化统计缓冲（5 个数组 + remap + num_valid）
__global__ void initStatsKernel(
    int* __restrict__ areas,
    int* __restrict__ min_x, int* __restrict__ max_x,
    int* __restrict__ min_y, int* __restrict__ max_y,
    int* __restrict__ remap,
    int* __restrict__ num_valid,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    areas[idx] = 0;
    min_x[idx] = INT_MAX;
    max_x[idx] = INT_MIN;
    min_y[idx] = INT_MAX;
    max_y[idx] = INT_MIN;
    remap[idx] = 0;
    if (idx == 0) *num_valid = 0;
}

/// 逐像素统计：5 次 atomic（面积 + 包围盒），无质心
__global__ void computeStatsKernel(
    const int* __restrict__ labels,
    int rows, int cols, int step_elem,
    int* __restrict__ areas,
    int* __restrict__ min_x, int* __restrict__ max_x,
    int* __restrict__ min_y, int* __restrict__ max_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int label = labels[y * step_elem + x];
    if (label <= 0) return;

    atomicAdd(&areas[label], 1);
    atomicMin(&min_x[label], x);
    atomicMax(&max_x[label], x);
    atomicMin(&min_y[label], y);
    atomicMax(&max_y[label], y);
}

/// 遍历标签空间，面积过滤 + 分配稠密编号
__global__ void buildRemapKernel(
    const int* __restrict__ areas,
    int* __restrict__ remap,
    int* __restrict__ num_valid,
    int min_area, int max_area,
    int scan_range,
    int compact_max)
{
    int label = blockIdx.x * blockDim.x + threadIdx.x + 1;  // skip background=0
    if (label >= scan_range) return;

    int a = areas[label];
    if (a >= min_area && a <= max_area && a > 0) {
        int current = atomicAdd(num_valid, 1);
        if (current < compact_max) {
            remap[label] = current + 1;  // 1-indexed
        }
    }
}

/// 全图重编号：查 remap 表，写入输出掩膜
__global__ void relabelKernel(
    const int* __restrict__ labels,
    int* __restrict__ relabeled,
    int rows, int cols, int step_elem,
    const int* __restrict__ remap)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int label = labels[y * step_elem + x];
    relabeled[y * step_elem + x] = (label > 0) ? remap[label] : 0;
}

/// 压缩统计：遍历标签空间，将有效标签的包围盒写入紧凑数组
__global__ void compactStatsKernel(
    const int* __restrict__ min_x, const int* __restrict__ max_x,
    const int* __restrict__ min_y, const int* __restrict__ max_y,
    const int* __restrict__ remap,
    int scan_range,
    GPUComponentResult* __restrict__ results)
{
    int label = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (label >= scan_range) return;

    int new_id = remap[label];
    if (new_id <= 0) return;

    results[new_id - 1].label  = new_id;
    results[new_id - 1].bbox_x = min_x[label];
    results[new_id - 1].bbox_y = min_y[label];
    results[new_id - 1].bbox_w = max_x[label] - min_x[label] + 1;
    results[new_id - 1].bbox_h = max_y[label] - min_y[label] + 1;
}

// ============================================================
// Impl 构造 / 析构
// ============================================================

RegionAnalyzerCUDA::Impl::Impl(const RegionAnalyzerParams& params)
    : params_(params), old_device_id(params.deviceId)
{
    params_.validate();

    int device_count = cv::cuda::getCudaEnabledDeviceCount();
    if (device_count <= 0) {
        throw std::runtime_error("No CUDA-capable GPU found");
    }

    if (params_.deviceId >= device_count) {
        throw std::invalid_argument(
            "RegionAnalyzerParams::deviceId=" + std::to_string(params_.deviceId)
            + " exceeds available device count=" + std::to_string(device_count));
    }
    cv::cuda::setDevice(params_.deviceId);
}

// ============================================================
// GPU 缓冲分配 / 释放
// ============================================================

void RegionAnalyzerCUDA::Impl::allocateStatsBuffers(int max_labels) {
    freeStatsBuffers();

    cudaError_t err;
    size_t bytes_int = (size_t)max_labels * sizeof(int);

    err = cudaMalloc(&d_areas_, bytes_int);     if (err) goto fail;
    err = cudaMalloc(&d_min_x_, bytes_int);     if (err) goto fail;
    err = cudaMalloc(&d_max_x_, bytes_int);     if (err) goto fail;
    err = cudaMalloc(&d_min_y_, bytes_int);     if (err) goto fail;
    err = cudaMalloc(&d_max_y_, bytes_int);     if (err) goto fail;
    err = cudaMalloc(&d_remap_, bytes_int);      if (err) goto fail;
    err = cudaMalloc(&d_num_valid_, sizeof(int)); if (err) goto fail;
    err = cudaMalloc(&d_valid_count_, sizeof(int)); if (err) goto fail;
    err = cudaMalloc(&d_valid_areas_, (size_t)COMPACT_MAX * sizeof(int)); if (err) goto fail;
    err = cudaMalloc(&d_compact_, (size_t)COMPACT_MAX * sizeof(GPUComponentResult)); if (err) goto fail;

    // Pinned host memory
    err = cudaMallocHost(&h_pinned_count_, sizeof(int));  if (err) goto fail;
    err = cudaMallocHost(&h_pinned_compact_, (size_t)COMPACT_MAX * sizeof(GPUComponentResult)); if (err) goto fail;

    stats_capacity_ = max_labels;
    CALIB_LOG_INFO("GPU stats buffers allocated: max_labels={}, total={:.1f} MB",
                   max_labels,
                   (6.0 * bytes_int + (double)COMPACT_MAX * sizeof(GPUComponentResult)) / 1048576.0);
    return;

fail:
    freeStatsBuffers();
    throw std::runtime_error(std::string("[07-RegionAnalyzerCUDA] GPU OOM in allocateStatsBuffers: ")
                             + cudaGetErrorString(err));
}

void RegionAnalyzerCUDA::Impl::freeStatsBuffers() {
    if (d_areas_)     { cudaFree(d_areas_);     d_areas_     = nullptr; }
    if (d_min_x_)     { cudaFree(d_min_x_);     d_min_x_     = nullptr; }
    if (d_max_x_)     { cudaFree(d_max_x_);     d_max_x_     = nullptr; }
    if (d_min_y_)     { cudaFree(d_min_y_);     d_min_y_     = nullptr; }
    if (d_max_y_)     { cudaFree(d_max_y_);     d_max_y_     = nullptr; }
    if (d_remap_)     { cudaFree(d_remap_);     d_remap_     = nullptr; }
    if (d_num_valid_) { cudaFree(d_num_valid_); d_num_valid_ = nullptr; }
    if (d_valid_count_) { cudaFree(d_valid_count_); d_valid_count_ = nullptr; }
    if (d_valid_areas_) { cudaFree(d_valid_areas_); d_valid_areas_ = nullptr; }
    if (d_compact_)   { cudaFree(d_compact_);   d_compact_   = nullptr; }
    if (h_pinned_count_)   { cudaFreeHost(h_pinned_count_);   h_pinned_count_   = nullptr; }
    if (h_pinned_compact_) { cudaFreeHost(h_pinned_compact_); h_pinned_compact_ = nullptr; }
    stats_capacity_ = 0;
}

// ============================================================
// Warmup()
// ============================================================
void RegionAnalyzerCUDA::Impl::Warmup(int rows, int cols) {
    CALIB_LOG_INFO("warmup() pre-allocating: {}x{}", rows, cols);

    d_labels_.create(rows, cols, CV_32SC1);
    d_relabeled_.create(rows, cols, CV_32SC1);

    // BKE 标签空间稀疏（max_label 可达像素总数），统计数组必须按像素总数分配
    int step_elem = static_cast<int>(d_labels_.step / sizeof(int));
    int max_labels = step_elem * rows + 1;
    allocateStatsBuffers(max_labels);

    warmup_rows_ = rows;
    warmup_cols_ = cols;

    {
        cv::cuda::GpuMat d_dummy_mask(rows, cols, CV_8UC1, cv::Scalar(0));
        cv::cuda::GpuMat d_dummy_labels;
        cv::cuda::Stream warmup_stream;
        cv::cuda::connectedComponents(d_dummy_mask, d_dummy_labels, 8, CV_32SC1);
        warmup_stream.waitForCompletion();
    }

#ifndef NDEBUG
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("[07-RegionAnalyzerCUDA] GPU OOM: warmup validation failed: ")
            + cudaGetErrorString(err));
    }
#endif

    warmed_up_ = true;
    CALIB_LOG_INFO("warmup() completed successfully");
}

// ============================================================
// SetParams()
// ============================================================
void RegionAnalyzerCUDA::Impl::SetParams(const RegionAnalyzerParams& params) {
#ifndef NDEBUG
    assert(!inProcess_.load() && "setParams() called while analyze() is running - NOT thread-safe!");
#endif

    params_ = params;
    params_.validate();

    if (params_.deviceId != old_device_id) {
        int device_count = cv::cuda::getCudaEnabledDeviceCount();
        if (params_.deviceId >= device_count) {
            throw std::invalid_argument(
                "RegionAnalyzerParams::deviceId=" + std::to_string(params_.deviceId)
                + " exceeds available device count=" + std::to_string(device_count));
        }
        cv::cuda::setDevice(params_.deviceId);
        CALIB_LOG_INFO("setParams() device switched: {} -> {}", old_device_id, params_.deviceId);
        old_device_id = params_.deviceId;
    }

    CALIB_LOG_INFO("setParams() updated: minArea={}, maxArea={}, deviceId={}",
                   params_.minArea, params_.maxArea, params_.deviceId);
}

// ============================================================
// Execute() - shared_ptr 重载
// ============================================================
RegionAnalysisResult RegionAnalyzerCUDA::Impl::Execute(
    const std::shared_ptr<cv::cuda::GpuMat>& d_mask,
    cv::cuda::Stream& stream)
{
    return Execute(*d_mask, stream);
}

// ============================================================
// Execute() - const GpuMat& 重载（全 GPU 精简管线）
// ============================================================
RegionAnalysisResult RegionAnalyzerCUDA::Impl::Execute(
    const cv::cuda::GpuMat& d_inputBinaryMask,
    cv::cuda::Stream& stream)
{
#ifndef NDEBUG
    assert(!inProcess_.load() && "Concurrent analyze() calls detected - NOT thread-safe!");

    struct ScopedFlag {
        std::atomic<bool>* flag;
        ScopedFlag(std::atomic<bool>* f) : flag(f) {}
        ~ScopedFlag() { flag->store(false); }
    };

    ScopedFlag guard(&inProcess_);
    inProcess_.store(true);
#endif

    RegionAnalysisResult result;

    try {
        const auto wall_t0 = std::chrono::steady_clock::now();
        int rows = d_inputBinaryMask.rows;
        int cols = d_inputBinaryMask.cols;

        // ── 逐步骤计时（ev0..ev7，7 个区间）＋整函数墙钟 ──
        cudaEvent_t ev0, ev1, ev2, ev3, ev4, ev5, ev6, ev7;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventCreate(&ev2); cudaEventCreate(&ev3);
        cudaEventCreate(&ev4); cudaEventCreate(&ev5);
        cudaEventCreate(&ev6); cudaEventCreate(&ev7);

        if (!warmed_up_ || warmup_rows_ != rows || warmup_cols_ != cols) {
            CALIB_LOG_WARN("GPU buffer (re)allocation during analyze() - "
                           "call warmup() beforehand for production use");
            d_labels_.create(rows, cols, CV_32SC1);
            d_relabeled_.create(rows, cols, CV_32SC1);
            int step_elem = static_cast<int>(d_labels_.step / sizeof(int));
            // BKE 标签稀疏，容量必须覆盖像素总数（与 Warmup 一致）
            int want = step_elem * rows + 1;
            if (want > stats_capacity_) {
                allocateStatsBuffers(want);
            }
            warmup_rows_ = rows;
            warmup_cols_ = cols;
            warmed_up_ = true;  // 自预热：同尺寸后续调用不再进入本分支
        }

        int scan_range = stats_capacity_;  // 扫描范围 = 已分配容量（BKE 标签稀疏，无法缩小）

        // === Step 1: GPU CCL ===
        cudaEventRecord(ev0);
        cv::cuda::connectedComponents(d_inputBinaryMask, d_labels_, 8, CV_32SC1);
        cudaEventRecord(ev1);

        // === Step 2: GPU initStats (5 数组 + remap + num_valid) ===
        {
            int threads = 256;
            int blocks = (scan_range + threads - 1) / threads;
            initStatsKernel<<<blocks, threads>>>(
                d_areas_,
                d_min_x_, d_max_x_, d_min_y_, d_max_y_,
                d_remap_, d_num_valid_, scan_range);
        }
        cudaEventRecord(ev2);

        // === Step 3: GPU computeStats (5 原子: area + bbox, 无质心) ===
        {
            int step_elem = static_cast<int>(d_labels_.step / sizeof(int));
            dim3 block2d(32, 8);
            dim3 grid2d((cols + 31) / 32, (rows + 7) / 8);
            computeStatsKernel<<<grid2d, block2d>>>(
                d_labels_.ptr<int>(), rows, cols, step_elem,
                d_areas_,
                d_min_x_, d_max_x_, d_min_y_, d_max_y_);
        }
        cudaEventRecord(ev3);

        // === Step 4: GPU buildRemap (面积过滤 + 稠密编号) ===
        {
            int threads = 256;
            int blocks = (scan_range + threads - 1) / threads;
            buildRemapKernel<<<blocks, threads>>>(
                d_areas_, d_remap_, d_num_valid_,
                params_.minArea, params_.maxArea,
                scan_range, COMPACT_MAX);
        }
        cudaEventRecord(ev4);

        // === Step 5: GPU relabel ===
        {
            int step_elem = static_cast<int>(d_labels_.step / sizeof(int));
            dim3 block2d(32, 8);
            dim3 grid2d((cols + 31) / 32, (rows + 7) / 8);
            relabelKernel<<<grid2d, block2d>>>(
                d_labels_.ptr<int>(), d_relabeled_.ptr<int>(),
                rows, cols, step_elem, d_remap_);
        }
        cudaEventRecord(ev5);

        // === Step 6: GPU compactStats ===
        {
            int threads = 256;
            int blocks = (scan_range + threads - 1) / threads;
            compactStatsKernel<<<blocks, threads>>>(
                d_min_x_, d_max_x_, d_min_y_, d_max_y_,
                d_remap_, scan_range, d_compact_);
        }
        cudaEventRecord(ev6);

        // === Step 7: 同步 + D2H (pinned memory) ===
        cudaDeviceSynchronize();
        cudaEventRecord(ev7);

        int num_valid = 0;
        cudaMemcpy(h_pinned_count_, d_num_valid_, sizeof(int), cudaMemcpyDeviceToHost);
        num_valid = *h_pinned_count_;

        // === Step 7b: topX 二次筛选（可选; 只保留面积最大的前 X 个域）===
        // 条件: topXCount>0 且首遍有效域数超限。实现: GPU 把有效域面积压缩到
        // 紧凑缓冲（仅几十~几百个 int）→ 小传输 D2H → CPU nth_element 求阈值
        // → 提高下界重跑 buildRemap/relabel/compactStats。
        // （旧版拉整张 12.6MB 面积表 pageable D2H + CPU 扫 315 万标签，~10ms/帧）
        // 边界并列（面积==阈值）全部保留 → 结果域数可能 >= X。
        int truncated_from = 0;
        if (params_.topXCount > 0 && num_valid > params_.topXCount) {
            cudaMemsetAsync(d_valid_count_, 0, sizeof(int));
            {
                int threads = 256;
                int blocks = (scan_range + threads - 1) / threads;
                collectValidAreasKernel<<<blocks, threads>>>(
                    d_areas_, scan_range,
                    params_.minArea, params_.maxArea,
                    d_valid_areas_, d_valid_count_, COMPACT_MAX);
            }

            int valid_pairs = 0;
            cudaMemcpy(&valid_pairs, d_valid_count_, sizeof(int),
                       cudaMemcpyDeviceToHost);
            if (valid_pairs > COMPACT_MAX) valid_pairs = COMPACT_MAX;

            std::vector<int> validAreas(static_cast<size_t>(valid_pairs));
            if (valid_pairs > 0) {
                cudaMemcpy(validAreas.data(), d_valid_areas_,
                           static_cast<size_t>(valid_pairs) * sizeof(int),
                           cudaMemcpyDeviceToHost);
            }

            if (validAreas.size() > static_cast<size_t>(params_.topXCount)) {
                const size_t k = static_cast<size_t>(params_.topXCount) - 1;
                std::nth_element(validAreas.begin(), validAreas.begin() + k,
                                 validAreas.end(), std::greater<int>());
                const int threshold = validAreas[k];
                const int newMin = std::max(params_.minArea, threshold);
                truncated_from = num_valid;

                // 只清 remap/num_valid（勿用 initStatsKernel——它会把 areas 一并清零）
                cudaMemsetAsync(d_remap_, 0,
                                static_cast<size_t>(scan_range) * sizeof(int));
                cudaMemsetAsync(d_num_valid_, 0, sizeof(int));
                {
                    int threads = 256;
                    int blocks = (scan_range + threads - 1) / threads;
                    buildRemapKernel<<<blocks, threads>>>(
                        d_areas_, d_remap_, d_num_valid_,
                        newMin, params_.maxArea,
                        scan_range, COMPACT_MAX);
                }
                {
                    int step_elem = static_cast<int>(d_labels_.step / sizeof(int));
                    dim3 block2d(32, 8);
                    dim3 grid2d((cols + 31) / 32, (rows + 7) / 8);
                    relabelKernel<<<grid2d, block2d>>>(
                        d_labels_.ptr<int>(), d_relabeled_.ptr<int>(),
                        rows, cols, step_elem, d_remap_);
                }
                {
                    int threads = 256;
                    int blocks = (scan_range + threads - 1) / threads;
                    compactStatsKernel<<<blocks, threads>>>(
                        d_min_x_, d_max_x_, d_min_y_, d_max_y_,
                        d_remap_, scan_range, d_compact_);
                }
                cudaDeviceSynchronize();

                cudaMemcpy(h_pinned_count_, d_num_valid_, sizeof(int),
                           cudaMemcpyDeviceToHost);
                num_valid = *h_pinned_count_;
                CALIB_LOG_INFO("topX filter: kept {} of {} components "
                               "(topXCount={}, area threshold={})",
                               num_valid, truncated_from,
                               params_.topXCount, threshold);
            }
        }

        if (num_valid > 0) {
            int copy_count = (num_valid < COMPACT_MAX) ? num_valid : COMPACT_MAX;
            cudaMemcpy(h_pinned_compact_, d_compact_,
                       (size_t)copy_count * sizeof(GPUComponentResult),
                       cudaMemcpyDeviceToHost);
            result.components.resize(num_valid);
            memcpy(result.components.data(), h_pinned_compact_,
                   (size_t)copy_count * sizeof(GPUComponentResult));
        }
        result.componentCount = num_valid;

        // 复制重编号掩膜到独立 GpuMat
        auto d_out = std::make_shared<cv::cuda::GpuMat>();
        d_relabeled_.copyTo(*d_out, stream);
        result.d_labeledMask = d_out;

        // ── 输出逐步骤计时（事件区间=GPU 步骤分解；WALL=整函数墙钟，含 topX/copyTo）──
        cudaEventSynchronize(ev7);
        float ms1, ms2, ms3, ms4, ms5, ms6, ms7;
        cudaEventElapsedTime(&ms1, ev0, ev1);  // CCL
        cudaEventElapsedTime(&ms2, ev1, ev2);  // initStats
        cudaEventElapsedTime(&ms3, ev2, ev3);  // computeStats
        cudaEventElapsedTime(&ms4, ev3, ev4);  // buildRemap
        cudaEventElapsedTime(&ms5, ev4, ev5);  // relabel
        cudaEventElapsedTime(&ms6, ev5, ev6);  // compactStats
        cudaEventElapsedTime(&ms7, ev6, ev7);  // sync + D2H
        float total = ms1 + ms2 + ms3 + ms4 + ms5 + ms6 + ms7;
        const double wall_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - wall_t0).count();

        CALIB_LOG_INFO(
            "=== analyze() per-step timing ===\n"
            "  Step 1: GPU CCL          {:>8.3f} ms  ({:>5.1f}%)\n"
            "  Step 2: initStats        {:>8.3f} ms  ({:>5.1f}%)\n"
            "  Step 3: computeStats     {:>8.3f} ms  ({:>5.1f}%)\n"
            "  Step 4: buildRemap       {:>8.3f} ms  ({:>5.1f}%)\n"
            "  Step 5: relabel          {:>8.3f} ms  ({:>5.1f}%)\n"
            "  Step 6: compactStats     {:>8.3f} ms  ({:>5.1f}%)\n"
            "  Step 7: sync+D2H(pinned) {:>8.3f} ms  ({:>5.1f}%)\n"
            "  -----------------------------------------\n"
            "  TOTAL(GPU steps)         {:>8.3f} ms\n"
            "  WALL(含topX+copyTo)      {:>8.3f} ms\n"
            "  components={}, image={}x{}, scan_range={}",
            ms1, ms1/total*100,
            ms2, ms2/total*100,
            ms3, ms3/total*100,
            ms4, ms4/total*100,
            ms5, ms5/total*100,
            ms6, ms6/total*100,
            ms7, ms7/total*100,
            total, wall_ms,
            num_valid, rows, cols, scan_range);

        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        cudaEventDestroy(ev2); cudaEventDestroy(ev3);
        cudaEventDestroy(ev4); cudaEventDestroy(ev5);
        cudaEventDestroy(ev6); cudaEventDestroy(ev7);

        // === 质量标记 ===
        result.success = true;
        if (result.componentCount == 0) {
            result.qualityFlag = calib::QualityFlag::Warning;
            result.message = "No connected components found after filtering";
            CALIB_LOG_WARN("Connected component count = 0, investigation recommended");
        } else if (result.componentCount > 200) {
            result.qualityFlag = calib::QualityFlag::Degraded;
            result.message = "Abnormal component count: " + std::to_string(result.componentCount);
            CALIB_LOG_WARN("Connected component count > 200 ({}), results may be unreliable",
                           result.componentCount);
        } else {
            result.qualityFlag = calib::QualityFlag::Normal;
            result.message = "Analysis successful";
        }

        CALIB_LOG_DEBUG("analyze() completed: {} components, qualityFlag={}",
                        result.componentCount,
                        static_cast<int>(result.qualityFlag));

    } catch (const cv::Exception& e) {
        result.success = false;
        result.message = std::string("OpenCV error: ") + e.what();
        CALIB_LOG_ERROR("analyze() OpenCV exception: {}", e.what());
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Error: ") + e.what();
        CALIB_LOG_ERROR("analyze() exception: {}", e.what());
    }

    return result;
}
