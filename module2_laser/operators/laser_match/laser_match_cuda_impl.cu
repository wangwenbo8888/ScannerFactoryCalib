/**
 * @file laser_match_cuda_impl.cu
 * @brief 激光线匹配CUDA算子 - CUDA实现（struct Impl 方法 + GPU Kernel）
 *
 * 算法步骤：
 *   Step 1: 量化左右点集行索引 (kernelQuantizeRowIdx)
 *   Step 2: 构建左点排序键 (kernelBuildSortKeys) + CUB 基数排序
 *   Step 3: 右点二分探测匹配 (kernelProbeMatchSorted)
 *   Step 4: CUB DeviceSelect::Flagged 压缩输出
 *
 * 2026-09-04 确定性重写（AGENTS 已知限制 #9 根治）：原实现为并行开放寻址哈希
 * （kernelBuildHash/kernelProbeMatch），slot 布局与 >max_probe 丢弃点随线程
 * 调度逐次变化——输入逐位相同、匹配成员集合每跑一变（test_frontend_ab_compare
 * 逐级量化哈希实锤）。本版改排序＋二分：排序键 (复合键<<32)|左点下标 全唯一，
 * 排序结果与调度序无关；**代表点＝同键段坐标字典序最小（序无关）**——曾用
 * "组内最小 idx"仍隐依赖输入点序（steger_fast 原子 scatter 输出序随执行环境
 * 变化, 跨 exe 实测 matched +64/+109）, 字典序准则使输出唯一确定于点集本身。
 * 不再有链上撞 empty 误失配与超深丢弃——召回 ≥ 原实现。⚠ 与 09 侧逐字拷贝
 * 分歧，同步债务见 AGENTS。
 */

#include "laser_match_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cub/cub.cuh>
#include <cmath>
#include <stdexcept>

using namespace calib;


CALIB_DEFINE_LOG_TAG(07, LaserMatchCuda);

// ============================================================================
// Configuration Constants
// ============================================================================

static constexpr int BLOCK_SIZE = 256;

// ============================================================================
// Device Helpers
// ============================================================================

__device__ __forceinline__ unsigned int makeCompositeKey(int row_idx, int frame_id) {
    return (static_cast<unsigned int>(row_idx) << 16) | (static_cast<unsigned int>(frame_id) & 0xFFFF);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void __launch_bounds__(256, 4) kernelQuantizeRowIdx(
    const float2* __restrict__ d_points,
    int count,
    float step,
    int* __restrict__ d_rowidx)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    d_rowidx[idx] = static_cast<int>(roundf(d_points[idx].y / step));
}

__global__ void __launch_bounds__(256, 4) kernelBuildSortKeys(
    const int* __restrict__ d_rowidx,
    const int* __restrict__ d_fids,
    int count,
    unsigned long long* __restrict__ d_sort_keys)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    // 排序键 = (复合键<<32) | 左点下标——键全唯一 → 排序结果与调度序/稳定性
    // 无关（确定）。同键段在排序后连续（代表点选取见 kernelProbeMatchSorted,
    // 为序无关的坐标字典序）。
    d_sort_keys[idx] =
        (static_cast<unsigned long long>(
             makeCompositeKey(d_rowidx[idx], d_fids[idx])) << 32)
        | static_cast<unsigned long long>(idx);
}

__global__ void __launch_bounds__(256, 4) kernelProbeMatchSorted(
    const int* __restrict__ d_right_rowidx,
    const int* __restrict__ d_right_fids,
    const float2* __restrict__ d_right_points,
    const float2* __restrict__ d_left_points,
    const unsigned long long* __restrict__ d_sorted_keys,
    int right_count,
    int left_count,
    float min_disp,
    float max_disp,
    int* __restrict__ d_flags,
    float2* __restrict__ d_match_left,
    float2* __restrict__ d_match_right,
    int* __restrict__ d_match_fids)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= right_count) {
        return;
    }

    const unsigned int key32 = makeCompositeKey(d_right_rowidx[idx], d_right_fids[idx]);

    // lower_bound 二分：首个复合键 >= key32 的元素 = 该键组组首（最小左点下标）。
    // 注意比较基准两侧同为 32 位复合键（排序键高 32 位）
    int lo = 0, hi = left_count;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (static_cast<unsigned int>(d_sorted_keys[mid] >> 32) < key32) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    d_flags[idx] = 0;
    if (lo >= left_count || static_cast<unsigned int>(d_sorted_keys[lo] >> 32) != key32) {
        return;   // 无同键左点
    }

    // 代表点选择（2026-09-04 二次修正: 序无关）: 扫描同键段取坐标字典序最小
    // (x, tie y) 的点。原"组内最小 idx"依赖输入点序——steger_fast 的原子 scatter
    // 输出序随执行环境变化（集合不变序变, 跨 exe 实测 matched +64/+109）, 造成
    // 同算法不同 exe 结果不同; 字典序准则使代表点唯一确定于点集本身。
    int rep = static_cast<int>(d_sorted_keys[lo] & 0xFFFFFFFFull);
    float2 repP = d_left_points[rep];
    for (int j = lo + 1; j < left_count; ++j) {
        if (static_cast<unsigned int>(d_sorted_keys[j] >> 32) != key32) {
            break;
        }
        const int cand = static_cast<int>(d_sorted_keys[j] & 0xFFFFFFFFull);
        const float2 cp = d_left_points[cand];
        if (cp.x < repP.x || (cp.x == repP.x && cp.y < repP.y)) {
            rep = cand;
            repP = cp;
        }
    }

    const float2 lp = repP;
    const float2 rp = d_right_points[idx];
    const float disparity = lp.x - rp.x;
    if (disparity >= min_disp && disparity <= max_disp) {
        d_flags[idx] = 1;
        d_match_left[idx] = lp;
        d_match_right[idx] = rp;
        d_match_fids[idx] = d_right_fids[idx];
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
// Impl Implementation
// ============================================================================

LaserMatchCuda::Impl::Impl(const LaserMatchParams& params)
    : params_(params)
{
    params_.validate();

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("[07-LaserMatchCuda] No CUDA devices found");
    }
    if (params_.deviceId >= deviceCount) {
        throw std::invalid_argument("[07-LaserMatchCuda] deviceId >= device count");
    }

    old_device_id_ = 0;
    cudaGetDevice(&old_device_id_);
    if (params_.deviceId != old_device_id_) {
        cudaSetDevice(params_.deviceId);
    }
}

LaserMatchCuda::Impl::~Impl() {
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        CALIB_LOG_ERROR("cudaDeviceSynchronize in destructor failed: {}",
                        cudaGetErrorString(sync_err));
    }

    safeCudaFree(d_cub_temp_);
    cub_temp_size_ = 0;
    last_max_count_ = 0;
}

bool LaserMatchCuda::Impl::allocateBuffers(int leftCount, int rightCount) {
    const int maxCount = (leftCount > rightCount) ? leftCount : rightCount;

    if (maxCount <= 0) {
        return true;
    }

    if (maxCount <= last_max_count_ &&
        !d_sort_keys_.empty() &&
        !d_flags_.empty()) {
        return true;
    }

    CALIB_LOG_DEBUG("Allocating GPU buffers for left={}, right={}", leftCount, rightCount);

    // 2026-09-04 越界修复：缓冲统一按 maxCount 分配。原 d_sort_keys_/d_sort_alt_
    // 按 leftCount、d_temp_*/d_flags_ 按 rightCount 分配，且 early-return 只看
    // maxCount —— 当本帧 left/right 单侧超过历史分配尺寸而 maxCount 未增长时
    // （实例: p5 分配 right=27637, p6 right=27834 而 maxCount 28859>28517 不重建），
    // kernel 写越界踩相邻显存（实测: flags 尾部被踩 → 上一帧残留点混入输出＝
    // "幽灵线号"点, 踩法随堆布局而变＝环境依赖之谜）。原哈希实现同病（matched
    // 抖动史上即有）。排序/压缩仅消费前 left/right_count 个元素, 按 maxCount
    // 分配不影响数值。
    // CV_32SC2 = 8B/元素, 复用为 unsigned long long 排序键
    d_sort_keys_.create(1, maxCount, CV_32SC2);
    d_sort_alt_.create(1, maxCount, CV_32SC2);

    d_left_rowidx_.create(1, maxCount, CV_32SC1);
    d_right_rowidx_.create(1, maxCount, CV_32SC1);

    d_flags_.create(1, maxCount, CV_32SC1);
    d_temp_left_.create(1, maxCount, CV_32FC2);
    d_temp_right_.create(1, maxCount, CV_32FC2);
    d_temp_fids_.create(1, maxCount, CV_32SC1);

    d_out_left_.create(1, maxCount, CV_32FC2);
    d_out_right_.create(1, maxCount, CV_32FC2);
    d_out_fids_.create(1, maxCount, CV_32SC1);
    d_out_count_.create(1, 1, CV_32SC1);

    last_max_count_ = maxCount;
    return true;
}

void LaserMatchCuda::Impl::Warmup(int leftCount, int rightCount) {
    if (leftCount <= 0 || rightCount <= 0) {
        CALIB_LOG_WARN("warmup(): invalid leftCount={}, rightCount={}, skipping", leftCount, rightCount);
        return;
    }

    if (warmed_up_ && warmup_left_ == leftCount && warmup_right_ == rightCount) {
        CALIB_LOG_DEBUG("warmup(): already warmed up for left={}, right={}", leftCount, rightCount);
        return;
    }

    allocateBuffers(leftCount, rightCount);
    warmed_up_ = true;
    warmup_left_ = leftCount;
    warmup_right_ = rightCount;

    CALIB_LOG_INFO("warmup(): allocated GPU buffers for left={}, right={}", leftCount, rightCount);
}

LaserMatchResult LaserMatchCuda::Impl::Execute(
    const cv::cuda::GpuMat& d_left_points,
    const cv::cuda::GpuMat& d_left_line_ids,
    const cv::cuda::GpuMat& d_right_points,
    const cv::cuda::GpuMat& d_right_line_ids,
    cv::cuda::Stream stream)
{
#ifndef NDEBUG
    ScopedFlag guard(&inProcess_);
#endif

    LaserMatchResult result;

    try {
        const int leftCount = d_left_points.rows * d_left_points.cols;
        const int rightCount = d_right_points.rows * d_right_points.cols;

        if (leftCount == 0 || rightCount == 0) {
            result.success = true;
            result.message = "Empty input, no matching";
            result.d_matched_left = std::make_shared<cv::cuda::GpuMat>();
            result.d_matched_right = std::make_shared<cv::cuda::GpuMat>();
            result.d_matched_line_ids = std::make_shared<cv::cuda::GpuMat>();
            result.matchCount = 0;
            return result;
        }

        if (!allocateBuffers(leftCount, rightCount)) {
            result.success = false;
            result.message = "GPU buffer allocation failed";
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

        const float2* d_left_ptr = d_left_points.ptr<float2>();
        const int* d_left_fid_ptr = d_left_line_ids.ptr<int>();
        const float2* d_right_ptr = d_right_points.ptr<float2>();
        const int* d_right_fid_ptr = d_right_line_ids.ptr<int>();

        const int left_grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const int right_grid = (rightCount + BLOCK_SIZE - 1) / BLOCK_SIZE;

        unsigned long long* d_keys_in =
            reinterpret_cast<unsigned long long*>(d_sort_keys_.ptr<int>());
        unsigned long long* d_keys_out =
            reinterpret_cast<unsigned long long*>(d_sort_alt_.ptr<int>());

        kernelQuantizeRowIdx<<<left_grid, BLOCK_SIZE, 0, cuda_stream>>>(
            d_left_ptr, leftCount, params_.epipolar_row_step,
            d_left_rowidx_.ptr<int>());

        kernelQuantizeRowIdx<<<right_grid, BLOCK_SIZE, 0, cuda_stream>>>(
            d_right_ptr, rightCount, params_.epipolar_row_step,
            d_right_rowidx_.ptr<int>());

        kernelBuildSortKeys<<<left_grid, BLOCK_SIZE, 0, cuda_stream>>>(
            d_left_rowidx_.ptr<int>(), d_left_fid_ptr, leftCount, d_keys_in);

        cudaError_t kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Kernel launch failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        // CUB 基数排序（升序, 全 64 位）——键全唯一, 结果与调度序无关（确定）
        size_t temp_bytes_sort = 0;
        cub::DeviceRadixSort::SortKeys(
            nullptr, temp_bytes_sort, d_keys_in, d_keys_out,
            leftCount, 0, sizeof(unsigned long long) * 8, cuda_stream);

        size_t temp_bytes_left = 0;
        cub::DeviceSelect::Flagged(
            nullptr, temp_bytes_left,
            d_temp_left_.ptr<float2>(), d_flags_.ptr<int>(),
            d_out_left_.ptr<float2>(), d_out_count_.ptr<int>(),
            rightCount, cuda_stream);

        size_t temp_bytes_right = 0;
        cub::DeviceSelect::Flagged(
            nullptr, temp_bytes_right,
            d_temp_right_.ptr<float2>(), d_flags_.ptr<int>(),
            d_out_right_.ptr<float2>(), d_out_count_.ptr<int>(),
            rightCount, cuda_stream);

        size_t temp_bytes_fids = 0;
        cub::DeviceSelect::Flagged(
            nullptr, temp_bytes_fids,
            d_temp_fids_.ptr<int>(), d_flags_.ptr<int>(),
            d_out_fids_.ptr<int>(), d_out_count_.ptr<int>(),
            rightCount, cuda_stream);

        size_t max_temp_bytes = temp_bytes_sort;
        if (temp_bytes_left > max_temp_bytes) max_temp_bytes = temp_bytes_left;
        if (temp_bytes_right > max_temp_bytes) max_temp_bytes = temp_bytes_right;
        if (temp_bytes_fids > max_temp_bytes) max_temp_bytes = temp_bytes_fids;

        if (max_temp_bytes > cub_temp_size_ || d_cub_temp_ == nullptr) {
            safeCudaFree(d_cub_temp_);

            cudaError_t alloc_err = cudaMalloc(&d_cub_temp_, max_temp_bytes);
            if (alloc_err != cudaSuccess) {
                result.success = false;
                result.message = std::string("CUB temp storage allocation failed: ") + cudaGetErrorString(alloc_err);
                CALIB_LOG_ERROR("process(): {}", result.message);
                return result;
            }
            cub_temp_size_ = max_temp_bytes;
        }

        cudaError_t cub_err = cub::DeviceRadixSort::SortKeys(
            d_cub_temp_, cub_temp_size_, d_keys_in, d_keys_out,
            leftCount, 0, sizeof(unsigned long long) * 8, cuda_stream);
        if (cub_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUB::SortKeys failed: ") + cudaGetErrorString(cub_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Kernel launch failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        kernelProbeMatchSorted<<<right_grid, BLOCK_SIZE, 0, cuda_stream>>>(
            d_right_rowidx_.ptr<int>(), d_right_fid_ptr, d_right_ptr,
            d_left_ptr,
            d_keys_out,
            rightCount, leftCount,
            params_.min_disparity, params_.max_disparity,
            d_flags_.ptr<int>(),
            d_temp_left_.ptr<float2>(), d_temp_right_.ptr<float2>(),
            d_temp_fids_.ptr<int>());

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Kernel launch failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        cub_err = cub::DeviceSelect::Flagged(
            d_cub_temp_, cub_temp_size_,
            d_temp_left_.ptr<float2>(), d_flags_.ptr<int>(),
            d_out_left_.ptr<float2>(), d_out_count_.ptr<int>(),
            rightCount, cuda_stream);
        if (cub_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUB::Flagged (left) failed: ") + cudaGetErrorString(cub_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        cub_err = cub::DeviceSelect::Flagged(
            d_cub_temp_, cub_temp_size_,
            d_temp_right_.ptr<float2>(), d_flags_.ptr<int>(),
            d_out_right_.ptr<float2>(), d_out_count_.ptr<int>(),
            rightCount, cuda_stream);
        if (cub_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUB::Flagged (right) failed: ") + cudaGetErrorString(cub_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        cub_err = cub::DeviceSelect::Flagged(
            d_cub_temp_, cub_temp_size_,
            d_temp_fids_.ptr<int>(), d_flags_.ptr<int>(),
            d_out_fids_.ptr<int>(), d_out_count_.ptr<int>(),
            rightCount, cuda_stream);
        if (cub_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUB::Flagged (fids) failed: ") + cudaGetErrorString(cub_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

#ifndef NDEBUG
        cudaStreamSynchronize(cuda_stream);
        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Kernel execution failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }
#endif

        int h_count = 0;
        cudaMemcpyAsync(&h_count, d_out_count_.ptr<int>(), sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);

        result.success = true;
        result.message = "Success";
        result.matchCount = h_count;
        result.d_matched_left = std::make_shared<cv::cuda::GpuMat>(d_out_left_.colRange(0, h_count).clone());
        result.d_matched_right = std::make_shared<cv::cuda::GpuMat>(d_out_right_.colRange(0, h_count).clone());
        result.d_matched_line_ids = std::make_shared<cv::cuda::GpuMat>(d_out_fids_.colRange(0, h_count).clone());

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

void LaserMatchCuda::Impl::SetParams(const LaserMatchParams& params) {
#ifndef NDEBUG
    if (inProcess_.load()) {
        CALIB_LOG_ERROR("setParams(): called while process() is running");
        throw std::runtime_error("[07-LaserMatchCuda] setParams() called during process()");
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
