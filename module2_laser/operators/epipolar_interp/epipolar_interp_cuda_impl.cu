/**
 * @file epipolar_interp_cuda_impl.cu
 * @brief 激光中心点极线插值CUDA算子 - CUDA实现（struct Impl 方法 + GPU Kernel）
 *
 * 算法步骤：
 *   Step 1: 对每对相邻点 (p_i, p_{i+1})，检查是否同帧
 *   Step 2: 检查 X 差值 < max_x_diff，Y 跨度 < max_y_span
 *   Step 3: 计算最近极线 y_target = ceil(y_min / step) * step
 *   Step 4: 验证 y_target 严格在 (y_min, y_max) 之间且距离 < 1.0
 *   Step 5: 线性插值 x_interp = x1 + t * (x2 - x1)
 *   Step 6: CUB DeviceSelect::Flagged 压缩输出
 */

#include "epipolar_interp_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cub/cub.cuh>
#include <cmath>
#include <climits>
#include <vector>
#include <map>
#include <utility>
#include <stdexcept>

using namespace calib;


CALIB_DEFINE_LOG_TAG(06, EpipolarInterpCuda);

// ============================================================================
// Configuration Constants
// ============================================================================

static constexpr int BLOCK_SIZE = 256;
static constexpr float EPSILON = 1e-6f;
static constexpr float DIST_THRESHOLD = 1.0f;

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
// CUDA Kernel
// ============================================================================

// ============ 逐极线选点式插值（v2，2026-08-23 重设计）============
// 设计目标（五条硬性约定）：
//   ① 极线上下必须都有点（开区间严格夹住）
//   ② 两点 X 差 <1.4px 且 Y 差 <1.4px（门禁参数，可配）
//   ③ 两点是距该极线最近的点（按 Y：y 有序流中夹住极线的相邻对）
//   ④ 极线步距可配（默认 0.7px）
//   ⑤ 每线每极线最多一个点（槽位唯一 = line × 极线行号，无重复产出）
// 实现：每线程负责 (一条线的起始点范围, 一批极线行)，对每条极线二分找
//       夹住它的相邻两点，校验门禁后线性插值，写入唯一槽。
// 输入要求：点流按 (line, y) 严格有序（行级合并后每线每行 1 点）。

__device__ __forceinline__ int findSpan(
    const float2* __restrict__ pts, int lo, int hi, float yq)
{
    // 二分：返回最大下标 i 使 pts[i].y <= yq（lo<=i<hi-1）
    // 即 pts[i] 是 yq 下方最近点、pts[i+1] 是上方最近点
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (pts[mid].y <= yq) lo = mid;
        else hi = mid;
    }
    return lo;
}

__global__ void __launch_bounds__(256, 4) kernelRowInterp(
    const float2* __restrict__ d_input,    // (line,y) 有序点流
    const int*   __restrict__ d_line_start,// 每条线在点流中的起始下标（+1 为终止）
    const int*   __restrict__ d_line_list, // 线号列表
    int line_count,
    int row_base,                          // 极线行号基准（y 最小处的行号）
    float epipolar_step,                   // ④ 极线步距（默认 0.7）
    float max_pt_diff,                     // ② 两点 X/Y 差上限（1.4）
    int* __restrict__ d_flags,             // 槽 = 全局极线行号 × line_count + lineIdx
    float2* __restrict__ d_interp_pts,
    int* __restrict__ d_interp_fids,
    int total_slots)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_slots) return;
    d_flags[idx] = 0;

    // 槽 → (极线行号, 线)
    int row = idx / line_count;
    int li  = idx - row * line_count;
    int lineId = d_line_list[li];
    int s = d_line_start[li];
    int e = d_line_start[li + 1];
    if (e - s < 2) return;   // 线内点不足

    float yq = (row + row_base) * epipolar_step;

    // ③ 二分找夹住 yq 的相邻两点（y 有序流中必然是最近的两点）
    int i = findSpan(d_input, s, e, yq);
    if (i < s || i + 1 >= e) return;
    float2 p1 = d_input[i];
    float2 p2 = d_input[i + 1];
    if (p1.y > p2.y) { float2 tmp = p1; p1 = p2; p2 = tmp; }

    // ① 极线必须严格在两点之间（上下都有点）
    if (!(yq > p1.y && yq < p2.y)) return;

    // ② 门禁：两点间 X/Y 差 <1.4px，且两点距极线的 Y 距离都 <0.7px（=步距，
    //    即两点分别紧贴极线上下、跨距不超过一个步距）
    if (fabsf(p1.x - p2.x) >= max_pt_diff) return;
    if (fabsf(p1.y - p2.y) >= max_pt_diff) return;
    if (fabsf(p1.y - yq) >= max_pt_diff) return;   // 贴线容差用绝对量(1.4px), 不随 step 缩
    if (fabsf(p2.y - yq) >= max_pt_diff) return;

    // 线性插值（连线与极线 y=yq 的交点）
    float t = (yq - p1.y) / (p2.y - p1.y);
    float xq = p1.x + t * (p2.x - p1.x);

    d_interp_pts[idx] = make_float2(xq, yq);
    d_interp_fids[idx] = lineId;
    d_flags[idx] = 1;      // ⑤ 槽唯一，天然每线每极线一点
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
// Impl Implementation
// ============================================================================

EpipolarInterpCuda::Impl::Impl(const EpipolarInterpParams& params)
    : params_(params)
{
    params_.validate();

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("[06-EpipolarInterpCuda] No CUDA devices found");
    }
    if (params_.deviceId >= deviceCount) {
        throw std::invalid_argument("[06-EpipolarInterpCuda] deviceId >= device count");
    }

    old_device_id_ = 0;
    cudaGetDevice(&old_device_id_);
    if (params_.deviceId != old_device_id_) {
        cudaSetDevice(params_.deviceId);
    }
}

EpipolarInterpCuda::Impl::~Impl() {
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        CALIB_LOG_ERROR("cudaDeviceSynchronize in destructor failed: {}",
                        cudaGetErrorString(sync_err));
    }

    safeCudaFree(d_cub_temp_storage_);
    cub_temp_size_ = 0;
    last_max_pairs_count_ = 0;
}

bool EpipolarInterpCuda::Impl::allocateBuffers(int pointCount) {
    const int num_pairs = pointCount - 1;

    if (num_pairs <= 0) {
        return true;
    }

    if (num_pairs <= last_max_pairs_count_ &&
        !d_flags_.empty() &&
        !d_temp_interp_.empty()) {
        return true;
    }

    CALIB_LOG_DEBUG("Allocating GPU buffers for {} pairs", num_pairs);

    // 每 pair 最多 4 个输出槽 → 缓冲 4 倍（多点产出 kernel）
    const int slots = num_pairs * 4;
    d_flags_.create(1, slots, CV_32SC1);
    d_temp_interp_.create(1, slots, CV_32FC2);
    d_temp_fids_.create(1, slots, CV_32SC1);
    d_output_.create(1, slots, CV_32FC2);
    d_output_fids_.create(1, slots, CV_32SC1);
    d_output_count_.create(1, 1, CV_32SC1);

    last_max_pairs_count_ = num_pairs;
    return true;
}

void EpipolarInterpCuda::Impl::Warmup(int pointCount) {
    if (pointCount <= 0) {
        CALIB_LOG_WARN("Warmup(): invalid pointCount={}, skipping", pointCount);
        return;
    }

    if (warmed_up_ && warmup_count_ == pointCount) {
        CALIB_LOG_DEBUG("Warmup(): already warmed up for {} points", pointCount);
        return;
    }

    allocateBuffers(pointCount);
    warmed_up_ = true;
    warmup_count_ = pointCount;

    CALIB_LOG_INFO("Warmup(): allocated GPU buffers for {} points", pointCount);
}

EpipolarInterpResult EpipolarInterpCuda::Impl::Execute(
    const cv::cuda::GpuMat& d_points,
    const cv::cuda::GpuMat& d_line_ids,
    cv::cuda::Stream& stream)
{
#ifndef NDEBUG
    ScopedFlag guard(&inProcess_);
#endif

    EpipolarInterpResult result;

    try {
        int pointCount = d_points.rows * d_points.cols;

        if (pointCount < 2) {
            result.success = true;
            result.message = "Less than 2 points, no interpolation";
            result.d_interpPoints = std::make_shared<cv::cuda::GpuMat>();
            result.interpCount = 0;
            return result;
        }

        if (!allocateBuffers(pointCount)) {
            result.success = false;
            result.message = "GPU buffer allocation failed";
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

        // ---- 行级合并（host 侧一次性预处理）：按 (line, floor(y)) 取均值 →
        //      每线每行唯一点，(line,y) 严格有序 —— kernel v2 的输入契约
        cv::Mat h_pts, h_ids;
        d_points.download(h_pts, stream);
        d_line_ids.download(h_ids, stream);
        cudaStreamSynchronize(cv::cuda::StreamAccessor::getStream(stream));
        {
            struct Acc { double sx=0, sy=0; int n=0; };
            std::map<std::pair<int,int>, Acc> acc;
            const cv::Vec2f* pp = h_pts.ptr<cv::Vec2f>();
            const int* pi = h_ids.ptr<int>();
            for (int k = 0; k < (int)h_pts.total(); ++k) {
                auto key = std::make_pair(pi[k], (int)std::floor(pp[k][1]));
                auto& a = acc[key];
                a.sx += pp[k][0];
                a.sy += pp[k][1];
                a.n += 1;
            }
            int w = 0;
            for (const auto& [key, a] : acc) {
                h_pts.ptr<cv::Vec2f>()[w] = cv::Vec2f(
                    (float)(a.sx / a.n), (float)(a.sy / a.n));
                h_ids.ptr<int>()[w] = key.first;
                ++w;
            }
            h_pts = h_pts.colRange(0, w);
            h_ids = h_ids.colRange(0, w);
        }
        pointCount = (int)h_pts.total();

        // 上传合并后的点流 + 计算每线 [start,end) 分段
        cv::cuda::GpuMat d_in_pts, d_in_ids;
        d_in_pts.upload(h_pts, stream);
        d_in_ids.upload(h_ids, stream);
        {
            std::vector<int> starts, lines;
            int prev = INT_MIN;
            const int* pi = h_ids.ptr<int>();
            for (int k = 0; k < pointCount; ++k) {
                if (pi[k] != prev) { starts.push_back(k); lines.push_back(pi[k]); prev = pi[k]; }
            }
            starts.push_back(pointCount);
            lineSegStart_.upload(cv::Mat(starts).reshape(1, 1), stream);
            lineSegList_.upload(cv::Mat(lines).reshape(1, 1), stream);
            lineCount_ = (int)lines.size();
        }

        // ---- 槽位规划：全局极线行号 = floor(y / step)，槽 = row × lineCount + li
        const float* ppy = h_pts.ptr<cv::Vec2f>()[0].val; // note: 仅取首元素探测
        float yMin = 1e30f, yMax = -1e30f;
        for (int k = 0; k < pointCount; ++k) {
            float y = h_pts.ptr<cv::Vec2f>()[k][1];
            yMin = fminf(yMin, y);
            yMax = fmaxf(yMax, y);
        }
        const int row0 = (int)std::floor(yMin / params_.epipolar_row_step);
        const int row1 = (int)std::ceil(yMax / params_.epipolar_row_step);
        const int totalRows = row1 - row0 + 1;
        const int totalSlots = totalRows * lineCount_;

        if (!allocateBuffers(totalSlots)) {
            result.success = false;
            result.message = "GPU buffer allocation failed";
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        int* d_flags_ptr = d_flags_.ptr<int>();
        float2* d_temp_ptr = d_temp_interp_.ptr<float2>();
        int* d_temp_fid_ptr = d_temp_fids_.ptr<int>();
        float2* d_output_ptr = d_output_.ptr<float2>();
        int* d_output_fid_ptr = d_output_fids_.ptr<int>();
        int* d_count_ptr = d_output_count_.ptr<int>();
        int* d_count_ptr2 = d_output_count_.ptr<int>();

        {
            const int grid_size = (totalSlots + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelRowInterp<<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
                d_in_pts.ptr<float2>(),
                lineSegStart_.ptr<int>(),
                lineSegList_.ptr<int>(),
                lineCount_,
                row0,
                params_.epipolar_row_step,
                params_.max_x_diff,      // 复用 maxXDiff 作为门禁（config 设 1.4）
                d_flags_ptr, d_temp_ptr, d_temp_fid_ptr,
                totalSlots);
        }

        cudaError_t kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Kernel launch failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        size_t temp_storage_bytes = 0;
        size_t temp_storage_fids = 0;

        cudaError_t cub_err = cub::DeviceSelect::Flagged(
            nullptr, temp_storage_bytes,
            d_temp_ptr, d_flags_ptr,
            d_output_ptr, d_count_ptr,
            totalSlots, cuda_stream);

        if (cub_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUB::Flagged size query failed: ") + cudaGetErrorString(cub_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        cub_err = cub::DeviceSelect::Flagged(
            nullptr, temp_storage_fids,
            d_temp_fid_ptr, d_flags_ptr,
            d_output_fid_ptr, d_count_ptr2,
            totalSlots, cuda_stream);

        if (cub_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUB::Flagged size query (fids) failed: ") + cudaGetErrorString(cub_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        size_t required_temp = (temp_storage_bytes > temp_storage_fids) ? temp_storage_bytes : temp_storage_fids;

        if (required_temp > cub_temp_size_ || d_cub_temp_storage_ == nullptr) {
            safeCudaFree(d_cub_temp_storage_);

            cub_err = cudaMalloc(&d_cub_temp_storage_, required_temp);
            if (cub_err != cudaSuccess) {
                result.success = false;
                result.message = std::string("CUB temp storage allocation failed: ") + cudaGetErrorString(cub_err);
                CALIB_LOG_ERROR("Execute(): {}", result.message);
                return result;
            }

            cub_temp_size_ = required_temp;
        }

        cub_err = cub::DeviceSelect::Flagged(
            d_cub_temp_storage_, cub_temp_size_,
            d_temp_ptr, d_flags_ptr,
            d_output_ptr, d_count_ptr,
            totalSlots, cuda_stream);

        if (cub_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUB::Flagged execution failed: ") + cudaGetErrorString(cub_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        cub_err = cub::DeviceSelect::Flagged(
            d_cub_temp_storage_, cub_temp_size_,
            d_temp_fid_ptr, d_flags_ptr,
            d_output_fid_ptr, d_count_ptr2,
            totalSlots, cuda_stream);

        if (cub_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUB::Flagged execution failed (fids): ") + cudaGetErrorString(cub_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        int h_count = 0;
        cudaMemcpyAsync(&h_count, d_count_ptr, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);

        result.success = true;
        result.message = "Success";
        result.interpCount = h_count;
        result.d_interpPoints = std::make_shared<cv::cuda::GpuMat>(d_output_.colRange(0, h_count).clone());
        result.d_interp_line_ids = std::make_shared<cv::cuda::GpuMat>(d_output_fids_.colRange(0, h_count).clone());

    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Exception: ") + e.what();
        CALIB_LOG_ERROR("Execute(): {}", result.message);
    } catch (...) {
        result.success = false;
        result.message = "Unknown exception";
        CALIB_LOG_ERROR("Execute(): {}", result.message);
    }

    return result;
}

void EpipolarInterpCuda::Impl::SetParams(const EpipolarInterpParams& params) {
#ifndef NDEBUG
    if (inProcess_.load()) {
        CALIB_LOG_ERROR("setParams(): called while process() is running");
        throw std::runtime_error("[06-EpipolarInterpCuda] setParams() called during process()");
    }
#endif

    params.validate();

    bool deviceChanged = (params.deviceId != params_.deviceId);

    params_ = params;

    if (deviceChanged) {
        cudaSetDevice(params_.deviceId);
    }

    warmed_up_ = false;
    warmup_count_ = 0;

    CALIB_LOG_INFO("SetParams(): params updated, warmup reset");
}

// ============================================================================
// Destroy()
// ============================================================================

void EpipolarInterpCuda::Impl::Destroy() {
    if (d_cub_temp_storage_) {
        cudaFree(d_cub_temp_storage_);
        d_cub_temp_storage_ = nullptr;
    }
    cub_temp_size_ = 0;
    d_flags_.release();
    d_temp_interp_.release();
    d_temp_fids_.release();
    d_output_.release();
    d_output_fids_.release();
    d_output_count_.release();
    warmed_up_ = false;
    warmup_count_ = 0;
    last_max_pairs_count_ = 0;
    CALIB_LOG_INFO("Destroy() completed");
}
