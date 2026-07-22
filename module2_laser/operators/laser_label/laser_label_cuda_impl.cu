/**
 * @file laser_label_cuda_impl.cu
 * @brief 激光线编号算子 CUDA 实现（struct Impl 方法 + GPU Kernel）
 *
 * 算法步骤：
 *   Step 1: 输入判断（仅 CV_32SC1，直接使用）
 *   Step 2: 计算中心列 center_x = cols / 2 + centerColOffset
 *   Step 3: Thrust reduce 获取最大标签值
 *   Step 4: InitScanBuffersKernel（初始化扫描缓冲区）
 *   Step 5: ScanCenterColumnKernel（中心列扫描取每个标签最小Y）
 *   Step 6: Thrust stable_sort_by_key（按Y值排序标签）
 *   Step 7: BuildMapTableKernel（构建旧标签→新编号映射）
 *   Step 8: RelabelKernel（全图重编号）
 */

#include "laser_label_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <stdexcept>

using namespace calib;


CALIB_DEFINE_LOG_TAG(09, LaserLabelerCUDA);

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void InitScanBuffersKernel(
    int* d_min_y_coords,
    int* d_label_ids,
    int* d_map_table,
    int num_labels,
    int img_rows)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_labels) return;

    d_min_y_coords[idx] = img_rows;
    d_label_ids[idx] = idx;
    d_map_table[idx] = 0;
}

__global__ void ScanCenterColumnKernel(
    const int* d_labels,
    int* d_min_y_coords,
    int img_rows,
    int img_cols,
    size_t labels_step,
    int center_x,
    int max_label)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= img_rows) return;

    const int* row = reinterpret_cast<const int*>(
        reinterpret_cast<const char*>(d_labels) + y * labels_step);
    int label = row[center_x];

    if (label > 0 && label < max_label) {
        atomicMin(&d_min_y_coords[label], y);
    }
}

__global__ void BuildMapTableKernel(
    const int* d_label_ids,
    const int* d_min_y_coords,
    int* d_map_table,
    int num_entries,
    int img_rows)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_entries) return;

    if (d_min_y_coords[idx] >= img_rows) return;

    int old_label = d_label_ids[idx];
    if (old_label > 0 && old_label < num_entries) {
        d_map_table[old_label] = idx + 1;
    }
}

__global__ void RelabelKernel(
    const int* d_input,
    int* d_output,
    const int* d_map_table,
    int img_rows,
    int img_cols,
    size_t in_step,
    size_t out_step)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= img_cols || y >= img_rows) return;

    const int* in_row = reinterpret_cast<const int*>(
        reinterpret_cast<const char*>(d_input) + y * in_step);
    int* out_row = reinterpret_cast<int*>(
        reinterpret_cast<char*>(d_output) + y * out_step);

    int old_label = in_row[x];
    if (old_label > 0) {
        out_row[x] = d_map_table[old_label];
    } else {
        out_row[x] = 0;
    }
}

// ============================================================================
// Impl 构造函数
// ============================================================================

LaserLabelerCUDA::Impl::Impl(const LaserLabelParams& params)
    : params_(params), old_device_id(params.deviceId)
{
    params_.validate();

    int device_count = cv::cuda::getCudaEnabledDeviceCount();
    if (device_count <= 0) {
        throw std::runtime_error("No CUDA-capable GPU found");
    }

    if (params_.deviceId >= device_count) {
        throw std::invalid_argument(
            "LaserLabelParams::deviceId=" + std::to_string(params_.deviceId)
            + " exceeds available device count=" + std::to_string(device_count));
    }
    cv::cuda::setDevice(params_.deviceId);

    try {
        d_min_y_coords_.resize(params_.maxLabels + 1);
        d_label_ids_.resize(params_.maxLabels + 1);
        d_map_table_.resize(params_.maxLabels + 1);
    } catch (const std::exception&) {
        throw std::runtime_error("Failed to allocate GPU memory for label buffers");
    }
}

// ============================================================================
// warmup()
// ============================================================================

void LaserLabelerCUDA::Impl::Warmup(int rows, int cols) {
    CALIB_LOG_INFO("warmup() pre-allocating: {}x{}", rows, cols);

    d_labels_buf_.create(rows, cols, CV_32SC1);
    d_output_buf_.create(rows, cols, CV_32SC1);

    warmup_rows_ = rows;
    warmup_cols_ = cols;

#ifndef NDEBUG
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("[09-LaserLabelerCUDA] GPU OOM: warmup validation failed: ")
            + cudaGetErrorString(err));
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("[09-LaserLabelerCUDA] warmup() CUDA error: ")
            + cudaGetErrorString(err));
    }
#endif

    warmed_up_ = true;
    CALIB_LOG_INFO("warmup() completed successfully");
}

// ============================================================================
// setParams()
// ============================================================================

void LaserLabelerCUDA::Impl::SetParams(const LaserLabelParams& params) {
#ifndef NDEBUG
    assert(!inProcess_.load() && "setParams() called while label() is running - NOT thread-safe!");
#endif

    params_ = params;
    params_.validate();

    if (params_.deviceId != old_device_id) {
        int device_count = cv::cuda::getCudaEnabledDeviceCount();
        if (params_.deviceId >= device_count) {
            throw std::invalid_argument(
                "LaserLabelParams::deviceId=" + std::to_string(params_.deviceId)
                + " exceeds available device count=" + std::to_string(device_count));
        }
        cv::cuda::setDevice(params_.deviceId);
        CALIB_LOG_INFO("setParams() device switched: {} -> {}", old_device_id, params_.deviceId);
        old_device_id = params_.deviceId;
    }

    try {
        d_min_y_coords_.resize(params_.maxLabels + 1);
        d_label_ids_.resize(params_.maxLabels + 1);
        d_map_table_.resize(params_.maxLabels + 1);
    } catch (const std::exception&) {
        throw std::runtime_error("Failed to resize GPU buffers in setParams()");
    }

    CALIB_LOG_INFO("setParams() updated: maxLabels={}, centerColOffset={}, deviceId={}",
                   params_.maxLabels, params_.centerColOffset, params_.deviceId);
}

// ============================================================================
// label() - 核心方法
// ============================================================================

LaserLabelResult LaserLabelerCUDA::Impl::Execute(
    const cv::cuda::GpuMat& d_inputMask,
    cv::cuda::Stream stream)
{
#ifndef NDEBUG
    assert(!inProcess_.load() && "Concurrent label() calls detected - NOT thread-safe!");

    struct ScopedFlag {
        std::atomic<bool>* flag;
        ScopedFlag(std::atomic<bool>* f) : flag(f) {}
        ~ScopedFlag() { flag->store(false); }
    };

    ScopedFlag guard(&inProcess_);
    inProcess_.store(true);
#endif

    LaserLabelResult result;

    try {
        int rows = d_inputMask.rows;
        int cols = d_inputMask.cols;
        int inputType = d_inputMask.type();

        cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

        // === Step 1: Input type handling (CV_32SC1 only) ===
        CALIB_LOG_DEBUG("Input is CV_32SC1, using directly");
        if (!warmed_up_ || warmup_rows_ != rows || warmup_cols_ != cols) {
            d_labels_buf_.create(rows, cols, CV_32SC1);
        }
        d_inputMask.copyTo(d_labels_buf_, stream);

        cudaError_t op_err = cudaGetLastError();
        if (op_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("CUDA error after CCL: ") + cudaGetErrorString(op_err);
            return result;
        }

        // === Step 2: Calculate center column ===
        int center_x = cols / 2 + params_.centerColOffset;
        if (center_x < 0 || center_x >= cols) {
            result.success = false;
            result.message = "Center column out of bounds: " + std::to_string(center_x);
            return result;
        }

        // === Step 3: Get max label value ===
        double min_val, max_val;
        cv::cuda::minMax(d_labels_buf_, &min_val, &max_val);
        int max_label_host = static_cast<int>(max_val);

        CALIB_LOG_INFO("Max label value: {}, min: {}", max_label_host, static_cast<int>(min_val));

        if (max_label_host == 0) {
            CALIB_LOG_WARN("No foreground components found");
            result.success = false;
            result.message = "No foreground components in input mask";
            result.qualityFlag = calib::QualityFlag::Warning;

            auto d_output = std::make_shared<cv::cuda::GpuMat>(rows, cols, CV_32SC1, cv::Scalar(0));
            result.d_labeledMask = d_output;
            return result;
        }

        if (max_label_host > params_.maxLabels) {
            CALIB_LOG_ERROR("Too many labels: {} > maxLabels={}", max_label_host, params_.maxLabels);
            result.success = false;
            result.message = "Too many labels: " + std::to_string(max_label_host)
                           + " exceeds maxLabels=" + std::to_string(params_.maxLabels);
            return result;
        }

        int actual_max_label = max_label_host + 1;

        // === Step 4: Init scan buffers ===
        int block_size_1d = 256;
        int grid_init = (actual_max_label + block_size_1d - 1) / block_size_1d;

        InitScanBuffersKernel<<<grid_init, block_size_1d, 0, cuda_stream>>>(
            thrust::raw_pointer_cast(d_min_y_coords_.data()),
            thrust::raw_pointer_cast(d_label_ids_.data()),
            thrust::raw_pointer_cast(d_map_table_.data()),
            actual_max_label,
            rows
        );

        op_err = cudaGetLastError();
        if (op_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("InitScanBuffersKernel failed: ") + cudaGetErrorString(op_err);
            return result;
        }

        // === Step 5: Scan center column ===
        int grid_scan = (rows + block_size_1d - 1) / block_size_1d;

        ScanCenterColumnKernel<<<grid_scan, block_size_1d, 0, cuda_stream>>>(
            d_labels_buf_.ptr<int>(),
            thrust::raw_pointer_cast(d_min_y_coords_.data()),
            rows,
            cols,
            d_labels_buf_.step,
            center_x,
            actual_max_label
        );

        op_err = cudaGetLastError();
        if (op_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("ScanCenterColumnKernel failed: ") + cudaGetErrorString(op_err);
            return result;
        }

        // === Step 6: Sort by Y coordinate ===
        thrust::stable_sort_by_key(
            thrust::cuda::par_nosync.on(cuda_stream),
            d_min_y_coords_.begin(), d_min_y_coords_.begin() + actual_max_label,
            d_label_ids_.begin()
        );

        // === Step 7: Build mapping table ===
        BuildMapTableKernel<<<grid_init, block_size_1d, 0, cuda_stream>>>(
            thrust::raw_pointer_cast(d_label_ids_.data()),
            thrust::raw_pointer_cast(d_min_y_coords_.data()),
            thrust::raw_pointer_cast(d_map_table_.data()),
            actual_max_label,
            rows
        );

        op_err = cudaGetLastError();
        if (op_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("BuildMapTableKernel failed: ") + cudaGetErrorString(op_err);
            return result;
        }

        // === Step 8: Apply relabeling ===
        if (d_output_buf_.empty() ||
            d_output_buf_.cols != cols || d_output_buf_.rows != rows) {
            d_output_buf_.create(rows, cols, CV_32SC1);
        }

        dim3 relabel_block(16, 16);
        dim3 relabel_grid((cols + relabel_block.x - 1) / relabel_block.x,
                          (rows + relabel_block.y - 1) / relabel_block.y);

        RelabelKernel<<<relabel_grid, relabel_block, 0, cuda_stream>>>(
            d_labels_buf_.ptr<int>(),
            d_output_buf_.ptr<int>(),
            thrust::raw_pointer_cast(d_map_table_.data()),
            rows,
            cols,
            d_labels_buf_.step,
            d_output_buf_.step
        );

        op_err = cudaGetLastError();
        if (op_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("RelabelKernel failed: ") + cudaGetErrorString(op_err);
            return result;
        }

        // === Output ===
        auto d_output = std::make_shared<cv::cuda::GpuMat>();
        d_output_buf_.copyTo(*d_output, stream);

        // Count components from mapping table
        thrust::host_vector<int> h_map_table(actual_max_label);
        thrust::copy(d_map_table_.begin(), d_map_table_.begin() + actual_max_label, h_map_table.begin());
        int component_count = 0;
        for (int i = 1; i < actual_max_label; ++i) {
            if (h_map_table[i] > 0) {
                component_count++;
            }
        }

        result.d_labeledMask = d_output;
        result.componentCount = component_count;
        result.success = true;

        if (component_count == 0) {
            result.qualityFlag = calib::QualityFlag::Warning;
            result.message = "No components intersect center column";
        } else if (component_count > 200) {
            result.qualityFlag = calib::QualityFlag::Degraded;
            result.message = "Abnormal component count: " + std::to_string(component_count);
        } else {
            result.qualityFlag = calib::QualityFlag::Normal;
            result.message = "Labeling successful";
        }

        CALIB_LOG_DEBUG("label() completed: {} components, qualityFlag={}",
                        component_count, static_cast<int>(result.qualityFlag));

    } catch (const cv::Exception& e) {
        result.success = false;
        result.message = std::string("OpenCV error: ") + e.what();
        CALIB_LOG_ERROR("label() OpenCV exception: {}", e.what());
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Error: ") + e.what();
        CALIB_LOG_ERROR("label() exception: {}", e.what());
    }

    return result;
}
