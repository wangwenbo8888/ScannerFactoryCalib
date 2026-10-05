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
#include <cmath>
#include <map>
#include <utility>
#include <vector>
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
    int sentinel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_labels) return;

    d_min_y_coords[idx] = sentinel;
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

// 横切: 扫描中心行, 取每个标签最小 X (scanDirection=1)
__global__ void ScanCenterRowKernel(
    const int* d_labels,
    int* d_min_x_coords,
    int img_cols,
    size_t labels_step,
    int center_y,
    int max_label)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= img_cols) return;

    const int* row = reinterpret_cast<const int*>(
        reinterpret_cast<const char*>(d_labels) + center_y * labels_step);
    int label = row[x];

    if (label > 0 && label < max_label) {
        atomicMin(&d_min_x_coords[label], x);
    }
}

__global__ void BuildMapTableKernel(
    const int* d_label_ids,
    const int* d_min_y_coords,
    int* d_map_table,
    int num_entries,
    int sentinel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_entries) return;

    if (d_min_y_coords[idx] >= sentinel) return;

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

// 拆分粘连标签: 同一标签 x >= split_x 的像素改为 new_label (仅该行上下所有像素,
// 因为粘连发生在整域, 用 x 阈值切整个域)
__global__ void SplitMergedLabelKernel(
    int* d_labels,
    int rows, int cols,
    size_t step,
    int old_label,
    int split_x,
    int new_label)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int* row = reinterpret_cast<int*>(
        reinterpret_cast<char*>(d_labels) + y * step);
    if (row[x] == old_label && x >= split_x)
        row[x] = new_label;
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

        // === Step 2: 计算扫描线坐标（竖切: 中心列 / 横切: 中心行） ===
        bool horizontal = (params_.scanDirection == 1);
        int center_x = cols / 2 + params_.centerColOffset;
        int center_y = rows / 2 + params_.centerRowOffset;
        if (!horizontal && (center_x < 0 || center_x >= cols)) {
            result.success = false;
            result.message = "Center column out of bounds: " + std::to_string(center_x);
            return result;
        }
        if (horizontal && (center_y < 0 || center_y >= rows)) {
            result.success = false;
            result.message = "Center row out of bounds: " + std::to_string(center_y);
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

        // === Step 3.5: 拆分扫描行上的粘连游程（仅横切 + splitMergedRuns） ===
        // 膨胀致相邻平行线连通域粘连时, 同一标签在扫描行上呈多个不连续游程段;
        // 对 ≥2 游程的标签, 以游程间隙中点为界把域拆成两个标签, 循环至全部单游程。
        if (horizontal && params_.splitMergedRuns) {
            // 下载扫描行
            std::vector<int> scanline(cols);
            cudaMemcpyAsync(scanline.data(),
                            reinterpret_cast<const int*>(
                                reinterpret_cast<const char*>(d_labels_buf_.ptr<int>())
                                + (size_t)center_y * d_labels_buf_.step),
                            cols * sizeof(int), cudaMemcpyDeviceToHost, cuda_stream);
            cudaStreamSynchronize(cuda_stream);

            // 游程分解: (label, runStart, runEnd) 仅记录长度>=1 的段
            struct Run { int label, x0, x1; };
            std::vector<Run> runs;
            int x = 0;
            while (x < cols) {
                if (scanline[x] > 0 && scanline[x] < actual_max_label) {
                    int lab = scanline[x];
                    int x0 = x;
                    while (x < cols && scanline[x] == lab) ++x;
                    runs.push_back({lab, x0, x});
                } else ++x;
            }
            // 每标签游程数
            std::map<int, std::vector<Run>> byLabel;
            for (auto& r : runs) byLabel[r.label].push_back(r);
            int next_label = actual_max_label;
            bool changed = false;
            for (auto& [lab, v] : byLabel) {
                if (v.size() < 2) continue;
                // 保留首游程为原标签, 其余每游程拆出新标签
                for (size_t k = 1; k < v.size(); ++k) {
                    // 拆分阈值: 前一段结束与本段开始的中点
                    int split_x = (v[k - 1].x1 + v[k].x0) / 2;
                    if (next_label > params_.maxLabels) break;
                    dim3 blk(16, 16);
                    dim3 grd((cols + 15) / 16, (rows + 15) / 16);
                    SplitMergedLabelKernel<<<grd, blk, 0, cuda_stream>>>(
                        d_labels_buf_.ptr<int>(), rows, cols, d_labels_buf_.step,
                        lab, split_x, next_label);
                    ++next_label;
                    changed = true;
                }
            }
            if (changed) {
                cudaStreamSynchronize(cuda_stream);
                op_err = cudaGetLastError();
                if (op_err != cudaSuccess) {
                    result.success = false;
                    result.message = std::string("SplitMergedLabelKernel failed: ")
                                   + cudaGetErrorString(op_err);
                    return result;
                }
                actual_max_label = next_label;
                CALIB_LOG_INFO("split merged runs: {} labels after split",
                               actual_max_label - 1);
            }
        }

        // === Step 4: Init scan buffers（哨兵值: 竖切=rows, 横切=cols） ===
        int block_size_1d = 256;
        int grid_init = (actual_max_label + block_size_1d - 1) / block_size_1d;
        int sentinel = horizontal ? cols : rows;

        InitScanBuffersKernel<<<grid_init, block_size_1d, 0, cuda_stream>>>(
            thrust::raw_pointer_cast(d_min_y_coords_.data()),
            thrust::raw_pointer_cast(d_label_ids_.data()),
            thrust::raw_pointer_cast(d_map_table_.data()),
            actual_max_label,
            sentinel
        );

        op_err = cudaGetLastError();
        if (op_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("InitScanBuffersKernel failed: ") + cudaGetErrorString(op_err);
            return result;
        }

        // === Step 5: 扫描扫描线（竖切: 中心列取minY / 横切: 中心行取minX） ===
        if (horizontal) {
            int grid_scan = (cols + block_size_1d - 1) / block_size_1d;
            ScanCenterRowKernel<<<grid_scan, block_size_1d, 0, cuda_stream>>>(
                d_labels_buf_.ptr<int>(),
                thrust::raw_pointer_cast(d_min_y_coords_.data()),
                cols,
                d_labels_buf_.step,
                center_y,
                actual_max_label
            );
        } else {
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
        }

        op_err = cudaGetLastError();
        if (op_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("ScanCenterLine kernel failed: ") + cudaGetErrorString(op_err);
            return result;
        }

        // === Step 6: Sort by Y coordinate ===
        thrust::stable_sort_by_key(
            thrust::cuda::par_nosync.on(cuda_stream),
            d_min_y_coords_.begin(), d_min_y_coords_.begin() + actual_max_label,
            d_label_ids_.begin()
        );

        // === Step 6.5: 真激光线甄别（可选, realLineTolerance>0 启用） ===
        // 局部等距判据: 排序后交点序列 p[0..n-1], 若 |p[i]-p[i-1] - (p[i+1]-p[i])| <= tol
        // → p[i] 为真线; 真线的左右紧邻域也判真(端点传播)。被剔域映射置 0。
        // host 侧判别（序列 <= maxLabels, 零 GPU 开销），直接构建映射表替代 Step 7。
        bool map_built_on_host = false;
        int rejected_count = 0;
        if (params_.realLineTolerance > 0) {
            thrust::host_vector<int> h_coords(
                d_min_y_coords_.begin(), d_min_y_coords_.begin() + actual_max_label);
            thrust::host_vector<int> h_labels(
                d_label_ids_.begin(), d_label_ids_.begin() + actual_max_label);

            // 有效交点序列（越过哨兵=未与扫描线相交, 与 BuildMapTableKernel 同口径）
            std::vector<std::pair<int, int>> seq;   // (坐标, 旧标签), 已按坐标升序
            seq.reserve(actual_max_label);
            for (int i = 0; i < actual_max_label; ++i) {
                if (h_coords[i] < sentinel && h_labels[i] > 0)
                    seq.emplace_back(h_coords[i], h_labels[i]);
            }

            if (seq.size() >= 3) {
                const int tol = params_.realLineTolerance;
                std::vector<char> isReal(seq.size(), 0);
                for (size_t i = 1; i + 1 < seq.size(); ++i) {
                    const int dL = seq[i].first - seq[i - 1].first;
                    const int dR = seq[i + 1].first - seq[i].first;
                    if (std::abs(dL - dR) <= tol) isReal[i] = 1;
                }
                std::vector<char> keep(isReal);
                for (size_t i = 0; i < isReal.size(); ++i) {
                    if (!isReal[i]) continue;
                    if (i > 0) keep[i - 1] = 1;
                    if (i + 1 < keep.size()) keep[i + 1] = 1;
                }

                thrust::host_vector<int> h_map(actual_max_label, 0);
                int newId = 0;
                for (size_t i = 0; i < seq.size(); ++i) {
                    if (keep[i]) h_map[static_cast<size_t>(seq[i].second)] = ++newId;
                }
                rejected_count = static_cast<int>(seq.size()) - newId;

                cudaMemcpyAsync(thrust::raw_pointer_cast(d_map_table_.data()),
                                h_map.data(),
                                static_cast<size_t>(actual_max_label) * sizeof(int),
                                cudaMemcpyHostToDevice, cuda_stream);
                map_built_on_host = true;

                CALIB_LOG_INFO("realLine filter: kept {} of {} intersections "
                               "(tolerance={} px, rejected {})",
                               newId, seq.size(), tol, rejected_count);
            } else {
                CALIB_LOG_INFO("realLine filter skipped: {} intersections (< 3)",
                               seq.size());
            }
        }

        // === Step 7: Build mapping table（Step 6.5 未启用时走 kernel） ===
        if (!map_built_on_host) {
        BuildMapTableKernel<<<grid_init, block_size_1d, 0, cuda_stream>>>(
            thrust::raw_pointer_cast(d_label_ids_.data()),
            thrust::raw_pointer_cast(d_min_y_coords_.data()),
            thrust::raw_pointer_cast(d_map_table_.data()),
            actual_max_label,
            sentinel
        );

        op_err = cudaGetLastError();
        if (op_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("BuildMapTableKernel failed: ") + cudaGetErrorString(op_err);
            return result;
        }
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
            result.message = rejected_count > 0
                ? "Labeling successful (realLine filter rejected "
                  + std::to_string(rejected_count) + " components)"
                : "Labeling successful";
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
