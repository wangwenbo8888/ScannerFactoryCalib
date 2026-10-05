/**
 * @file laser_match_scan_cuda_pimpl.h
 * @brief 激光线匹配扫描CUDA算子 - 内部桥接头文件（声明 struct Impl 完整结构）
 *
 * 本文件包含 CUDA 类型，仅供内部实现使用。
 * .cpp 和 .cu 均 include 此文件。
 */

#pragma once

#include <opencv2/core/cuda.hpp>
#include <nlohmann/json.hpp>
#include <memory>
#include <atomic>
#include <vector>
#include <string>

#ifndef NDEBUG
#include <cuda_runtime.h>
#endif

#include "laser_match_scan_cuda.h"

namespace calib {

struct LaserMatchScanCuda::Impl {
    static constexpr int MAX_EPIPOLAR_ROWS = 8192;
static constexpr int MAX_LOOKUP_CANDS = 64;   // 每(行,x)表候选上限（25线×双根×邻格, 去重后实际~25-50）

    LaserMatchScanParams params_;

    struct TempTableEntry {
        double temperature = 0.0;
        std::vector<float> mapData;
        int entryCount = 0;
        int numLines = 0;
    };
    std::vector<TempTableEntry> tempTable_;
    bool tableLoaded_ = false;

    cv::cuda::GpuMat d_map_table_;
    cv::cuda::GpuMat d_map_line_start_;
    cv::cuda::GpuMat d_map_line_count_;
    cv::cuda::GpuMat d_map_byrow_;          // table re-sorted by (row,xL), CV_32FC4 (xL,uR,lineId,yL)
    cv::cuda::GpuMat d_map_row_start_;       // CSR by epipolar row (MAX_EPIPOLAR_ROWS, CV_32SC1)
    cv::cuda::GpuMat d_map_row_count_;       // CSR row counts (MAX_EPIPOLAR_ROWS, CV_32SC1)
    int activeMapCount_ = 0;
    int activeNumLines_ = 0;
    double currentTemp_ = 25.0;
    bool tempSet_ = false;

    cv::cuda::GpuMat d_left_sorted_pts_;
    cv::cuda::GpuMat d_left_sorted_lids_;
    cv::cuda::GpuMat d_left_sorted_idx_;
    cv::cuda::GpuMat d_left_cand_uR_;      // [i*MAX_LOOKUP_CANDS + k] 多候选 uR 预测
    cv::cuda::GpuMat d_left_cand_lid_;     // 同布局 候选线号
    cv::cuda::GpuMat d_left_cand_count_;   // [i] 候选数
    cv::cuda::GpuMat d_out_line_ids_sorted_;  // looked-up line_id per sorted left point (CV_32SC1)
    cv::cuda::GpuMat d_out_line_ids_orig_;    // looked-up line_id scattered to original index (CV_32SC1)
    cv::cuda::GpuMat d_right_sorted_pts_;
    cv::cuda::GpuMat d_right_sorted_lids_;
    cv::cuda::GpuMat d_right_sorted_idx_;

    cv::cuda::GpuMat d_left_keys_in_;
    cv::cuda::GpuMat d_left_keys_out_;
    cv::cuda::GpuMat d_right_keys_in_;
    cv::cuda::GpuMat d_right_keys_out_;
    cv::cuda::GpuMat d_left_idx_in_;
    cv::cuda::GpuMat d_right_idx_in_;
    cv::cuda::GpuMat d_max_row_key_;

    cv::cuda::GpuMat d_left_row_start_;
    cv::cuda::GpuMat d_left_row_count_;
    cv::cuda::GpuMat d_right_row_start_;
    cv::cuda::GpuMat d_right_row_count_;
    int numEpipolarRows_ = 0;

    cv::cuda::GpuMat d_left_status_;
    cv::cuda::GpuMat d_right_status_;

    cv::cuda::GpuMat d_match_left_idx_;
    cv::cuda::GpuMat d_match_count_;
    void* d_cub_temp_ = nullptr;
    size_t cub_temp_size_ = 0;

    bool warmed_up_ = false;
    int warmup_left_ = 0;
    int warmup_right_ = 0;
    int old_device_id_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const LaserMatchScanParams& params);
    ~Impl();

    bool LoadTempTable(const std::string& jsonPath);
    bool SetTempTable(std::shared_ptr<const LaserPlaneMapTempTable> table);
    void SetCurrentTemperature(double temperature);

    LaserMatchScanResult process(
        const cv::cuda::GpuMat& d_left_points,
        const cv::cuda::GpuMat& d_left_line_ids,
        const cv::cuda::GpuMat& d_right_points,
        const cv::cuda::GpuMat& d_right_line_ids,
        cv::cuda::Stream& stream);

    void warmup(int maxLeftPoints, int maxRightPoints);
    void setParams(const LaserMatchScanParams& params);
    const LaserMatchScanParams& getParams() const { return params_; }

    bool allocateBuffers(int leftCount, int rightCount);
    void uploadMapTable(const TempTableEntry& entry, cv::cuda::Stream& stream);
};

} // namespace calib
