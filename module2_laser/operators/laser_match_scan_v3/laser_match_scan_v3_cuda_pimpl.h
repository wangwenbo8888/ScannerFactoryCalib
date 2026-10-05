/**
 * @file laser_match_scan_v3_cuda_pimpl.h
 * @brief 激光线匹配扫描CUDA算子 V2 — 内部桥接头文件（声明 struct Impl 完整结构）
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

#include "laser_match_scan_v3_cuda.h"

namespace calib {

struct LaserMatchScanCudaV3::Impl {
    static constexpr int MAX_EPIPOLAR_ROWS = 8192;
    static constexpr int MAX_LOOKUP_CANDS = 64;       // 每(行,x)表候选上限（同 v1）
    static constexpr int SMEM_RIGHT_CAP = 256;        // kernelMatchV2 smem 每行右点容量(4KB)；超出置 overflow 标志
    static constexpr unsigned int KEY_SENTINEL = 0xFFFFFFFFu;  // row>=8192 丢弃标记

    LaserMatchScanParamsV3 params_;

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

    // ── 排序链（32bit 键：row<<19 | xq@1/256px）──
    cv::cuda::GpuMat d_left_keys_in_;        // CV_32SC1 (unsigned int)
    cv::cuda::GpuMat d_left_keys_out_;
    cv::cuda::GpuMat d_right_keys_in_;
    cv::cuda::GpuMat d_right_keys_out_;
    cv::cuda::GpuMat d_left_idx_in_;         // keygen 顺带写 sequence
    cv::cuda::GpuMat d_right_idx_in_;
    cv::cuda::GpuMat d_left_sorted_idx_;
    cv::cuda::GpuMat d_right_sorted_idx_;
    cv::cuda::GpuMat d_left_sorted_pts_;     // CV_32FC2
    cv::cuda::GpuMat d_left_sorted_lids_;    // CV_32SC1
    cv::cuda::GpuMat d_right_sorted_pts_;
    cv::cuda::GpuMat d_right_sorted_lids_;

    cv::cuda::GpuMat d_left_row_start_;      // CSR (MAX_EPIPOLAR_ROWS)
    cv::cuda::GpuMat d_left_row_count_;
    cv::cuda::GpuMat d_right_row_start_;
    cv::cuda::GpuMat d_right_row_count_;

    // ── 查表产物 ──
    // V3 card table (orig space; nearest card per lid)
    cv::cuda::GpuMat d_card_lid_;           // leftCount*V3_MAX_CARDS, CV_32SC1
    cv::cuda::GpuMat d_card_hitRx_;         // leftCount*V3_MAX_CARDS, CV_32FC1
    cv::cuda::GpuMat d_card_count_;         // leftCount, CV_32SC1
    // V3 segment pipeline
    cv::cuda::GpuMat d_seg_map_;            // leftCount, CV_32SC1 (orig -> segIdx)
    cv::cuda::GpuMat d_seg_votes_;          // segCount*V3_MAX_LIDS, CV_32SC1
    cv::cuda::GpuMat d_seg_winner_;         // segCount, CV_32SC1
    cv::cuda::GpuMat d_seg_drop_;           // segCount, CV_32SC1
    cv::cuda::GpuMat d_cell_seg_;           // MAX_EPIPOLAR_ROWS*V3_MAX_LIDS, CV_32SC1
    cv::cuda::GpuMat d_out_flags_;          // leftCount, CV_32SC1
    cv::cuda::GpuMat d_out_left_;           // leftCount, CV_32FC2 (gather temp)
    cv::cuda::GpuMat d_out_right_;          // leftCount, CV_32FC2
    cv::cuda::GpuMat d_out_lid_;            // leftCount, CV_32SC1
    cv::cuda::GpuMat d_out_track_;          // leftCount, CV_32SC1
    void* d_select_temp_ = nullptr;
    size_t select_temp_size_ = 0;
    size_t select_temp_cap_ = 0;
    // matched outputs (capacity-style persistent buffers)
    cv::cuda::GpuMat d_matched_left_cap_;    // leftCount, CV_32FC2
    cv::cuda::GpuMat d_matched_right_cap_;   // leftCount, CV_32FC2
    cv::cuda::GpuMat d_matched_lids_cap_;    // leftCount, CV_32SC1
    cv::cuda::GpuMat d_matched_tracks_cap_;  // leftCount, CV_32SC1

    // ── status ping-pong 双缓冲（免 clone）──
    cv::cuda::GpuMat d_left_status_[2];
    cv::cuda::GpuMat d_right_status_[2];
    int status_gen_ = 0;

    // ── 计数批量回传：[0]=match_count [1]=excl_left [2]=excl_right [3]=overflow ──
    cv::cuda::GpuMat d_counts_;              // 1x4 CV_32SC1
    int* h_counts_ = nullptr;                // pinned host
    bool overflow_warned_ = false;

    void* d_cub_temp_ = nullptr;
    size_t cub_temp_size_ = 0;

    bool warmed_up_ = false;
    int warmup_left_ = 0;
    int warmup_right_ = 0;
    int old_device_id_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const LaserMatchScanParamsV3& params);
    ~Impl();

    bool LoadTempTable(const std::string& jsonPath);
    bool SetTempTable(std::shared_ptr<const LaserPlaneMapTempTable> table);
    void SetCurrentTemperature(double temperature);

    LaserMatchScanResultV3 process(
        const cv::cuda::GpuMat& d_left_points,
        const cv::cuda::GpuMat& d_left_line_ids,
        const cv::cuda::GpuMat& d_right_points,
        const cv::cuda::GpuMat& d_right_line_ids,
        cv::cuda::Stream& stream);

    void warmup(int maxLeftPoints, int maxRightPoints);
    void setParams(const LaserMatchScanParamsV3& params);
    const LaserMatchScanParamsV3& getParams() const { return params_; }

    bool allocateBuffers(int leftCount, int rightCount);
    void uploadMapTable(const TempTableEntry& entry, cv::cuda::Stream& stream);
};

} // namespace calib
