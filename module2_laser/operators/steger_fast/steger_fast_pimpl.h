/**
 * @file steger_fast_pimpl.h
 * @brief Steger激光中心亚像素提取算子（优化版）- 内部桥接头文件
 *
 * 声明 struct Impl 完整结构；仅供 steger_fast.cpp / steger_fast_impl.cu 使用。
 */

#pragma once

#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <atomic>

#include "steger_fast.h"

namespace calib {

struct FastPoint {
    int label;
    float px;
    float py;
};

// —— dedup v2（位移式）中间结构 ——
struct DedupPt {                  // 两次排序的值（trivially copyable）
    int rrid;                     // 光栅序号 y*cols+x（全链确定性锚）
    int label;
    float px, py;
};

struct StegerExtractorFast::Impl {
    static constexpr int kLabelBins = 4096;
    static constexpr int kMaxKernelSize = 31;

    StegerParams params_;
    int actualKernelSize_ = 0;

    cv::cuda::GpuMat d_tg_, d_tgx_, d_tgxx_;
    cv::cuda::GpuMat d_ix_, d_iy_, d_ixx_, d_iyy_, d_ixy_;

    float* d_kg_ = nullptr;
    float* d_kgx_ = nullptr;
    float* d_kgxx_ = nullptr;

    FastPoint* d_points_ = nullptr;
    FastPoint* d_sorted_ = nullptr;
    float2* d_coords_ = nullptr;
    int* d_ids_ = nullptr;
    int capacity_ = 0;

    int* d_count_ = nullptr;
    int* d_reject_ = nullptr;   // 逐点宽度门禁拒绝计数（maxLineWidth>0 时累加）
    int* d_hist_ = nullptr;
    int* d_base_ = nullptr;
    int* d_cursor_ = nullptr;

    // —— dedup v2 缓冲（ensureBuffers 按 capacity 分配；CUB 临时空间全部调用
    //    （两次 SortPairs＋一次 ExclusiveSum）stream 串行复用同一块，
    //    尺寸取 max，按 capacity 一次性空查询）——
    unsigned long long* d_ckey_a_ = nullptr;   // D1 键 (cell|rrid) 双缓冲
    unsigned long long* d_ckey_b_ = nullptr;
    unsigned* d_cidx_a_ = nullptr;             // D1 值：排序输入下标 i（u32 载荷）
    unsigned* d_cidx_b_ = nullptr;
    DedupPt* d_cval_a_ = nullptr;              // D1 打包载荷（按原下标）
    DedupPt* d_cval_b_ = nullptr;              // D1 排序后载荷（gather 物化）
    float2* d_tan_ = nullptr;                  // 切向，按 D1 排序位索引（0..n-1——
                                                //   不用 rrid：round 恢复有半 ulp 撞号洞）
    char*   d_tan_valid_ = nullptr;
    int*    d_adj_ = nullptr;                  // 定长邻接行：stride 24（几何上界 8——
                                                //   半径 1.6 圆盘含自己 ≤9 格点、每像素
                                                //   ≤1 点；3 倍余量）
    int*    d_rowlen_ = nullptr;               // 逐行有效长度（D3 atomicAdd 槽位计数）
    int     adj_stride_ = 24;
    int*    d_lab_a_ = nullptr;                // 标签双缓冲（按下标索引，值为下标——传播用）
    int*    d_lab_b_ = nullptr;
    int*    d_rootkey_ = nullptr;              // root→簇最小 rrid（atomicMin，结果唯一）
    unsigned long long* d_lkey_a_ = nullptr;   // D5 键 (clusterKey<<22|rrid) 双缓冲
    unsigned long long* d_lkey_b_ = nullptr;
    unsigned* d_lidx_a_ = nullptr;             // D5 值：D1 排序后下标（u32 载荷）
    unsigned* d_lidx_b_ = nullptr;
    int* d_seg_id_ = nullptr;      // 逐点段边界标志（1=段首；ExclusiveSum 输入）
    int* d_seg_scan_ = nullptr;    // 逐点段号（ExclusiveSum 输出——不得与输入别名，
                                   //   CUB 多 tile 扫描下别名会损坏结果）
    int* d_seg_cnt_ = nullptr;     // 逐段输出点数（SegCount kernel 写）
    int* d_seg_base_ = nullptr;    // 逐段输出基址（ExclusiveSum(seg_cnt)）
    FastPoint* d_dedup_points_ = nullptr;
    int* d_dedup_count_ = nullptr;
    int* h_dedup_count_ = nullptr;
    int* d_flags_ = nullptr;       // [0]=邻接行溢出 [1]=传播未收敛（任一非 0 → 本帧保守直通）
    int* h_flags_ = nullptr;
    void* d_cub_temp_ = nullptr;
    size_t cub_temp_bytes_ = 0;
    cudaEvent_t ev_dedup_ = nullptr;

    int* h_count_ = nullptr;
    int h_reject_ = 0;          // 宽度门禁拒绝计数宿主侧落点（普通 int, D2H 直拷）
    int* h_hist_ = nullptr;
    int* h_base_ = nullptr;
    FastPoint* h_pts_ = nullptr;

    cudaEvent_t ev_start_ = nullptr, ev_row_ = nullptr, ev_col_ = nullptr,
                ev_hess_ = nullptr, ev_grp_ = nullptr, ev_out_ = nullptr, ev_end_ = nullptr;
    bool events_ready_ = false;

    bool warmed_up_ = false;
    int warmup_rows_ = 0;
    int warmup_cols_ = 0;
    int old_device_id_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const StegerParams& params);
    ~Impl();

    StegerResult Execute(const cv::cuda::GpuMat& d_grayImage,
                         const cv::cuda::GpuMat& d_labeledMask,
                         bool maskIs8U,
                         cv::cuda::Stream& stream);
    StegerResult extractFlat(const cv::cuda::GpuMat& d_grayImage,
                             const cv::cuda::GpuMat& d_binaryMask,
                             cv::cuda::Stream& stream);
    void Destroy();
    void Warmup(int rows, int cols);
    void SetParams(const StegerParams& params);
    const StegerParams& GetParams() const { return params_; }

    void buildGaussianKernels();
    int computeKernelSize() const;
    void ensureBuffers(int rows, int cols);
    void createEvents();
};

} // namespace calib
