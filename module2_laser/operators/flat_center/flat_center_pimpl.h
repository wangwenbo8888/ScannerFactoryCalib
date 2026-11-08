/**
 * @file flat_center_pimpl.h
 * @brief 平顶激光中心提取算子（flat_center）- 内部桥接头文件
 *
 * 声明 struct FlatCenterExtractor::Impl 完整结构；仅供
 * flat_center.cpp / flat_center_impl.cu 使用（照 steger_fast_pimpl.h 惯例）。
 */

#pragma once

#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <atomic>

#include "flat_center.h"

namespace calib {

// 【P2-2 参数块】图模式动态参数块（host pinned 镜像 h_params_ ↔ 设备 d_params_）：
// 图内首节点 memcpyAsync(h→d)，内核经 Block 壳单指针读——指针/标量/维度每次
// graphLaunch 现读，lane 轮转输入指针随之自愈（烧死缺陷修复）；grid 维度仍
// 需图外比对毁图重捕（维度烧进 gridDim，参数块救不了）。
struct FlatParamsDev {
    // 指针区（8B 对齐优先）
    const unsigned char* g;        // 灰度
    size_t gstep;
    const unsigned char* mask;     // 二值掩膜（8U）
    size_t mstep;
    float* gs;                     // 预平滑缓冲（d_gsmooth_）
    float2* coords;                // 出点
    int* ids;                      // 出线号
    int* stats;                    // [0]=count [1..6]=drop
    // 维度区
    int rows, cols;
    int maxpts;
    // 标量阈值区
    float low_thresh, half_ratio, plateau_ratio;
    int plateau_min_w;
    float min_contrast, outlier_gate, max_line_width;
    int min_run;
    int smooth3;                   // 0/1（int 化保结构体布局确定）
};

struct FlatCenterExtractor::Impl {
    static constexpr int kLabelBins = 4096;   // K2 hist bins（照 steger 硬编码常量；
                                              // FlatCenterParams.maxLabels 仅契约兼容）

    FlatCenterParams params_;

    // 【G′·C4′ paramsEpoch_】参数/缓冲代际（SetParams/ensureBuffers 扩容/freeAll
    // 递增）——harness 大图签名聚合消费（F 批 fuseNeedRecapture_ 范式平移）；
    // 稳态恒定零开销
    uint64_t paramsEpoch_ = 0;

    // —— 设备缓冲（ensureBuffers 按 capacity 分配，grow-only）——
    float2* d_coords_ = nullptr;   // 源序出点 (x, cy)
    int*     d_ids_ = nullptr;     // 源序线编号
    int capacity_ = 0;             // = rows×cols（硬上界 ~786k 段/帧 ≪ 3.1M）

    int* d_stats_ = nullptr;       // 连续 7 槽：[0]=count、[1..6]=六门 drop 计数
                                   // （minRun/contrast/bright/edge/gate/width）——同批一次 D2H
    int* d_hist_ = nullptr;        // ByLabel：kLabelBins bins
    int* d_base_ = nullptr;        // ByLabel：逐 label 基址
    int* d_cursor_ = nullptr;      // ByLabel：scatter 游标
    float2* d_gcoords_ = nullptr;  // ByLabel：分组后输出（scatter 目的，独立于源序缓冲）
    int* d_gids_ = nullptr;
    float* d_gsmooth_ = nullptr;   // 【flat 优化 C】预平滑灰度（row-major float，
                                   // 逐位同 ReadGS(smooth3)；K1 段内多趟重读免
                                   // 3 抽头放大——JMW_FLATPRE=0 关闭走原路径 A/B）

    // —— 宿主镜像 ——
    int* h_stats_ = nullptr;       // pinned 7 槽（照 steger cudaHostAlloc 惯例）
    int h_hist_[kLabelBins] = {0};
    int h_base_[kLabelBins] = {0};
    float2* h_coords_ = nullptr;   // collectHostPoints_ / ByLabel host 构建缓冲（cap 宽）
    int* h_ids_ = nullptr;

    // —— bench（JMW_FLATTIME=1 时记录；零点早退帧清零防残留）——
    cudaEvent_t ev_start_ = nullptr, ev_scan_ = nullptr, ev_end_ = nullptr;
    bool events_ready_ = false;
    float bench_ms_[3] = {0.f, 0.f, 0.f};   // 0=Total 1=Scan 2=Group/Out

    // 【flat 优化·宏打包】[PreSmooth+K1] 免改参数图（全参数跨帧稳定：灰度/掩膜
    // 为每 lane 固定缓冲指针、模板/规模常量）——每帧 2 发射→1 图发射。
    // JMW_FLATGRAPH=0 关闭（对照）；SetParams 触发重建
    cudaGraph_t flatGraph_ = nullptr;
    cudaGraphExec_t flatGraphExec_ = nullptr;

    // 【P2-2 参数块】h_params_ pinned（memcpy 节点每次 launch 现读 src）；
    // d_params_ lazy 分配（仅图模式首帧——默认关臂零开销 census 逐位不变），
    // freeAll 释放（毁图同步）。图捕获时维度三元组记录：任一变化→毁图重捕。
    FlatParamsDev* d_params_ = nullptr;
    FlatParamsDev* h_params_ = nullptr;   // cudaMallocHost pinned 镜像
    int graph_rows_ = 0, graph_cols_ = 0, graph_bands_ = 0;
    bool paramsFailed_ = false;           // 参数块分配失败→本实例图模式永久回退

    bool bench_ = false;
    bool collectHostPoints_ = false;   // 默认 false（设计 §0：默认只出 GPU 视图）
    bool warmed_up_ = false;
    int warmup_rows_ = 0, warmup_cols_ = 0;
    int old_device_id_ = 0;

    // Flat 快路输出视图：包 d_coords_/d_ids_ 原始设备内存（capacity 宽），每帧取
    // [0,N) ROI 返回（帧内消费契约同 steger d_out_cp_/d_out_pl_ 注记）。
    // ByLabel 用 d_gcoords_/d_gids_ 同款视图（d_out_gc_/d_out_gl_）。
    cv::cuda::GpuMat d_out_cp_, d_out_pl_, d_out_gc_, d_out_gl_;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const FlatCenterParams& params);
    ~Impl();

    StegerResult Execute(const cv::cuda::GpuMat& d_grayImage,
                         const cv::cuda::GpuMat& d_labeledMask,
                         bool maskIs8U,
                         cv::cuda::Stream& stream);          // ByLabel 主路径
    StegerResult extractFlat(const cv::cuda::GpuMat& d_grayImage,
                              const cv::cuda::GpuMat& d_binaryMask,
                              cv::cuda::Stream& stream);       // Flat 快路
    /// 【flat 优化·合并唤醒】提交段（不同步；容量宽临时 ROI）＋收尾段（调用方
    /// 同步流后读 stats 重建 [0,N) 视图）——extractFlat＝两段直连
    StegerResult extractFlatSubmit(const cv::cuda::GpuMat& d_grayImage,
                                    const cv::cuda::GpuMat& d_binaryMask,
                                    cv::cuda::Stream& stream);
    void finishFlat(StegerResult& result);
    void Destroy();
    void Warmup(int rows, int cols);
    void SetParams(const FlatCenterParams& params);
    const FlatCenterParams& GetParams() const { return params_; }

    void ensureBuffers(int rows, int cols);
    void createEvents();
    void freeAll();

    // 【G′·三合一大图】裸发射口（harness FrontBigGraph 捕获窗专用）
    bool prepareBigGraph_(const cv::cuda::GpuMat& d_grayImage,
                          const cv::cuda::GpuMat& d_binaryMask);
    void refreshGraphParams_(const cv::cuda::GpuMat& d_grayImage,
                             const cv::cuda::GpuMat& d_binaryMask);
    bool emitGraphBody_(cudaStream_t cs);
    StegerResult makeDeferredShell_();
};

} // namespace calib
