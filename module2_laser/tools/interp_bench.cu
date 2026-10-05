// interp_bench.cu — epipolar_interp 单算子分步计时 + 优化分析基准
// 输入: laser 数据目录（left_skew 同布局: config.json + camera_calib.json + pose_*/)
// 输出: <outDir>/timing.csv + spdlog 汇总
// 计时: 整算子 Execute() wall 对照 + 复刻算子内部 4 步逐段计时（ev=GPU 事件 / wall=墙钟）
//   S1 排序(buildSortKeys/thrustSort/gather)
//   S2 host 线号统计(dlFids/hostLineScan/dlPts/hostYRange/alloc+ulTables)
//   S3 极线插值 kernel
//   S4 CUB 压缩(cubQuery/cubSelectx2/dlCount/resultClone)
// 复刻 kernel 与 operators/epipolar_interp/epipolar_interp_cuda_impl.cu 逐字一致。
// 用法: fcstepdump --interp-bench <inDir> [outDir iters]

#include "calib_io.h"

#include "mask_extract_cuda.h"
#include "region_analyze_cuda.h"
#include "laser_label_cuda.h"
#include "steger_extract_cuda.h"
#include "undistort_points_cuda.h"
#include "epipolar_interp_cuda.h"
#include "epipolar_interp_opt_cuda.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

using namespace fc;
using namespace calib;

namespace {

constexpr int BLOCK_SIZE = 256;
constexpr float EPSILON = 1e-4f;
constexpr float DEGEN_EPS = 1e-6f;

enum {
    S_EXEC = 0,
    S1_KEYS, S1_SORT, S1_GATHER,
    S2_DL_FIDS, S2_HOST_LINES, S2_DL_PTS, S2_HOST_YRANGE, S2_UL_TABLES,
    S3_KERNEL,
    S4_CUB_QUERY, S4_CUB_SELECT, S4_DL_COUNT, S4_CLONE,
    S_COUNT
};
const char* kNames[S_COUNT] = {
    "Execute(whole)",
    "S1a_buildSortKeys", "S1b_thrustSort", "S1c_gather",
    "S2a_dlFids+sync", "S2b_hostLineScan", "S2c_dlPts+sync", "S2d_hostYRange", "S2e_alloc+ulTables",
    "S3_interpKernel",
    "S4a_cubQuery", "S4b_cubSelectx2", "S4c_dlCount+sync", "S4d_resultClone"
};

// ============================================================================
// 复刻 kernel（与 epipolar_interp_cuda_impl.cu 逐字一致）
// ============================================================================

__global__ void kernelBuildSortKeys(
    const float2* __restrict__ d_pts,
    const int* __restrict__ d_fids,
    long long* __restrict__ d_keys,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float y = d_pts[idx].y;
    unsigned int yi = (unsigned int)(__float_as_int(y) & 0x7fffffff);
    if (__float_as_int(y) < 0) yi = ~yi;
    long long key = ((long long)d_fids[idx] << 32) | (unsigned int)yi;
    d_keys[idx] = key;
}

__global__ void gatherKernel(
    const float2* __restrict__ d_pts_src,
    const int* __restrict__ d_fids_src,
    const int* __restrict__ d_indices,
    float2* __restrict__ d_pts_dst,
    int* __restrict__ d_fids_dst,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int src = d_indices[i];
    d_pts_dst[i] = d_pts_src[src];
    d_fids_dst[i] = d_fids_src[src];
}

__global__ void __launch_bounds__(256, 4) kernelEpipolarInterp(
    const float2* __restrict__ d_pts_sorted,
    const int* __restrict__ d_fids_sorted,
    const int* __restrict__ d_line_begin,
    const int* __restrict__ d_line_ids_map,
    int num_lines,
    int num_rows,
    int row_begin,
    float row_step,
    float window,
    float max_x_diff,
    int* __restrict__ d_flags,
    float2* __restrict__ d_out_pts,
    int* __restrict__ d_out_fids)
{
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= num_lines * num_rows) return;

    int line_idx = linear / num_rows;
    int row_idx = linear - line_idx * num_rows;

    int fid = d_line_ids_map[line_idx];
    int begin = d_line_begin[line_idx];
    int end = d_line_begin[line_idx + 1];
    int cnt = end - begin;
    if (cnt <= 0) { d_flags[linear] = 0; return; }

    float y_k = (float)(row_begin + row_idx) * row_step;

    int lo = begin, hi = end;
    float ylw = y_k - window;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (d_pts_sorted[mid].y < ylw) lo = mid + 1;
        else hi = mid;
    }
    int hit_idx = -1;
    float hit_dy = 1e30f;
    int up_idx = -1;
    float up_dy = 1e30f;
    int down_idx = -1;
    float down_dy = 1e30f;

    for (int i = lo; i < end; ++i) {
        float y = d_pts_sorted[i].y;
        if (y > y_k + window) break;
        float dy = y - y_k;
        float ady = fabsf(dy);
        if (ady < EPSILON) {
            if (ady < hit_dy) { hit_dy = ady; hit_idx = i; }
        } else if (dy < 0.0f) {
            if (ady < up_dy) { up_dy = ady; up_idx = i; }
        } else {
            if (ady < down_dy) { down_dy = ady; down_idx = i; }
        }
    }

    float2 out;
    if (hit_idx >= 0) {
        out = d_pts_sorted[hit_idx];
        out.y = y_k;
    } else {
        if (up_idx < 0 || down_idx < 0) { d_flags[linear] = 0; return; }
        float2 pu = d_pts_sorted[up_idx];
        float2 pd = d_pts_sorted[down_idx];
        if (fabsf(pu.x - pd.x) > max_x_diff) { d_flags[linear] = 0; return; }
        float denom = pd.y - pu.y;
        if (fabsf(denom) < DEGEN_EPS) { d_flags[linear] = 0; return; }
        float t = (y_k - pu.y) / denom;
        out.x = pu.x + t * (pd.x - pu.x);
        out.y = y_k;
    }

    d_out_pts[linear] = out;
    d_out_fids[linear] = fid;
    d_flags[linear] = 1;
}

// ============================================================================
// 分步计时器（ev + wall 双计量，warmup 期间只同步不累计）
// ============================================================================

struct StepTimer {
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    cudaStream_t cs = nullptr;
    bool active = false;
    double evMs[S_COUNT] = {0};
    double wallMs[S_COUNT] = {0};
    std::chrono::steady_clock::time_point wt0;

    void init(cudaStream_t s) {
        cs = s;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
    }
    void begin() {
        cudaEventRecord(ev0, cs);
        wt0 = std::chrono::steady_clock::now();
    }
    void end(int k) {
        cudaEventRecord(ev1, cs);
        cudaEventSynchronize(ev1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        double w = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - wt0).count();
        if (!active) return;
        evMs[k] += ms;
        wallMs[k] += w;
    }
    void reset() {
        for (int k = 0; k < S_COUNT; ++k) { evMs[k] = 0; wallMs[k] = 0; }
    }
};

struct ReplBuf {
    cv::cuda::GpuMat keys, sortedPts, sortedFids, indices;
    cv::cuda::GpuMat flags, tempInterp, tempFids, output, outputFids, outputCount;
    cv::cuda::GpuMat outClonePts, outCloneIds;
    void* cubTemp = nullptr;
    size_t cubSize = 0;
};

} // namespace

// ============================================================================
// runInterpBench
// ============================================================================

int runInterpBench(const std::string& inDir, const std::string& outDir, int iters) {
    auto input = loadLaserInput(inDir);
    if (!input) { spdlog::error("load input failed"); return 1; }
    std::filesystem::create_directories(outDir);
    const auto& cfg = input->config;
    const auto& h = input->handoff;

    cv::cuda::Stream stream;
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, cfg.deviceId);
    spdlog::info("=== interp_bench === GPU={} poses={} iters={}",
                 prop.name, input->poseFrames.size(), iters);

    // ---- 上游算子（参数照搬 fcstepdump）----
    MaskExtractParams maskParams;
    maskParams.threshold = 50;
    maskParams.erodeSize = 1;
    maskParams.laserDilateSize = 19;
    maskParams.postErodeSize = 13;
    MaskExtractCUDA maskL(maskParams), maskR(maskParams);

    RegionAnalyzerParams cclParams;
    cclParams.deviceId = cfg.deviceId;
    cclParams.minArea = 0;
    cclParams.topXCount = 27;
    RegionAnalyzerCUDA cclL(cclParams), cclR(cclParams);

    LaserLabelParams labelParams;
    labelParams.deviceId = cfg.deviceId;
    labelParams.scanDirection = cfg.labelScanDirection;
    labelParams.realLineTolerance = 10;
    labelParams.centerRowOffset = cfg.labelCenterRowOffset;
    LaserLabelerCUDA labelL(labelParams), labelR(labelParams);

    StegerParams stegerParams;
    stegerParams.deviceId = cfg.deviceId;
    StegerExtractorCUDA stegerL(stegerParams), stegerR(stegerParams);

    cv::Mat P1_3x3 = h.P1.clone(); if (P1_3x3.cols > 3) P1_3x3.at<double>(0, 3) = 0.0;
    cv::Mat P2_3x3 = h.P2.clone(); if (P2_3x3.cols > 3) P2_3x3.at<double>(0, 3) = 0.0;
    UndistortPointsParams undistL;
    undistL.cameraMatrix = h.cameraMatrixL;
    undistL.distCoeffs   = h.distCoeffsL;
    undistL.R            = h.R1;
    undistL.P            = P1_3x3;
    undistL.deviceId     = cfg.deviceId;
    undistL.validate();
    UndistortPointsParams undistR;
    undistR.cameraMatrix = h.cameraMatrixR;
    undistR.distCoeffs   = h.distCoeffsR;
    undistR.R            = h.R2;
    undistR.P            = P2_3x3;
    undistR.deviceId     = cfg.deviceId;
    undistR.validate();
    UndistortPointsCuda undistLOp(undistL), undistROp(undistR);

    EpipolarInterpParams epipolarParams;
    epipolarParams.deviceId = cfg.deviceId;
    epipolarParams.lineIdCheck = true;
    EpipolarInterpCuda epipolarLOp(epipolarParams), epipolarROp(epipolarParams);
    EpipolarInterpOptCuda epipolarOptLOp(epipolarParams), epipolarOptROp(epipolarParams);

    StepTimer timer;
    timer.init(cs);
    ReplBuf rb;

    // ---- 上游链 4-1..4-5: 每帧每侧跑一次, 缓存 epipolar_interp 输入 ----
    auto prepInput = [&](const cv::Mat& gray, MaskExtractCUDA& maskOp, RegionAnalyzerCUDA& cclOp,
                         LaserLabelerCUDA& labelOp, StegerExtractorCUDA& stegerOp,
                         UndistortPointsCuda& undistOp,
                         std::shared_ptr<cv::cuda::GpuMat>& dPts,
                         std::shared_ptr<cv::cuda::GpuMat>& dIds) -> bool {
        auto m = maskOp.Execute(gray, stream);
        if (!m.success || !m.d_cleanedMask || m.d_cleanedMask->empty()) return false;
        auto c = cclOp.Execute(*m.d_cleanedMask, stream);
        if (!c.success || !c.d_labeledMask) return false;
        auto l = labelOp.Execute(*c.d_labeledMask, stream);
        if (!l.success || !l.d_labeledMask) return false;
        auto s = stegerOp.Execute(*m.d_grayImage, *l.d_labeledMask, stream, GroupMode::ByLabel);
        if (!s.success || !s.d_centerPoints || !s.d_line_ids) return false;
        auto u = undistOp.Execute(*s.d_centerPoints, *s.d_line_ids, stream);
        if (!u.success || !u.d_rectifiedPoints || !u.d_line_ids) return false;
        dPts = u.d_rectifiedPoints;
        dIds = u.d_line_ids;
        return !dPts->empty() && (dPts->rows * dPts->cols) >= 2;
    };

    // ---- 复刻算子内部流水（单次, 按 timer 分段）----
    auto runReplica = [&](const cv::cuda::GpuMat& dPoints, const cv::cuda::GpuMat& dIds,
                          int& outCount, int& outNumLines, int& outNumRows) {
        const int pointCount = dPoints.rows * dPoints.cols;
        const float step = epipolarParams.epipolar_row_step;
        const float window = epipolarParams.effectiveWindow();
        const int gridN = (pointCount + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // 复刻缓冲稳态化（同算子 allocateBuffers 语义: 首次分配, 之后 no-op）
        rb.keys.create(1, pointCount, CV_64FC1);
        rb.sortedPts.create(1, pointCount, CV_32FC2);
        rb.sortedFids.create(1, pointCount, CV_32SC1);
        rb.indices.create(1, pointCount, CV_32SC1);

        // ---- Step 1: 按 (fid, y) 排序 ----
        timer.begin();
        kernelBuildSortKeys<<<gridN, BLOCK_SIZE, 0, cs>>>(
            dPoints.ptr<float2>(), dIds.ptr<int>(),
            (long long*)rb.keys.ptr<double>(), pointCount);
        timer.end(S1_KEYS);

        timer.begin();
        {
            thrust::device_ptr<long long> keys_begin((long long*)rb.keys.ptr<double>());
            thrust::device_ptr<int> idx_begin((int*)rb.indices.ptr<int>());
            thrust::sequence(thrust::cuda::par.on(cs), idx_begin, idx_begin + pointCount);
            thrust::sort_by_key(thrust::cuda::par.on(cs),
                                keys_begin, keys_begin + pointCount, idx_begin);
        }
        timer.end(S1_SORT);

        timer.begin();
        gatherKernel<<<gridN, BLOCK_SIZE, 0, cs>>>(
            dPoints.ptr<float2>(), dIds.ptr<int>(),
            (int*)rb.indices.ptr<int>(),
            rb.sortedPts.ptr<float2>(), rb.sortedFids.ptr<int>(), pointCount);
        timer.end(S1_GATHER);

        // ---- Step 2: host 侧线号区间统计 ----
        std::vector<int> h_fids_sorted;
        std::vector<float2> h_pts;
        timer.begin();
        h_fids_sorted.resize(pointCount);
        cudaMemcpyAsync(h_fids_sorted.data(), rb.sortedFids.ptr<int>(),
                        pointCount * sizeof(int), cudaMemcpyDeviceToHost, cs);
        cudaStreamSynchronize(cs);
        timer.end(S2_DL_FIDS);

        std::vector<int> h_line_ids_map, h_line_begin;
        timer.begin();
        {
            int i = 0;
            while (i < pointCount) {
                int fid = h_fids_sorted[i];
                int j = i;
                while (j < pointCount && h_fids_sorted[j] == fid) ++j;
                h_line_ids_map.push_back(fid);
                h_line_begin.push_back(i);
                i = j;
            }
            h_line_begin.push_back(pointCount);
        }
        timer.end(S2_HOST_LINES);
        const int num_lines = (int)h_line_ids_map.size();

        timer.begin();
        h_pts.resize(pointCount);
        cudaMemcpyAsync(h_pts.data(), rb.sortedPts.ptr<float2>(),
                        pointCount * sizeof(float2), cudaMemcpyDeviceToHost, cs);
        cudaStreamSynchronize(cs);
        timer.end(S2_DL_PTS);

        float y_min = 0.f, y_max = 0.f;
        timer.begin();
        {
            std::vector<float> h_ys(pointCount);
            for (int i = 0; i < pointCount; ++i) h_ys[i] = h_pts[i].y;
            y_min = *std::min_element(h_ys.begin(), h_ys.end());
            y_max = *std::max_element(h_ys.begin(), h_ys.end());
        }
        timer.end(S2_HOST_YRANGE);
        const int row_begin = (int)std::floor(y_min / step);
        const int row_end = (int)std::ceil(y_max / step);
        const int num_rows = row_end - row_begin + 1;
        const int total_lines = num_lines * num_rows;

        // 线表 GpuMat 每次调用新分配（与算子 Execute 内局部 GpuMat 一致）
        cv::cuda::GpuMat d_line_begin, d_line_ids_map;
        timer.begin();
        {
            d_line_begin.create(1, num_lines + 1, CV_32SC1);
            d_line_ids_map.create(1, num_lines, CV_32SC1);
            cudaMemcpyAsync(d_line_begin.ptr<int>(), h_line_begin.data(),
                            (num_lines + 1) * sizeof(int), cudaMemcpyHostToDevice, cs);
            cudaMemcpyAsync(d_line_ids_map.ptr<int>(), h_line_ids_map.data(),
                            num_lines * sizeof(int), cudaMemcpyHostToDevice, cs);
        }
        timer.end(S2_UL_TABLES);

        // ---- Step 3: 极线插值 kernel ----
        timer.begin();
        {
            rb.flags.create(1, total_lines, CV_32SC1);
            rb.tempInterp.create(1, total_lines, CV_32FC2);
            rb.tempFids.create(1, total_lines, CV_32SC1);
            rb.output.create(1, total_lines, CV_32FC2);
            rb.outputFids.create(1, total_lines, CV_32SC1);
            rb.outputCount.create(1, 1, CV_32SC1);
            const int gridL = (total_lines + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelEpipolarInterp<<<gridL, BLOCK_SIZE, 0, cs>>>(
                rb.sortedPts.ptr<float2>(), rb.sortedFids.ptr<int>(),
                d_line_begin.ptr<int>(), d_line_ids_map.ptr<int>(),
                num_lines, num_rows, row_begin,
                step, window, epipolarParams.max_x_diff,
                rb.flags.ptr<int>(), rb.tempInterp.ptr<float2>(),
                rb.tempFids.ptr<int>());
        }
        timer.end(S3_KERNEL);

        // ---- Step 4: CUB 压缩 ----
        timer.begin();
        {
            size_t t1 = 0, t2 = 0;
            cub::DeviceSelect::Flagged(nullptr, t1,
                rb.tempInterp.ptr<float2>(), rb.flags.ptr<int>(),
                rb.output.ptr<float2>(), rb.outputCount.ptr<int>(), total_lines, cs);
            cub::DeviceSelect::Flagged(nullptr, t2,
                rb.tempFids.ptr<int>(), rb.flags.ptr<int>(),
                rb.outputFids.ptr<int>(), rb.outputCount.ptr<int>(), total_lines, cs);
            const size_t need = t1 > t2 ? t1 : t2;
            if (need > rb.cubSize || rb.cubTemp == nullptr) {
                if (rb.cubTemp) cudaFree(rb.cubTemp);
                cudaMalloc(&rb.cubTemp, need);
                rb.cubSize = need;
            }
        }
        timer.end(S4_CUB_QUERY);

        timer.begin();
        {
            cub::DeviceSelect::Flagged(rb.cubTemp, rb.cubSize,
                rb.tempInterp.ptr<float2>(), rb.flags.ptr<int>(),
                rb.output.ptr<float2>(), rb.outputCount.ptr<int>(), total_lines, cs);
            cub::DeviceSelect::Flagged(rb.cubTemp, rb.cubSize,
                rb.tempFids.ptr<int>(), rb.flags.ptr<int>(),
                rb.outputFids.ptr<int>(), rb.outputCount.ptr<int>(), total_lines, cs);
        }
        timer.end(S4_CUB_SELECT);

        int h_count = 0;
        timer.begin();
        cudaMemcpyAsync(&h_count, rb.outputCount.ptr<int>(), sizeof(int),
                        cudaMemcpyDeviceToHost, cs);
        cudaStreamSynchronize(cs);
        timer.end(S4_DL_COUNT);

        timer.begin();
        {
            rb.outClonePts = rb.output.colRange(0, h_count).clone();
            rb.outCloneIds = rb.outputFids.colRange(0, h_count).clone();
        }
        timer.end(S4_CLONE);

        outCount = h_count;
        outNumLines = num_lines;
        outNumRows = num_rows;
    };

    // ---- CSV ----
    std::ofstream csv(outDir + "/timing.csv");
    csv << "frame,side,points,num_lines,num_rows,interp_out,exec_wall_ms,opt_wall_ms,verify_ok,pt_mismatch,max_pt_diff,repl_wall_ms";
    for (int k = 1; k < S_COUNT; ++k)
        csv << ',' << kNames[k] << "_ev" << ',' << kNames[k] << "_wall";
    csv << '\n';

    double gEv[S_COUNT] = {0}, gWall[S_COUNT] = {0};
    double gExec = 0, gRepl = 0, gOpt = 0;
    long gRows = 0, gPts = 0, gOut = 0, gVerifyOk = 0, gPtMismatch = 0;

    const int kWarmup = 3;
    auto tNow = [] { return std::chrono::steady_clock::now(); };
    auto msBetween = [](auto t0, auto t1) {
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    for (size_t pi = 0; pi < input->poseFrames.size(); ++pi) {
        const std::string& poseName = input->poseDirs[pi];
        const auto& tubes = input->poseFrames[pi];
        for (size_t ti = 0; ti < tubes.size(); ++ti) {
            const cv::Mat imgs[2] = {tubes[ti].leftGray, tubes[ti].rightGray};
            const char* side[2] = {"L", "R"};
            const std::string stem = poseName + "_tube" + std::to_string(ti);

            for (int s = 0; s < 2; ++s) {
                std::shared_ptr<cv::cuda::GpuMat> dPts, dIds;
                MaskExtractCUDA& maskOp = s ? maskR : maskL;
                RegionAnalyzerCUDA& cclOp = s ? cclR : cclL;
                LaserLabelerCUDA& labelOp = s ? labelR : labelL;
                StegerExtractorCUDA& stegerOp = s ? stegerR : stegerL;
                UndistortPointsCuda& undistOp = s ? undistROp : undistLOp;
                EpipolarInterpCuda& interpOp = s ? epipolarROp : epipolarLOp;
                EpipolarInterpOptCuda& interpOptOp = s ? epipolarOptROp : epipolarOptLOp;

                if (!prepInput(imgs[s], maskOp, cclOp, labelOp, stegerOp, undistOp, dPts, dIds)) {
                    spdlog::warn("{} {}: upstream failed or <2 points, skip", stem, side[s]);
                    continue;
                }
                const int pointCount = dPts->rows * dPts->cols;

                // warmup（不计时, 稳态化缓冲）
                timer.active = false;
                int rc = 0, rl = 0, rr = 0, execCount = -1;
                for (int w = 0; w < kWarmup; ++w) {
                    auto r = interpOp.Execute(*dPts, *dIds, stream);
                    execCount = r.interpCount;
                    auto rn = interpOptOp.Execute(*dPts, *dIds, stream);
                    runReplica(*dPts, *dIds, rc, rl, rr);
                }
                if (rc != execCount)
                    spdlog::warn("{} {}: replica count {} != Execute count {}",
                                 stem, side[s], rc, execCount);

                // 验证: 同输入下 原版 vs 优化版 输出逐点一致
                bool verifyOk = false;
                int ptMismatch = -1;
                double maxDiff = -1.0;
                {
                    auto rO = interpOp.Execute(*dPts, *dIds, stream);
                    auto rN = interpOptOp.Execute(*dPts, *dIds, stream);
                    stream.waitForCompletion();
                    verifyOk = rO.success && rN.success
                               && rO.interpCount == rN.interpCount;
                    ptMismatch = 0;
                    maxDiff = 0.0;
                    cv::Mat po, fo, pn, fn;
                    if (rO.success && rN.success
                        && rO.interpCount > 0 && rN.interpCount > 0) {
                        rO.d_interpPoints->download(po);
                        rO.d_interp_line_ids->download(fo);
                        rN.d_interpPoints->download(pn);
                        rN.d_interp_line_ids->download(fn);
                    }
                    if (verifyOk && rO.interpCount > 0 && !po.empty()) {
                        const cv::Point2f* a = po.ptr<cv::Point2f>();
                        const cv::Point2f* b = pn.ptr<cv::Point2f>();
                        const int* ia = fo.ptr<int>();
                        const int* ib = fn.ptr<int>();
                        for (int i = 0; i < rO.interpCount; ++i) {
                            if (ia[i] != ib[i]) { verifyOk = false; break; }
                            const double dx = std::abs(a[i].x - b[i].x);
                            const double dy = std::abs(a[i].y - b[i].y);
                            const double d = dx > dy ? dx : dy;
                            if (d > maxDiff) maxDiff = d;
                            if (dx > 1e-6 || dy > 1e-6) ++ptMismatch;
                        }
                        verifyOk = verifyOk && ptMismatch == 0;
                    }
                    // 图像输出: 原版(绿) vs 优化版(红) 叠加；对齐良好呈黄绿色，错位点分离显色
                    if (!po.empty() && !pn.empty()) {
                        cv::Mat img = cv::Mat::zeros(h.imageSize, CV_8UC3);
                        const cv::Point2f* a = po.ptr<cv::Point2f>();
                        const cv::Point2f* b = pn.ptr<cv::Point2f>();
                        for (int i = 0; i < rO.interpCount; ++i)
                            cv::circle(img, cv::Point(cvRound(a[i].x), cvRound(a[i].y)),
                                       1, cv::Scalar(0, 255, 0), cv::FILLED);
                        for (int i = 0; i < rN.interpCount; ++i)
                            cv::circle(img, cv::Point(cvRound(b[i].x), cvRound(b[i].y)),
                                       1, cv::Scalar(0, 0, 255), cv::FILLED);
                        cv::imwrite(outDir + "/" + stem + "_" + side[s]
                                    + "_6interp_orig_green_vs_opt_red.png", img);
                    }
                    if (!verifyOk)
                        spdlog::warn("{} {}: VERIFY FAIL count {}/{} mismatch={} maxDiff={:.3e}",
                                     stem, side[s], rO.interpCount, rN.interpCount,
                                     ptMismatch, maxDiff);
                }

                // 计时
                timer.active = true;
                timer.reset();
                double execWall = 0;
                for (int it = 0; it < iters; ++it) {
                    auto t0 = tNow();
                    auto r = interpOp.Execute(*dPts, *dIds, stream);
                    stream.waitForCompletion();
                    execWall += msBetween(t0, tNow());
                }
                double optWall = 0;
                for (int it = 0; it < iters; ++it) {
                    auto t0 = tNow();
                    auto rn = interpOptOp.Execute(*dPts, *dIds, stream);
                    stream.waitForCompletion();
                    optWall += msBetween(t0, tNow());
                }
                double replWall = 0;
                for (int it = 0; it < iters; ++it) {
                    auto t0 = tNow();
                    runReplica(*dPts, *dIds, rc, rl, rr);
                    replWall += msBetween(t0, tNow());
                }
                timer.active = false;

                double sumSteps = 0;
                for (int k = 1; k < S_COUNT; ++k) sumSteps += timer.wallMs[k];

                csv << stem << ',' << side[s] << ',' << pointCount << ',' << rl << ',' << rr
                    << ',' << rc << ',' << std::fixed << std::setprecision(3)
                    << execWall / iters << ',' << optWall / iters
                    << ',' << (verifyOk ? 1 : 0) << ',' << ptMismatch
                    << ',' << std::setprecision(6) << maxDiff << ',' << std::setprecision(3)
                    << replWall / iters;
                for (int k = 1; k < S_COUNT; ++k)
                    csv << ',' << timer.evMs[k] / iters << ',' << timer.wallMs[k] / iters;
                csv << '\n';

                for (int k = 0; k < S_COUNT; ++k) { gEv[k] += timer.evMs[k]; gWall[k] += timer.wallMs[k]; }
                gExec += execWall; gRepl += replWall; gOpt += optWall;
                ++gRows; gPts += pointCount; gOut += rc;
                if (verifyOk) ++gVerifyOk;
                gPtMismatch += ptMismatch;

                spdlog::info("{} {}: pts={} lines={} rows={} interp={} exec={:.3f}ms opt={:.3f}ms ({:.2f}x) verify={}",
                             stem, side[s], pointCount, rl, rr, rc,
                             execWall / iters, optWall / iters,
                             optWall > 1e-9 ? execWall / optWall : 0.0,
                             verifyOk ? "OK" : "FAIL");
            }
        }
    }

    // ---- 汇总 ----
    if (gRows == 0) { spdlog::error("no frame benchmarked"); return 1; }
    const double n = (double)gRows * iters;
    double sumWall = 0;
    for (int k = 1; k < S_COUNT; ++k) sumWall += gWall[k];
    spdlog::info("=== interp_bench summary: {} frame-sides x {} iters (avg ms) ===", gRows, iters);
    spdlog::info("{:<22}{:>10}{:>10}{:>9}", "step", "wall_ms", "ev_ms", "pct%");
    for (int k = 1; k < S_COUNT; ++k)
        spdlog::info("{:<22}{:>10.3f}{:>10.3f}{:>8.1f}%",
                     kNames[k], gWall[k] / n, gEv[k] / n,
                     100.0 * gWall[k] / std::max(1e-9, sumWall));
    spdlog::info("{:<22}{:>10.3f}", "sum(steps)", sumWall / n);
    spdlog::info("{:<22}{:>10.3f}{}", "replica(whole)", gRepl / n,
                 "  [sum vs whole = interleaving overhead]");
    spdlog::info("{:<22}{:>10.3f}", "Execute(whole)", gExec / n);
    spdlog::info("{:<22}{:>10.3f}   [{:.2f}x vs orig]", "OptExecute(whole)", gOpt / n,
                 gOpt > 1e-9 ? gExec / gOpt : 0.0);
    spdlog::info("verify: {}/{} frame-sides OK, total pt mismatches: {}", gVerifyOk, gRows, gPtMismatch);
    spdlog::info("avg points={} avg interp_out={}", gPts / gRows, gOut / gRows);

    return 0;
}
