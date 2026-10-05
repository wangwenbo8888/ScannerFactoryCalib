#include "plane_map_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <Eigen/Dense>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>

using namespace calib;


CALIB_DEFINE_LOG_TAG(12, PlaneMapCuda);

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

// ---------------------------------------------------------------------------
// Section 2: Helper functions
// ---------------------------------------------------------------------------

static void matx33ToFloat9(const cv::Matx33d& m, float* out) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out[i * 3 + j] = static_cast<float>(m(i, j));
}

static void matx34ToFloat12(const cv::Matx34d& m, float* out) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            out[i * 4 + j] = static_cast<float>(m(i, j));
}

static cv::Matx33d computeF_VL(const cv::Matx33d& virtualK,
                                const cv::Matx33d& virtualR,
                                const cv::Vec3d& virtualT) {
    double tx = virtualT(0), ty = virtualT(1), tz = virtualT(2);
    cv::Matx33d Tx( 0, -tz,  ty,
                   tz,   0, -tx,
                  -ty,  tx,   0);
    cv::Matx33d E = Tx * virtualR;
    cv::Matx33d Kinv = virtualK.inv();
    return Kinv.t() * E * Kinv;
}

static cv::Matx33d computeF_VR(const cv::Matx33d& virtualK,
                                const cv::Matx33d& virtualR,
                                const cv::Vec3d& virtualT,
                                const cv::Matx33d& R1,
                                const cv::Matx33d& R2,
                                const cv::Matx34d& P1,
                                const cv::Matx34d& P2) {
    cv::Matx33d R_LR = R2 * R1.t();
    cv::Vec3d T_LR(P2(0,3) - R_LR(0,0)*P1(0,3) - R_LR(0,1)*P1(1,3) - R_LR(0,2)*P1(2,3),
                   P2(1,3) - R_LR(1,0)*P1(0,3) - R_LR(1,1)*P1(1,3) - R_LR(1,2)*P1(2,3),
                   P2(2,3) - R_LR(2,0)*P1(0,3) - R_LR(2,1)*P1(1,3) - R_LR(2,2)*P1(2,3));
    cv::Vec3d T_VR(T_LR(0) - (R_LR(0,0)*virtualT(0) + R_LR(0,1)*virtualT(1) + R_LR(0,2)*virtualT(2)),
                   T_LR(1) - (R_LR(1,0)*virtualT(0) + R_LR(1,1)*virtualT(1) + R_LR(1,2)*virtualT(2)),
                   T_LR(2) - (R_LR(2,0)*virtualT(0) + R_LR(2,1)*virtualT(1) + R_LR(2,2)*virtualT(2)));
    double tx = T_VR(0), ty = T_VR(1), tz = T_VR(2);
    cv::Matx33d TxV( 0, -tz,  ty,
                    tz,   0, -tx,
                   -ty,  tx,   0);
    cv::Matx33d E = TxV * R_LR * virtualR;
    cv::Matx33d Kinv = virtualK.inv();
    return Kinv.t() * E * Kinv;
}

// ---------------------------------------------------------------------------
// Section 3: CUDA kernels
// ---------------------------------------------------------------------------

__device__ static void clampDepthRange(
    float num_0, float den_0, float num_t, float den_t,
    float lo, float hi, float& tMin, float& tMax)
{
    float a = num_t - lo * den_t;
    float b = num_0 - lo * den_0;
    float c = num_t - hi * den_t;
    float d = num_0 - hi * den_0;

    if (fabsf(a) < 1e-12f && fabsf(c) < 1e-12f) return;

    float r1 = -1e30f, r2 = 1e30f;
    if (fabsf(a) > 1e-12f) {
        float r = -b / a;
        if (a > 0) r1 = r; else r2 = r;
    }
    if (fabsf(c) > 1e-12f) {
        float r = -d / c;
        if (c > 0) r1 = r; else r2 = r;
    }
    tMin = fmaxf(tMin, r1);
    tMax = fminf(tMax, r2);
}

__global__ void kernelProjective(
    const float* __restrict__ d_pixels,
    const int N,
    const float* __restrict__ Kv_inv,
    const float* __restrict__ Rvt,
    const float* __restrict__ Tv,
    const float* __restrict__ R1,
    const float* __restrict__ R2,
    const float* __restrict__ P1,
    const float* __restrict__ P2,
    const int width, const int height,
    const float depthMin, const float depthMax,
    const int depthSamples,
    const float gridStep,
    float* __restrict__ d_candidates,
    int* __restrict__ d_valid_counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float u_v = d_pixels[idx * 3 + 0];
    float v_v = d_pixels[idx * 3 + 1];
    float lid = d_pixels[idx * 3 + 2];

    float pn_x = Kv_inv[0]*u_v + Kv_inv[1]*v_v + Kv_inv[2];
    float pn_y = Kv_inv[3]*u_v + Kv_inv[4]*v_v + Kv_inv[5];
    float pn_z = Kv_inv[6]*u_v + Kv_inv[7]*v_v + Kv_inv[8];

    float dx = Rvt[0]*pn_x + Rvt[1]*pn_y + Rvt[2]*pn_z;
    float dy = Rvt[3]*pn_x + Rvt[4]*pn_y + Rvt[5]*pn_z;
    float dz = Rvt[6]*pn_x + Rvt[7]*pn_y + Rvt[8]*pn_z;

    float p0x = Tv[0], p0y = Tv[1], p0z = Tv[2];

    float RLx_t = R1[0]*dx + R1[1]*dy + R1[2]*dz;
    float RLy_t = R1[3]*dx + R1[4]*dy + R1[5]*dz;
    float RLz_t = R1[6]*dx + R1[7]*dy + R1[8]*dz;

    float RLx_0 = R1[0]*p0x + R1[1]*p0y + R1[2]*p0z;
    float RLy_0 = R1[3]*p0x + R1[4]*p0y + R1[5]*p0z;
    float RLz_0 = R1[6]*p0x + R1[7]*p0y + R1[8]*p0z;

    float uL_num_t = P1[0]*RLx_t + P1[1]*RLy_t + P1[2]*RLz_t;
    float uL_den_t = RLz_t;
    float uL_num_0 = P1[0]*RLx_0 + P1[1]*RLy_0 + P1[2]*RLz_0 + P1[3];
    float uL_den_0 = RLz_0;

    float vL_num_t = P1[4]*RLx_t + P1[5]*RLy_t + P1[6]*RLz_t;
    float vL_den_t = RLz_t;
    float vL_num_0 = P1[4]*RLx_0 + P1[5]*RLy_0 + P1[6]*RLz_0 + P1[7];
    float vL_den_0 = RLz_0;

    float tMin = depthMin;
    float tMax = depthMax;
    bool valid = true;

    clampDepthRange(uL_num_0, uL_den_0, uL_num_t, uL_den_t, 0.0f, (float)(width - 1), tMin, tMax);
    clampDepthRange(vL_num_0, vL_den_0, vL_num_t, vL_den_t, 0.0f, (float)(height - 1), tMin, tMax);

    if (tMin >= tMax || tMax < depthMin || tMin > depthMax) valid = false;

    float RRx_t = R2[0]*dx + R2[1]*dy + R2[2]*dz;
    float RRy_t = R2[3]*dx + R2[4]*dy + R2[5]*dz;
    float RRz_t = R2[6]*dx + R2[7]*dy + R2[8]*dz;

    float RRx_0 = R2[0]*p0x + R2[1]*p0y + R2[2]*p0z;
    float RRy_0 = R2[3]*p0x + R2[4]*p0y + R2[5]*p0z;
    float RRz_0 = R2[6]*p0x + R2[7]*p0y + R2[8]*p0z;

    float uR_num_t = P2[0]*RRx_t + P2[1]*RRy_t + P2[2]*RRz_t;
    float uR_den_t = RRz_t;
    float uR_num_0 = P2[0]*RRx_0 + P2[1]*RRy_0 + P2[2]*RRz_0 + P2[3];
    float uR_den_0 = RRz_0;

    if (valid) {
        clampDepthRange(uR_num_0, uR_den_0, uR_num_t, uR_den_t, 0.0f, (float)(width - 1), tMin, tMax);
        if (tMin >= tMax) valid = false;
    }

    int count = 0;
    int sMin = 0, sMax = depthSamples;
    float depthRange = depthMax - depthMin;
    float invDepthRange = (depthSamples > 1) ? (depthSamples - 1) / depthRange : 0.0f;

    if (valid) {
        float clampedMin = fmaxf(tMin, depthMin);
        float clampedMax = fminf(tMax, depthMax);
        sMin = (int)fmaxf(0.0f, floorf((clampedMin - depthMin) * invDepthRange));
        sMax = (int)fminf((float)(depthSamples - 1), ceilf((clampedMax - depthMin) * invDepthRange)) + 1;
        sMin = max(0, sMin);
        sMax = min(depthSamples, sMax);
    }

    for (int s = 0; s < sMin; ++s) {
        int out_idx = idx * depthSamples + s;
        d_candidates[out_idx * 4 + 3] = -1.0f;
    }

    for (int s = sMin; s < sMax; ++s) {
        float t = depthMin + depthRange * s / fmaxf(depthSamples - 1, 1);
        float Px = p0x + t * dx;
        float Py = p0y + t * dy;
        float Pz = p0z + t * dz;

        if (Pz < 1e-6f) {
            int out_idx = idx * depthSamples + s;
            d_candidates[out_idx * 4 + 3] = -1.0f;
            continue;
        }

        float RLx = RLx_0 + t * RLx_t;
        float RLy = RLy_0 + t * RLy_t;
        float RLz = RLz_0 + t * RLz_t;

        if (fabsf(RLz) < 1e-6f) {
            int out_idx = idx * depthSamples + s;
            d_candidates[out_idx * 4 + 3] = -1.0f;
            continue;
        }

        float invRLz = 1.0f / RLz;
        float uL = (uL_num_0 + t * uL_num_t) * invRLz;
        float vL = (vL_num_0 + t * vL_num_t) * invRLz;

        if (uL < 0 || uL >= width || vL < 0 || vL >= height) {
            int out_idx = idx * depthSamples + s;
            d_candidates[out_idx * 4 + 3] = -1.0f;
            continue;
        }

        float RRx = RRx_0 + t * RRx_t;
        float RRy = RRy_0 + t * RRy_t;
        float RRz = RRz_0 + t * RRz_t;

        if (fabsf(RRz) < 1e-6f) {
            int out_idx = idx * depthSamples + s;
            d_candidates[out_idx * 4 + 3] = -1.0f;
            continue;
        }

        float invRRz = 1.0f / RRz;
        float uR = (uR_num_0 + t * uR_num_t) * invRRz;

        if (uR < 0 || uR >= width) {
            int out_idx = idx * depthSamples + s;
            d_candidates[out_idx * 4 + 3] = -1.0f;
            continue;
        }

        float uL_g = roundf(uL / gridStep) * gridStep;
        float vL_g = roundf(vL / gridStep) * gridStep;
        float uR_g = roundf(uR / gridStep) * gridStep;

        int out_idx = idx * depthSamples + count;
        d_candidates[out_idx * 4 + 0] = uL_g;
        d_candidates[out_idx * 4 + 1] = vL_g;
        d_candidates[out_idx * 4 + 2] = uR_g;
        d_candidates[out_idx * 4 + 3] = lid;
        count++;
    }

    for (int s = count; s < depthSamples; ++s) {
        int out_idx = idx * depthSamples + s;
        d_candidates[out_idx * 4 + 3] = -1.0f;
    }

    d_valid_counts[idx] = count;
}

__global__ void kernelFundamental(
    const float* __restrict__ d_pixels,
    const int N,
    const float* __restrict__ FVL,
    const float* __restrict__ FVR,
    const int width, const int height,
    const float gridStep,
    const float epipolarStep,
    const int maxSamples,
    float* __restrict__ d_candidates,
    int* __restrict__ d_valid_counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float u_v = d_pixels[idx * 3 + 0];
    float v_v = d_pixels[idx * 3 + 1];
    float lid = d_pixels[idx * 3 + 2];

    float aL = FVL[0]*u_v + FVL[1]*v_v + FVL[2];
    float bL = FVL[3]*u_v + FVL[4]*v_v + FVL[5];
    float cL = FVL[6]*u_v + FVL[7]*v_v + FVL[8];

    float aR = FVR[0]*u_v + FVR[1]*v_v + FVR[2];
    float bR = FVR[3]*u_v + FVR[4]*v_v + FVR[5];
    float cR = FVR[6]*u_v + FVR[7]*v_v + FVR[8];

    int count = 0;

    bool useV = (fabsf(bL) > fabsf(aL)) && (fabsf(bR) > fabsf(aR));

    if (useV) {
        for (float vL = 0; vL < height && count < maxSamples; vL += epipolarStep) {
            float uL = -(aL * vL + cL) / bL;
            if (uL < 0 || uL >= width) continue;

            float uR = -(aR * vL + cR) / bR;
            if (uR < 0 || uR >= width) continue;

            float uL_g = roundf(uL / gridStep) * gridStep;
            float vL_g = roundf(vL / gridStep) * gridStep;
            float uR_g = roundf(uR / gridStep) * gridStep;

            int out_idx = idx * maxSamples + count;
            d_candidates[out_idx * 4 + 0] = uL_g;
            d_candidates[out_idx * 4 + 1] = vL_g;
            d_candidates[out_idx * 4 + 2] = uR_g;
            d_candidates[out_idx * 4 + 3] = lid;
            count++;
        }
    }

    d_valid_counts[idx] = count;
    for (int s = count; s < maxSamples; ++s) {
        int out_idx = idx * maxSamples + s;
        d_candidates[out_idx * 4 + 3] = -1.0f;
    }
}

__global__ void kernelCompactCandidates(
    const float* __restrict__ d_candidates,
    const int totalCandidates,
    float* __restrict__ d_compact,
    int* __restrict__ d_compact_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalCandidates) return;

    float lid = d_candidates[idx * 4 + 3];
    if (lid < 0) return;

    int pos = atomicAdd(d_compact_count, 1);
    d_compact[pos * 4 + 0] = d_candidates[idx * 4 + 0];
    d_compact[pos * 4 + 1] = d_candidates[idx * 4 + 1];
    d_compact[pos * 4 + 2] = d_candidates[idx * 4 + 2];
    d_compact[pos * 4 + 3] = lid;
}

__global__ void kernelComputeStats(
    const float* __restrict__ d_data,
    const int N,
    const int numLines,
    int* __restrict__ d_numPairs,
    float* __restrict__ d_uMin,
    float* __restrict__ d_uMax,
    float* __restrict__ d_vMin,
    float* __restrict__ d_vMax)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float uL  = d_data[idx * 4 + 0];
    float vL  = d_data[idx * 4 + 1];
    float lid = d_data[idx * 4 + 3];

    int lineId = static_cast<int>(lid);
    if (lineId < 0 || lineId >= numLines) return;

    atomicAdd(&d_numPairs[lineId], 1);

    unsigned int oldval, newval, readval;

    readval = atomicCAS(reinterpret_cast<unsigned int*>(&d_uMin[lineId]),
                         __float_as_uint(FLT_MAX), __float_as_uint(uL));
    if (__uint_as_float(readval) > uL) {
        oldval = readval;
        do {
            newval = __float_as_uint(fminf(__uint_as_float(oldval), uL));
            oldval = atomicCAS(reinterpret_cast<unsigned int*>(&d_uMin[lineId]), oldval, newval);
        } while (oldval != readval && __uint_as_float(oldval) > uL);
    }

    readval = atomicCAS(reinterpret_cast<unsigned int*>(&d_uMax[lineId]),
                         __float_as_uint(-FLT_MAX), __float_as_uint(uL));
    if (__uint_as_float(readval) < uL) {
        oldval = readval;
        do {
            newval = __float_as_uint(fmaxf(__uint_as_float(oldval), uL));
            oldval = atomicCAS(reinterpret_cast<unsigned int*>(&d_uMax[lineId]), oldval, newval);
        } while (oldval != readval && __uint_as_float(oldval) < uL);
    }

    readval = atomicCAS(reinterpret_cast<unsigned int*>(&d_vMin[lineId]),
                         __float_as_uint(FLT_MAX), __float_as_uint(vL));
    if (__uint_as_float(readval) > vL) {
        oldval = readval;
        do {
            newval = __float_as_uint(fminf(__uint_as_float(oldval), vL));
            oldval = atomicCAS(reinterpret_cast<unsigned int*>(&d_vMin[lineId]), oldval, newval);
        } while (oldval != readval && __uint_as_float(oldval) > vL);
    }

    readval = atomicCAS(reinterpret_cast<unsigned int*>(&d_vMax[lineId]),
                         __float_as_uint(-FLT_MAX), __float_as_uint(vL));
    if (__uint_as_float(readval) < vL) {
        oldval = readval;
        do {
            newval = __float_as_uint(fmaxf(__uint_as_float(oldval), vL));
            oldval = atomicCAS(reinterpret_cast<unsigned int*>(&d_vMax[lineId]), oldval, newval);
        } while (oldval != readval && __uint_as_float(oldval) < vL);
    }
}

__global__ void kernelExtractRightU(
    const float* __restrict__ d_data, int N,
    float* __restrict__ d_right_u)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    d_right_u[idx] = d_data[idx * 4 + 2];
}

__global__ void kernelMaxLineId(
    const float* __restrict__ d_data, int N,
    int* __restrict__ d_result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int lid = static_cast<int>(d_data[idx * 4 + 3]);
    atomicMax(d_result, lid);
}

// ---------------------------------------------------------------------------
// Section 4: Impl ctor/dtor/warmup/setParams
// ---------------------------------------------------------------------------

PlaneMapCuda::Impl::Impl(const PlaneMapParams& params)
    : params_(params)
{
    params_.validate();

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("[12-PlaneMapCuda] No CUDA devices found");
    }
    if (params_.deviceId >= deviceCount) {
        throw std::invalid_argument("[12-PlaneMapCuda] deviceId >= device count");
    }

    old_device_id_ = 0;
    cudaGetDevice(&old_device_id_);
    if (params_.deviceId != old_device_id_) {
        cudaSetDevice(params_.deviceId);
    }
}

PlaneMapCuda::Impl::~Impl() {
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        CALIB_LOG_ERROR("cudaDeviceSynchronize in destructor failed: {}",
                        cudaGetErrorString(sync_err));
    }
}

void PlaneMapCuda::Impl::Warmup(int numVirtualPixels, int maxLineId) {
    if (numVirtualPixels <= 0 || maxLineId < 0) {
        CALIB_LOG_WARN("warmup(): invalid numVirtualPixels={} or maxLineId={}, skipping",
                       numVirtualPixels, maxLineId);
        return;
    }
    int maxSamples = params_.depthSamples;
    d_pixels_raw_buf_.create(1, numVirtualPixels * 3, CV_32FC1);
    d_cand_buf_.create(1, numVirtualPixels * maxSamples * 4, CV_32FC1);
    d_compact_buf_.create(1, numVirtualPixels * maxSamples * 4, CV_32FC1);
    d_compact_count_buf_.create(1, 1, CV_32SC1);
    d_Kv_inv_buf_.create(1, 9, CV_32FC1);
    d_Rvt_buf_.create(1, 9, CV_32FC1);
    d_Tv_buf_.create(1, 3, CV_32FC1);
    d_R1_buf_.create(1, 9, CV_32FC1);
    d_R2_buf_.create(1, 9, CV_32FC1);
    d_P1_buf_.create(1, 12, CV_32FC1);
    d_P2_buf_.create(1, 12, CV_32FC1);
    d_FVL_buf_.create(1, 9, CV_32FC1);
    d_FVR_buf_.create(1, 9, CV_32FC1);
    d_valid_counts_buf_.create(1, numVirtualPixels, CV_32SC1);
    warmed_up_ = true;
    warmup_count_ = numVirtualPixels;
    warmup_max_lid_ = maxLineId;
    CALIB_LOG_INFO("warmup(): pre-allocated GPU buffers for numVirtualPixels={}, maxLineId={}",
                   numVirtualPixels, maxLineId);
}

void PlaneMapCuda::Impl::SetParams(const PlaneMapParams& params) {
#ifndef NDEBUG
    if (inProcess_.load()) {
        CALIB_LOG_ERROR("setParams(): called while process() is running");
        throw std::runtime_error("[12-PlaneMapCuda] setParams() called during process()");
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
    warmup_max_lid_ = 0;
    CALIB_LOG_INFO("setParams(): params updated, warmup reset");
}

// ---------------------------------------------------------------------------
// Section 5: Impl::process() — full implementation
// ---------------------------------------------------------------------------

namespace {
struct CandidateKey {
    float uL, vL, uR, lid;
};

inline bool candidateLess(const CandidateKey& a, const CandidateKey& b) {
    if (a.lid != b.lid) return a.lid < b.lid;
    if (a.vL != b.vL) return a.vL < b.vL;
    return a.uL < b.uL;
}

inline bool candidateEqual(const CandidateKey& a, const CandidateKey& b) {
    return a.lid == b.lid && a.vL == b.vL && a.uL == b.uL;
}

struct GpuSortUniqueResult {
    int uniqueCount;
};

GpuSortUniqueResult gpuSortUnique(float* d_compact_ptr, int compactCount, CUstream_st* cu_stream) {
    float4* d_data = reinterpret_cast<float4*>(d_compact_ptr);

    thrust::stable_sort(
        thrust::cuda::par.on(cu_stream),
        thrust::device_pointer_cast(d_data),
        thrust::device_pointer_cast(d_data + compactCount),
        [] __device__ (const float4& a, const float4& b) -> bool {
            if (a.w != b.w) return a.w < b.w;
            if (a.y != b.y) return a.y < b.y;
            return a.x < b.x;
        });

    auto new_end = thrust::unique(
        thrust::cuda::par.on(cu_stream),
        thrust::device_pointer_cast(d_data),
        thrust::device_pointer_cast(d_data + compactCount),
        [] __device__ (const float4& a, const float4& b) -> bool {
            return a.w == b.w && a.y == b.y && a.x == b.x;
        });

    int uniqueCount = static_cast<int>(thrust::raw_pointer_cast(new_end) - d_data);
    return GpuSortUniqueResult{uniqueCount};
}
}

PlaneMapResult PlaneMapCuda::Impl::Execute(
    const cv::cuda::GpuMat& d_virtual_pixels,
    const cv::Matx33d& virtualK,
    const cv::Matx33d& virtualR,
    const cv::Vec3d& virtualT,
    const calib::StereoCalibration& calib,
    cv::cuda::Stream stream)
{
#ifndef NDEBUG
    ScopedFlag guard(&inProcess_);
#endif

    PlaneMapResult result;

    try {
        const int N = d_virtual_pixels.rows * d_virtual_pixels.cols;

        if (N == 0) {
            result.success = true;
            result.message = "Empty input, no pixels to process";
            return result;
        }

        cv::Matx33d R1 = calib.getR1();
        cv::Matx33d R2 = calib.getR2();
        cv::Matx34d P1 = calib.getP1();
        cv::Matx34d P2 = calib.getP2();
        cv::Size imageSize = calib.imageSize;

        const int width = imageSize.width;
        const int height = imageSize.height;

        cv::Matx33d virtualK_inv = virtualK.inv();
        cv::Matx33d Rvt = virtualR.t();

        float h_Kv_inv[9], h_Rvt[9], h_Tv[3];
        float h_R1[9], h_R2[9], h_P1[12], h_P2[12];

        matx33ToFloat9(virtualK_inv, h_Kv_inv);
        matx33ToFloat9(Rvt, h_Rvt);
        h_Tv[0] = static_cast<float>(virtualT(0));
        h_Tv[1] = static_cast<float>(virtualT(1));
        h_Tv[2] = static_cast<float>(virtualT(2));
        matx33ToFloat9(R1, h_R1);
        matx33ToFloat9(R2, h_R2);
        matx34ToFloat12(P1, h_P1);
        matx34ToFloat12(P2, h_P2);

        float h_FVL[9], h_FVR[9];
        cv::Matx33d F_VL = computeF_VL(virtualK, virtualR, virtualT);
        cv::Matx33d F_VR = computeF_VR(virtualK, virtualR, virtualT, R1, R2, P1, P2);
        matx33ToFloat9(F_VL, h_FVL);
        matx33ToFloat9(F_VR, h_FVR);

        const int maxSamples = params_.depthSamples;
        const int totalSlots = N * maxSamples;

        cv::cuda::GpuMat d_pixels_raw;
        cv::cuda::GpuMat d_cand;
        cv::cuda::GpuMat d_compact;
        cv::cuda::GpuMat d_compact_count;
        cv::cuda::GpuMat d_valid_counts;

        if (warmed_up_ && warmup_count_ >= N) {
            d_pixels_raw = d_pixels_raw_buf_.colRange(0, N * 3);
            d_cand = d_cand_buf_.colRange(0, totalSlots * 4);
            d_compact = d_compact_buf_.colRange(0, totalSlots * 4);
            d_compact_count = d_compact_count_buf_;
            d_valid_counts = d_valid_counts_buf_.colRange(0, N);
        } else {
            d_pixels_raw.create(1, N * 3, CV_32FC1);
            d_cand.create(1, totalSlots * 4, CV_32FC1);
            d_compact.create(1, totalSlots * 4, CV_32FC1);
            d_compact_count.create(1, 1, CV_32SC1);
            d_valid_counts.create(1, N, CV_32SC1);
        }

        d_Kv_inv_buf_.upload(cv::Mat(1, 9, CV_32FC1, h_Kv_inv), stream);
        d_Rvt_buf_.upload(cv::Mat(1, 9, CV_32FC1, h_Rvt), stream);
        d_Tv_buf_.upload(cv::Mat(1, 3, CV_32FC1, h_Tv), stream);
        d_R1_buf_.upload(cv::Mat(1, 9, CV_32FC1, h_R1), stream);
        d_R2_buf_.upload(cv::Mat(1, 9, CV_32FC1, h_R2), stream);
        d_P1_buf_.upload(cv::Mat(1, 12, CV_32FC1, h_P1), stream);
        d_P2_buf_.upload(cv::Mat(1, 12, CV_32FC1, h_P2), stream);
        d_FVL_buf_.upload(cv::Mat(1, 9, CV_32FC1, h_FVL), stream);
        d_FVR_buf_.upload(cv::Mat(1, 9, CV_32FC1, h_FVR), stream);

        size_t pixels_bytes = N * 3 * sizeof(float);
        cudaError_t err = cudaMemcpy(d_pixels_raw.ptr<float>(),
                                     d_virtual_pixels.ptr<float>(),
                                     pixels_bytes, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy D2D pixels failed: ") + cudaGetErrorString(err));
        }

        cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);

        const int kBlock = 256;
        const int kGrid = (N + kBlock - 1) / kBlock;

        if (params_.method == PlaneMapMethod::Projective) {
            kernelProjective<<<kGrid, kBlock, 0, cu_stream>>>(
                d_pixels_raw.ptr<float>(), N,
                d_Kv_inv_buf_.ptr<float>(), d_Rvt_buf_.ptr<float>(), d_Tv_buf_.ptr<float>(),
                d_R1_buf_.ptr<float>(), d_R2_buf_.ptr<float>(),
                d_P1_buf_.ptr<float>(), d_P2_buf_.ptr<float>(),
                width, height,
                params_.depthMin, params_.depthMax,
                params_.depthSamples, params_.gridStep,
                d_cand.ptr<float>(), d_valid_counts.ptr<int>());
        } else {
            kernelFundamental<<<kGrid, kBlock, 0, cu_stream>>>(
                d_pixels_raw.ptr<float>(), N,
                d_FVL_buf_.ptr<float>(), d_FVR_buf_.ptr<float>(),
                width, height,
                params_.gridStep, params_.epipolarStep,
                maxSamples,
                d_cand.ptr<float>(), d_valid_counts.ptr<int>());
        }

        err = cudaStreamSynchronize(cu_stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("kernel launch/sync failed: ") + cudaGetErrorString(err));
        }

        err = cudaMemsetAsync(d_compact_count.ptr<int>(), 0, sizeof(int), cu_stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemset d_compact_count failed: ") + cudaGetErrorString(err));
        }

        int kGrid2 = (totalSlots + kBlock - 1) / kBlock;
        kernelCompactCandidates<<<kGrid2, kBlock, 0, cu_stream>>>(
            d_cand.ptr<float>(), totalSlots,
            d_compact.ptr<float>(), d_compact_count.ptr<int>());

        err = cudaStreamSynchronize(cu_stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("kernelCompactCandidates sync failed: ") + cudaGetErrorString(err));
        }

        int compactCount = 0;
        err = cudaMemcpyAsync(&compactCount, d_compact_count.ptr<int>(), sizeof(int),
                              cudaMemcpyDeviceToHost, cu_stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy compact_count D2H failed: ") + cudaGetErrorString(err));
        }
        err = cudaStreamSynchronize(cu_stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaStreamSynchronize after compact_count download failed: ") + cudaGetErrorString(err));
        }

        CALIB_LOG_DEBUG("Compact candidate count: {}", compactCount);

        if (compactCount == 0) {
            result.success = true;
            result.message = "No valid candidates generated";
            result.totalPairs = 0;
            return result;
        }

        float* d_compact_ptr = d_compact.ptr<float>();

        auto sort_result = gpuSortUnique(d_compact_ptr, compactCount, cu_stream);

        err = cudaStreamSynchronize(cu_stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("GPU sort/unique sync failed: ") + cudaGetErrorString(err));
        }

        int uniqueCount = sort_result.uniqueCount;
        CALIB_LOG_DEBUG("Unique candidate count after GPU dedup: {}", uniqueCount);

        if (uniqueCount == 0) {
            result.success = true;
            result.message = "No unique candidates after dedup";
            result.totalPairs = 0;
            return result;
        }

        result.d_left_to_right = std::make_shared<cv::cuda::GpuMat>();
        result.d_right_u = std::make_shared<cv::cuda::GpuMat>();

        result.d_left_to_right->create(uniqueCount, 1, CV_32FC4);
        result.d_right_u->create(uniqueCount, 1, CV_32FC1);

        err = cudaMemcpyAsync(result.d_left_to_right->ptr<float>(), d_compact_ptr,
                              uniqueCount * 4 * sizeof(float),
                              cudaMemcpyDeviceToDevice, cu_stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy output D2D failed: ") + cudaGetErrorString(err));
        }

        int kGridExtract = (uniqueCount + kBlock - 1) / kBlock;
        kernelExtractRightU<<<kGridExtract, kBlock, 0, cu_stream>>>(
            d_compact_ptr, uniqueCount, result.d_right_u->ptr<float>());

        cv::cuda::GpuMat d_maxLineId(1, 1, CV_32SC1);
        cudaMemsetAsync(d_maxLineId.ptr<int>(), 0, sizeof(int), cu_stream);
        kernelMaxLineId<<<kGridExtract, kBlock, 0, cu_stream>>>(
            d_compact_ptr, uniqueCount, d_maxLineId.ptr<int>());

        int maxLineId = 0;
        err = cudaMemcpyAsync(&maxLineId, d_maxLineId.ptr<int>(), sizeof(int),
                              cudaMemcpyDeviceToHost, cu_stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy maxLineId D2H failed: ") + cudaGetErrorString(err));
        }
        err = cudaStreamSynchronize(cu_stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaStreamSynchronize extract/maxLineId failed: ") + cudaGetErrorString(err));
        }

        int numLines = maxLineId + 1;

        cv::cuda::GpuMat d_stats_numPairs(1, numLines, CV_32SC1);
        cv::cuda::GpuMat d_stats_uMin(1, numLines, CV_32FC1);
        cv::cuda::GpuMat d_stats_uMax(1, numLines, CV_32FC1);
        cv::cuda::GpuMat d_stats_vMin(1, numLines, CV_32FC1);
        cv::cuda::GpuMat d_stats_vMax(1, numLines, CV_32FC1);

        cudaMemsetAsync(d_stats_numPairs.ptr<int>(), 0, numLines * sizeof(int), cu_stream);
        thrust::fill(thrust::cuda::par.on(cu_stream),
                     thrust::device_pointer_cast(d_stats_uMin.ptr<float>()),
                     thrust::device_pointer_cast(d_stats_uMin.ptr<float>() + numLines), FLT_MAX);
        thrust::fill(thrust::cuda::par.on(cu_stream),
                     thrust::device_pointer_cast(d_stats_uMax.ptr<float>()),
                     thrust::device_pointer_cast(d_stats_uMax.ptr<float>() + numLines), -FLT_MAX);
        thrust::fill(thrust::cuda::par.on(cu_stream),
                     thrust::device_pointer_cast(d_stats_vMin.ptr<float>()),
                     thrust::device_pointer_cast(d_stats_vMin.ptr<float>() + numLines), FLT_MAX);
        thrust::fill(thrust::cuda::par.on(cu_stream),
                     thrust::device_pointer_cast(d_stats_vMax.ptr<float>()),
                     thrust::device_pointer_cast(d_stats_vMax.ptr<float>() + numLines), -FLT_MAX);

        int kGrid3 = (uniqueCount + kBlock - 1) / kBlock;
        kernelComputeStats<<<kGrid3, kBlock, 0, cu_stream>>>(
            d_compact_ptr, uniqueCount, numLines,
            d_stats_numPairs.ptr<int>(),
            d_stats_uMin.ptr<float>(), d_stats_uMax.ptr<float>(),
            d_stats_vMin.ptr<float>(), d_stats_vMax.ptr<float>());

        err = cudaStreamSynchronize(cu_stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("kernelComputeStats sync failed: ") + cudaGetErrorString(err));
        }

        std::vector<int> h_numPairs(numLines);
        std::vector<float> h_uMin(numLines), h_uMax(numLines);
        std::vector<float> h_vMin(numLines), h_vMax(numLines);
        cudaMemcpy(h_numPairs.data(), d_stats_numPairs.ptr<int>(), numLines * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_uMin.data(), d_stats_uMin.ptr<float>(), numLines * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_uMax.data(), d_stats_uMax.ptr<float>(), numLines * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vMin.data(), d_stats_vMin.ptr<float>(), numLines * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vMax.data(), d_stats_vMax.ptr<float>(), numLines * sizeof(float), cudaMemcpyDeviceToHost);

        for (int lid = 0; lid < numLines; ++lid) {
            if (h_numPairs[lid] > 0) {
                LineMapStats st;
                st.lineId = lid;
                st.numPairs = h_numPairs[lid];
                st.uMin = h_uMin[lid];
                st.uMax = h_uMax[lid];
                st.vMin = h_vMin[lid];
                st.vMax = h_vMax[lid];
                result.lineStats.push_back(std::move(st));
            }
        }
        std::sort(result.lineStats.begin(), result.lineStats.end(),
                  [](const LineMapStats& a, const LineMapStats& b) { return a.lineId < b.lineId; });

        result.totalPairs = uniqueCount;
        result.success = true;
        result.message = fmt::format("OK: {} unique pairs across {} lines",
                                     uniqueCount, result.lineStats.size());

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
