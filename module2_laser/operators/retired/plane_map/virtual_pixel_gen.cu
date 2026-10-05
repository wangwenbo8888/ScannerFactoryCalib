#include "virtual_pixel_gen.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace calib;


CALIB_DEFINE_LOG_TAG(12, VirtualPixelGen);

#ifndef NDEBUG
class ScopedPixelGenFlag {
public:
    explicit ScopedPixelGenFlag(std::atomic<bool>* flag) : flag_(flag) {
        flag_->store(true);
    }
    ~ScopedPixelGenFlag() { flag_->store(false); }
    ScopedPixelGenFlag(const ScopedPixelGenFlag&) = delete;
    ScopedPixelGenFlag& operator=(const ScopedPixelGenFlag&) = delete;
private:
    std::atomic<bool>* flag_;
};
#endif

__global__ void kernelGenerateVirtualPixels(
    const int width, const int height,
    const float gridStep,
    const int* __restrict__ d_lineIds,
    const int numLines,
    const int pixelsPerLine,
    float* __restrict__ d_output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = numLines * pixelsPerLine;
    if (idx >= totalPixels) return;

    int lineIdx = idx / pixelsPerLine;
    int pixelIdx = idx % pixelsPerLine;

    int colsPerLine = static_cast<int>(width / gridStep) + 1;
    int rowsPerLine = static_cast<int>(height / gridStep) + 1;

    int col = pixelIdx % colsPerLine;
    int row = pixelIdx / colsPerLine;

    float u = col * gridStep;
    float v = row * gridStep;
    float lid = static_cast<float>(d_lineIds[lineIdx]);

    d_output[idx * 3 + 0] = u;
    d_output[idx * 3 + 1] = v;
    d_output[idx * 3 + 2] = lid;
}

struct VirtualPixelGenerator::Impl {
    VirtualPixelGenParams params_;
    cv::cuda::GpuMat d_output_buf_;
    bool warmed_up_ = false;
    int warmup_count_ = 0;
    int old_device_id_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inGenerate_{false};
#endif

    explicit Impl(const VirtualPixelGenParams& params)
        : params_(params)
    {
        params_.validate();

        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            throw std::runtime_error("[12-VirtualPixelGen] No CUDA devices found");
        }
        if (params_.deviceId >= deviceCount) {
            throw std::invalid_argument("[12-VirtualPixelGen] deviceId >= device count");
        }

        old_device_id_ = 0;
        cudaGetDevice(&old_device_id_);
        if (params_.deviceId != old_device_id_) {
            cudaSetDevice(params_.deviceId);
        }
    }

    ~Impl() {
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            CALIB_LOG_ERROR("cudaDeviceSynchronize in destructor failed: {}",
                            cudaGetErrorString(sync_err));
        }
    }

    void Warmup(int maxPixels) {
        if (maxPixels <= 0) {
            CALIB_LOG_WARN("warmup(): invalid maxPixels={}, skipping", maxPixels);
            return;
        }
        warmed_up_ = true;
        warmup_count_ = maxPixels;
        d_output_buf_.create(1, maxPixels, CV_32FC3);
        CALIB_LOG_INFO("warmup(): pre-allocated GPU buffer for maxPixels={}", maxPixels);
    }

    void SetParams(const VirtualPixelGenParams& params) {
#ifndef NDEBUG
        if (inGenerate_.load()) {
            CALIB_LOG_ERROR("setParams(): called while generate() is running");
            throw std::runtime_error("[12-VirtualPixelGen] setParams() called during generate()");
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
        d_output_buf_.release();
        CALIB_LOG_INFO("setParams(): params updated, warmup reset");
    }

    const VirtualPixelGenParams& GetParams() const { return params_; }

    VirtualPixelGenResult generateImpl(
        const cv::Matx33d& virtualK,
        const cv::Size& imageSize,
        const std::vector<int>& lineIds,
        cv::cuda::Stream& stream)
    {
#ifndef NDEBUG
        ScopedPixelGenFlag guard(&inGenerate_);
#endif

        VirtualPixelGenResult result;

        if (lineIds.empty()) {
            CALIB_LOG_WARN("generate(): empty lineIds, returning empty result");
            result.success = true;
            result.message = "empty lineIds";
            return result;
        }

        if (imageSize.width <= 0 || imageSize.height <= 0) {
            CALIB_LOG_ERROR("generate(): invalid imageSize={}x{}", imageSize.width, imageSize.height);
            throw std::invalid_argument("[12-VirtualPixelGen] imageSize must have positive width and height");
        }

        const int width = imageSize.width;
        const int height = imageSize.height;
        const int numLines = static_cast<int>(lineIds.size());
        const int colsPerLine = static_cast<int>(width / params_.gridStep) + 1;
        const int rowsPerLine = static_cast<int>(height / params_.gridStep) + 1;
        const int pixelsPerLine = colsPerLine * rowsPerLine;
        const int totalPixels = numLines * pixelsPerLine;

        CALIB_LOG_DEBUG("generate(): numLines={}, colsPerLine={}, rowsPerLine={}, totalPixels={}",
                        numLines, colsPerLine, rowsPerLine, totalPixels);

        cv::cuda::GpuMat d_output(1, totalPixels, CV_32FC3);

        int* d_lineIds = nullptr;
        cudaError_t err = cudaMalloc(&d_lineIds, numLines * sizeof(int));
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc d_lineIds failed: ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(d_lineIds, lineIds.data(), numLines * sizeof(int),
                              cudaMemcpyHostToDevice,
                              cv::cuda::StreamAccessor::getStream(stream));
        if (err != cudaSuccess) {
            cudaFree(d_lineIds);
            throw std::runtime_error(std::string("cudaMemcpy lineIds H2D failed: ") + cudaGetErrorString(err));
        }

        cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);

        const int kBlock = 256;
        const int kGrid = (totalPixels + kBlock - 1) / kBlock;

        kernelGenerateVirtualPixels<<<kGrid, kBlock, 0, cu_stream>>>(
            width, height,
            params_.gridStep,
            d_lineIds, numLines, pixelsPerLine,
            d_output.ptr<float>());

        err = cudaStreamSynchronize(cu_stream);
        cudaFree(d_lineIds);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("kernelGenerateVirtualPixels launch/sync failed: ") + cudaGetErrorString(err));
        }

        CALIB_LOG_INFO("generate(): produced {} virtual pixels for {} lines",
                       totalPixels, numLines);

        result.success = true;
        result.message = "Virtual pixels generated";
        result.d_virtualPixels = std::move(d_output);
        return result;
    }
};

#if BUILD_CUDA

VirtualPixelGenerator::VirtualPixelGenerator(const VirtualPixelGenParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("VirtualPixelGenerator initialized: deviceId={}, gridStep={}",
                   params.deviceId, params.gridStep);
}

VirtualPixelGenerator::~VirtualPixelGenerator() = default;

VirtualPixelGenResult VirtualPixelGenerator::Execute(
    const cv::Matx33d& virtualK,
    const cv::Size& imageSize,
    const std::vector<int>& lineIds,
    cv::cuda::Stream& stream)
{
    CALIB_LOG_DEBUG("generate() called: imageSize={}x{}, numLineIds={}",
                    imageSize.width, imageSize.height, lineIds.size());
    return pImpl_->generateImpl(virtualK, imageSize, lineIds, stream);
}

VirtualPixelGenResult VirtualPixelGenerator::Execute(
    const cv::Matx33d& virtualK,
    const cv::Size& imageSize,
    const std::vector<int>& lineIds)
{
    cv::cuda::Stream stream;
    return Execute(virtualK, imageSize, lineIds, stream);
}

void VirtualPixelGenerator::Destroy() {
    pImpl_.reset();
}

void VirtualPixelGenerator::Warmup(int maxPixels) {
    CALIB_LOG_INFO("warmup() called: maxPixels={}", maxPixels);
    pImpl_->Warmup(maxPixels);
}

void VirtualPixelGenerator::SetParams(const VirtualPixelGenParams& params) {
    CALIB_LOG_INFO("setParams(): deviceId={}, gridStep={}",
                   params.deviceId, params.gridStep);
    pImpl_->SetParams(params);
}

const VirtualPixelGenParams& VirtualPixelGenerator::GetParams() const {
    return pImpl_->GetParams();
}

#else

struct VirtualPixelGenerator::Impl {};

VirtualPixelGenerator::VirtualPixelGenerator(const VirtualPixelGenParams& params)
    : pImpl_(std::make_unique<Impl>())
{
    CALIB_LOG_WARN("VirtualPixelGenerator: BUILD_CUDA=OFF, all operations will throw");
    params.validate();
}

VirtualPixelGenerator::~VirtualPixelGenerator() = default;

VirtualPixelGenResult VirtualPixelGenerator::Execute(
    const cv::Matx33d&, const cv::Size&,
    const std::vector<int>&, cv::cuda::Stream&) {
    VirtualPixelGenResult r;
    r.success = false;
    r.message = "BUILD_CUDA=OFF";
    return r;
}

VirtualPixelGenResult VirtualPixelGenerator::Execute(
    const cv::Matx33d&, const cv::Size&,
    const std::vector<int>&) {
    VirtualPixelGenResult r;
    r.success = false;
    r.message = "BUILD_CUDA=OFF";
    return r;
}

void VirtualPixelGenerator::Destroy() {}

void VirtualPixelGenerator::Warmup(int) {
    throw std::runtime_error("[12-VirtualPixelGen] CUDA not available (BUILD_CUDA=OFF)");
}

void VirtualPixelGenerator::SetParams(const VirtualPixelGenParams&) {
    throw std::runtime_error("[12-VirtualPixelGen] CUDA not available (BUILD_CUDA=OFF)");
}

const VirtualPixelGenParams& VirtualPixelGenerator::GetParams() const {
    throw std::runtime_error("[12-VirtualPixelGen] CUDA not available (BUILD_CUDA=OFF)");
}

#endif
