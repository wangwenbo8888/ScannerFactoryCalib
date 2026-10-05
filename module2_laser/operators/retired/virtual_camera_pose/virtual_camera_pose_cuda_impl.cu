/**
 * @file virtual_camera_pose_cuda_impl.cu
 * @brief 激光器虚拟相机光心和初步外参CUDA算子 - CUDA实现（struct Impl 方法）
 *
 * 算法步骤：
 *   Step 1: GPU → CPU 数据传输
 *   Step 2: 按 line_id 分组端点
 *   Step 3: 逐线 RANSAC + PCA 拟合3D直线
 *   Step 4: 解析求距离所有直线之和最小的交点
 *   Step 5: 组装输出和诊断信息
 */

#include "virtual_camera_pose_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <Eigen/Dense>
#include <unordered_map>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace calib;


CALIB_DEFINE_LOG_TAG(10, VirtualCameraPoseCuda);

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
// Helper: point-to-line distance
// ============================================================================

static inline double pointToLineDistance(const Eigen::Vector3d& p,
                                         const Eigen::Vector3d& linePoint,
                                         const Eigen::Vector3d& lineDir) {
    Eigen::Vector3d v = p - linePoint;
    Eigen::Vector3d cross = v.cross(lineDir);
    return cross.norm();
}

// ============================================================================
// RANSAC Line Fitting
// ============================================================================

struct FittedLine {
    Eigen::Vector3d point;
    Eigen::Vector3d direction;
    int inlierCount;
    double fittingError;
};

static bool ransacLineFit(const std::vector<Eigen::Vector3d>& points,
                          const VirtualCameraPoseParams& params,
                          FittedLine& result) {
    if (static_cast<int>(points.size()) < params.minPointsPerLine)
        return false;

    const int n = static_cast<int>(points.size());
    const double threshold = params.ransacThreshold;
    const double confidence = params.ransacConfidence;
    const int maxIter = params.ransacMaxIterations;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, n - 1);

    int bestInlierCount = 0;
    std::vector<int> bestInliers;

    int adaptiveMaxIter = maxIter;

    for (int iter = 0; iter < adaptiveMaxIter; ++iter) {
        int i1 = dist(rng);
        int i2 = dist(rng);
        if (i1 == i2) continue;

        Eigen::Vector3d dir = (points[i2] - points[i1]).normalized();
        if (dir.squaredNorm() < 1e-12) continue;

        Eigen::Vector3d lp = points[i1];

        std::vector<int> inliers;
        inliers.reserve(n);
        for (int j = 0; j < n; ++j) {
            double d = pointToLineDistance(points[j], lp, dir);
            if (d < threshold) {
                inliers.push_back(j);
            }
        }

        if (static_cast<int>(inliers.size()) > bestInlierCount) {
            bestInlierCount = static_cast<int>(inliers.size());
            bestInliers = std::move(inliers);

            double w = static_cast<double>(bestInlierCount) / n;
            if (w > 0.0) {
                double denom = 1.0 - std::pow(w, 2.0);
                if (denom > 1e-12) {
                    int newMax = static_cast<int>(
                        std::log(1.0 - confidence) / std::log(denom)) + 1;
                    adaptiveMaxIter = std::min(adaptiveMaxIter, std::max(newMax, 1));
                }
            }
        }
    }

    if (bestInlierCount < params.minPointsPerLine)
        return false;

    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (int idx : bestInliers) {
        centroid += points[idx];
    }
    centroid /= static_cast<double>(bestInlierCount);

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (int idx : bestInliers) {
        Eigen::Vector3d v = points[idx] - centroid;
        cov += v * v.transpose();
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Vector3d direction = solver.eigenvectors().col(2).normalized();

    double totalErr = 0.0;
    for (int idx : bestInliers) {
        double d = pointToLineDistance(points[idx], centroid, direction);
        totalErr += d * d;
    }
    double avgErr = std::sqrt(totalErr / bestInlierCount);

    result.point = centroid;
    result.direction = direction;
    result.inlierCount = bestInlierCount;
    result.fittingError = avgErr;
    return true;
}

// ============================================================================
// Impl Implementation
// ============================================================================

VirtualCameraPoseCuda::Impl::Impl(const VirtualCameraPoseParams& params)
    : params_(params)
{
    params_.validate();

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("[10-VirtualCameraPoseCuda] No CUDA devices found");
    }
    if (params_.deviceId >= deviceCount) {
        throw std::invalid_argument("[10-VirtualCameraPoseCuda] deviceId >= device count");
    }

    old_device_id_ = 0;
    cudaGetDevice(&old_device_id_);
    if (params_.deviceId != old_device_id_) {
        cudaSetDevice(params_.deviceId);
    }
}

VirtualCameraPoseCuda::Impl::~Impl() {
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        CALIB_LOG_ERROR("cudaDeviceSynchronize in destructor failed: {}",
                        cudaGetErrorString(sync_err));
    }
}

void VirtualCameraPoseCuda::Impl::Warmup(int numEndpoints, int maxLineId) {
    if (numEndpoints <= 0 || maxLineId < 0) {
        CALIB_LOG_WARN("warmup(): invalid numEndpoints={} or maxLineId={}, skipping",
                       numEndpoints, maxLineId);
        return;
    }

    d_endpoints_buf_.create(1, numEndpoints, CV_32FC3);
    d_line_ids_buf_.create(1, numEndpoints, CV_32SC1);

    warmed_up_ = true;
    warmup_count_ = numEndpoints;
    warmup_max_lid_ = maxLineId;

    CALIB_LOG_INFO("warmup(): allocated GPU buffers for numEndpoints={}, maxLineId={}",
                   numEndpoints, maxLineId);
}

void VirtualCameraPoseCuda::Impl::SetParams(const VirtualCameraPoseParams& params) {
#ifndef NDEBUG
    if (inProcess_.load()) {
        CALIB_LOG_ERROR("setParams(): called while process() is running");
        throw std::runtime_error("[10-VirtualCameraPoseCuda] setParams() called during process()");
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

VirtualCameraPoseResult VirtualCameraPoseCuda::Impl::Execute(
    const cv::cuda::GpuMat& d_endpoints,
    const cv::cuda::GpuMat& d_line_ids,
    const cv::Matx33d& stereoK,
    const cv::Matx33d& stereoR,
    cv::cuda::Stream stream)
{
#ifndef NDEBUG
    ScopedFlag guard(&inProcess_);
#endif

    VirtualCameraPoseResult result;
    result.virtualK = stereoK;
    result.virtualR = stereoR;

    try {
        const int count = d_endpoints.rows * d_endpoints.cols;

        if (count == 0) {
            result.success = true;
            result.message = "Empty input, no endpoints to process";
            return result;
        }

        cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

        // Step 1: GPU -> CPU transfer
        cv::Mat h_endpoints, h_line_ids;
        d_endpoints.download(h_endpoints, stream);
        d_line_ids.download(h_line_ids, stream);
        cudaStreamSynchronize(cuda_stream);

        h_endpoints = h_endpoints.reshape(3);
        h_line_ids = h_line_ids.reshape(1);

        result.totalEndpoints = count;

        CALIB_LOG_DEBUG("Downloaded {} endpoints to CPU", count);

        // Step 2: Group by line_id
        std::unordered_map<int, std::vector<Eigen::Vector3d>> lineGroups;
        for (int i = 0; i < count; ++i) {
            cv::Vec3f p = h_endpoints.at<cv::Vec3f>(i);
            int lid = h_line_ids.at<int>(i);
            lineGroups[lid].emplace_back(p[0], p[1], p[2]);
        }

        std::vector<std::pair<int, std::vector<Eigen::Vector3d>>> validLines;
        for (auto& [lid, pts] : lineGroups) {
            if (static_cast<int>(pts.size()) >= params_.minPointsPerLine) {
                validLines.emplace_back(lid, std::move(pts));
            } else {
                CALIB_LOG_DEBUG("Skipping line {}: only {} points (< minPointsPerLine={})",
                                lid, pts.size(), params_.minPointsPerLine);
            }
        }

        if (static_cast<int>(validLines.size()) < params_.minLinesForSolve) {
            result.success = false;
            result.message = "Insufficient valid lines: " +
                             std::to_string(validLines.size()) +
                             " < minLinesForSolve=" +
                             std::to_string(params_.minLinesForSolve);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        CALIB_LOG_DEBUG("Valid lines for fitting: {}", validLines.size());

        // Step 3: Per-line RANSAC + PCA fit
        std::vector<FittedLine> fittedLines;
        result.lineInlierCounts.resize(validLines.size());

        for (size_t i = 0; i < validLines.size(); ++i) {
            FittedLine fl;
            if (ransacLineFit(validLines[i].second, params_, fl)) {
                fittedLines.push_back(fl);
                result.lineInlierCounts[i] = fl.inlierCount;
                CALIB_LOG_DEBUG("Line {}: inliers={}, fittingError={:.4f}",
                                validLines[i].first, fl.inlierCount, fl.fittingError);
            } else {
                result.lineInlierCounts[i] = 0;
                CALIB_LOG_WARN("Line {}: RANSAC fit failed, skipping", validLines[i].first);
            }
        }

        if (static_cast<int>(fittedLines.size()) < params_.minLinesForSolve) {
            result.success = false;
            result.message = "Insufficient fitted lines: " +
                             std::to_string(fittedLines.size()) +
                             " < minLinesForSolve=" +
                             std::to_string(params_.minLinesForSolve);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        // Step 4: Solve for intersection point C
        // A * C = b where A = sum(I - d*dT), b = sum((I - d*dT) * a)
        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();

        for (const auto& fl : fittedLines) {
            Eigen::Matrix3d I_minus_ddT = Eigen::Matrix3d::Identity()
                                           - fl.direction * fl.direction.transpose();
            A += I_minus_ddT;
            b += I_minus_ddT * fl.point;
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenChecker(A);
        double minEigenvalue = eigenChecker.eigenvalues()(0);
        if (minEigenvalue < 1e-10) {
            result.success = false;
            result.message = "Solve failed: singular system (min eigenvalue=" +
                             std::to_string(minEigenvalue) + ")";
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        Eigen::Vector3d C = A.ldlt().solve(b);

        CALIB_LOG_INFO("Virtual camera center: ({:.4f}, {:.4f}, {:.4f})",
                       C.x(), C.y(), C.z());

        // Step 5: Assemble output and diagnostics
        result.virtualT = cv::Vec3d(C.x(), C.y(), C.z());
        result.numLines = static_cast<int>(fittedLines.size());

        double totalFittingError = 0.0;
        double totalDistToCenter = 0.0;
        for (const auto& fl : fittedLines) {
            totalFittingError += fl.fittingError;
            totalDistToCenter += pointToLineDistance(C, fl.point, fl.direction);
        }
        result.avgLineFittingError = totalFittingError / fittedLines.size();
        result.avgDistToCenter = totalDistToCenter / fittedLines.size();

        result.success = true;
        result.message = "Success";

        if (result.avgDistToCenter > 5.0) {
            result.qualityFlag = calib::QualityFlag::Warning;
        } else if (result.avgDistToCenter > 2.0) {
            result.qualityFlag = calib::QualityFlag::Degraded;
        }

        CALIB_LOG_INFO("avgLineFittingError={:.4f}, avgDistToCenter={:.4f}, quality={}",
                       result.avgLineFittingError, result.avgDistToCenter,
                       static_cast<int>(result.qualityFlag));

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
