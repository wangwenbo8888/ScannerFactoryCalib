#include "pose_optimize_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace calib;


CALIB_DEFINE_LOG_TAG(11, PoseOptimizeCuda);

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

struct QuadCurve {
    double coeffs[3];
    char mainAxis;
    double fittingError;
};

static QuadCurve fitQuadraticCurve(const std::vector<Eigen::Vector2d>& pixels) {
    QuadCurve curve;
    curve.coeffs[0] = curve.coeffs[1] = curve.coeffs[2] = 0.0;
    curve.mainAxis = 'u';
    curve.fittingError = 0.0;

    if (pixels.size() < 3) return curve;

    double meanU = 0, meanV = 0;
    for (const auto& p : pixels) { meanU += p.x(); meanV += p.y(); }
    meanU /= pixels.size(); meanV /= pixels.size();

    double varU = 0, varV = 0;
    for (const auto& p : pixels) {
        varU += (p.x() - meanU) * (p.x() - meanU);
        varV += (p.y() - meanV) * (p.y() - meanV);
    }

    bool useU = (varU >= varV);
    curve.mainAxis = useU ? 'u' : 'v';

    int n = static_cast<int>(pixels.size());
    Eigen::MatrixXd A(n, 3);
    Eigen::VectorXd b(n);

    for (int i = 0; i < n; ++i) {
        double indep = useU ? pixels[i].x() : pixels[i].y();
        double dep = useU ? pixels[i].y() : pixels[i].x();
        A(i, 0) = indep * indep;
        A(i, 1) = indep;
        A(i, 2) = 1.0;
        b(i) = dep;
    }

    Eigen::Vector3d coeffs = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    curve.coeffs[0] = coeffs(0);
    curve.coeffs[1] = coeffs(1);
    curve.coeffs[2] = coeffs(2);

    double totalErr = 0;
    for (int i = 0; i < n; ++i) {
        double indep = useU ? pixels[i].x() : pixels[i].y();
        double dep = useU ? pixels[i].y() : pixels[i].x();
        double pred = coeffs(0) * indep * indep + coeffs(1) * indep + coeffs(2);
        totalErr += (dep - pred) * (dep - pred);
    }
    curve.fittingError = std::sqrt(totalErr / n);

    return curve;
}

static double pointToCurveResidual(const Eigen::Vector2d& p, const QuadCurve& curve) {
    double indep = (curve.mainAxis == 'u') ? p.x() : p.y();
    double dep = (curve.mainAxis == 'u') ? p.y() : p.x();
    double pred = curve.coeffs[0] * indep * indep + curve.coeffs[1] * indep + curve.coeffs[2];
    return dep - pred;
}

struct LMOptFunctor {
    typedef double Scalar;
    enum { InputsAtCompileTime = 3 };
    enum { ValuesAtCompileTime = Eigen::Dynamic };

    const std::vector<std::vector<Eigen::Vector3f>>& lineGroups;
    const Eigen::Matrix3d& R;
    double fx, fy, cx, cy;
    int totalResiduals;

    LMOptFunctor(const std::vector<std::vector<Eigen::Vector3f>>& groups,
                 const Eigen::Matrix3d& R_, double fx_, double fy_, double cx_, double cy_)
        : lineGroups(groups), R(R_), fx(fx_), fy(fy_), cx(cx_), cy(cy_)
    {
        totalResiduals = 0;
        for (const auto& g : groups) totalResiduals += static_cast<int>(g.size());
    }

    int inputs() const { return 3; }
    int values() const { return totalResiduals; }

    int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const {
        Eigen::Vector3d T(x(0), x(1), x(2));

        int idx = 0;
        for (const auto& group : lineGroups) {
            std::vector<Eigen::Vector2d> pixels(group.size());
            for (size_t i = 0; i < group.size(); ++i) {
                Eigen::Vector3d pv = R * group[i].cast<double>() + T;
                if (std::fabs(pv.z()) < 1e-10) {
                    pixels[i] = Eigen::Vector2d(0, 0);
                } else {
                    pixels[i] = Eigen::Vector2d(fx * pv.x() / pv.z() + cx,
                                                fy * pv.y() / pv.z() + cy);
                }
            }

            QuadCurve curve = fitQuadraticCurve(pixels);

            for (size_t i = 0; i < group.size(); ++i) {
                fvec(idx++) = pointToCurveResidual(pixels[i], curve);
            }
        }
        return 0;
    }

    int df(const Eigen::VectorXd& x, Eigen::MatrixXd& fjac) const {
        const double eps = 1e-8;
        Eigen::VectorXd f0(values());
        operator()(x, f0);

        for (int j = 0; j < 3; ++j) {
            Eigen::VectorXd x_eps = x;
            x_eps(j) += eps;
            Eigen::VectorXd f1(values());
            operator()(x_eps, f1);
            fjac.col(j) = (f1 - f0) / eps;
        }
        return 0;
    }
};

PoseOptimizeCuda::Impl::Impl(const PoseOptimizeParams& params)
    : params_(params)
{
    params_.validate();

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("[11-PoseOptimizeCuda] No CUDA devices found");
    }
    if (params_.deviceId >= deviceCount) {
        throw std::invalid_argument("[11-PoseOptimizeCuda] deviceId >= device count");
    }

    old_device_id_ = 0;
    cudaGetDevice(&old_device_id_);
    if (params_.deviceId != old_device_id_) {
        cudaSetDevice(params_.deviceId);
    }
}

PoseOptimizeCuda::Impl::~Impl() {
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        CALIB_LOG_ERROR("cudaDeviceSynchronize in destructor failed: {}",
                        cudaGetErrorString(sync_err));
    }
}

void PoseOptimizeCuda::Impl::Warmup(int numPoints, int maxLineId) {
    if (numPoints <= 0 || maxLineId < 0) {
        CALIB_LOG_WARN("warmup(): invalid numPoints={} or maxLineId={}, skipping",
                       numPoints, maxLineId);
        return;
    }

    d_points_buf_.create(1, numPoints, CV_32FC3);
    d_line_ids_buf_.create(1, numPoints, CV_32SC1);

    warmed_up_ = true;
    warmup_count_ = numPoints;
    warmup_max_lid_ = maxLineId;

    CALIB_LOG_INFO("warmup(): allocated GPU buffers for numPoints={}, maxLineId={}",
                   numPoints, maxLineId);
}

void PoseOptimizeCuda::Impl::SetParams(const PoseOptimizeParams& params) {
#ifndef NDEBUG
    if (inProcess_.load()) {
        CALIB_LOG_ERROR("setParams(): called while process() is running");
        throw std::runtime_error("[11-PoseOptimizeCuda] setParams() called during process()");
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

PoseOptimizeResult PoseOptimizeCuda::Impl::Execute(
    const cv::cuda::GpuMat& d_points,
    const cv::cuda::GpuMat& d_line_ids,
    const cv::Matx33d& virtualK,
    const cv::Matx33d& virtualR,
    const cv::Vec3d& initialT,
    cv::cuda::Stream stream)
{
#ifndef NDEBUG
    ScopedFlag guard(&inProcess_);
#endif

    PoseOptimizeResult result;
    result.virtualK = virtualK;
    result.virtualR = virtualR;
    result.initialT = initialT;

    try {
        const int count = d_points.rows * d_points.cols;

        if (count == 0) {
            result.success = true;
            result.message = "Empty input, no points to process";
            return result;
        }

        cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

        cv::Mat h_points, h_line_ids;
        d_points.download(h_points, stream);
        d_line_ids.download(h_line_ids, stream);
        cudaStreamSynchronize(cuda_stream);

        h_points = h_points.reshape(3);
        h_line_ids = h_line_ids.reshape(1);

        CALIB_LOG_DEBUG("Downloaded {} points to CPU", count);

        std::unordered_map<int, std::vector<Eigen::Vector3f>> lineGroupsMap;
        for (int i = 0; i < count; ++i) {
            cv::Vec3f p = h_points.at<cv::Vec3f>(i);
            int lid = h_line_ids.at<int>(i);
            lineGroupsMap[lid].emplace_back(p[0], p[1], p[2]);
        }

        std::vector<std::pair<int, std::vector<Eigen::Vector3f>>> validPairs;
        for (auto& [lid, pts] : lineGroupsMap) {
            if (static_cast<int>(pts.size()) >= params_.minPointsPerLine) {
                validPairs.emplace_back(lid, std::move(pts));
            } else {
                CALIB_LOG_DEBUG("Skipping line {}: only {} points (< minPointsPerLine={})",
                                lid, pts.size(), params_.minPointsPerLine);
            }
        }

        if (static_cast<int>(validPairs.size()) < params_.minLinesForOptimize) {
            result.success = false;
            result.message = "Insufficient valid lines: " +
                             std::to_string(validPairs.size()) +
                             " < minLinesForOptimize=" +
                             std::to_string(params_.minLinesForOptimize);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        std::vector<int> lineIds;
        std::vector<std::vector<Eigen::Vector3f>> lineGroups;
        for (auto& [lid, pts] : validPairs) {
            lineIds.push_back(lid);
            lineGroups.push_back(std::move(pts));
        }

        Eigen::Matrix3d R;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R(i, j) = virtualR(i, j);

        double fx = virtualK(0, 0);
        double fy = virtualK(1, 1);
        double cx = virtualK(0, 2);
        double cy = virtualK(1, 2);

        Eigen::Vector3d T_init(initialT(0), initialT(1), initialT(2));

        auto computeTotalError = [&](const Eigen::Vector3d& T_cur) -> double {
            double totalErr = 0.0;
            for (const auto& group : lineGroups) {
                std::vector<Eigen::Vector2d> pixels(group.size());
                for (size_t i = 0; i < group.size(); ++i) {
                    Eigen::Vector3d pv = R * group[i].cast<double>() + T_cur;
                    if (std::fabs(pv.z()) < 1e-10) {
                        pixels[i] = Eigen::Vector2d(0, 0);
                    } else {
                        pixels[i] = Eigen::Vector2d(fx * pv.x() / pv.z() + cx,
                                                    fy * pv.y() / pv.z() + cy);
                    }
                }
                QuadCurve curve = fitQuadraticCurve(pixels);
                for (size_t i = 0; i < group.size(); ++i) {
                    double r = pointToCurveResidual(pixels[i], curve);
                    totalErr += r * r;
                }
            }
            return totalErr;
        };

        double initErr = computeTotalError(T_init);
        result.initialReprojectionError = initErr;

        CALIB_LOG_INFO("Initial reprojection error: {:.6f}", initErr);

        LMOptFunctor functor(lineGroups, R, fx, fy, cx, cy);

        Eigen::VectorXd x(3);
        x(0) = T_init(0);
        x(1) = T_init(1);
        x(2) = T_init(2);

        Eigen::LevenbergMarquardt<LMOptFunctor> lm(functor);
        lm.parameters.maxfev = params_.maxIterations;
        lm.parameters.xtol = params_.convergenceThreshold;
        lm.parameters.ftol = params_.convergenceThreshold;

        Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);

        Eigen::Vector3d T_opt(x(0), x(1), x(2));

        CALIB_LOG_INFO("LM optimization status: {}, iterations: {}, fn evaluations: {}",
                       static_cast<int>(status), lm.iter, lm.nfev);
        CALIB_LOG_INFO("Optimized T: ({:.6f}, {:.6f}, {:.6f})",
                       T_opt.x(), T_opt.y(), T_opt.z());

        double finalErr = computeTotalError(T_opt);
        result.totalReprojectionError = finalErr;

        CALIB_LOG_INFO("Final reprojection error: {:.6f}", finalErr);

        result.virtualT = cv::Vec3d(T_opt.x(), T_opt.y(), T_opt.z());
        result.numLines = static_cast<int>(lineGroups.size());

        for (size_t li = 0; li < lineGroups.size(); ++li) {
            std::vector<Eigen::Vector2d> pixels(lineGroups[li].size());
            for (size_t i = 0; i < lineGroups[li].size(); ++i) {
                Eigen::Vector3d pv = R * lineGroups[li][i].cast<double>() + T_opt;
                if (std::fabs(pv.z()) < 1e-10) {
                    pixels[i] = Eigen::Vector2d(0, 0);
                } else {
                    pixels[i] = Eigen::Vector2d(fx * pv.x() / pv.z() + cx,
                                                fy * pv.y() / pv.z() + cy);
                }
            }

            QuadCurve curve = fitQuadraticCurve(pixels);

            LaserLineCurve lc;
            lc.lineId = lineIds[li];
            lc.coeffs[0] = curve.coeffs[0];
            lc.coeffs[1] = curve.coeffs[1];
            lc.coeffs[2] = curve.coeffs[2];
            lc.mainAxis = curve.mainAxis;
            lc.fittingError = curve.fittingError;
            lc.pointCount = static_cast<int>(lineGroups[li].size());
            result.lineCurves.push_back(lc);
        }

        result.success = true;
        result.message = "Success";

        double improvementRatio = (initErr > 1e-12) ? (finalErr / initErr) : 1.0;
        if (improvementRatio > 0.9) {
            result.qualityFlag = calib::QualityFlag::Warning;
        } else if (improvementRatio > 0.5) {
            result.qualityFlag = calib::QualityFlag::Degraded;
        }

        CALIB_LOG_INFO("quality={}", static_cast<int>(result.qualityFlag));

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
