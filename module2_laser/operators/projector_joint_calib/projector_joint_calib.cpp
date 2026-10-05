#include "projector_joint_calib.h"

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <spdlog/spdlog.h>
#include <array>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>

#if defined(BUILD_CERES)
#include <ceres/ceres.h>
#endif

namespace calib {
namespace {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

// 归一化参数：u'=(u-mu_u)/su, v'=(v-mu_v)/sv —— 消除 u²~1e6 导致的 JtJ 病态
struct NormParams { double mu_u, mu_v, su, sv; };

struct ProjSample { double up, vp, Zp; };   // 归一化坐标 (up,vp) + 投影机系深度 Zp

inline ProjSample projectNorm(const cv::Vec3d& P, const cv::Vec3d& t,
                              double f, const cv::Point2d& pp,
                              const NormParams& np) {
    const double Xp = P[0] - t[0];
    const double Yp = P[1] - t[1];
    const double Zp = P[2] - t[2];
    const double u = f * Xp / Zp + pp.x;
    const double v = f * Yp / Zp + pp.y;
    return { (u - np.mu_u) / np.su, (v - np.mu_v) / np.sv, Zp };
}

inline void evalCurve(const double C[6], double up, double vp,
                      double& F, double& Fu, double& Fv) {
    const double A = C[0], B = C[1], Cc = C[2], D = C[3], E = C[4], F0 = C[5];
    F  = A * up * up + B * up * vp + Cc * vp * vp + D * up + E * vp + F0;
    Fu = 2.0 * A * up + B * vp + D;
    Fv = B * up + 2.0 * Cc * vp + E;
}

NormParams computeNorm(const std::vector<cv::Vec3d>& pts, const cv::Vec3d& t,
                       double f, const cv::Point2d& pp) {
    const double n = static_cast<double>(pts.size());
    double mu_u = 0.0, mu_v = 0.0;
    for (const auto& P : pts) {
        const double Xp = P[0] - t[0], Yp = P[1] - t[1], Zp = P[2] - t[2];
        mu_u += f * Xp / Zp + pp.x;
        mu_v += f * Yp / Zp + pp.y;
    }
    mu_u /= n; mu_v /= n;
    double su = 0.0, sv = 0.0;
    for (const auto& P : pts) {
        const double Xp = P[0] - t[0], Yp = P[1] - t[1], Zp = P[2] - t[2];
        const double du = f * Xp / Zp + pp.x - mu_u;
        const double dv = f * Yp / Zp + pp.y - mu_v;
        su += du * du; sv += dv * dv;
    }
    su = std::sqrt(su / n); sv = std::sqrt(sv / n);
    if (su < 1e-9) su = 1.0;
    if (sv < 1e-9) sv = 1.0;
    return { mu_u, mu_v, su, sv };
}

// 加权 Sampson 残差 + L2 正则（txt §三.3 策略B）：r = [√w·F'/‖∇F'‖ (N), √λ·A', √λ·B', √λ·Cc']
// 正则压二次项，强迫优化器先调 t，避免弯曲吸收 t_z 误差
void computeResiduals(const std::vector<cv::Vec3d>& pts, const cv::Vec3d& t,
                      const double C[6], double f, const cv::Point2d& pp,
                      const NormParams& np, double lambdaReg,
                      Eigen::VectorXd& r, double& cost, double* costSOut = nullptr) {
    const int n = static_cast<int>(pts.size());
    r.resize(n + 3);
    double costS = 0.0;
    for (int i = 0; i < n; ++i) {
        const ProjSample s = projectNorm(pts[i], t, f, pp, np);
        double F, Fu, Fv;
        evalCurve(C, s.up, s.vp, F, Fu, Fv);
        double denom = std::sqrt(Fu * Fu + Fv * Fv);
        if (denom < 1e-12) denom = 1e-12;
        double w = s.Zp / f;
        if (w < 1e-6) w = 1e-6;
        const double d = F / denom;
        r(i) = std::sqrt(w) * d;
        costS += w * d * d;
    }
    const double sl = std::sqrt(lambdaReg);
    r(n)     = sl * C[0];   // A'
    r(n + 1) = sl * C[1];   // B'
    r(n + 2) = sl * C[2];   // Cc'
    cost = costS + lambdaReg * (C[0] * C[0] + C[1] * C[1] + C[2] * C[2]);
    if (costSOut) *costSOut = costS;
}

#if defined(BUILD_CERES)
// Ceres AutoDiff functor: Sampson 残差（每点1个）
struct SampsonResidual {
    const double Px, Py, Pz;       // 3D 点（左相机系）
    const double f, cx, cy;        // 焦距 + 主点
    const double mu_u, su, mu_v, sv; // 归一化参数
    SampsonResidual(double px, double py, double pz, double f_, double cx_, double cy_,
                    double muu, double su_, double muv, double sv_)
        : Px(px), Py(py), Pz(pz), f(f_), cx(cx_), cy(cy_), mu_u(muu), su(su_), mu_v(muv), sv(sv_) {}
    template <typename T>
    bool operator()(const T* const t, const T* const C, T* residual) const {
        T Xp = T(Px) - t[0];
        T Yp = T(Py) - t[1];
        T Zp = T(Pz) - t[2];
        T u = T(f) * Xp / Zp + T(cx);
        T v = T(f) * Yp / Zp + T(cy);
        T up = (u - T(mu_u)) / T(su);
        T vp = (v - T(mu_v)) / T(sv);
        T F  = C[0]*up*up + C[1]*up*vp + C[2]*vp*vp + C[3]*up + C[4]*vp + C[5];
        T Fu = T(2.0)*C[0]*up + C[1]*vp + C[3];
        T Fv = C[1]*up + T(2.0)*C[2]*vp + C[4];
        T gn = ceres::sqrt(Fu*Fu + Fv*Fv + T(1e-24));
        T d = F / gn;
        T w = Zp / T(f);
        w = (w < T(1e-6)) ? T(1e-6) : w;
        residual[0] = ceres::sqrt(w) * d;
        return true;
    }
};

// Ceres AutoDiff functor: L2 正则（压二次项 [A,B,Cc]）
struct CurveRegularizer {
    const double lambda;
    explicit CurveRegularizer(double l) : lambda(l) {}
    template <typename T>
    bool operator()(const T* const C, T* residual) const {
        T sl = ceres::sqrt(T(lambda));
        residual[0] = sl * C[0];
        residual[1] = sl * C[1];
        residual[2] = sl * C[2];
        return true;
    }
};
#endif
void denormalizeCurve(const double Cp[6], const NormParams& np, double C[6]) {
    Eigen::Matrix3d Mp;
    Mp << Cp[0],       Cp[1] / 2.0, Cp[3] / 2.0,
          Cp[1] / 2.0, Cp[2],       Cp[4] / 2.0,
          Cp[3] / 2.0, Cp[4] / 2.0, Cp[5];
    Eigen::Matrix3d Tinv;
    Tinv << 1.0 / np.su, 0.0,          -np.mu_u / np.su,
            0.0,          1.0 / np.sv,  -np.mu_v / np.sv,
            0.0,          0.0,           1.0;
    const Eigen::Matrix3d M = Tinv.transpose() * Mp * Tinv;
    C[0] = M(0, 0); C[1] = 2.0 * M(0, 1); C[2] = M(1, 1);
    C[3] = 2.0 * M(0, 2); C[4] = 2.0 * M(1, 2); C[5] = M(2, 2);
    double nrm = 0.0;
    for (int i = 0; i < 6; ++i) nrm += C[i] * C[i];
    nrm = std::sqrt(nrm);
    if (nrm < 1e-15) { for (int i = 0; i < 6; ++i) C[i] = 0.0; C[5] = 1.0; return; }
    for (int i = 0; i < 6; ++i) C[i] /= nrm;
}

// 原坐标无权 Sampson 几何残差 RMS（像素级，输出诊断用）
double sampsonRmsRaw(const std::vector<cv::Vec3d>& pts, const cv::Vec3d& t,
                     const double C[6], double f, const cv::Point2d& pp) {
    double sum = 0.0;
    int n = 0;
    const double A = C[0], B = C[1], Cc = C[2], D = C[3], E = C[4], F0 = C[5];
    for (const auto& P : pts) {
        const double Xp = P[0] - t[0], Yp = P[1] - t[1], Zp = P[2] - t[2];
        const double u = f * Xp / Zp + pp.x;
        const double v = f * Yp / Zp + pp.y;
        const double F = A * u * u + B * u * v + Cc * v * v + D * u + E * v + F0;
        const double Fu = 2.0 * A * u + B * v + D;
        const double Fv = B * u + 2.0 * Cc * v + E;
        double denom = std::sqrt(Fu * Fu + Fv * Fv);
        if (denom < 1e-12) denom = 1e-12;
        const double d = F / denom;
        sum += d * d;
        ++n;
    }
    return n > 0 ? std::sqrt(sum / n) : 0.0;
}

} // namespace


ProjectorJointCalib::ProjectorJointCalib(const ProjectorJointCalibParams& params)
    : params_(params)
{
    params_.validate();
}

void ProjectorJointCalib::SetParams(const ProjectorJointCalibParams& params) {
    params_ = params;
    params_.validate();
}

const ProjectorJointCalibParams& ProjectorJointCalib::GetParams() const {
    return params_;
}

ProjectorJointCalibResult ProjectorJointCalib::Execute(const ProjectorJointCalibInput& input) {
    ProjectorJointCalibResult result;
    result.initialT = input.initialT;

    try {
        if (input.poses.empty()) {
            result.success = true;
            result.message = "Empty input, no poses";
            return result;
        }
        if (input.f <= 0.0) {
            result.success = false;
            result.message = "Invalid focal length (must be > 0)";
            return result;
        }

        const double f = input.f;
        const cv::Point2d pp = input.principalPoint;
        const double inlierThresh = params_.planeFitInlierThresh;

        // —— Step 1: 逐姿态 SVD 平面降噪 ——
        std::vector<cv::Vec3d> cleanPts;
        int validPoses = 0;
        for (const auto& pose : input.poses) {
            if (static_cast<int>(pose.points3d.size()) < params_.minPointsPerPose) continue;
            const size_t m = pose.points3d.size();
            std::vector<Vec3> pts(m);
            for (size_t i = 0; i < m; ++i)
                pts[i] = Vec3(pose.points3d[i][0], pose.points3d[i][1], pose.points3d[i][2]);

            Vec3 c = Vec3::Zero();
            for (const auto& p : pts) c += p;
            c /= static_cast<double>(m);

            Mat3 cov = Mat3::Zero();
            for (const auto& p : pts) { const Vec3 d = p - c; cov += d * d.transpose(); }
            Eigen::SelfAdjointEigenSolver<Mat3> es(cov);
            const Vec3 nrm = es.eigenvectors().col(0);   // 最小特征值方向 = 平面法向
            const Vec3 uAxis = es.eigenvectors().col(2); // 最大特征值 = 激光线主方向
            const Vec3 vAxis = es.eigenvectors().col(1); // 中特征值 = 弯曲方向

            // Step 1 平面降噪：内点过滤 + 投影到平面
            std::vector<Vec3> planePts;
            for (const auto& p : pts) {
                const Vec3 d = p - c;
                if (std::fabs(d.dot(nrm)) > inlierThresh) continue;
                planePts.push_back(p - d.dot(nrm) * nrm);
            }

            // Step 1.5 平面曲线降噪（去横向噪声，SVD 平面降噪未覆盖）
            std::vector<Vec3> denoisedPts;
            if (planePts.size() >= 3) {
                const size_t np = planePts.size();
                Eigen::VectorXd alpha(np), beta(np);
                for (size_t i = 0; i < np; ++i) {
                    const Vec3 d = planePts[i] - c;
                    alpha(i) = d.dot(uAxis);
                    beta(i)  = d.dot(vAxis);
                }
                // α 归一化（避免 α³~1e7 导致 M 病态，bdcSvd 数值不稳）
                const double meanA = alpha.mean();
                double stdA = std::sqrt((alpha.array() - meanA).square().sum() / static_cast<double>(np));
                if (stdA < 1e-9) stdA = 1.0;
                const Eigen::VectorXd alphaN = (alpha.array() - meanA) / stdA;
                // 多项式拟合（curveDegree=2 抛物线3参数, =3 三次4参数）
                const int nCols = (params_.curveDegree <= 2) ? 3 : 4;
                Eigen::MatrixXd M(np, nCols);
                for (size_t i = 0; i < np; ++i) {
                    const double a = alphaN(i);
                    if (nCols == 3) { M(i,0)=a*a; M(i,1)=a; M(i,2)=1.0; }
                    else { M(i,0)=a*a*a; M(i,1)=a*a; M(i,2)=a; M(i,3)=1.0; }
                }
                Eigen::VectorXd coef = M.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(beta);
                // 残差过滤（1轮，3σ 剔除离群 + 重拟合）
                Eigen::VectorXd resid = beta - M * coef;
                double sigmaR = std::sqrt(resid.squaredNorm() / static_cast<double>(np));
                if (sigmaR > 1e-9) {
                    std::vector<int> inIdx;
                    for (size_t i = 0; i < np; ++i)
                        if (std::fabs(resid(i)) < 3.0 * sigmaR) inIdx.push_back(static_cast<int>(i));
                    if (inIdx.size() >= static_cast<size_t>(nCols) && inIdx.size() < np) {
                        Eigen::MatrixXd Mi(inIdx.size(), nCols);
                        Eigen::VectorXd bi(inIdx.size());
                        for (size_t j = 0; j < inIdx.size(); ++j) {
                            const double a = alphaN(inIdx[j]);
                            if (nCols == 3) { Mi(j,0)=a*a; Mi(j,1)=a; Mi(j,2)=1.0; }
                            else { Mi(j,0)=a*a*a; Mi(j,1)=a*a; Mi(j,2)=a; Mi(j,3)=1.0; }
                            bi(j) = beta(inIdx[j]);
                        }
                        coef = Mi.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(bi);
                    }
                }
                // 垂直投影降噪 + 转回 3D（仅内点）
                for (size_t i = 0; i < np; ++i) {
                    const double a3 = alphaN(i);
                    const double betaFit = (nCols == 3)
                        ? (coef(0)*a3*a3 + coef(1)*a3 + coef(2))
                        : (coef(0)*a3*a3*a3 + coef(1)*a3*a3 + coef(2)*a3 + coef(3));
                    const double r = beta(i) - betaFit;
                    if (sigmaR > 1e-9 && std::fabs(r) > 3.0 * sigmaR) continue;
                    const Vec3 pD = c + alpha(i) * uAxis + betaFit * vAxis;
                    denoisedPts.push_back(pD);
                }
            } else {
                denoisedPts = planePts;  // 点太少，降级用平面降噪点
            }

            for (const auto& p : denoisedPts)
                cleanPts.emplace_back(p.x(), p.y(), p.z());
            ++validPoses;
        }
        result.poseCount = validPoses;

        if (validPoses < params_.minPoses) {
            result.success = false;
            result.message = "Insufficient poses: " + std::to_string(validPoses)
                           + " < minPoses=" + std::to_string(params_.minPoses);
            return result;
        }
        if (cleanPts.empty()) {
            result.success = false;
            result.message = "No inlier points survived plane denoise";
            return result;
        }
        result.denoisedPoints = cleanPts;  // 诊断：Step 1.5 曲线降噪后的点
        result.totalPointCount = static_cast<int>(cleanPts.size());
        const int N = static_cast<int>(cleanPts.size());

        // —— Step 2: 坐标归一化（基于 t_init 反投影统计）——
        cv::Vec3d t = input.initialT;
        const NormParams np = computeNorm(cleanPts, t, f, pp);

        // 曲线初始化：归一化坐标过原点直线 F' = v'（‖C'‖=1，对齐 txt §三.3 步骤二）
        double C[6] = {0.0, 0.0, 0.0, 0.0, 1.0, 0.0};

        double CinitRaw[6];
        denormalizeCurve(C, np, CinitRaw);
        result.initialSampsonRms = sampsonRmsRaw(cleanPts, t, CinitRaw, f, pp);

        Eigen::MatrixXd lastJtJ = Eigen::MatrixXd::Zero(9, 9);  // 退化检测（Ceres 路径保持 0）

#if defined(BUILD_CERES)
        if (params_.useCeres) {
            // —— Ceres 后端：AutoDiff Sampson + 固定正则 + trust region ——
            double tArr[3] = {t[0], t[1], t[2]};
            double CArr[6]; for (int k = 0; k < 6; ++k) CArr[k] = C[k];
            ceres::Problem problem;
            for (const auto& P : cleanPts) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<SampsonResidual, 1, 3, 6>(
                        new SampsonResidual(P[0], P[1], P[2], f, pp.x, pp.y,
                                            np.mu_u, np.su, np.mu_v, np.sv)),
                    nullptr, tArr, CArr);
            }
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<CurveRegularizer, 3, 6>(
                    new CurveRegularizer(params_.lambda0)),
                nullptr, CArr);
            ceres::Solver::Options opts;
            opts.max_num_iterations = params_.maxIterations;
            opts.linear_solver_type = ceres::DENSE_QR;
            opts.function_tolerance = params_.convergenceThreshold;
            opts.gradient_tolerance = params_.convergenceThreshold;
            opts.minimizer_progress_to_stdout = false;
            ceres::Solver::Summary summ;
            ceres::Solve(opts, &problem, &summ);
            t[0] = tArr[0]; t[1] = tArr[1]; t[2] = tArr[2];
            for (int k = 0; k < 6; ++k) C[k] = CArr[k];
        } else
#endif
        {
        // —— Step 3: 联合 LM 优化 (t:3 + C:6 = 9 DOF) + L2 正则退火（txt §三.3 策略B）——
        // lambdaReg 压二次项 [A',B',Cc']：初期强正则强迫先调 t（避免弯曲吸收 t_z 误差），
        // 每 iter 衰减释放弯曲，最终趋近纯 Sampson 代价。
        double lambdaReg = (params_.curveDegree <= 2) ? 0.0 : params_.lambda0;  // 显式2阶无耦合，不需正则
        const double lambdaRegMin = 1e-6;   // P1a: 阈值修复后重测（之前崩因 planePts 仅 20%）
        Eigen::VectorXd r0;
        double cost0;
        computeResiduals(cleanPts, t, C, f, pp, np, lambdaReg, r0, cost0);

        std::vector<double> costHist;
        double lambda = 1e-3;
        for (int iter = 0; iter < params_.maxIterations; ++iter) {
            const int M = N + 3;
            // 数值雅可比（含 3 个正则残差行）
            Eigen::MatrixXd J(M, 9);
            for (int j = 0; j < 9; ++j) {
                const int ci = j - 3;
                if (j >= 3 && params_.curveDegree <= 2 && (ci == 1 || ci == 2)) {
                    J.col(j).setZero();   // B,Cc 固定 0（显式2阶约束）
                    continue;
                }
                cv::Vec3d tp = t;
                double Cp[6];
                for (int k = 0; k < 6; ++k) Cp[k] = C[k];
                const double eps = 1e-7;
                if (j < 3) tp[j] += eps;
                else       Cp[j - 3] += eps;
                Eigen::VectorXd r1;
                double c1;
                computeResiduals(cleanPts, tp, Cp, f, pp, np, lambdaReg, r1, c1);
                J.col(j) = (r1 - r0) / eps;
            }

            const Eigen::MatrixXd JtJ = J.transpose() * J;
            const Eigen::VectorXd Jtr = J.transpose() * r0;
            const Eigen::MatrixXd I9 = Eigen::MatrixXd::Identity(9, 9);
            lastJtJ = JtJ;   // 保存用于退化检测

            cv::Vec3d tBest = t;
            double Cbest[6];
            for (int k = 0; k < 6; ++k) Cbest[k] = C[k];
            double costBest = cost0;
            bool improved = false;

            for (int trial = 0; trial < 8; ++trial) {
                Eigen::VectorXd delta = (JtJ + lambda * I9).ldlt().solve(-Jtr);
                if (!delta.allFinite()) { lambda = std::min(lambda * 3.0, 1e8); continue; }

                cv::Vec3d tn = t;
                double Cn[6];
                for (int k = 0; k < 6; ++k) Cn[k] = C[k];
                for (int k = 0; k < 3; ++k) tn[k] += delta(k);
                for (int k = 0; k < 6; ++k) Cn[k] += delta(3 + k);

                if (params_.curveDegree <= 2) {
                    // 显式2阶：归一化 E=1（保持 v 系数=1），强制 B=Cc=0
                    if (std::fabs(Cn[4]) < 1e-12) { lambda = std::min(lambda * 3.0, 1e8); continue; }
                    const double eInv = 1.0 / Cn[4];
                    for (int k = 0; k < 6; ++k) Cn[k] *= eInv;
                    Cn[1] = 0.0; Cn[2] = 0.0;
                } else {
                    double cnrm = 0.0;
                    for (int k = 0; k < 6; ++k) cnrm += Cn[k] * Cn[k];
                    cnrm = std::sqrt(cnrm);
                    if (cnrm < 1e-12) { lambda = std::min(lambda * 3.0, 1e8); continue; }
                    for (int k = 0; k < 6; ++k) Cn[k] /= cnrm;
                }

                Eigen::VectorXd rn;
                double cn;
                computeResiduals(cleanPts, tn, Cn, f, pp, np, lambdaReg, rn, cn);

                if (cn < cost0) {
                    improved = true;
                    if (cn < costBest) {
                        costBest = cn; tBest = tn;
                        for (int k = 0; k < 6; ++k) Cbest[k] = Cn[k];
                    }
                    lambda = std::max(lambda * 0.3, 1e-12);
                    break;
                } else {
                    lambda = std::min(lambda * 3.0, 1e8);
                }
            }

            double stepNorm = 0.0;
            for (int k = 0; k < 3; ++k) { const double d = tBest[k] - t[k]; stepNorm += d * d; }
            for (int k = 0; k < 6; ++k) { const double d = Cbest[k] - C[k]; stepNorm += d * d; }
            stepNorm = std::sqrt(stepNorm);

            t = tBest;
            for (int k = 0; k < 6; ++k) C[k] = Cbest[k];
            lambdaReg = std::max(lambdaReg * params_.lambdaDecay, lambdaRegMin);
            computeResiduals(cleanPts, t, C, f, pp, np, lambdaReg, r0, cost0);
            costHist.push_back(cost0);
            if (stepNorm < params_.convergenceThreshold) break;
            if (iter >= 30) {
                const double ref = costHist[iter - 30];
                const double relDrop30 = (ref > 1e-15) ? (ref - cost0) / ref : 0.0;
                if (relDrop30 < 0.05) break;   // 30-iter 窗口 cost 平台：防 tz 末期漂移
            }
            if (!improved && lambda >= 1e8) break;
        }
        }  // end else (手写 LM)

        // —— Step 4: 输出与诊断 ——
        result.projectorT = t;
        double Craw[6];
        denormalizeCurve(C, np, Craw);
        for (int k = 0; k < 6; ++k) result.emissionCurve.coeffs[k] = Craw[k];
        result.emissionCurve.discriminant = Craw[1] * Craw[1] - 4.0 * Craw[0] * Craw[2];
        result.emissionCurve.sampsonRms = sampsonRmsRaw(cleanPts, t, Craw, f, pp);
        result.emissionCurve.pointCount = N;
        result.finalSampsonRms = result.emissionCurve.sampsonRms;
        result.improvementRatio = (result.initialSampsonRms > 1e-12)
            ? result.finalSampsonRms / result.initialSampsonRms : 1.0;

        result.success = true;
        result.message = "Success";
        if (result.improvementRatio > 0.9) result.qualityFlag = QualityFlag::Warning;
        else if (result.improvementRatio > 0.5) result.qualityFlag = QualityFlag::Degraded;

        // 姿态退化检测：算最后 iter 的 JᵀJ 条件数，过大则 t_z 不可辨识。
        // 阈值 1e10：实测正常标定 cond~1e9，正面退化姿态 cond~1e12（几何均值≈6e10）。
        // 注：原阈值 1e6 远低于正常水平，导致所有正常标定都被误报为"退化"。
        constexpr double kDegenerateCondThreshold = 1e10;
        {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(lastJtJ);
            if (es.info() == Eigen::Success) {
                const double lmax = std::max(es.eigenvalues()(8), 1e-15);
                const double lmin = std::max(es.eigenvalues()(0), 1e-15);
                result.jacobianConditionNumber = lmax / lmin;
            }
            if (result.jacobianConditionNumber > kDegenerateCondThreshold) {
                result.qualityFlag = QualityFlag::Warning;
                result.message = "Pose degraded: t_z unidentifiable (cond="
                               + std::to_string(result.jacobianConditionNumber) + ")";
            }
        }

        // 伪极小值/异常解检测：曲线拟合残差远超正常水平 → 优化可能落到错解
        // （如初值落在伪极小值陷阱：实测 σ=0.2 正常 rms≈0.045，陷阱/跑飞 rms≈0.48，约 10 倍差距）。
        // cond 查不出此类错解（陷阱 cond 反而更低），rms 是有效探测器。阈值可按噪声档调。
        if (result.finalSampsonRms > params_.anomalyRmsThreshold) {
            result.qualityFlag = QualityFlag::Warning;
            result.message = "Anomalous fit: Sampson RMS=" + std::to_string(result.finalSampsonRms)
                           + " > " + std::to_string(params_.anomalyRmsThreshold)
                           + " (possible spurious minimum)";
        }

    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Exception: ") + e.what();
    } catch (...) {
        result.success = false;
        result.message = "Unknown exception";
    }
    return result;
}

OperatorInfo getProjectorJointCalibInfo() {
    return {"ProjectorJointCalib", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

// ============================================================================
// ExecuteMultiLine — 共享 t(3) + 每线独立曲线 C_l(6) 联合 LM
//   坐标归一化与单线版同构, 但基于全部点统一计算(共享同一 NormParams);
//   正则压每线二次项; 联合雅可比列数 3+6L, 数值差分。
// ============================================================================

namespace {

// 逐姿态平面+曲线降噪（与 Execute Step1/1.5 同逻辑, 提为公共函数）
// poseThicknessOut: 每姿态内点 RMS 离面厚度 (mm)——帧精度代理 (F6 W_s=1/σ_s²)
// poseCountOut: 每姿态降噪后保留点数（与 cleanPts 追加顺序严格对齐）
std::vector<cv::Vec3d> denoisePoses(const std::vector<PosePointSet>& poses,
                                    double inlierThresh, int curveDegree,
                                    int minPointsPerPose, int& validPoses,
                                    std::vector<double>* poseThicknessOut = nullptr,
                                    std::vector<int>* poseCountOut = nullptr) {
    std::vector<cv::Vec3d> cleanPts;
    validPoses = 0;
    if (poseThicknessOut) poseThicknessOut->clear();
    if (poseCountOut) poseCountOut->clear();
    for (const auto& pose : poses) {
        if (static_cast<int>(pose.points3d.size()) < minPointsPerPose) continue;
        const size_t m = pose.points3d.size();
        std::vector<Vec3> pts(m);
        for (size_t i = 0; i < m; ++i)
            pts[i] = Vec3(pose.points3d[i][0], pose.points3d[i][1], pose.points3d[i][2]);
        Vec3 c = Vec3::Zero();
        for (const auto& p : pts) c += p;
        c /= static_cast<double>(m);
        Mat3 cov = Mat3::Zero();
        for (const auto& p : pts) { const Vec3 d = p - c; cov += d * d.transpose(); }
        Eigen::SelfAdjointEigenSolver<Mat3> es(cov);
        const Vec3 nrm = es.eigenvectors().col(0);
        const Vec3 uAxis = es.eigenvectors().col(2);
        const Vec3 vAxis = es.eigenvectors().col(1);

        std::vector<Vec3> planePts;
        double sqSum = 0.0;                                  // 帧厚度（内点离面²和）
        for (const auto& p : pts) {
            const Vec3 d = p - c;
            const double off = std::fabs(d.dot(nrm));
            if (off > inlierThresh) continue;
            sqSum += off * off;
            planePts.push_back(p - d.dot(nrm) * nrm);
        }
        if (poseThicknessOut)
            poseThicknessOut->push_back(planePts.empty() ? 0.0
                : std::sqrt(sqSum / static_cast<double>(planePts.size())));
        std::vector<Vec3> denoisedPts;
        if (planePts.size() >= 3) {
            const size_t np = planePts.size();
            Eigen::VectorXd alpha(np), beta(np);
            for (size_t i = 0; i < np; ++i) {
                const Vec3 d = planePts[i] - c;
                alpha(i) = d.dot(uAxis);
                beta(i)  = d.dot(vAxis);
            }
            const double meanA = alpha.mean();
            double stdA = std::sqrt((alpha.array() - meanA).square().sum() / static_cast<double>(np));
            if (stdA < 1e-9) stdA = 1.0;
            const Eigen::VectorXd alphaN = (alpha.array() - meanA) / stdA;
            const int nCols = (curveDegree <= 2) ? 3 : 4;
            Eigen::MatrixXd M(np, nCols);
            for (size_t i = 0; i < np; ++i) {
                const double a = alphaN(i);
                if (nCols == 3) { M(i,0)=a*a; M(i,1)=a; M(i,2)=1.0; }
                else { M(i,0)=a*a*a; M(i,1)=a*a; M(i,2)=a; M(i,3)=1.0; }
            }
            Eigen::VectorXd coef = M.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(beta);
            Eigen::VectorXd resid = beta - M * coef;
            double sigmaR = std::sqrt(resid.squaredNorm() / static_cast<double>(np));
            if (sigmaR > 1e-9) {
                std::vector<int> inIdx;
                for (size_t i = 0; i < np; ++i)
                    if (std::fabs(resid(i)) < 3.0 * sigmaR) inIdx.push_back(static_cast<int>(i));
                if (inIdx.size() >= static_cast<size_t>(nCols) && inIdx.size() < np) {
                    Eigen::MatrixXd Mi(inIdx.size(), nCols);
                    Eigen::VectorXd bi(inIdx.size());
                    for (size_t j = 0; j < inIdx.size(); ++j) {
                        const double a = alphaN(inIdx[j]);
                        if (nCols == 3) { Mi(j,0)=a*a; Mi(j,1)=a; Mi(j,2)=1.0; }
                        else { Mi(j,0)=a*a*a; Mi(j,1)=a*a; Mi(j,2)=a; Mi(j,3)=1.0; }
                        bi(j) = beta(inIdx[j]);
                    }
                    coef = Mi.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(bi);
                }
            }
            for (size_t i = 0; i < np; ++i) {
                const double a3 = alphaN(i);
                const double betaFit = (nCols == 3)
                    ? (coef(0)*a3*a3 + coef(1)*a3 + coef(2))
                    : (coef(0)*a3*a3*a3 + coef(1)*a3*a3 + coef(2)*a3 + coef(3));
                const double r = beta(i) - betaFit;
                if (sigmaR > 1e-9 && std::fabs(r) > 3.0 * sigmaR) continue;
                const Vec3 pD = c + alpha(i) * uAxis + betaFit * vAxis;
                denoisedPts.push_back(pD);
            }
        } else {
            denoisedPts = planePts;
        }
        for (const auto& p : denoisedPts)
            cleanPts.emplace_back(p.x(), p.y(), p.z());
        if (poseCountOut)
            poseCountOut->push_back(static_cast<int>(denoisedPts.size()));
        ++validPoses;
    }
    return cleanPts;
}

// 多线联合 Sampson 残差: 点归 (lineIdx, ptIdx); 参数 [t(3), C_0..C_{L-1} (每线6)]
// F6: ① 姿态权 W_s（逐点所属姿态, 帧精度 σ_s⁻²）② 鲁棒核 w_rob(|d|; Huber/Cauchy)
struct MultiLineData {
    const std::vector<std::vector<cv::Vec3d>>* linePts;  // 每线降噪点
    const std::vector<NormParams>* nps;                  // 每线独立归一化
    double f = 0.0;
    cv::Point2d pp;
    double lambdaReg = 0.0;
    // F6 新增:
    const std::vector<std::vector<double>>* poseW = nullptr; // [line][pose] 帧权（点按 pose 顺序排布）
    const std::vector<std::vector<int>>* ptsPerPose = nullptr; // [line][pose] 每姿态点数
    double huberDelta = 0.0;       // >0: Huber 核生效; 0: 纯 L2
    double cauchyC = 0.0;          // >0 且 huberDelta<=0: Cauchy 核生效
    const std::vector<double>* robustW = nullptr;   // IRLS 外部权（空=不用）
};

// 核权: Huber w=min(1,δ/|d|); Cauchy w=1/(1+(d/c)²)
inline double robustWeight(double d, double huberDelta, double cauchyC) {
    const double ad = std::fabs(d);
    if (cauchyC > 0.0) return 1.0 / (1.0 + (d / cauchyC) * (d / cauchyC));
    if (huberDelta > 0.0 && ad > huberDelta) return huberDelta / ad;
    return 1.0;
}

void computeMultiResiduals(const MultiLineData& d, const cv::Vec3d& t,
                           const std::vector<std::array<double, 6>>& Cs,
                           Eigen::VectorXd& r, double& cost) {
    const size_t L = d.linePts->size();
    size_t nTotal = 0;
    for (size_t l = 0; l < L; ++l) nTotal += (*d.linePts)[l].size();
    r.resize(static_cast<Eigen::Index>(nTotal + 3 * L));
    double costS = 0.0;
    Eigen::Index ri = 0;
    for (size_t l = 0; l < L; ++l) {
        const auto& pts = (*d.linePts)[l];
        const NormParams& np = (*d.nps)[l];
        const double C[6] = {Cs[l][0], Cs[l][1], Cs[l][2], Cs[l][3], Cs[l][4], Cs[l][5]};
        size_t ptIdx = 0;
        const size_t nPoses = d.ptsPerPose && !d.ptsPerPose->empty()
            ? (*d.ptsPerPose)[l].size() : 0;
        for (size_t pi = 0; pi < nPoses; ++pi) {
            const double Ws = (d.poseW && !d.poseW->empty())
                ? (*d.poseW)[l][pi] : 1.0;
            const int nPts = (*d.ptsPerPose)[l][pi];
            for (int k = 0; k < nPts; ++k, ++ptIdx) {
                const ProjSample s = projectNorm(pts[ptIdx], t, d.f, d.pp, np);
                double F, Fu, Fv;
                evalCurve(C, s.up, s.vp, F, Fu, Fv);
                double denom = std::sqrt(Fu * Fu + Fv * Fv);
                if (denom < 1e-12) denom = 1e-12;
                double w = s.Zp / d.f;
                if (w < 1e-6) w = 1e-6;
                const double dd = F / denom;
                double wt = Ws * w;
                if (d.robustW) wt *= (*d.robustW)[static_cast<size_t>(ri)];
                else wt *= robustWeight(dd, d.huberDelta, d.cauchyC);
                r(ri) = std::sqrt(wt) * dd;
                costS += wt * dd * dd;
                ++ri;
            }
        }
        // 无姿态分组信息时兜底（不应发生, 保防御）
        for (; ptIdx < pts.size(); ++ptIdx, ++ri) {
            const ProjSample s = projectNorm(pts[ptIdx], t, d.f, d.pp, np);
            double F, Fu, Fv;
            evalCurve(C, s.up, s.vp, F, Fu, Fv);
            double denom = std::sqrt(Fu * Fu + Fv * Fv);
            if (denom < 1e-12) denom = 1e-12;
            double w = s.Zp / d.f;
            if (w < 1e-6) w = 1e-6;
            const double dd = F / denom;
            r(ri) = std::sqrt(w) * dd;
            costS += w * dd * dd;
        }
    }
    const double sl = std::sqrt(d.lambdaReg);
    for (size_t l = 0; l < L; ++l) {
        r(ri++) = sl * Cs[l][0];
        r(ri++) = sl * Cs[l][1];
        r(ri++) = sl * Cs[l][2];
    }
    cost = costS + d.lambdaReg * [&]{
        double s = 0.0;
        for (size_t l = 0; l < L; ++l)
            s += Cs[l][0]*Cs[l][0] + Cs[l][1]*Cs[l][1] + Cs[l][2]*Cs[l][2];
        return s;
    }();
}

// 线内残差（A·线内雅可比）: 仅算第 li 线——点行段写入 r_out 的 [off, off+n_li),
// 该线 3 条正则行值经 regOut 返回; 循环体逐字取自 computeMultiResiduals 第 li 线块
// （robustW 按全局行号 ri=off+局部行 索引, 算术与全量版逐位一致）; cost_out 为该线
// 部分代价（雅可比差分不消费）。off=该线点行起始偏移（linePts 尺寸前缀和）。
void computeMultiResidualsLine(const MultiLineData& d, size_t li,
                               const cv::Vec3d& t,
                               const std::vector<std::array<double, 6>>& Cs,
                               Eigen::VectorXd& r_out, double& cost_out,
                               Eigen::Index off, double regOut[3]) {
    double costS = 0.0;
    Eigen::Index ri = off;
    const auto& pts = (*d.linePts)[li];
    const NormParams& np = (*d.nps)[li];
    const double C[6] = {Cs[li][0], Cs[li][1], Cs[li][2], Cs[li][3], Cs[li][4], Cs[li][5]};
    size_t ptIdx = 0;
    const size_t nPoses = d.ptsPerPose && !d.ptsPerPose->empty()
        ? (*d.ptsPerPose)[li].size() : 0;
    for (size_t pi = 0; pi < nPoses; ++pi) {
        const double Ws = (d.poseW && !d.poseW->empty())
            ? (*d.poseW)[li][pi] : 1.0;
        const int nPts = (*d.ptsPerPose)[li][pi];
        for (int k = 0; k < nPts; ++k, ++ptIdx) {
            const ProjSample s = projectNorm(pts[ptIdx], t, d.f, d.pp, np);
            double F, Fu, Fv;
            evalCurve(C, s.up, s.vp, F, Fu, Fv);
            double denom = std::sqrt(Fu * Fu + Fv * Fv);
            if (denom < 1e-12) denom = 1e-12;
            double w = s.Zp / d.f;
            if (w < 1e-6) w = 1e-6;
            const double dd = F / denom;
            double wt = Ws * w;
            if (d.robustW) wt *= (*d.robustW)[static_cast<size_t>(ri)];
            else wt *= robustWeight(dd, d.huberDelta, d.cauchyC);
            r_out(ri) = std::sqrt(wt) * dd;
            costS += wt * dd * dd;
            ++ri;
        }
    }
    // 无姿态分组信息时兜底（不应发生, 保防御）
    for (; ptIdx < pts.size(); ++ptIdx, ++ri) {
        const ProjSample s = projectNorm(pts[ptIdx], t, d.f, d.pp, np);
        double F, Fu, Fv;
        evalCurve(C, s.up, s.vp, F, Fu, Fv);
        double denom = std::sqrt(Fu * Fu + Fv * Fv);
        if (denom < 1e-12) denom = 1e-12;
        double w = s.Zp / d.f;
        if (w < 1e-6) w = 1e-6;
        const double dd = F / denom;
        r_out(ri) = std::sqrt(w) * dd;
        costS += w * dd * dd;
    }
    const double sl = std::sqrt(d.lambdaReg);
    regOut[0] = sl * Cs[li][0];
    regOut[1] = sl * Cs[li][1];
    regOut[2] = sl * Cs[li][2];
    cost_out = costS;
}

} // namespace

MultiLineResult ProjectorJointCalib::ExecuteMultiLine(const MultiLineInput& input) {
    MultiLineResult result;
    result.projectorT = input.initialT;

    // —— enableTiming 相位计时骨架（默认 false: 数值路径零变化, 仅多一条汇总日志）——
    const auto tNow = [] { return std::chrono::steady_clock::now(); };
    auto tMs = [&tNow](std::chrono::steady_clock::time_point tp) {
        return std::chrono::duration<double, std::milli>(tNow() - tp).count();
    };
    const auto tWall0 = tNow();
    double msDenoise = 0.0, msNorm = 0.0, msJac = 0.0, msJtJ = 0.0;
    double msTrials = 0.0, msMisc = 0.0;
    int iterCount = 0, evalCount = 0;

    try {
        if (input.lines.empty()) {
            result.success = true;
            result.message = "Empty input, no lines";
            return result;
        }
        if (input.f <= 0.0) {
            result.success = false;
            result.message = "Invalid focal length (must be > 0)";
            return result;
        }
        const double f = input.f;
        const cv::Point2d pp = input.principalPoint;
        const double inlierThresh = params_.planeFitInlierThresh;

        // —— 逐线降噪（每线内部跨姿态）—— F6: 带出逐姿态点数/σ_s（帧权源, 顺序严格对齐）
        std::vector<std::vector<cv::Vec3d>> linePts;
        std::vector<std::vector<int>> ptsPerPose;          // [line][pose] 降噪后点数
        std::vector<std::vector<double>> poseW;             // [line][pose] W_s
        std::vector<std::vector<double>> poseSigma;         // [line][pose] σ_s (诊断)
        std::vector<int> usedIdx;
        int totalGroups = 0;
        double sigmaRef = 0.0;                              // σ_s 全局中位数（归一化基准）
        const auto tDenoise0 = tNow();
        {
            std::vector<double> allSigma;
            for (size_t l = 0; l < input.lines.size(); ++l) {
                int vp = 0;
                std::vector<double> thickness;
                std::vector<int> counts;
                auto pts = denoisePoses(input.lines[l], inlierThresh,
                                        params_.curveDegree, params_.minPointsPerPose,
                                        vp, &thickness, &counts);
                if (vp < params_.minPoses || pts.empty()) continue;   // 弱线弃用
                linePts.push_back(std::move(pts));
                ptsPerPose.push_back(std::move(counts));
                poseSigma.push_back(thickness);
                poseW.emplace_back(thickness.size(), 1.0);
                usedIdx.push_back(static_cast<int>(l));
                totalGroups += vp;
                for (double s : thickness) allSigma.push_back(s);
            }
            if (!allSigma.empty()) {
                std::sort(allSigma.begin(), allSigma.end());
                sigmaRef = allSigma[allSigma.size() / 2];
            }
            // 帧权: W_s = (σ_ref/σ_s)² clamp [0.2,5] —— 平面厚(重建差)的帧降权
            if (params_.poseWeightEnabled && sigmaRef > 1e-9) {
                for (size_t l = 0; l < poseSigma.size(); ++l)
                    for (size_t pi = 0; pi < poseSigma[l].size(); ++pi) {
                        const double s = poseSigma[l][pi];
                        if (s <= 1e-9) { poseW[l][pi] = 1.0; continue; }
                        double w = (sigmaRef / s) * (sigmaRef / s);
                        poseW[l][pi] = std::max(0.2, std::min(5.0, w));
                    }
            }
        }
        msDenoise += tMs(tDenoise0);
        const size_t L = linePts.size();
        result.usedLineIdx = usedIdx;
        result.lineCount = static_cast<int>(L);
        result.poseCount = totalGroups;
        if (L == 0) {
            result.success = false;
            result.message = "No usable lines (each needs >= minPoses valid poses)";
            return result;
        }
        size_t nTotal = 0;
        for (const auto& p : linePts) nTotal += p.size();
        result.totalPointCount = static_cast<int>(nTotal);
        std::vector<Eigen::Index> lineOff(L);
        Eigen::Index offAcc = 0;
        for (size_t l = 0; l < L; ++l) {
            lineOff[l] = offAcc;
            offAcc += static_cast<Eigen::Index>(linePts[l].size());
        }

        // —— 逐线归一化（各线像素分布不同, 独立 NormParams）——
        const auto tNorm0 = tNow();
        cv::Vec3d t = input.initialT;
        std::vector<NormParams> nps;
        for (const auto& pts : linePts) nps.push_back(computeNorm(pts, t, f, pp));

        // 曲线初始化: 每线 F' = v'
        std::vector<std::array<double, 6>> Cs(L);
        for (size_t l = 0; l < L; ++l) Cs[l] = {0, 0, 0, 0, 1, 0};

        // 初始 rms（每线去归一化曲线, 合并统计）
        auto rawRmsAll = [&](const cv::Vec3d& tv,
                             const std::vector<std::array<double, 6>>& Csv) {
            double sum = 0.0; size_t n = 0;
            for (size_t l = 0; l < L; ++l) {
                double Craw[6];
                denormalizeCurve(Csv[l].data(), nps[l], Craw);
                const std::vector<cv::Vec3d>& pts = linePts[l];
                for (const auto& P : pts) {
                    const double Xp = P[0] - tv[0], Yp = P[1] - tv[1], Zp = P[2] - tv[2];
                    const double u = f * Xp / Zp + pp.x;
                    const double v = f * Yp / Zp + pp.y;
                    const double F = Craw[0]*u*u + Craw[1]*u*v + Craw[2]*v*v
                                   + Craw[3]*u + Craw[4]*v + Craw[5];
                    const double Fu = 2.0*Craw[0]*u + Craw[1]*v + Craw[3];
                    const double Fv = Craw[1]*u + 2.0*Craw[2]*v + Craw[4];
                    double denom = std::sqrt(Fu*Fu + Fv*Fv);
                    if (denom < 1e-12) denom = 1e-12;
                    const double dd = F / denom;
                    sum += dd * dd; ++n;
                }
            }
            return n > 0 ? std::sqrt(sum / n) : 0.0;
        };
        result.initialSampsonRms = rawRmsAll(t, Cs);
        msNorm += tMs(tNorm0);

        // —— 联合 LM: 参数 [t(3), C_0..C_{L-1}] —— F6: L2 粗收敛→Huber→Cauchy 退火
        MultiLineData mdata{&linePts, &nps, f, pp, 0.0};
        mdata.lambdaReg = (params_.curveDegree <= 2) ? 0.0 : params_.lambda0;
        mdata.poseW = &poseW;
        mdata.ptsPerPose = &ptsPerPose;
        const double lambdaRegMin = 1e-6;
        const int l2Iters = 10;                       // L2 粗收敛迭代数
        double huberDelta = params_.robustEnabled ? params_.huberDelta0 : 0.0;

        const int P = 3 + static_cast<int>(6 * L);
        Eigen::VectorXd r0;
        double cost0;
        computeMultiResiduals(mdata, t, Cs, r0, cost0);
        ++evalCount;

        // —— B·块稀疏 Gauss-Newton 累加（useBlockJtJ=true，主循环与 IRLS 两处共用）——
        // 按线 J_l = [J_t_l | J_c_l]（(n_l+3)×9，列序 t(3)+C_l(6)；尾 3 行为该线正则行——
        // t 列恒 0，C 列 k∈{0,1,2} 为差分 sl、k∈{3,4,5} 为 0）；
        // 串行固定序 l=0..L-1：H += J_lᵀ·J_l、g += J_lᵀ·r0_l（9×9 块散布 {t,t}/{t,c_l}/{c_l,c_l}）。
        // J 条目与稠密路径同 eps 同扰动逐位一致，仅 H/g 归约序不同；不分配 M×P 稠密 J。
        auto accumBlockHg = [&](const cv::Vec3d& tCur,
                                const std::vector<std::array<double, 6>>& CsCur,
                                const Eigen::VectorXd& r0Cur,
                                Eigen::MatrixXd& HOut, Eigen::VectorXd& gOut) {
            const double eps = 1e-7;
            HOut.setZero();
            gOut.setZero();
            const auto tTA = tNow();
            std::array<Eigen::VectorXd, 3> r1t;
            for (int j = 0; j < 3; ++j) {      // t 列：3 次全量差分（按线切片消费）
                cv::Vec3d tp = tCur;
                tp[j] += eps;
                double c1;
                computeMultiResiduals(mdata, tp, CsCur, r1t[static_cast<size_t>(j)], c1);
            }
            msJac += tMs(tTA);
            Eigen::VectorXd lineBuf(static_cast<Eigen::Index>(nTotal));
            for (size_t l = 0; l < L; ++l) {
                const Eigen::Index offL = lineOff[l];
                const Eigen::Index nL = static_cast<Eigen::Index>(linePts[l].size());
                const Eigen::Index regOff = static_cast<Eigen::Index>(nTotal) + 3 * static_cast<Eigen::Index>(l);
                const auto tTB = tNow();
                Eigen::MatrixXd Jl(nL + 3, 9);
                Jl.setZero();
                for (int j = 0; j < 3; ++j)
                    Jl.col(j).head(nL) = (r1t[static_cast<size_t>(j)].segment(offL, nL)
                                          - r0Cur.segment(offL, nL)) / eps;
                for (int ci = 0; ci < 6; ++ci) {   // C_l 列：线内差分（A·computeMultiResidualsLine）
                    std::vector<std::array<double, 6>> Cp = CsCur;
                    Cp[l][static_cast<size_t>(ci)] += eps;
                    double c1;
                    double reg1[3];
                    computeMultiResidualsLine(mdata, l, tCur, Cp, lineBuf, c1, offL, reg1);
                    Jl.col(3 + ci).head(nL) = (lineBuf.segment(offL, nL)
                                               - r0Cur.segment(offL, nL)) / eps;
                    for (int k = 0; k < 3; ++k)    // 正则行导数（同差分口径；k∈{3,4,5} 恒 0 不写）
                        Jl(nL + k, 3 + ci) = (reg1[k] - r0Cur(regOff + k)) / eps;
                }
                msJac += tMs(tTB);
                const auto tTC = tNow();
                Eigen::VectorXd r0l(nL + 3);
                r0l.head(nL) = r0Cur.segment(offL, nL);
                r0l.tail(3) = r0Cur.segment(regOff, 3);
                const int c0 = 3 + 6 * static_cast<int>(l);
                const int cols[9] = {0, 1, 2, c0, c0 + 1, c0 + 2, c0 + 3, c0 + 4, c0 + 5};
                for (int a = 0; a < 9; ++a) {
                    gOut(cols[a]) += Jl.col(a).dot(r0l);
                    for (int b = a; b < 9; ++b) {
                        const double v = Jl.col(a).dot(Jl.col(b));
                        HOut(cols[a], cols[b]) += v;
                        if (b != a) HOut(cols[b], cols[a]) += v;
                    }
                }
                msJtJ += tMs(tTC);
            }
        };

        // 稠密回退（useBlockJtJ=false）：原 M×P 雅可比逐列差分——保真分支，逐字沿用
        auto buildDenseJ = [&](const Eigen::VectorXd& r0Cur) {
            const auto tJD0 = tNow();
            const Eigen::Index M = r0Cur.size();
            Eigen::MatrixXd J(M, P);
            J.setZero();
            for (int j = 0; j < P; ++j) {
                const double eps = 1e-7;
                if (j < 3) {
                    cv::Vec3d tp = t;
                    std::vector<std::array<double, 6>> Cp = Cs;
                    tp[j] += eps;
                    Eigen::VectorXd r1;
                    double c1;
                    computeMultiResiduals(mdata, tp, Cp, r1, c1);
                    J.col(j) = (r1 - r0Cur) / eps;
                } else {
                    const int li = (j - 3) / 6, ci = (j - 3) % 6;
                    std::vector<std::array<double, 6>> Cp = Cs;
                    Cp[static_cast<size_t>(li)][static_cast<size_t>(ci)] += eps;
                    const size_t ls = static_cast<size_t>(li);
                    const Eigen::Index offL = lineOff[ls];
                    const Eigen::Index nL = static_cast<Eigen::Index>(linePts[ls].size());
                    Eigen::VectorXd r1(M);
                    double c1;
                    double reg1[3];
                    computeMultiResidualsLine(mdata, ls, t, Cp, r1, c1, offL, reg1);
                    for (Eigen::Index k = 0; k < nL; ++k)
                        J(offL + k, j) = (r1(offL + k) - r0Cur(offL + k)) / eps;
                    const Eigen::Index regOff = static_cast<Eigen::Index>(nTotal) + 3 * static_cast<Eigen::Index>(li);
                    J(regOff, j) = (reg1[0] - r0Cur(regOff)) / eps;
                    J(regOff + 1, j) = (reg1[1] - r0Cur(regOff + 1)) / eps;
                    J(regOff + 2, j) = (reg1[2] - r0Cur(regOff + 2)) / eps;
                }
            }
            msJac += tMs(tJD0);
            return J;
        };

        double lambda = 1e-3;
        double lastCond = 0.0;
        std::vector<double> costHist;
        for (int iter = 0; iter < params_.maxIterations; ++iter) {
            ++iterCount;
            const auto tMisc0 = tNow();
            // —— 核退火调度: 前 10 iter 纯 L2; 之后 Huber; 残差中位数 < cauchyToL2Thresh 后切 Cauchy ——
            if (params_.robustEnabled && iter == l2Iters) {
                mdata.huberDelta = huberDelta;
                mdata.cauchyC = 0.0;
                computeMultiResiduals(mdata, t, Cs, r0, cost0);   // 重算带核残差
                ++evalCount;
                lambda = 1e-3;                                    // 换代价函数后重置 LM 阻尼
            }
            msMisc += tMs(tMisc0);

            Eigen::MatrixXd H(P, P);
            Eigen::VectorXd g(P);
            if (params_.useBlockJtJ) {
                accumBlockHg(t, Cs, r0, H, g);   // 计时（jac/jtj）内聚于累加函数
                evalCount += P;
            } else {
                const Eigen::MatrixXd J = buildDenseJ(r0);
                evalCount += P;
                const auto tJtJ0 = tNow();
                H = J.transpose() * J;
                g = J.transpose() * r0;
                msJtJ += tMs(tJtJ0);
            }
            const Eigen::MatrixXd IP = Eigen::MatrixXd::Identity(P, P);
            {
                const auto tJtJ0 = tNow();
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
                if (es.info() == Eigen::Success) {
                    const double lmax = std::max(es.eigenvalues()(P - 1), 1e-15);
                    const double lmin = std::max(es.eigenvalues()(0), 1e-15);
                    lastCond = lmax / lmin;
                }
                msJtJ += tMs(tJtJ0);
            }

            cv::Vec3d tBest = t;
            std::vector<std::array<double, 6>> Cbest = Cs;
            double costBest = cost0;
            bool improved = false;

            const auto tTrials0 = tNow();
            for (int trial = 0; trial < 8; ++trial) {
                Eigen::VectorXd delta = (H + lambda * IP).ldlt().solve(-g);
                if (!delta.allFinite()) { lambda = std::min(lambda * 3.0, 1e8); continue; }

                cv::Vec3d tn = t;
                std::vector<std::array<double, 6>> Cn = Cs;
                for (int k = 0; k < 3; ++k) tn[k] += delta(k);
                bool badCurve = false;
                for (size_t l = 0; l < L; ++l) {
                    double cnrm = 0.0;
                    for (int k = 0; k < 6; ++k) {
                        Cn[l][static_cast<size_t>(k)] += delta(3 + static_cast<Eigen::Index>(6 * l + k));
                        cnrm += Cn[l][static_cast<size_t>(k)] * Cn[l][static_cast<size_t>(k)];
                    }
                    cnrm = std::sqrt(cnrm);
                    if (cnrm < 1e-12) { badCurve = true; break; }
                    for (int k = 0; k < 6; ++k) Cn[l][static_cast<size_t>(k)] /= cnrm;
                }
                if (badCurve) { lambda = std::min(lambda * 3.0, 1e8); continue; }

                Eigen::VectorXd rn;
                double cn;
                computeMultiResiduals(mdata, tn, Cn, rn, cn);
                ++evalCount;
                if (cn < cost0) {
                    improved = true;
                    if (cn < costBest) {
                        costBest = cn; tBest = tn; Cbest = Cn;
                    }
                    lambda = std::max(lambda * 0.3, 1e-12);
                    break;
                } else {
                    lambda = std::min(lambda * 3.0, 1e8);
                }
            }

            const auto tMisc1 = tNow();
            double stepNorm = 0.0;
            for (int k = 0; k < 3; ++k) { const double d = tBest[k] - t[k]; stepNorm += d * d; }
            for (size_t l = 0; l < L; ++l)
                for (int k = 0; k < 6; ++k) {
                    const double d = Cbest[l][static_cast<size_t>(k)] - Cs[l][static_cast<size_t>(k)];
                    stepNorm += d * d;
                }
            stepNorm = std::sqrt(stepNorm);

            t = tBest;
            Cs = Cbest;
            mdata.lambdaReg = std::max(mdata.lambdaReg * params_.lambdaDecay, lambdaRegMin);
            computeMultiResiduals(mdata, t, Cs, r0, cost0);
            ++evalCount;
            costHist.push_back(cost0);
            // Cauchy 切换: 残差中位数（归一化坐标）低于阈值后收紧核
            if (params_.robustEnabled && mdata.huberDelta > 0.0) {
                Eigen::VectorXd absR = r0.head(r0.size() - static_cast<Eigen::Index>(3 * L)).cwiseAbs();
                std::sort(absR.data(), absR.data() + absR.size());
                const double medR = absR.size() > 0 ? absR(absR.size() / 2) : 0.0;
                if (medR < params_.cauchyToL2Thresh * 0.5) {
                    mdata.huberDelta = 0.0;
                    mdata.cauchyC = std::max(0.2, medR * 2.0);
                    computeMultiResiduals(mdata, t, Cs, r0, cost0);
                    ++evalCount;
                    lambda = 1e-3;
                }
            }
            msMisc += tMs(tMisc1);
            if (stepNorm < params_.convergenceThreshold) break;
            if (iter >= 30) {
                const double ref = costHist[static_cast<size_t>(iter - 30)];
                const double relDrop30 = (ref > 1e-15) ? (ref - cost0) / ref : 0.0;
                if (relDrop30 < 0.05) break;
            }
            if (!improved && lambda >= 1e8) break;
        }

        // —— F6 IRLS: 残差驱动的权刷新再优化（初权错判的帧由数据自证修正）——
        for (int round = 0; round < params_.irlsMaxRounds; ++round) {
            const auto tMisc2 = tNow();
            // 用当前残差生成 Cauchy 型外部权
            const size_t nTotalR = static_cast<size_t>(r0.size()) - 3 * L;
            std::vector<double> wNew(nTotalR, 1.0);
            {
                std::vector<double> absR(nTotalR);
                for (size_t i = 0; i < nTotalR; ++i) absR[i] = std::fabs(r0(static_cast<Eigen::Index>(i)));
                std::vector<double> sorted = absR;
                std::sort(sorted.begin(), sorted.end());
                const double med = sorted.empty() ? 0.0 : sorted[sorted.size() / 2];
                const double c = std::max(0.1, med);
                for (size_t i = 0; i < nTotalR; ++i)
                    wNew[i] = 1.0 / (1.0 + (absR[i] / c) * (absR[i] / c));
            }
            // 权变化检查（<5% 停）
            double maxChange = 0.0;
            if (mdata.robustW) {
                for (size_t i = 0; i < nTotalR; ++i)
                    maxChange = std::max(maxChange,
                        std::fabs(wNew[i] - (*mdata.robustW)[i]));
            }
            std::vector<double> wPrev = mdata.robustW ? *mdata.robustW : std::vector<double>();
            mdata.robustW = &wNew;
            computeMultiResiduals(mdata, t, Cs, r0, cost0);
            ++evalCount;
            msMisc += tMs(tMisc2);
            // 短程再优化（30 iter 上限）
            double lambdaI = 1e-3;
            for (int iter = 0; iter < 30; ++iter) {
                ++iterCount;
                Eigen::MatrixXd H(P, P);
                Eigen::VectorXd g(P);
                if (params_.useBlockJtJ) {
                    accumBlockHg(t, Cs, r0, H, g);   // 计时（jac/jtj）内聚于累加函数
                    evalCount += P;
                } else {
                    const Eigen::MatrixXd J = buildDenseJ(r0);
                    evalCount += P;
                    const auto tJtJ0 = tNow();
                    H = J.transpose() * J;
                    g = J.transpose() * r0;
                    msJtJ += tMs(tJtJ0);
                }
                const Eigen::MatrixXd IP = Eigen::MatrixXd::Identity(P, P);
                bool improvedI = false;
                const auto tTrials0 = tNow();
                for (int trial = 0; trial < 8; ++trial) {
                    Eigen::VectorXd delta = (H + lambdaI * IP).ldlt().solve(-g);
                    if (!delta.allFinite()) { lambdaI = std::min(lambdaI * 3.0, 1e8); continue; }
                    cv::Vec3d tn = t;
                    std::vector<std::array<double, 6>> Cn = Cs;
                    for (int k = 0; k < 3; ++k) tn[k] += delta(k);
                    bool badCurve = false;
                    for (size_t l = 0; l < L; ++l) {
                        double cnrm = 0.0;
                        for (int k = 0; k < 6; ++k) {
                            Cn[l][static_cast<size_t>(k)] += delta(3 + static_cast<Eigen::Index>(6 * l + k));
                            cnrm += Cn[l][static_cast<size_t>(k)] * Cn[l][static_cast<size_t>(k)];
                        }
                        cnrm = std::sqrt(cnrm);
                        if (cnrm < 1e-12) { badCurve = true; break; }
                        for (int k = 0; k < 6; ++k) Cn[l][static_cast<size_t>(k)] /= cnrm;
                    }
                    if (badCurve) { lambdaI = std::min(lambdaI * 3.0, 1e8); continue; }
                    Eigen::VectorXd rn;
                    double cn;
                    computeMultiResiduals(mdata, tn, Cn, rn, cn);
                    ++evalCount;
                    if (cn < cost0) {
                        t = tn; Cs = Cn; cost0 = cn; r0 = rn;
                        lambdaI = std::max(lambdaI * 0.3, 1e-12);
                        improvedI = true;
                        break;
                    }
                    lambdaI = std::min(lambdaI * 3.0, 1e8);
                }
                msTrials += tMs(tTrials0);
                if (!improvedI || lambdaI >= 1e8) break;
            }
            ++result.irlsRounds;
            mdata.robustW = nullptr;   // 本轮权消耗完, 下轮由新残差再生成
            (void)maxChange; (void)wPrev;
            if (maxChange < 0.05 && !wPrev.empty()) break;
        }

        // —— F6 诊断: perLine（rms/p95/离群率/帧权均值）+ perPose（带符号偏差）——
        {
            result.perLine.resize(L);
            result.perPose.clear();
            for (size_t l = 0; l < L; ++l) {
                const auto& pts = linePts[l];
                const NormParams& np = nps[l];
                const double C[6] = {Cs[l][0], Cs[l][1], Cs[l][2], Cs[l][3], Cs[l][4], Cs[l][5]};
                std::vector<double> res;
                res.reserve(pts.size());
                size_t ptIdx = 0;
                for (size_t pi = 0; pi < ptsPerPose[l].size(); ++pi) {
                    double sumSigned = 0.0, sumSq = 0.0;
                    const int nP = ptsPerPose[l][pi];
                    for (int k = 0; k < nP; ++k, ++ptIdx) {
                        const ProjSample s = projectNorm(pts[ptIdx], t, f, pp, np);
                        double F, Fu, Fv;
                        evalCurve(C, s.up, s.vp, F, Fu, Fv);
                        double denom = std::sqrt(Fu * Fu + Fv * Fv);
                        if (denom < 1e-12) denom = 1e-12;
                        const double dd = F / denom;
                        res.push_back(dd);
                        sumSigned += dd;
                        sumSq += dd * dd;
                    }
                    MultiLineResult::PoseDiag pd;
                    pd.poseIdx = static_cast<int>(pi);
                    pd.meanResidual = nP > 0 ? sumSigned / nP : 0.0;
                    pd.rms = nP > 0 ? std::sqrt(sumSq / nP) : 0.0;
                    result.perPose.push_back(pd);
                }
                auto& ld = result.perLine[l];
                ld.lineIdx = usedIdx[l];
                ld.pointCount = static_cast<int>(res.size());
                double wSum = 0.0;
                for (double w : poseW[l]) wSum += w;
                ld.meanPoseWeight = poseW[l].empty() ? 1.0 : wSum / poseW[l].size();
                if (!res.empty()) {
                    double sq = 0.0;
                    int outliers = 0;
                    for (double v : res) { sq += v * v; if (std::fabs(v) > 2.0) ++outliers; }
                    ld.rms = std::sqrt(sq / res.size());
                    std::sort(res.begin(), res.end());
                    // p95 by absolute value
                    std::vector<double> absSorted;
                    absSorted.reserve(res.size());
                    for (double v : res) absSorted.push_back(std::fabs(v));
                    std::sort(absSorted.begin(), absSorted.end());
                    ld.p95 = absSorted[absSorted.size() * 95 / 100];
                    ld.outlierRatio = static_cast<double>(outliers) / res.size();
                }
            }
            result.huberDelta = mdata.huberDelta;
        }

        // —— 输出 ——
        result.projectorT = t;
        result.emissionCurves.resize(L);
        for (size_t l = 0; l < L; ++l) {
            double Craw[6];
            denormalizeCurve(Cs[l].data(), nps[l], Craw);
            for (int k = 0; k < 6; ++k) result.emissionCurves[l].coeffs[k] = Craw[k];
            result.emissionCurves[l].discriminant = Craw[1]*Craw[1] - 4.0*Craw[0]*Craw[2];
            result.emissionCurves[l].pointCount = static_cast<int>(linePts[l].size());
        }
        result.finalSampsonRms = rawRmsAll(t, Cs);
        result.improvementRatio = (result.initialSampsonRms > 1e-12)
            ? result.finalSampsonRms / result.initialSampsonRms : 1.0;
        result.jacobianConditionNumber = lastCond;

        result.success = true;
        result.message = "Success (multi-line joint)";
        if (result.improvementRatio > 0.9) result.qualityFlag = QualityFlag::Warning;
        else if (result.improvementRatio > 0.5) result.qualityFlag = QualityFlag::Degraded;
        constexpr double kDegenerateCondThreshold = 1e10;
        if (result.jacobianConditionNumber > kDegenerateCondThreshold) {
            result.qualityFlag = QualityFlag::Warning;
            result.message = "Pose degraded: t_z unidentifiable (cond="
                           + std::to_string(result.jacobianConditionNumber) + ")";
        }
        if (result.finalSampsonRms > params_.anomalyRmsThreshold) {
            result.qualityFlag = QualityFlag::Warning;
            result.message = "Anomalous fit: Sampson RMS=" + std::to_string(result.finalSampsonRms)
                           + " > " + std::to_string(params_.anomalyRmsThreshold)
                           + " (possible spurious minimum)";
        }

    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Exception: ") + e.what();
    } catch (...) {
        result.success = false;
        result.message = "Unknown exception";
    }
    if (params_.enableTiming) {
        spdlog::info("[PJC-timing] wall={:.1f}ms denoise={:.1f}ms norm={:.1f}ms "
                     "iters={} evals={} | per-iter: jac={:.1f}ms jtj={:.1f}ms "
                     "trials={:.1f}ms misc={:.1f}ms (均为总累计, jac/iter={:.1f}ms)",
                     tMs(tWall0), msDenoise, msNorm, iterCount, evalCount,
                     msJac, msJtJ, msTrials, msMisc,
                     iterCount > 0 ? msJac / static_cast<double>(iterCount) : 0.0);
    }
    return result;
}

} // namespace calib
