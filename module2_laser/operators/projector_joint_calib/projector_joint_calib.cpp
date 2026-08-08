#include "projector_joint_calib.h"

#include <Eigen/Dense>
#include <opencv2/core.hpp>
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

} // namespace calib
