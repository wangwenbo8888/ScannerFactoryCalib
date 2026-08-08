#define _USE_MATH_DEFINES
#include <cmath>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <fstream>
#include "projector_joint_calib.h"

using namespace calib;


namespace {

struct SyntheticScene {
    ProjectorJointCalibInput input;
    cv::Vec3d t_true;
    double curveSag;
};

// 合成场景生成器：
//  - 投影机光心（左相机系）= t_true
//  - CMOS 固定发射曲线：v = cy + sag*((u-cx)/halfSpan)^2  （抛物线，弧矢高 sag）
//  - 多姿态：每姿态一块平板（法向 n_k、距光心深度 depth_k），射线-平面求交生成 3D 点
//  - 反投影回 CMOS 应聚拢到原曲线 → round-trip 可验证 t 与曲线恢复
SyntheticScene makeScene(int numPoses, int pointsPerPose, double noiseSigma, uint32_t seed, bool degraded = false, bool truncated = false) {
    SyntheticScene sc;
    // 真实硬件参数：激光光心距左相机 X~80mm（侧方），Y/Z 各偏离~3mm
    sc.t_true = cv::Vec3d(80.0, 3.0, 3.0);
    sc.input.f = 1500.0;                          // 300mm 景深视野~400mm（2048px 传感器反推）
    sc.input.principalPoint = cv::Point2d(1024.0, 768.0);
    const double halfSpan = 800.0;                           // 激光线占大部分视野（2048px 传感器，±800px≈78%）
    sc.curveSag = 30.0;                           // CMOS 曲线弧矢高 30px（~6mm 物理弯曲@300mm）

    std::mt19937 rng(seed);
    const bool addNoise = noiseSigma > 0.0;
    std::normal_distribution<double> noiseDist(0.0, addNoise ? noiseSigma : 1.0);
    auto noise = [&]() -> double { return addNoise ? noiseDist(rng) : 0.0; };

    struct PoseSpec { double nx, ny, nz, depth; };
    std::vector<PoseSpec> specs;
    if (degraded) {
        // 退化：全正面 + 深度接近（ρ≈0，t_z 不可辨识）
        specs = {
            {0.0, 0.0, 1.0, 300.0}, {0.0, 0.0, 1.0, 298.0},
            {0.0, 0.0, 1.0, 302.0}, {0.0, 0.0, 1.0, 299.0},
            {0.0, 0.0, 1.0, 301.0}, {0.0, 0.0, 1.0, 297.0},
            {0.0, 0.0, 1.0, 303.0}, {0.0, 0.0, 1.0, 296.0},
        };
    } else {
        // 25 种姿态：景深 200-500mm + 多方向面外倾斜（前8种覆盖全深度跨度，保证不同 N 下 ρ 一致）
        specs = {
            {0.00, 0.00, 1.00, 200.0}, {0.00, 0.00, 1.00, 500.0},
            {0.15, 0.00, 0.989, 300.0}, {0.00, 0.20, 0.980, 350.0},
            {-0.18, 0.10, 0.978, 250.0}, {0.12, -0.15, 0.981, 450.0},
            {-0.10, 0.08, 0.992, 280.0}, {0.20, 0.12, 0.973, 400.0},
            {0.10, 0.10, 0.990, 320.0}, {-0.15, -0.05, 0.987, 380.0},
            {0.05, 0.18, 0.982, 430.0}, {-0.08, 0.15, 0.984, 270.0},
            {0.18, -0.08, 0.980, 470.0}, {-0.12, 0.12, 0.985, 230.0},
            {0.08, -0.18, 0.981, 340.0}, {0.15, 0.05, 0.988, 410.0},
            {-0.05, -0.15, 0.987, 290.0}, {0.20, 0.05, 0.979, 360.0},
            {-0.10, -0.10, 0.990, 440.0}, {0.12, 0.15, 0.981, 260.0},
            {-0.18, 0.08, 0.980, 480.0}, {0.05, -0.12, 0.989, 330.0},
            {-0.15, 0.15, 0.979, 370.0}, {0.10, 0.20, 0.975, 420.0},
            {-0.08, -0.08, 0.994, 310.0},
        };
    }

    const double cx = sc.input.principalPoint.x;
    const double cy = sc.input.principalPoint.y;
    const double f  = sc.input.f;

    for (int k = 0; k < numPoses; ++k) {
        const PoseSpec& sp = specs[k % specs.size()];
        Eigen::Vector3d n(sp.nx, sp.ny, sp.nz);
        n.normalize();
        const double depth = sp.depth;

        PosePointSet pset;
        // 截断：每姿态两端随机去掉 5%~20%（各姿态不同）
        int iStart = 0, iEnd = pointsPerPose;
        if (truncated) {
            std::uniform_real_distribution<double> frac(0.05, 0.20);
            iStart = static_cast<int>(frac(rng) * pointsPerPose);
            iEnd   = pointsPerPose - static_cast<int>(frac(rng) * pointsPerPose);
        }
        for (int i = iStart; i < iEnd; ++i) {
            const double u = cx - halfSpan + (2.0 * halfSpan) * i / (pointsPerPose - 1);
            const double v = cy + sc.curveSag * std::pow((u - cx) / halfSpan, 2.0);

            Eigen::Vector3d r(u - cx, v - cy, f);
            r.normalize();

            const double ndotr = n.dot(r);
            if (ndotr < 1e-6) continue;
            const double s = depth / ndotr;

            const Eigen::Vector3d P(sc.t_true[0] + s * r.x() + noise(),
                                    sc.t_true[1] + s * r.y() + noise(),
                                    sc.t_true[2] + s * r.z() + noise());
            pset.points3d.push_back(cv::Vec3f(static_cast<float>(P.x()),
                                              static_cast<float>(P.y()),
                                              static_cast<float>(P.z())));
            pset.lineIds.push_back(0);
        }
        if (static_cast<int>(pset.points3d.size()) >= 10)
            sc.input.poses.push_back(std::move(pset));
    }

    sc.input.initialT = sc.t_true + cv::Vec3d(5.0, -3.0, 8.0);
    return sc;
}

} // namespace


class ProjectorJointCalibTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_.maxIterations = 200;
        params_.convergenceThreshold = 1e-9;
        params_.minPoses = 5;
        params_.minPointsPerPose = 30;
        op_.reset(new ProjectorJointCalib(params_));
    }
    void TearDown() override { op_.reset(); }

    ProjectorJointCalibResult run(const ProjectorJointCalibInput& in) {
        return op_->Execute(in);
    }

    ProjectorJointCalibParams params_;
    std::unique_ptr<ProjectorJointCalib> op_;
};


TEST_F(ProjectorJointCalibTest, ParamsValidation) {
    ProjectorJointCalibParams p;

    p.convergenceThreshold = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.convergenceThreshold = 1e-8;
    p.maxIterations = 0;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.maxIterations = 100;
    p.minPoses = 2;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.minPoses = 5;
    p.minPointsPerPose = 5;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.minPointsPerPose = 30;
    p.planeFitInlierThresh = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.planeFitInlierThresh = 0.05;
    EXPECT_NO_THROW(p.validate());
}

TEST_F(ProjectorJointCalibTest, ParamsJson) {
    ProjectorJointCalibParams p;
    p.maxIterations = 150;
    p.convergenceThreshold = 1e-7;
    p.minPoses = 6;
    p.minPointsPerPose = 40;
    p.planeFitInlierThresh = 0.08;

    auto j = p.toJson();
    auto p2 = ProjectorJointCalibParams::fromJson(j);

    EXPECT_EQ(p2.maxIterations, 150);
    EXPECT_DOUBLE_EQ(p2.convergenceThreshold, 1e-7);
    EXPECT_EQ(p2.minPoses, 6);
    EXPECT_EQ(p2.minPointsPerPose, 40);
    EXPECT_DOUBLE_EQ(p2.planeFitInlierThresh, 0.08);
}

TEST_F(ProjectorJointCalibTest, EmptyInput) {
    ProjectorJointCalibInput in;
    in.f = 2000.0;
    in.principalPoint = cv::Point2d(1024.0, 768.0);
    auto r = run(in);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.poseCount, 0);
}

TEST_F(ProjectorJointCalibTest, InsufficientPoses) {
    auto sc = makeScene(2, 51, 0.0, 1);  // 仅 2 姿态 < minPoses=5
    auto r = run(sc.input);
    EXPECT_FALSE(r.success);
}

TEST_F(ProjectorJointCalibTest, SyntheticExact) {
    auto sc = makeScene(8, 1500, 0.0, 42);  // 8 姿态，200点/姿态，无噪声（真实硬件参数）
    auto r = run(sc.input);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_NEAR(r.projectorT[0], sc.t_true[0], 1.0);  // tx < 1mm
    EXPECT_NEAR(r.projectorT[1], sc.t_true[1], 1.0);  // ty < 1mm
    // tz 阈值 5mm：固定 lambdaRegMin=0.005 在无噪声场景的正则副作用（~4.5mm 偏差）；
    // 二期自适应正则下限（∝残差）或碗状拓扑惩罚可收紧到 <1mm。
    EXPECT_NEAR(r.projectorT[2], sc.t_true[2], 5.0);
    EXPECT_LT(r.finalSampsonRms, r.initialSampsonRms);
    EXPECT_LE(r.emissionCurve.discriminant, 1.0);      // 碗状（≤0 容差）
    EXPECT_GT(r.totalPointCount, 0);
}

TEST_F(ProjectorJointCalibTest, SyntheticWithNoise) {
    auto sc = makeScene(8, 1500, 0.2, 7);  // σ=0.2mm 真实双目重建噪声
    auto r = run(sc.input);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_NEAR(r.projectorT[0], sc.t_true[0], 3.0);
    EXPECT_NEAR(r.projectorT[1], sc.t_true[1], 3.0);
    EXPECT_NEAR(r.projectorT[2], sc.t_true[2], 8.0);
    EXPECT_LT(r.finalSampsonRms, r.initialSampsonRms);
}

TEST_F(ProjectorJointCalibTest, MultiplePosesDecoupling) {
    auto sc = makeScene(10, 1500, 0.15, 99);  // 10 姿态，σ=0.15（真实多姿态）
    auto r = run(sc.input);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_NEAR(r.projectorT[0], sc.t_true[0], 1.5);
    EXPECT_NEAR(r.projectorT[1], sc.t_true[1], 1.5);
    EXPECT_NEAR(r.projectorT[2], sc.t_true[2], 5.0);
    EXPECT_LT(r.finalSampsonRms, r.initialSampsonRms);
    EXPECT_LE(r.emissionCurve.discriminant, 1.0);
}

TEST_F(ProjectorJointCalibTest, DegradedPosesWarning) {
    // 退化姿态：全正面、深度接近（ρ≈0），t_z 不可辨识
    auto sc = makeScene(8, 1500, 0.0, 42, true);
    auto r = run(sc.input);
    ASSERT_TRUE(r.success) << r.message;              // 算法仍运行（不崩）
    EXPECT_GT(r.jacobianConditionNumber, 1e6);        // 条件数大（退化标志）
    EXPECT_EQ(r.qualityFlag, QualityFlag::Warning);   // 告警 t_z 不可信
}

// 导出原始（曲线降噪前）+ 拟合后（曲线降噪后）两组 ASC，供对比降噪效果
TEST_F(ProjectorJointCalibTest, DumpSampleData) {
    auto sc = makeScene(8, 1500, 0.2, 42);   // 8 姿态，1500点/姿态，σ=0.2 真实噪声
    auto r = run(sc.input);                   // 执行算子（含 Step 1.5 曲线降噪）
    ASSERT_TRUE(r.success) << r.message;

    // 原始数据（曲线降噪前，带横向噪声）
    const std::string rawAsc = "projector_calib_raw.asc";
    std::ofstream rawOfs(rawAsc);
    int rawCnt = 0;
    for (size_t p = 0; p < sc.input.poses.size(); ++p) {
        for (size_t i = 0; i < sc.input.poses[p].points3d.size(); ++i) {
            const auto& pt = sc.input.poses[p].points3d[i];
            rawOfs << pt[0] << " " << pt[1] << " " << pt[2] << " " << p << "\n";
            ++rawCnt;
        }
    }
    rawOfs.close();

    // 曲线降噪后数据（Step 1.5 后，横向噪声去除 + 离群剔除）
    const std::string denAsc = "projector_calib_denoised.asc";
    std::ofstream denOfs(denAsc);
    for (size_t i = 0; i < r.denoisedPoints.size(); ++i) {
        const auto& pt = r.denoisedPoints[i];
        denOfs << pt[0] << " " << pt[1] << " " << pt[2] << " 0\n";
    }
    denOfs.close();

    std::cerr << "[Dump] raw:      " << rawCnt << " points -> " << rawAsc << " (X Y Z pose_id)\n";
    std::cerr << "[Dump] denoised: " << r.denoisedPoints.size() << " points -> " << denAsc << " (X Y Z 0)\n";
    std::cerr << "[Dump] 去除(离群+降噪): " << (rawCnt - static_cast<int>(r.denoisedPoints.size())) << " points\n";
    SUCCEED() << "Dumped raw + denoised";
}

// 控制变量：同 σ=0.2/seed=7，变姿态数，验证 tz_err 是否 ∝ 1/N
TEST_F(ProjectorJointCalibTest, SigmaCompare) {
    for (int nPoses : {8, 12, 16, 20, 25}) {
        auto sc = makeScene(nPoses, 1500, 0.2, 7);  // 固定 σ=0.2，seed=7，变姿态数
        auto r = run(sc.input);
        if (r.success) {
            const double tzErr = std::fabs(r.projectorT[2] - sc.t_true[2]);
            std::cerr << "[poses] nPoses=" << nPoses << " tz_err=" << tzErr << " mm\n";
        }
    }
    SUCCEED() << "Poses compare done";
}

// curveDegree 选项对比：3阶/隐式6参数 vs 2阶/显式（Step1.5 抛物线 + Step4 B=Cc=0）
TEST_F(ProjectorJointCalibTest, CurveDegreeCompare) {
    for (int deg : {3, 2}) {
        params_.curveDegree = deg;
        op_.reset(new ProjectorJointCalib(params_));
        auto sc = makeScene(8, 1500, 0.2, 7);
        auto r = run(sc.input);
        if (r.success) {
            const double txErr = std::fabs(r.projectorT[0] - sc.t_true[0]);
            const double tzErr = std::fabs(r.projectorT[2] - sc.t_true[2]);
            std::cerr << "[deg] curveDegree=" << deg << " tx_err=" << txErr << " tz_err=" << tzErr << " mm\n";
        }
    }
    SUCCEED() << "Curve degree compare done";
}

// 后端对比：手写 LM vs Ceres（需 BUILD_CERES）
TEST_F(ProjectorJointCalibTest, BackendCompare) {
    for (int useCeres : {0, 1}) {
        params_.useCeres = (useCeres != 0);
        op_.reset(new ProjectorJointCalib(params_));
        auto sc = makeScene(8, 1500, 0.2, 7);
        auto r = run(sc.input);
        if (r.success) {
            const double txErr = std::fabs(r.projectorT[0] - sc.t_true[0]);
            const double tzErr = std::fabs(r.projectorT[2] - sc.t_true[2]);
            std::cerr << "[backend] useCeres=" << useCeres
                      << " tx_err=" << txErr << " tz_err=" << tzErr << " mm\n";
        } else {
            std::cerr << "[backend] useCeres=" << useCeres << " FAILED: " << r.message << "\n";
        }
    }
    SUCCEED() << "Backend compare done";
}

// 截断激光线测试：两端各缺 5%~20%（每姿态不同），验证算法有效性 + 导出 ASC
TEST_F(ProjectorJointCalibTest, TruncatedLines) {
    // 多样本截断测试（5 seed），确认 tz 偏差分布
    std::cerr << "[trunc] === 多样本截断 vs 完整 ===\n";
    double tzSumFull = 0, tzSumTrunc = 0;
    int nValid = 0;
    for (uint32_t seed : {7u, 42u, 99u, 123u, 321u}) {
        params_.useCeres = false;
        params_.curveDegree = 3;
        op_.reset(new ProjectorJointCalib(params_));

        // 完整
        auto scF = makeScene(8, 1500, 0.2, seed, false, false);
        auto rF = run(scF.input);
        // 截断
        auto scT = makeScene(8, 1500, 0.2, seed, false, true);
        auto rT = run(scT.input);

        if (rF.success && rT.success) {
            double tzF = std::fabs(rF.projectorT[2] - scF.t_true[2]);
            double tzT = std::fabs(rT.projectorT[2] - scT.t_true[2]);
            double txF = std::fabs(rF.projectorT[0] - scF.t_true[0]);
            double txT = std::fabs(rT.projectorT[0] - scT.t_true[0]);
            tzSumFull += tzF; tzSumTrunc += tzT; ++nValid;
            std::cerr << "[trunc] seed=" << seed
                      << " full:  tx=" << txF << " tz=" << tzF
                      << " | trunc: tx=" << txT << " tz=" << tzT << "\n";
        }
    }
    if (nValid > 0) {
        std::cerr << "[trunc] avg full_tz=" << tzSumFull/nValid
                  << " avg trunc_tz=" << tzSumTrunc/nValid
                  << " ratio=" << tzSumTrunc/tzSumFull << "\n";
    }

    // 导出截断 ASC（seed=42）
    auto sc = makeScene(8, 1500, 0.2, 42u, false, true);
    auto r = run(sc.input);
    std::ofstream rawOfs("projector_calib_truncated_raw.asc");
    int rawCnt = 0;
    for (size_t p = 0; p < sc.input.poses.size(); ++p)
        for (size_t i = 0; i < sc.input.poses[p].points3d.size(); ++i) {
            const auto& pt = sc.input.poses[p].points3d[i];
            rawOfs << pt[0] << " " << pt[1] << " " << pt[2] << " " << p << "\n";
            ++rawCnt;
        }
    rawOfs.close();
    std::ofstream denOfs("projector_calib_truncated_denoised.asc");
    for (size_t i = 0; i < r.denoisedPoints.size(); ++i) {
        const auto& pt = r.denoisedPoints[i];
        denOfs << pt[0] << " " << pt[1] << " " << pt[2] << " 0\n";
    }
    denOfs.close();
    std::cerr << "[trunc] raw=" << rawCnt << " denoised=" << r.denoisedPoints.size() << "\n";
    SUCCEED() << "Truncated test done";
}

// 带宽法实验：固定 tx/ty（LM 结果），网格搜索 tz 最小带宽
TEST_F(ProjectorJointCalibTest, BandwidthTzSearch) {
    // 带宽计算函数：给定 T，反投影所有点到 CMOS，按 u 分 bin 统计 v 的标准差
    auto computeBandwidth = [](const std::vector<cv::Vec3d>& pts,
                               double tx, double ty, double tz,
                               double f, double cx, double cy) -> double {
        if (pts.empty()) return 1e9;
        // 反投影
        std::vector<std::pair<double,double>> uv;
        uv.reserve(pts.size());
        for (const auto& P : pts) {
            const double Zp = P[2] - tz;
            if (std::fabs(Zp) < 1e-6) continue;
            double u = f * (P[0] - tx) / Zp + cx;
            double v = f * (P[1] - ty) / Zp + cy;
            uv.emplace_back(u, v);
        }
        if (uv.size() < 50) return 1e9;
        // 按 u 排序
        std::sort(uv.begin(), uv.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        // 分 50 个 bin，每个 bin 算 v 的标准差
        const int nBins = 50;
        const double uMin = uv.front().first;
        const double uMax = uv.back().first;
        const double binW = (uMax - uMin) / nBins;
        if (binW < 1e-6) return 1e9;
        double totalSpread = 0;
        int validBins = 0;
        size_t idx = 0;
        for (int b = 0; b < nBins; ++b) {
            double binStart = uMin + b * binW;
            double binEnd = binStart + binW;
            std::vector<double> vs;
            while (idx < uv.size() && uv[idx].first < binEnd) {
                if (uv[idx].first >= binStart) vs.push_back(uv[idx].second);
                ++idx;
            }
            if (vs.size() >= 5) {
                double mean = 0; for (double v : vs) mean += v; mean /= vs.size();
                double ss = 0; for (double v : vs) ss += (v - mean) * (v - mean);
                totalSpread += std::sqrt(ss / vs.size());
                ++validBins;
            }
        }
        return (validBins > 0) ? totalSpread / validBins : 1e9;
    };

    // 5 seed 测试
    std::cerr << "[bw] === 带宽法 t_z 网格搜索 ===\n";
    for (uint32_t seed : {7u, 42u, 99u, 123u, 321u}) {
        auto sc = makeScene(8, 1500, 0.2, seed, false, false);
        // 先跑 LM
        params_.useCeres = false;
        op_.reset(new ProjectorJointCalib(params_));
        auto r = run(sc.input);
        if (!r.success) continue;

        double txLM = r.projectorT[0], tyLM = r.projectorT[1], tzLM = r.projectorT[2];
        double tzTrue = sc.t_true[2];

        // 收集 cleanPts（从 denoisedPoints）
        std::vector<cv::Vec3d> pts(r.denoisedPoints.begin(), r.denoisedPoints.end());

        // 网格搜索 tz（固定 tx/ty = LM 结果）
        double bestTz = tzLM, bestBw = 1e9;
        for (double tz = tzTrue - 5.0; tz <= tzTrue + 5.0; tz += 0.1) {
            double bw = computeBandwidth(pts, txLM, tyLM, tz,
                                         sc.input.f, sc.input.principalPoint.x, sc.input.principalPoint.y);
            if (bw < bestBw) { bestBw = bw; bestTz = tz; }
        }

        double tzErrLM = std::fabs(tzLM - tzTrue);
        double tzErrBw = std::fabs(bestTz - tzTrue);
        std::cerr << "[bw] seed=" << seed
                  << " tz_true=" << tzTrue
                  << " tz_LM=" << tzLM << "(err=" << tzErrLM << ")"
                  << " tz_BW=" << bestTz << "(err=" << tzErrBw << ")"
                  << " bw_min=" << bestBw
                  << (tzErrBw < tzErrLM ? " BW更好" : " LM更好") << "\n";
    }
    SUCCEED() << "Bandwidth test done";
}

// ============================================================================
// 质量标记验证：两处改进落地后的行为校验
//   1) cond 阈值 1e10：正常标定 cond~1e9 不再误报退化；正面退化姿态 cond~1e12 仍告警
//   2) rms 异常检测：初值落入伪极小值陷阱时 finalSampsonRms 飙升 → 判 Warning
// （伪极小值的 cond 反而更低~1e7，cond 查不出；rms 是有效探测器）
// ============================================================================
TEST_F(ProjectorJointCalibTest, QualityFlagValidation) {
    // A. 正常标定：cond~1e9 < 1e10，rms 正常 → 不触发退化/异常告警
    {
        auto sc = makeScene(8, 1500, 0.2, 7u, false, false);
        auto r = run(sc.input);
        ASSERT_TRUE(r.success) << r.message;
        EXPECT_LT(r.jacobianConditionNumber, 1e10) << "正常 cond 应 < 1e10";
        EXPECT_LT(r.finalSampsonRms, params_.anomalyRmsThreshold) << "正常 rms 应低于异常阈值";
        EXPECT_NE(r.qualityFlag, QualityFlag::Warning)
            << "正常标定不应 Warning（msg=" << r.message << "）";
        std::cerr << "[qf] 正常: cond=" << r.jacobianConditionNumber
                  << " rms=" << r.finalSampsonRms
                  << " flag=" << static_cast<int>(r.qualityFlag) << " msg=" << r.message << "\n";
    }
    // B. 退化姿态（全正面）：cond~1e12 > 1e10 → Warning
    {
        auto sc = makeScene(8, 1500, 0.0, 42u, true, false);
        auto r = run(sc.input);
        ASSERT_TRUE(r.success) << r.message;
        EXPECT_GT(r.jacobianConditionNumber, 1e10);
        EXPECT_EQ(r.qualityFlag, QualityFlag::Warning);
        std::cerr << "[qf] 退化: cond=" << r.jacobianConditionNumber
                  << " flag=" << static_cast<int>(r.qualityFlag) << " msg=" << r.message << "\n";
    }
    // C. 伪极小值陷阱：初值 = 默认+(0,4,2)（已知跑飞方向）→ rms 飙升 → Warning
    {
        auto sc = makeScene(8, 1500, 0.2, 7u, false, false);
        sc.input.initialT = sc.input.initialT + cv::Vec3d(0, 4, 2);
        auto r = run(sc.input);
        ASSERT_TRUE(r.success) << r.message;
        EXPECT_GT(r.finalSampsonRms, params_.anomalyRmsThreshold)
            << "陷阱 rms 应超阈值（实际 " << r.finalSampsonRms << "）";
        EXPECT_EQ(r.qualityFlag, QualityFlag::Warning)
            << "陷阱应被 rms 异常检测告警";
        std::cerr << "[qf] 陷阱: tz_err=" << std::fabs(r.projectorT[2] - sc.t_true[2])
                  << " rms=" << r.finalSampsonRms
                  << " cond=" << r.jacobianConditionNumber
                  << " flag=" << static_cast<int>(r.qualityFlag) << "\n";
    }
    SUCCEED() << "Quality flag validation done";
}
