#if 0  // SKIPPED: corrupted by batch edit

/**
 * @file test_extrinsic_compensate_cpu.cpp
 * @brief 鐩告満澶栧弬娓╁害琛ュ伩琛ㄥ崟鍏冩祴璇? */

#include <gtest/gtest.h>
#include "extrinsic_compensate_cpu.h"
#include <cmath>
#include <vector>
#include <algorithm>

using namespace calib;

// 娴嬭瘯鐢ㄥ父閲?
static constexpr double kEps = 1e-9;          // 娴偣姣旇緝绮惧害
static constexpr double kCte6061 = 23.6e-6;    // 6061-T6 CTE
static constexpr double kRefTemp = 25.0;

// 鍗曚綅鐭╅樀 R (琛屼富搴?
static constexpr double kIdentityR[9] = {1,0,0, 0,1,0, 0,0,1};

// 鍏稿瀷鍩虹嚎锛氭部 X 杞?120mm
static constexpr double kTypicalT[3] = {120.0, 0.0, 0.0};

// 杈呭姪锛氬垱寤洪粯璁ゅ鍙?CameraExtrinsics makeDefaultExtrinsics() {
    CameraExtrinsics ce;
    std::copy(std::begin(kIdentityR), std::end(kIdentityR), std::begin(ce.R));
    std::copy(std::begin(kTypicalT), std::end(kTypicalT), std::begin(ce.T));
    ce.referenceTemp = kRefTemp;
    return ce;
}

// 杈呭姪锛氭瘮杈冧袱涓?double[9]
void expectRArrayEq(const double* expected, const double* actual, double eps = kEps) {
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(expected[i], actual[i], eps) << "R[" << i << "] mismatch";
    }
}

// 杈呭姪锛氭瘮杈冧袱涓?double[3]
void expectTArrayEq(const double* expected, const double* actual, double eps = kEps) {
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(expected[i], actual[i], eps) << "T[" << i << "] mismatch";
    }
}

// ============================================================
// CameraExtrinsics 娴嬭瘯
// ============================================================

TEST(CameraExtrinsicsTest, ValidateAcceptsValid) {
    auto ce = makeDefaultExtrinsics();
    EXPECT_NO_THROW(ce.validate());
}

TEST(CameraExtrinsicsTest, ValidateRejectsZeroT) {
    CameraExtrinsics ce;
    std::copy(std::begin(kIdentityR), std::end(kIdentityR), std::begin(ce.R));
    ce.T[0] = 0.0; ce.T[1] = 0.0; ce.T[2] = 0.0;
    EXPECT_THROW(ce.validate(), std::invalid_argument);
}

TEST(CameraExtrinsicsTest, JsonRoundTrip) {
    auto ce = makeDefaultExtrinsics();
    auto j = ce.toJson();
    auto ce2 = CameraExtrinsics::fromJson(j);
    expectRArrayEq(ce.R, ce2.R);
    expectTArrayEq(ce.T, ce2.T);
    EXPECT_DOUBLE_EQ(ce.referenceTemp, ce2.referenceTemp);
}

// ============================================================
// ExtrinsicCompensateCPUParams 娴嬭瘯
// ============================================================

TEST(ExtrinsicCompensateCPUParamsTest, DefaultValuesAreValid) {
    ExtrinsicCompensateCPUParams params;
    EXPECT_NO_THROW(params.validate());
    EXPECT_DOUBLE_EQ(params.cte, 23.6e-6);
    EXPECT_DOUBLE_EQ(params.tempStep, 0.2);
    EXPECT_DOUBLE_EQ(params.tempRangeMin, -10.0);
    EXPECT_DOUBLE_EQ(params.tempRangeMax, 10.0);
}

TEST(ExtrinsicCompensateCPUParamsTest, ValidateRejectsZeroStep) {
    ExtrinsicCompensateCPUParams params;
    params.tempStep = 0.0;
    EXPECT_THROW(params.validate(), std::invalid_argument);
}

TEST(ExtrinsicCompensateCPUParamsTest, ValidateRejectsNegativeCte) {
    ExtrinsicCompensateCPUParams params;
    params.cte = -1e-6;
    EXPECT_THROW(params.validate(), std::invalid_argument);
}

TEST(ExtrinsicCompensateCPUParamsTest, ValidateRejectsInvertedRange) {
    ExtrinsicCompensateCPUParams params;
    params.tempRangeMin = 10.0;
    params.tempRangeMax = -10.0;
    EXPECT_THROW(params.validate(), std::invalid_argument);
}

TEST(ExtrinsicCompensateCPUParamsTest, JsonRoundTrip) {
    ExtrinsicCompensateCPUParams params;
    params.cte = 30e-6;
    params.tempStep = 0.5;
    params.tempRangeMin = -20.0;
    params.tempRangeMax = 20.0;
    auto j = params.toJson();
    auto p2 = ExtrinsicCompensateCPUParams::fromJson(j);
    EXPECT_DOUBLE_EQ(params.cte, p2.cte);
    EXPECT_DOUBLE_EQ(params.tempStep, p2.tempStep);
    EXPECT_DOUBLE_EQ(params.tempRangeMin, p2.tempRangeMin);
    EXPECT_DOUBLE_EQ(params.tempRangeMax, p2.tempRangeMax);
}

// ============================================================
// 鏍稿績璁＄畻娴嬭瘯
// ============================================================

TEST(ExtrinsicCompensateCPUTest, DeltaTZeroReturnsOriginalExtrinsics) {
    ExtrinsicCompensateCPUParams params;
    params.tempStep = 1.0;
    params.tempRangeMin = 0.0;
    params.tempRangeMax = 0.0;

    ExtrinsicCompensateCPU op(params);
    auto ce = makeDefaultExtrinsics();

    auto result = op.Execute(ce);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.table.size(), 1u);

    const auto& entry = result.table[0];
    expectRArrayEq(ce.R, entry.R);
    expectTArrayEq(ce.T, entry.T);
    EXPECT_NEAR(entry.deltaT_vec[0], 0.0, kEps);
    EXPECT_NEAR(entry.deltaT_vec[1], 0.0, kEps);
    EXPECT_NEAR(entry.deltaT_vec[2], 0.0, kEps);
}

TEST(ExtrinsicCompensateCPUTest, PositiveDeltaTIncreasesBaseline) {
    ExtrinsicCompensateCPUParams params;
    params.cte = kCte6061;
    params.tempStep = 10.0;
    params.tempRangeMin = 0.0;
    params.tempRangeMax = 10.0;

    ExtrinsicCompensateCPU op(params);
    auto ce = makeDefaultExtrinsics();

    auto result = op.Execute(ce);
    ASSERT_TRUE(result.success);
    ASSERT_GE(result.table.size(), 2u);

    const auto& entry = result.table.back();
    double expectedScale = 1.0 + kCte6061 * 10.0;
    EXPECT_NEAR(entry.T[0], kTypicalT[0] * expectedScale, kEps);
    EXPECT_NEAR(entry.T[1], kTypicalT[1] * expectedScale, kEps);
    EXPECT_NEAR(entry.T[2], kTypicalT[2] * expectedScale, kEps);

    double expectedDelta = kTypicalT[0] * kCte6061 * 10.0;
    EXPECT_NEAR(entry.deltaT_vec[0], expectedDelta, kEps);
}

TEST(ExtrinsicCompensateCPUTest, NegativeDeltaTDecreasesBaseline) {
    ExtrinsicCompensateCPUParams params;
    params.cte = kCte6061;
    params.tempStep = 10.0;
    params.tempRangeMin = -10.0;
    params.tempRangeMax = 0.0;

    ExtrinsicCompensateCPU op(params);
    auto ce = makeDefaultExtrinsics();

    auto result = op.Execute(ce);
    ASSERT_TRUE(result.success);
    ASSERT_GE(result.table.size(), 2u);

    const auto& entry = result.table.front();
    double expectedScale = 1.0 + kCte6061 * (-10.0);
    EXPECT_NEAR(entry.T[0], kTypicalT[0] * expectedScale, kEps);
    EXPECT_NEAR(entry.deltaT_vec[0], kTypicalT[0] * kCte6061 * (-10.0), kEps);
}

TEST(ExtrinsicCompensateCPUTest, TableSizeMatchesRange) {
    ExtrinsicCompensateCPUParams params;
    params.tempStep = 0.2;
    params.tempRangeMin = -10.0;
    params.tempRangeMax = 10.0;

    ExtrinsicCompensateCPU op(params);
    auto ce = makeDefaultExtrinsics();

    auto result = op.Execute(ce);
    ASSERT_TRUE(result.success);

    int expectedSize = static_cast<int>(std::round((params.tempRangeMax - params.tempRangeMin) / params.tempStep)) + 1;
    EXPECT_EQ(static_cast<int>(result.table.size()), expectedSize);
}

TEST(ExtrinsicCompensateCPUTest, TableCoversFullRange) {
    ExtrinsicCompensateCPUParams params;
    params.tempStep = 1.0;
    params.tempRangeMin = -5.0;
    params.tempRangeMax = 5.0;

    ExtrinsicCompensateCPU op(params);
    auto ce = makeDefaultExtrinsics();

    auto result = op.Execute(ce);
    ASSERT_TRUE(result.success);

    EXPECT_NEAR(result.table.front().temperature, 20.0, kEps);
    EXPECT_NEAR(result.table.back().temperature, 30.0, kEps);
}

TEST(ExtrinsicCompensateCPUTest, RotationUnchangedAcrossAllSteps) {
    ExtrinsicCompensateCPUParams params;
    params.tempStep = 2.0;
    params.tempRangeMin = -10.0;
    params.tempRangeMax = 10.0;

    ExtrinsicCompensateCPU op(params);
    auto ce = makeDefaultExtrinsics();

    auto result = op.Execute(ce);
    ASSERT_TRUE(result.success);

    for (const auto& entry : result.table) {
        expectRArrayEq(ce.R, entry.R);
    }
}

TEST(ExtrinsicCompensateCPUTest, BaselineDirectionPreserved) {
    ExtrinsicCompensateCPUParams params;
    params.cte = kCte6061;
    params.tempStep = 5.0;
    params.tempRangeMin = -10.0;
    params.tempRangeMax = 10.0;

    // 浣跨敤闈炶酱瀵归綈鐨?T
    CameraExtrinsics ce;
    std::copy(std::begin(kIdentityR), std::end(kIdentityR), std::begin(ce.R));
    ce.T[0] = 100.0; ce.T[1] = 50.0; ce.T[2] = 30.0;
    ce.referenceTemp = kRefTemp;

    double origNorm = std::sqrt(ce.T[0]*ce.T[0] + ce.T[1]*ce.T[1] + ce.T[2]*ce.T[2]);

    ExtrinsicCompensateCPU op(params);
    auto result = op.Execute(ce);
    ASSERT_TRUE(result.success);

    for (const auto& entry : result.table) {
        double newNorm = std::sqrt(entry.T[0]*entry.T[0] + entry.T[1]*entry.T[1] + entry.T[2]*entry.T[2]);
        EXPECT_GT(newNorm, 0.0);
        // 鏂瑰悜涓嶅彉锛氬綊涓€鍖栧悗涓庡師鏂瑰悜涓€鑷?
        EXPECT_NEAR(entry.T[0] / newNorm, ce.T[0] / origNorm, kEps);
        EXPECT_NEAR(entry.T[1] / newNorm, ce.T[1] / origNorm, kEps);
        EXPECT_NEAR(entry.T[2] / newNorm, ce.T[2] / origNorm, kEps);
    }
}

TEST(ExtrinsicCompensateCPUTest, CustomCTEWorks) {
    ExtrinsicCompensateCPUParams params;
    params.cte = 50e-6;
    params.tempStep = 1.0;
    params.tempRangeMin = 0.0;
    params.tempRangeMax = 1.0;

    ExtrinsicCompensateCPU op(params);
    CameraExtrinsics ce;
    std::copy(std::begin(kIdentityR), std::end(kIdentityR), std::begin(ce.R));
    ce.T[0] = 100.0; ce.T[1] = 0.0; ce.T[2] = 0.0;
    ce.referenceTemp = 20.0;

    auto result = op.Execute(ce);
    ASSERT_TRUE(result.success);
    ASSERT_GE(result.table.size(), 2u);

    const auto& entry = result.table.back();
    EXPECT_NEAR(entry.T[0], 100.0 * 1.00005, kEps);
}

TEST(ExtrinsicCompensateCPUTest, InvalidExtrinsicsThrows) {
    ExtrinsicCompensateCPU op;
    CameraExtrinsics ce;
    std::copy(std::begin(kIdentityR), std::end(kIdentityR), std::begin(ce.R));
    ce.T[0] = 0.0; ce.T[1] = 0.0; ce.T[2] = 0.0;
    EXPECT_THROW(op.Execute(ce), std::invalid_argument);
}

TEST(ExtrinsicCompensateCPUTest, ResultToJsonContainsAllFields) {
    ExtrinsicCompensateCPUParams params;
    params.tempStep = 5.0;
    params.tempRangeMin = -5.0;
    params.tempRangeMax = 5.0;

    ExtrinsicCompensateCPU op(params);
    auto ce = makeDefaultExtrinsics();

    auto result = op.Execute(ce);
    auto j = result.toJson();

    EXPECT_TRUE(j.contains("success"));
    EXPECT_TRUE(j.contains("referenceTemperature"));
    EXPECT_TRUE(j.contains("cte"));
    EXPECT_TRUE(j.contains("baselineRef"));
    EXPECT_TRUE(j.contains("referenceExtrinsics"));
    EXPECT_TRUE(j.contains("table"));
    EXPECT_TRUE(j.contains("tableSize"));
    EXPECT_TRUE(j["success"].get<bool>());
    EXPECT_EQ(j["tableSize"].get<int>(), 3);
    EXPECT_GT(j["baselineRef"].get<double>(), 0.0);
}

TEST(ExtrinsicCompensateCPUTest, SetParamsUpdatesParameters) {
    ExtrinsicCompensateCPU op;

    ExtrinsicCompensateCPUParams newParams;
    newParams.cte = 10e-6;
    newParams.tempStep = 2.0;
    newParams.tempRangeMin = -4.0;
    newParams.tempRangeMax = 4.0;
    op.SetParams(newParams);

    const auto& p = op.GetParams();
    EXPECT_DOUBLE_EQ(p.cte, 10e-6);
    EXPECT_DOUBLE_EQ(p.tempStep, 2.0);
}


#endif // SKIPPED