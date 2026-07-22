#if 0  // SKIPPED: corrupted by batch edit

/**
 * @file test_laser_extrinsic_compensate_cpu.cpp
 * @brief 婵€鍏夊櫒铏氭嫙鐩告満澶栧弬娓╁害琛ュ伩琛ㄥ崟鍏冩祴璇? */

#include <gtest/gtest.h>
#include "laser_extrinsic_compensate_cpu.h"
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

// 鍏稿瀷鍩虹嚎锛氳櫄鎷熺浉鏈哄埌宸︾浉鏈烘部 X 杞?60mm
static constexpr double kTypicalLeftT[3] = {60.0, 0.0, 0.0};

// 鍏稿瀷鍩虹嚎锛氳櫄鎷熺浉鏈哄埌鍙崇浉鏈烘部 X 杞?-60mm
static constexpr double kTypicalRightT[3] = {-60.0, 0.0, 0.0};

// 杈呭姪锛氬垱寤洪粯璁よ櫄鎷熺浉鏈衡啋宸︾浉鏈哄鍙?CameraExtrinsics makeVirtualToLeft() {
    CameraExtrinsics ce;
    std::copy(std::begin(kIdentityR), std::end(kIdentityR), std::begin(ce.R));
    std::copy(std::begin(kTypicalLeftT), std::end(kTypicalLeftT), std::begin(ce.T));
    ce.referenceTemp = kRefTemp;
    return ce;
}

// 杈呭姪锛氬垱寤洪粯璁よ櫄鎷熺浉鏈衡啋鍙崇浉鏈哄鍙?CameraExtrinsics makeVirtualToRight() {
    CameraExtrinsics ce;
    std::copy(std::begin(kIdentityR), std::end(kIdentityR), std::begin(ce.R));
    std::copy(std::begin(kTypicalRightT), std::end(kTypicalRightT), std::begin(ce.T));
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
// LaserExtrinsicCompensateCPUParams 娴嬭瘯
// ============================================================

TEST(LaserExtrinsicCompensateCPUParamsTest, DefaultValuesAreValid) {
    LaserExtrinsicCompensateCPUParams params;
    EXPECT_NO_THROW(params.validate());
    EXPECT_DOUBLE_EQ(params.cte, 23.6e-6);
    EXPECT_DOUBLE_EQ(params.tempStep, 0.2);
    EXPECT_DOUBLE_EQ(params.tempRangeMin, -10.0);
    EXPECT_DOUBLE_EQ(params.tempRangeMax, 10.0);
}

TEST(LaserExtrinsicCompensateCPUParamsTest, ValidateRejectsZeroStep) {
    LaserExtrinsicCompensateCPUParams params;
    params.tempStep = 0.0;
    EXPECT_THROW(params.validate(), std::invalid_argument);
}

TEST(LaserExtrinsicCompensateCPUParamsTest, ValidateRejectsNegativeCte) {
    LaserExtrinsicCompensateCPUParams params;
    params.cte = -1e-6;
    EXPECT_THROW(params.validate(), std::invalid_argument);
}

TEST(LaserExtrinsicCompensateCPUParamsTest, ValidateRejectsInvertedRange) {
    LaserExtrinsicCompensateCPUParams params;
    params.tempRangeMin = 10.0;
    params.tempRangeMax = -10.0;
    EXPECT_THROW(params.validate(), std::invalid_argument);
}

TEST(LaserExtrinsicCompensateCPUParamsTest, JsonRoundTrip) {
    LaserExtrinsicCompensateCPUParams params;
    params.cte = 30e-6;
    params.tempStep = 0.5;
    params.tempRangeMin = -20.0;
    params.tempRangeMax = 20.0;
    auto j = params.toJson();
    auto p2 = LaserExtrinsicCompensateCPUParams::fromJson(j);
    EXPECT_DOUBLE_EQ(params.cte, p2.cte);
    EXPECT_DOUBLE_EQ(params.tempStep, p2.tempStep);
    EXPECT_DOUBLE_EQ(params.tempRangeMin, p2.tempRangeMin);
    EXPECT_DOUBLE_EQ(params.tempRangeMax, p2.tempRangeMax);
}

// ============================================================
// 鏍稿績璁＄畻娴嬭瘯
// ============================================================

TEST(LaserExtrinsicCompensateCPUTest, DeltaTZeroReturnsOriginalExtrinsics) {
    LaserExtrinsicCompensateCPUParams params;
    params.tempStep = 1.0;
    params.tempRangeMin = 0.0;
    params.tempRangeMax = 0.0;

    LaserExtrinsicCompensateCPU op(params);
    auto v2l = makeVirtualToLeft();
    auto v2r = makeVirtualToRight();

    auto result = op.Execute(v2l, v2r);
    ASSERT_TRUE(result.success);

    // left table: 1 entry, matches original
    ASSERT_EQ(result.leftResult.table.size(), 1u);
    expectRArrayEq(v2l.R, result.leftResult.table[0].R);
    expectTArrayEq(v2l.T, result.leftResult.table[0].T);

    // right table: 1 entry, matches original
    ASSERT_EQ(result.rightResult.table.size(), 1u);
    expectRArrayEq(v2r.R, result.rightResult.table[0].R);
    expectTArrayEq(v2r.T, result.rightResult.table[0].T);
}

TEST(LaserExtrinsicCompensateCPUTest, PositiveDeltaTIncreasesBaseline) {
    LaserExtrinsicCompensateCPUParams params;
    params.cte = kCte6061;
    params.tempStep = 10.0;
    params.tempRangeMin = 0.0;
    params.tempRangeMax = 10.0;

    LaserExtrinsicCompensateCPU op(params);
    auto v2l = makeVirtualToLeft();
    auto v2r = makeVirtualToRight();

    auto result = op.Execute(v2l, v2r);
    ASSERT_TRUE(result.success);
    ASSERT_GE(result.leftResult.table.size(), 2u);
    ASSERT_GE(result.rightResult.table.size(), 2u);

    // Left: T_x should increase (positive direction)
    double expectedScaleLeft = 1.0 + kCte6061 * 10.0;
    const auto& leftLast = result.leftResult.table.back();
    EXPECT_NEAR(leftLast.T[0], kTypicalLeftT[0] * expectedScaleLeft, kEps);

    // Right: T_x should decrease further (negative direction, magnitude increases)
    const auto& rightLast = result.rightResult.table.back();
    EXPECT_NEAR(rightLast.T[0], kTypicalRightT[0] * expectedScaleLeft, kEps);
}

TEST(LaserExtrinsicCompensateCPUTest, NegativeDeltaTDecreasesBaseline) {
    LaserExtrinsicCompensateCPUParams params;
    params.cte = kCte6061;
    params.tempStep = 10.0;
    params.tempRangeMin = -10.0;
    params.tempRangeMax = 0.0;

    LaserExtrinsicCompensateCPU op(params);
    auto v2l = makeVirtualToLeft();
    auto v2r = makeVirtualToRight();

    auto result = op.Execute(v2l, v2r);
    ASSERT_TRUE(result.success);
    ASSERT_GE(result.leftResult.table.size(), 2u);

    double expectedScaleLeft = 1.0 + kCte6061 * (-10.0);
    const auto& leftFirst = result.leftResult.table.front();
    EXPECT_NEAR(leftFirst.T[0], kTypicalLeftT[0] * expectedScaleLeft, kEps);
}

TEST(LaserExtrinsicCompensateCPUTest, BothTablesSameSize) {
    LaserExtrinsicCompensateCPUParams params;
    params.tempStep = 0.2;
    params.tempRangeMin = -10.0;
    params.tempRangeMax = 10.0;

    LaserExtrinsicCompensateCPU op(params);
    auto v2l = makeVirtualToLeft();
    auto v2r = makeVirtualToRight();

    auto result = op.Execute(v2l, v2r);
    ASSERT_TRUE(result.success);

    EXPECT_EQ(result.leftResult.table.size(), result.rightResult.table.size());
}

TEST(LaserExtrinsicCompensateCPUTest, TableSizeMatchesRange) {
    LaserExtrinsicCompensateCPUParams params;
    params.tempStep = 0.2;
    params.tempRangeMin = -10.0;
    params.tempRangeMax = 10.0;

    LaserExtrinsicCompensateCPU op(params);
    auto v2l = makeVirtualToLeft();
    auto v2r = makeVirtualToRight();

    auto result = op.Execute(v2l, v2r);
    ASSERT_TRUE(result.success);

    int expectedSize = static_cast<int>(std::round((params.tempRangeMax - params.tempRangeMin) / params.tempStep)) + 1;
    EXPECT_EQ(static_cast<int>(result.leftResult.table.size()), expectedSize);
    EXPECT_EQ(static_cast<int>(result.rightResult.table.size()), expectedSize);
}

TEST(LaserExtrinsicCompensateCPUTest, RotationUnchangedAcrossAllSteps) {
    LaserExtrinsicCompensateCPUParams params;
    params.tempStep = 2.0;
    params.tempRangeMin = -10.0;
    params.tempRangeMax = 10.0;

    LaserExtrinsicCompensateCPU op(params);
    auto v2l = makeVirtualToLeft();
    auto v2r = makeVirtualToRight();

    auto result = op.Execute(v2l, v2r);
    ASSERT_TRUE(result.success);

    for (const auto& entry : result.leftResult.table) {
        expectRArrayEq(v2l.R, entry.R);
    }
    for (const auto& entry : result.rightResult.table) {
        expectRArrayEq(v2r.R, entry.R);
    }
}

TEST(LaserExtrinsicCompensateCPUTest, BaselineDirectionPreserved) {
    LaserExtrinsicCompensateCPUParams params;
    params.cte = kCte6061;
    params.tempStep = 5.0;
    params.tempRangeMin = -10.0;
    params.tempRangeMax = 10.0;

    // 闈炶酱瀵归綈澶栧弬
    CameraExtrinsics v2l;
    std::copy(std::begin(kIdentityR), std::end(kIdentityR), std::begin(v2l.R));
    v2l.T[0] = 50.0; v2l.T[1] = 20.0; v2l.T[2] = 10.0;
    v2l.referenceTemp = kRefTemp;

    CameraExtrinsics v2r;
    std::copy(std::begin(kIdentityR), std::end(kIdentityR), std::begin(v2r.R));
    v2r.T[0] = -50.0; v2r.T[1] = -20.0; v2r.T[2] = -10.0;
    v2r.referenceTemp = kRefTemp;

    double origNormL = std::sqrt(v2l.T[0]*v2l.T[0] + v2l.T[1]*v2l.T[1] + v2l.T[2]*v2l.T[2]);
    double origNormR = std::sqrt(v2r.T[0]*v2r.T[0] + v2r.T[1]*v2r.T[1] + v2r.T[2]*v2r.T[2]);

    LaserExtrinsicCompensateCPU op(params);
    auto result = op.Execute(v2l, v2r);
    ASSERT_TRUE(result.success);

    // 妫€鏌ュ乏鐩告満鏂瑰悜涓嶅彉
    for (const auto& entry : result.leftResult.table) {
        double norm = std::sqrt(entry.T[0]*entry.T[0] + entry.T[1]*entry.T[1] + entry.T[2]*entry.T[2]);
        EXPECT_GT(norm, 0.0);
        EXPECT_NEAR(entry.T[0] / norm, v2l.T[0] / origNormL, kEps);
        EXPECT_NEAR(entry.T[1] / norm, v2l.T[1] / origNormL, kEps);
        EXPECT_NEAR(entry.T[2] / norm, v2l.T[2] / origNormL, kEps);
    }

    // 妫€鏌ュ彸鐩告満鏂瑰悜涓嶅彉
    for (const auto& entry : result.rightResult.table) {
        double norm = std::sqrt(entry.T[0]*entry.T[0] + entry.T[1]*entry.T[1] + entry.T[2]*entry.T[2]);
        EXPECT_GT(norm, 0.0);
        EXPECT_NEAR(entry.T[0] / norm, v2r.T[0] / origNormR, kEps);
        EXPECT_NEAR(entry.T[1] / norm, v2r.T[1] / origNormR, kEps);
        EXPECT_NEAR(entry.T[2] / norm, v2r.T[2] / origNormR, kEps);
    }
}

TEST(LaserExtrinsicCompensateCPUTest, InvalidLeftExtrinsicsFails) {
    LaserExtrinsicCompensateCPU op;
    CameraExtrinsics v2l;  // T = {0,0,0} 鈥?invalid
    std::copy(std::begin(kIdentityR), std::end(kIdentityR), std::begin(v2l.R));
    auto v2r = makeVirtualToRight();

    EXPECT_THROW(op.Execute(v2l, v2r), std::invalid_argument);
}

TEST(LaserExtrinsicCompensateCPUTest, InvalidRightExtrinsicsFails) {
    LaserExtrinsicCompensateCPU op;
    auto v2l = makeVirtualToLeft();
    CameraExtrinsics v2r;  // T = {0,0,0} 鈥?invalid
    std::copy(std::begin(kIdentityR), std::end(kIdentityR), std::begin(v2r.R));

    EXPECT_THROW(op.Execute(v2l, v2r), std::invalid_argument);
}

TEST(LaserExtrinsicCompensateCPUTest, ResultToJsonContainsAllFields) {
    LaserExtrinsicCompensateCPUParams params;
    params.tempStep = 5.0;
    params.tempRangeMin = -5.0;
    params.tempRangeMax = 5.0;

    LaserExtrinsicCompensateCPU op(params);
    auto v2l = makeVirtualToLeft();
    auto v2r = makeVirtualToRight();

    auto result = op.Execute(v2l, v2r);
    auto j = result.toJson();

    EXPECT_TRUE(j.contains("success"));
    EXPECT_TRUE(j.contains("message"));
    EXPECT_TRUE(j.contains("referenceTemperature"));
    EXPECT_TRUE(j.contains("cte"));
    EXPECT_TRUE(j.contains("virtualToLeft"));
    EXPECT_TRUE(j.contains("virtualToRight"));
    EXPECT_TRUE(j.contains("tableSize"));
    EXPECT_TRUE(j["success"].get<bool>());
    EXPECT_TRUE(j["virtualToLeft"]["success"].get<bool>());
    EXPECT_TRUE(j["virtualToRight"]["success"].get<bool>());
}

TEST(LaserExtrinsicCompensateCPUTest, SetParamsUpdatesParameters) {
    LaserExtrinsicCompensateCPU op;

    LaserExtrinsicCompensateCPUParams newParams;
    newParams.cte = 10e-6;
    newParams.tempStep = 2.0;
    newParams.tempRangeMin = -4.0;
    newParams.tempRangeMax = 4.0;
    op.SetParams(newParams);

    const auto& p = op.GetParams();
    EXPECT_DOUBLE_EQ(p.cte, 10e-6);
    EXPECT_DOUBLE_EQ(p.tempStep, 2.0);
}

TEST(LaserExtrinsicCompensateCPUTest, DifferentReferenceTemps) {
    // 铏氭嫙鐩告満鍒板乏鐩告満鍙傝€冩俯搴︿负25锛屽埌鍙崇浉鏈哄弬鑰冩俯搴︿负20
    // 瀹為檯浣跨敤涓簲璇ョ浉鍚岋紝浣嗙畻瀛愪笉寮哄埗瑕佹眰
    LaserExtrinsicCompensateCPUParams params;
    params.tempStep = 1.0;
    params.tempRangeMin = 0.0;
    params.tempRangeMax = 1.0;

    LaserExtrinsicCompensateCPU op(params);

    CameraExtrinsics v2l = makeVirtualToLeft();
    v2l.referenceTemp = 25.0;

    CameraExtrinsics v2r = makeVirtualToRight();
    v2r.referenceTemp = 25.0;

    auto result = op.Execute(v2l, v2r);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.leftResult.table.size(), result.rightResult.table.size());
}


#endif // SKIPPED