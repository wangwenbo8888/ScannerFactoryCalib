/**
 * @file test_intrinsic_compensate_cpu.cpp
 * @brief 鐩告満鍐呭弬娓╁害琛ュ伩琛ㄥ崟鍏冩祴璇? */

#include <gtest/gtest.h>
#include "intrinsic_compensate_cpu.h"
#include <cmath>
#include <vector>

using namespace calib;

// 娴嬭瘯鐢ㄥ父閲?
static constexpr double kEps = 1e-9;          // 娴偣姣旇緝绮惧害
static constexpr double kCte6061 = 23.6e-6;    // 6061-T6 CTE
static constexpr double kFx0 = 2500.0;
static constexpr double kFy0 = 2500.0;
static constexpr double kCx0 = 1024.0;
static constexpr double kCy0 = 768.0;
static constexpr double kRefTemp = 25.0;

// ============================================================
// CameraIntrinsics 娴嬭瘯
// ============================================================

TEST(CameraIntrinsicsTest, ValidateAcceptsValid) {
    CameraIntrinsics ci{kFx0, kFy0, kCx0, kCy0, kRefTemp};
    EXPECT_NO_THROW(ci.validate());
}

TEST(CameraIntrinsicsTest, ValidateRejectsZeroFx) {
    CameraIntrinsics ci{0.0, kFy0, kCx0, kCy0, kRefTemp};
    EXPECT_THROW(ci.validate(), std::invalid_argument);
}

TEST(CameraIntrinsicsTest, ValidateRejectsNegativeFy) {
    CameraIntrinsics ci{kFx0, -100.0, kCx0, kCy0, kRefTemp};
    EXPECT_THROW(ci.validate(), std::invalid_argument);
}

TEST(CameraIntrinsicsTest, JsonRoundTrip) {
    CameraIntrinsics ci{kFx0, kFy0, kCx0, kCy0, kRefTemp};
    auto j = ci.toJson();
    auto ci2 = CameraIntrinsics::fromJson(j);
    EXPECT_DOUBLE_EQ(ci.fx, ci2.fx);
    EXPECT_DOUBLE_EQ(ci.fy, ci2.fy);
    EXPECT_DOUBLE_EQ(ci.cx, ci2.cx);
    EXPECT_DOUBLE_EQ(ci.cy, ci2.cy);
    EXPECT_DOUBLE_EQ(ci.referenceTemp, ci2.referenceTemp);
}

// ============================================================
// IntrinsicCompensateCPUParams 娴嬭瘯
// ============================================================

TEST(IntrinsicCompensateCPUParamsTest, DefaultValuesAreValid) {
    IntrinsicCompensateCPUParams params;
    EXPECT_NO_THROW(params.validate());
    EXPECT_DOUBLE_EQ(params.cte, 23.6e-6);
    EXPECT_DOUBLE_EQ(params.tempStep, 0.2);
    EXPECT_DOUBLE_EQ(params.tempRangeMin, -10.0);
    EXPECT_DOUBLE_EQ(params.tempRangeMax, 10.0);
}

TEST(IntrinsicCompensateCPUParamsTest, ValidateRejectsZeroStep) {
    IntrinsicCompensateCPUParams params;
    params.tempStep = 0.0;
    EXPECT_THROW(params.validate(), std::invalid_argument);
}

TEST(IntrinsicCompensateCPUParamsTest, ValidateRejectsNegativeCte) {
    IntrinsicCompensateCPUParams params;
    params.cte = -1e-6;
    EXPECT_THROW(params.validate(), std::invalid_argument);
}

TEST(IntrinsicCompensateCPUParamsTest, ValidateRejectsInvertedRange) {
    IntrinsicCompensateCPUParams params;
    params.tempRangeMin = 10.0;
    params.tempRangeMax = -10.0;
    EXPECT_THROW(params.validate(), std::invalid_argument);
}

TEST(IntrinsicCompensateCPUParamsTest, JsonRoundTrip) {
    IntrinsicCompensateCPUParams params;
    params.cte = 30e-6;
    params.tempStep = 0.5;
    params.tempRangeMin = -20.0;
    params.tempRangeMax = 20.0;
    auto j = params.toJson();
    auto p2 = IntrinsicCompensateCPUParams::fromJson(j);
    EXPECT_DOUBLE_EQ(params.cte, p2.cte);
    EXPECT_DOUBLE_EQ(params.tempStep, p2.tempStep);
    EXPECT_DOUBLE_EQ(params.tempRangeMin, p2.tempRangeMin);
    EXPECT_DOUBLE_EQ(params.tempRangeMax, p2.tempRangeMax);
}

// ============================================================
// 鏍稿績璁＄畻娴嬭瘯
// ============================================================

TEST(IntrinsicCompensateCPUTest, DeltaTZeroReturnsOriginalIntrinsics) {
    IntrinsicCompensateCPUParams params;
    params.tempStep = 1.0;
    params.tempRangeMin = 0.0;
    params.tempRangeMax = 0.0;

    IntrinsicCompensateCPU op(params);
    CameraIntrinsics ci{kFx0, kFy0, kCx0, kCy0, kRefTemp};

    auto result = op.Execute(ci);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.table.size(), 1u);

    const auto& entry = result.table[0];
    EXPECT_NEAR(entry.fx, kFx0, kEps);
    EXPECT_NEAR(entry.fy, kFy0, kEps);
    EXPECT_NEAR(entry.cx, kCx0, kEps);
    EXPECT_NEAR(entry.cy, kCy0, kEps);
    EXPECT_NEAR(entry.deltaFx, 0.0, kEps);
    EXPECT_NEAR(entry.deltaFy, 0.0, kEps);
    EXPECT_NEAR(entry.deltaCx, 0.0, kEps);
    EXPECT_NEAR(entry.deltaCy, 0.0, kEps);
}

TEST(IntrinsicCompensateCPUTest, PositiveDeltaTIncreasesFocalLength) {
    IntrinsicCompensateCPUParams params;
    params.cte = kCte6061;
    params.tempStep = 10.0;
    params.tempRangeMin = 0.0;
    params.tempRangeMax = 10.0;

    IntrinsicCompensateCPU op(params);
    CameraIntrinsics ci{kFx0, kFy0, kCx0, kCy0, kRefTemp};

    auto result = op.Execute(ci);
    ASSERT_TRUE(result.success);
    ASSERT_GE(result.table.size(), 2u);

    const auto& entry = result.table.back();
    double expectedScale = 1.0 + kCte6061 * 10.0;
    EXPECT_NEAR(entry.fx, kFx0 * expectedScale, kEps);
    EXPECT_NEAR(entry.fy, kFy0 * expectedScale, kEps);
    EXPECT_NEAR(entry.cx, kCx0 * expectedScale, kEps);
    EXPECT_NEAR(entry.cy, kCy0 * expectedScale, kEps);

    double expectedDeltaFx = kFx0 * kCte6061 * 10.0;
    EXPECT_NEAR(entry.deltaFx, expectedDeltaFx, kEps);
}

TEST(IntrinsicCompensateCPUTest, NegativeDeltaTDecreasesFocalLength) {
    IntrinsicCompensateCPUParams params;
    params.cte = kCte6061;
    params.tempStep = 10.0;
    params.tempRangeMin = -10.0;
    params.tempRangeMax = 0.0;

    IntrinsicCompensateCPU op(params);
    CameraIntrinsics ci{kFx0, kFy0, kCx0, kCy0, kRefTemp};

    auto result = op.Execute(ci);
    ASSERT_TRUE(result.success);
    ASSERT_GE(result.table.size(), 2u);

    const auto& entry = result.table.front();
    double expectedScale = 1.0 + kCte6061 * (-10.0);
    EXPECT_NEAR(entry.fx, kFx0 * expectedScale, kEps);
    EXPECT_NEAR(entry.deltaFx, kFx0 * kCte6061 * (-10.0), kEps);
}

TEST(IntrinsicCompensateCPUTest, TableSizeMatchesRange) {
    IntrinsicCompensateCPUParams params;
    params.tempStep = 0.2;
    params.tempRangeMin = -10.0;
    params.tempRangeMax = 10.0;

    IntrinsicCompensateCPU op(params);
    CameraIntrinsics ci{kFx0, kFy0, kCx0, kCy0, kRefTemp};

    auto result = op.Execute(ci);
    ASSERT_TRUE(result.success);

    int expectedSize = static_cast<int>(std::round((params.tempRangeMax - params.tempRangeMin) / params.tempStep)) + 1;
    EXPECT_EQ(static_cast<int>(result.table.size()), expectedSize);
}

TEST(IntrinsicCompensateCPUTest, TableCoversFullRange) {
    IntrinsicCompensateCPUParams params;
    params.tempStep = 1.0;
    params.tempRangeMin = -5.0;
    params.tempRangeMax = 5.0;

    IntrinsicCompensateCPU op(params);
    CameraIntrinsics ci{kFx0, kFy0, kCx0, kCy0, kRefTemp};

    auto result = op.Execute(ci);
    ASSERT_TRUE(result.success);

    EXPECT_NEAR(result.table.front().temperature, 20.0, kEps);
    EXPECT_NEAR(result.table.back().temperature, 30.0, kEps);
}

TEST(IntrinsicCompensateCPUTest, CustomCTEWorks) {
    IntrinsicCompensateCPUParams params;
    params.cte = 50e-6;
    params.tempStep = 1.0;
    params.tempRangeMin = 0.0;
    params.tempRangeMax = 1.0;

    IntrinsicCompensateCPU op(params);
    CameraIntrinsics ci{1000.0, 1000.0, 500.0, 500.0, 20.0};

    auto result = op.Execute(ci);
    ASSERT_TRUE(result.success);
    ASSERT_GE(result.table.size(), 2u);

    const auto& entry = result.table.back();
    EXPECT_NEAR(entry.fx, 1000.0 * 1.00005, kEps);
}

TEST(IntrinsicCompensateCPUTest, InvalidIntrinsicsThrows) {
    IntrinsicCompensateCPU op;
    CameraIntrinsics ci{0.0, 0.0, 0.0, 0.0, 25.0};
    EXPECT_THROW(op.Execute(ci), std::invalid_argument);
}

TEST(IntrinsicCompensateCPUTest, ResultToJsonContainsAllFields) {
    IntrinsicCompensateCPUParams params;
    params.tempStep = 5.0;
    params.tempRangeMin = -5.0;
    params.tempRangeMax = 5.0;

    IntrinsicCompensateCPU op(params);
    CameraIntrinsics ci{kFx0, kFy0, kCx0, kCy0, kRefTemp};

    auto result = op.Execute(ci);
    auto j = result.toJson();

    EXPECT_TRUE(j.contains("success"));
    EXPECT_TRUE(j.contains("referenceTemperature"));
    EXPECT_TRUE(j.contains("cte"));
    EXPECT_TRUE(j.contains("referenceIntrinsics"));
    EXPECT_TRUE(j.contains("table"));
    EXPECT_TRUE(j.contains("tableSize"));
    EXPECT_TRUE(j["success"].get<bool>());
    EXPECT_EQ(j["tableSize"].get<int>(), 3);
}

TEST(IntrinsicCompensateCPUTest, SetParamsUpdatesParameters) {
    IntrinsicCompensateCPU op;

    IntrinsicCompensateCPUParams newParams;
    newParams.cte = 10e-6;
    newParams.tempStep = 2.0;
    newParams.tempRangeMin = -4.0;
    newParams.tempRangeMax = 4.0;
    op.SetParams(newParams);

    const auto& p = op.GetParams();
    EXPECT_DOUBLE_EQ(p.cte, 10e-6);
    EXPECT_DOUBLE_EQ(p.tempStep, 2.0);
}
