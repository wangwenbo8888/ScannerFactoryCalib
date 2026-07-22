/**
 * @file test_mask_extract_cuda_precision.cpp
 * @brief 激光掩膜提取算子 - 精度测试（CPU 参考实现 vs CUDA 结果）
 *
 * 精度容差档次：档次②（整像素/几何类）
 * - CPU vs CUDA 误差 < 0.1px（或整像素一致性）
 * - 跨 SM 架构间 CUDA 互比差异 < 0.1px
 *
 * 测试策略：
 * 1. 使用相同的输入图像
 * 2. CPU 端使用 OpenCV CPU API 执行相同的形态学流水线
 * 3. CUDA 端使用 MaskExtractCUDA
 * 4. 逐像素比较结果，允许边界处微小差异
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>

#include "../mask_extract_cuda.h"

using namespace calib;


// ============================================================
// CPU 参考实现
// ============================================================

/**
 * @brief CPU 端激光掩膜提取参考实现
 *
 * 执行与 GPU 端完全相同的形态学流水线：
 * 二值化 → 腐蚀 → 膨胀 → 面积过滤（跳过）
 */
static cv::Mat extractMaskCPU(const cv::Mat& grayImage, const MaskExtractParams& params) {
    // Step 1: 二值化
    cv::Mat binary;
    cv::threshold(grayImage, binary, params.threshold, 255.0, cv::THRESH_BINARY);

    // Step 2: 腐蚀（去噪）
    cv::Mat erodeKernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(params.erodeSize, params.erodeSize)
    );
    cv::Mat eroded;
    cv::erode(binary, eroded, erodeKernel);

    // Step 3: 膨胀（恢复激光形状）
    cv::Mat dilateKernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(params.laserDilateSize, params.laserDilateSize)
    );
    cv::Mat dilated;
    cv::dilate(eroded, dilated, dilateKernel);

    return dilated;
}

/**
 * @brief 比较两个二值掩膜的像素一致性
 * @return 不一致像素数量
 */
static int countMismatchPixels(const cv::Mat& cpuMask, const cv::Mat& gpuMask) {
    CV_Assert(cpuMask.size() == gpuMask.size());
    CV_Assert(cpuMask.type() == CV_8UC1);
    CV_Assert(gpuMask.type() == CV_8UC1);

    cv::Mat diff;
    cv::absdiff(cpuMask, gpuMask, diff);
    return cv::countNonZero(diff);
}

// ============================================================
// 测试辅助：生成测试图像
// ============================================================

/**
 * @brief 生成带有模拟激光线的测试图像
 * @param rows 图像行数
 * @param cols 图像列数
 * @param lineCount 激光线数量
 * @param brightness 激光线亮度 [0, 255]
 */
static cv::Mat generateLaserTestImage(int rows, int cols, int lineCount, int brightness) {
    cv::Mat image = cv::Mat::zeros(rows, cols, CV_8UC1);

    // 添加高斯噪声背景
    cv::Mat noise(rows, cols, CV_8UC1);
    cv::randn(noise, 20, 10);
    image = noise;

    // 添加水平激光线
    int spacing = rows / (lineCount + 1);
    for (int i = 1; i <= lineCount; i++) {
        int y = spacing * i;
        // 激光线宽度约 3-5 像素
        cv::rectangle(image, cv::Point(10, y - 2), cv::Point(cols - 10, y + 2),
                      cv::Scalar(brightness), -1);
    }

    return image;
}

// ============================================================
// 精度测试（需要 GPU）
// ============================================================
#ifdef WITH_CUDA_TESTS

class MaskExtractPrecisionTest : public ::testing::Test {
protected:
    void SetUp() override {
        params_.threshold = 80;
        params_.erodeSize = 5;
        params_.laserDilateSize = 3;
        params_.minArea = 100;
        params_.maxArea = 100000;
    }

    MaskExtractParams params_;
};

/**
 * @brief 测试1: 单条激光线 - CPU vs CUDA 一致性
 *
 * 在 256x256 图像上放置一条水平激光线，
 * 比较CPU和GPU结果的像素一致性。
 * 档次②要求：整像素一致性（误差 < 0.1px）
 */
TEST_F(MaskExtractPrecisionTest, SingleLineCPUvsCUDA) {
    cv::Mat testImage = generateLaserTestImage(256, 256, 1, 200);

    // CPU 参考结果
    cv::Mat cpuMask = extractMaskCPU(testImage, params_);

    // CUDA 结果
    MaskExtractCUDA extractor(params_);
    extractor.Warmup(256, 256);
    auto result = extractor.Execute(testImage);

    ASSERT_TRUE(result.success);
    ASSERT_NE(result.d_laserMask, nullptr);

    // 下载 GPU 结果到 CPU
    cv::Mat gpuMask;
    result.d_laserMask->download(gpuMask);

    // 比较
    int mismatches = countMismatchPixels(cpuMask, gpuMask);
    double totalPixels = cpuMask.rows * cpuMask.cols;
    double mismatchRatio = static_cast<double>(mismatches) / totalPixels;

    // 档次②: 整像素一致性，允许 < 0.1% 的边界差异
    EXPECT_LT(mismatchRatio, 0.001)
        << "Mismatch pixels: " << mismatches << " / " << totalPixels;
}

/**
 * @brief 测试2: 多条激光线 - CPU vs CUDA 一致性
 */
TEST_F(MaskExtractPrecisionTest, MultipleLinesCPUvsCUDA) {
    cv::Mat testImage = generateLaserTestImage(512, 512, 5, 180);

    cv::Mat cpuMask = extractMaskCPU(testImage, params_);

    MaskExtractCUDA extractor(params_);
    extractor.Warmup(512, 512);
    auto result = extractor.Execute(testImage);

    ASSERT_TRUE(result.success);

    cv::Mat gpuMask;
    result.d_laserMask->download(gpuMask);

    int mismatches = countMismatchPixels(cpuMask, gpuMask);
    double totalPixels = cpuMask.rows * cpuMask.cols;
    double mismatchRatio = static_cast<double>(mismatches) / totalPixels;

    EXPECT_LT(mismatchRatio, 0.001)
        << "Mismatch pixels: " << mismatches << " / " << totalPixels;
}

/**
 * @brief 测试3: 高阈值场景
 */
TEST_F(MaskExtractPrecisionTest, HighThresholdCPUvsCUDA) {
    params_.threshold = 150;
    cv::Mat testImage = generateLaserTestImage(256, 256, 3, 200);

    cv::Mat cpuMask = extractMaskCPU(testImage, params_);

    MaskExtractCUDA extractor(params_);
    extractor.Warmup(256, 256);
    auto result = extractor.Execute(testImage);

    ASSERT_TRUE(result.success);

    cv::Mat gpuMask;
    result.d_laserMask->download(gpuMask);

    int mismatches = countMismatchPixels(cpuMask, gpuMask);
    double totalPixels = cpuMask.rows * cpuMask.cols;
    double mismatchRatio = static_cast<double>(mismatches) / totalPixels;

    EXPECT_LT(mismatchRatio, 0.001)
        << "Mismatch pixels: " << mismatches << " / " << totalPixels;
}

/**
 * @brief 测试4: 不同核大小 - CPU vs CUDA 一致性
 */
TEST_F(MaskExtractPrecisionTest, DifferentKernelSizesCPUvsCUDA) {
    params_.erodeSize = 7;
    params_.laserDilateSize = 5;

    cv::Mat testImage = generateLaserTestImage(256, 256, 2, 220);

    cv::Mat cpuMask = extractMaskCPU(testImage, params_);

    MaskExtractCUDA extractor(params_);
    extractor.Warmup(256, 256);
    auto result = extractor.Execute(testImage);

    ASSERT_TRUE(result.success);

    cv::Mat gpuMask;
    result.d_laserMask->download(gpuMask);

    int mismatches = countMismatchPixels(cpuMask, gpuMask);
    double totalPixels = cpuMask.rows * cpuMask.cols;
    double mismatchRatio = static_cast<double>(mismatches) / totalPixels;

    EXPECT_LT(mismatchRatio, 0.001)
        << "Mismatch pixels: " << mismatches << " / " << totalPixels;
}

/**
 * @brief 测试5: 全黑图像 - CPU vs CUDA 均应输出全黑掩膜
 */
TEST_F(MaskExtractPrecisionTest, AllBlackImageCPUvsCUDA) {
    cv::Mat testImage = cv::Mat::zeros(256, 256, CV_8UC1);

    cv::Mat cpuMask = extractMaskCPU(testImage, params_);

    MaskExtractCUDA extractor(params_);
    extractor.Warmup(256, 256);
    auto result = extractor.Execute(testImage);

    ASSERT_TRUE(result.success);

    cv::Mat gpuMask;
    result.d_laserMask->download(gpuMask);

    // 全黑图像 → 全黑掩膜
    EXPECT_EQ(cv::countNonZero(cpuMask), 0);
    EXPECT_EQ(cv::countNonZero(gpuMask), 0);
}

/**
 * @brief 测试6: cleanedMask 一致性
 */
TEST_F(MaskExtractPrecisionTest, CleanedMaskConsistency) {
    cv::Mat testImage = generateLaserTestImage(256, 256, 2, 200);

    MaskExtractCUDA extractor(params_);
    extractor.Warmup(256, 256);
    auto result = extractor.Execute(testImage);

    ASSERT_TRUE(result.success);
    ASSERT_NE(result.d_laserMask, nullptr);
    ASSERT_NE(result.d_cleanedMask, nullptr);

    cv::Mat laserMask, cleanedMask;
    result.d_laserMask->download(laserMask);
    result.d_cleanedMask->download(cleanedMask);

    // 当前版本 cleanedMask = laserMask（面积过滤为 TODO）
    int mismatches = countMismatchPixels(laserMask, cleanedMask);
    EXPECT_EQ(mismatches, 0)
        << "laserMask and cleanedMask should be identical (area filter is TODO)";
}

#endif // WITH_CUDA_TESTS
