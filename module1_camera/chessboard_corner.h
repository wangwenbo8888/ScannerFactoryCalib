#pragma once
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <string>

namespace fc {

struct ChessboardCornerParams {
    cv::Size patternSize{11, 8};       // 内角点 (width, height)
    int sbFlags = cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_NORMALIZE_IMAGE;
    cv::Size subpixWin{11, 11};
    cv::Size subpixZeroZone{-1, -1};
    cv::TermCriteria subpixTerm{cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 1e-4};
};

struct ChessboardCornerResult {
    bool found = false;
    std::vector<cv::Point2f> corners;  // 行主序, size == patternSize.area()
    double meanSubpixDelta = 0.0;       // 亚像素平均修正量(px)，诊断用
};

// ★ 单 CPU 函数：提取 + 编号（findChessboardCornersSB + cornerSubPix）
// 失败时 result.found=false，返回 false
bool extractChessboardCorners(const cv::Mat& gray,
                              const ChessboardCornerParams& params,
                              ChessboardCornerResult& result);

// L/R 编号一致性归一化：保证两图 corners[i] 对应同物理角点
// 思路：比较两图 4 角凸包走向，必要时翻转/逆序使走向一致；返回是否一致
bool normalizeLRCornerOrder(ChessboardCornerResult& left,
                            ChessboardCornerResult& right);

} // namespace fc
