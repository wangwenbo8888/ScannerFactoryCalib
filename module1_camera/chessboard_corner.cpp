#include "chessboard_corner.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace fc {

bool extractChessboardCorners(const cv::Mat& gray,
                              const ChessboardCornerParams& params,
                              ChessboardCornerResult& result) {
    result.found = false;
    result.corners.clear();
    result.meanSubpixDelta = 0.0;

    if (gray.empty()) {
        spdlog::warn("[chessboard_corner] empty image");
        return false;
    }
    cv::Mat gray8;
    if (gray.type() != CV_8U) {
        gray.convertTo(gray8, CV_8U);
    } else {
        gray8 = gray;
    }

    std::vector<cv::Point2f> coarse;
    bool found = cv::findChessboardCornersSB(gray8, params.patternSize, coarse, params.sbFlags);
    if (!found || static_cast<int>(coarse.size()) != params.patternSize.area()) {
        spdlog::debug("[chessboard_corner] not found or count mismatch");
        return false;
    }

    // 亚像素精化前快照，用于估算修正量
    std::vector<cv::Point2f> before = coarse;
    cv::cornerSubPix(gray8, coarse, params.subpixWin, params.subpixZeroZone, params.subpixTerm);

    double acc = 0.0;
    for (size_t i = 0; i < coarse.size(); ++i) {
        acc += std::hypot(coarse[i].x - before[i].x, coarse[i].y - before[i].y);
    }
    result.meanSubpixDelta = acc / coarse.size();

    result.corners = std::move(coarse);
    result.found = true;
    return true;
}

bool normalizeLRCornerOrder(ChessboardCornerResult& left,
                            ChessboardCornerResult& right) {
    if (!left.found || !right.found) return false;
    if (left.corners.size() != right.corners.size()) return false;
    if (left.corners.size() < 2) return false;

    // 用第一个角点 → 最后一角点的向量方向作为朝向判据
    auto firstLast = [](const std::vector<cv::Point2f>& v) {
        return v.back() - v.front();
    };
    cv::Point2f vL = firstLast(left.corners);
    cv::Point2f vR = firstLast(right.corners);
    double dot = vL.dot(vR);
    double mag = std::hypot(vL.x, vL.y) * std::hypot(vR.x, vR.y);
    if (mag == 0.0) return false;
    double cosang = dot / mag;
    // 若两向量夹角 > 90°（cos<0），起步方向相反 → 把右图逆序
    if (cosang < 0.0) {
        std::reverse(right.corners.begin(), right.corners.end());
    }
    return true;
}

} // namespace fc
