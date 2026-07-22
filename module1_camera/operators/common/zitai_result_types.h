/**
 * @file zitai_result_types.h
 * @brief 标定流程（姿态）跨算子共享的结果类型定义
 *
 * 设计原则�? * - 每个算子�?result struct 及其子类型在此统一定义
 * - 算子自身�?Params、Stats、Operator class 留在各算子头文件�? * - 下游算子只需 include 本文件，无需 include 上游算子头文�? * - 修改任一算子的内部实现（Params/Operator class）不会触发下游重编译
 */

#pragma once


#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include "calib_types.h"

namespace calib {

// ============================================================
// 07 EllipseFitCPU 的输出类�?// ============================================================

struct EllipseFitCPUResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    double centerX = 0.0;
    double centerY = 0.0;
    double majorAxis = 0.0;
    double minorAxis = 0.0;
    double angle = 0.0;
    int inlierCount = 0;
    int totalPointCount = 0;

    std::vector<cv::Point2f> inlierPoints;

    EllipseFitCPUResult() = default;
    ~EllipseFitCPUResult() = default;

    EllipseFitCPUResult(EllipseFitCPUResult&&) = default;
    EllipseFitCPUResult& operator=(EllipseFitCPUResult&&) = default;

    EllipseFitCPUResult(const EllipseFitCPUResult&) = delete;
    EllipseFitCPUResult& operator=(const EllipseFitCPUResult&) = delete;

    cv::Point2f centerPoint2f() const {
        return cv::Point2f(static_cast<float>(centerX),
                           static_cast<float>(centerY));
    }
};

inline std::vector<cv::Point2f> extractEllipseCenters(
    const std::vector<EllipseFitCPUResult>& results)
{
    std::vector<cv::Point2f> centers;
    centers.reserve(results.size());
    for (const auto& r : results)
        if (r.success)
            centers.emplace_back(static_cast<float>(r.centerX),
                                 static_cast<float>(r.centerY));
    return centers;
}

// ============================================================
// 09 EpipolarIntersectCPU 的输出类�?// ============================================================

struct EpipolarIntersectPoint {
    double x = 0.0;
    double y = 0.0;
    double yEpipolar = 0.0;
    int epipolarIndex = 0;
};

struct EllipseIntersectResult {
    double centerX = 0.0;
    double centerY = 0.0;
    double majorAxis = 0.0;
    double minorAxis = 0.0;
    double angle = 0.0;

    std::vector<EpipolarIntersectPoint> intersectPts;

    EllipseIntersectResult() = default;
    ~EllipseIntersectResult() = default;

    EllipseIntersectResult(EllipseIntersectResult&&) = default;
    EllipseIntersectResult& operator=(EllipseIntersectResult&&) = default;

    EllipseIntersectResult(const EllipseIntersectResult&) = delete;
    EllipseIntersectResult& operator=(const EllipseIntersectResult&) = delete;
};

// ============================================================
// 10 EdgeMatchCPU 的输出类�?// ============================================================

struct EdgeMatchPair {
    double leftX = 0.0;
    double leftY = 0.0;
    double rightX = 0.0;
    double rightY = 0.0;
    float disparity = 0.0f;
    float confidence = 0.0f;
    int epipolarIndex = 0;
    int leftEllipseIdx = -1;
    int rightEllipseIdx = -1;
    int side = 0;
};

struct EllipseEdgeMatchResult {
    int leftEllipseIdx = -1;
    int rightEllipseIdx = -1;
    double leftCenterX = 0.0;
    double leftCenterY = 0.0;
    double rightCenterX = 0.0;
    double rightCenterY = 0.0;
    std::vector<EdgeMatchPair> matchedPairs;

    EllipseEdgeMatchResult() = default;
    ~EllipseEdgeMatchResult() = default;

    EllipseEdgeMatchResult(EllipseEdgeMatchResult&&) = default;
    EllipseEdgeMatchResult& operator=(EllipseEdgeMatchResult&&) = default;

    EllipseEdgeMatchResult(const EllipseEdgeMatchResult&) = delete;
    EllipseEdgeMatchResult& operator=(const EllipseEdgeMatchResult&) = delete;
};

struct EdgeMatchStats {
    double totalTimeMs = 0.0;
    size_t totalEllipsePairs = 0;
    size_t totalEpipolarLines = 0;
    size_t matchedPairs = 0;
    size_t skippedPairs = 0;
    float matchRate = 0.0f;
    float avgDisparity = 0.0f;
    float disparityStd = 0.0f;
    float avgConfidence = 0.0f;
};

struct EdgeMatchCPUResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::vector<EllipseEdgeMatchResult> ellipseResults;
    EdgeMatchStats statistics;

    EdgeMatchCPUResult() = default;
    ~EdgeMatchCPUResult() = default;

    EdgeMatchCPUResult(EdgeMatchCPUResult&&) = default;
    EdgeMatchCPUResult& operator=(EdgeMatchCPUResult&&) = default;

    EdgeMatchCPUResult(const EdgeMatchCPUResult&) = delete;
    EdgeMatchCPUResult& operator=(const EdgeMatchCPUResult&) = delete;
};

// ============================================================
// 11 PointReconstructCPU 的输出类�?// ============================================================

struct ReconstructedPoint3D {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double leftU = 0.0;
    double leftV = 0.0;
    double rightU = 0.0;
    double rightV = 0.0;
    float reprojError = 0.0f;
};

struct PlaneFitResult {
    double nx = 0.0;
    double ny = 0.0;
    double nz = 0.0;
    double d = 0.0;
    double centroidX = 0.0;
    double centroidY = 0.0;
    double centroidZ = 0.0;
    double fitError = 0.0;
    double singularValues[3] = {0.0, 0.0, 0.0};
};

struct CircleFitResult {
    double centerLocalX = 0.0;
    double centerLocalY = 0.0;
    double radius = 0.0;
    double fitError = 0.0;
    double centerX = 0.0;
    double centerY = 0.0;
    double centerZ = 0.0;
};

struct MarkerReconstructResult {
    int leftEllipseIdx = -1;
    int rightEllipseIdx = -1;
    std::vector<ReconstructedPoint3D> reconstructedPoints;
    PlaneFitResult planeFit;
    CircleFitResult circleFit;
    double centerX = 0.0;
    double centerY = 0.0;
    double centerZ = 0.0;
    double normalX = 0.0;
    double normalY = 0.0;
    double normalZ = 0.0;
    bool validPlane = false;
    bool validCircle = false;

    MarkerReconstructResult() = default;
    ~MarkerReconstructResult() = default;

    MarkerReconstructResult(MarkerReconstructResult&&) = default;
    MarkerReconstructResult& operator=(MarkerReconstructResult&&) = default;

    MarkerReconstructResult(const MarkerReconstructResult&) = delete;
    MarkerReconstructResult& operator=(const MarkerReconstructResult&) = delete;
};

struct PointReconstructStats {
    double totalTimeMs = 0.0;
    double triangulateTimeMs = 0.0;
    double planeFitTimeMs = 0.0;
    double circleFitTimeMs = 0.0;
    double projectionTimeMs = 0.0;
    size_t totalMarkerPairs = 0;
    size_t validMarkerCount = 0;
    size_t totalReconstructedPoints = 0;
    float avgReprojError = 0.0f;
    double avgPlaneFitError = 0.0;
    double avgCircleFitError = 0.0;
    double avgRadius = 0.0;
    double radiusStd = 0.0;
};

struct PointReconstructCPUResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::vector<MarkerReconstructResult> markerResults;
    PointReconstructStats statistics;

    PointReconstructCPUResult() = default;
    ~PointReconstructCPUResult() = default;

    PointReconstructCPUResult(PointReconstructCPUResult&&) = default;
    PointReconstructCPUResult& operator=(PointReconstructCPUResult&&) = default;

    PointReconstructCPUResult(const PointReconstructCPUResult&) = delete;
    PointReconstructCPUResult& operator=(const PointReconstructCPUResult&) = delete;
};

} // namespace calib