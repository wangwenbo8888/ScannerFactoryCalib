#pragma once
#include <opencv2/core.hpp>
#include <memory>
#include <vector>

namespace calib {

struct CclOutput {
    std::shared_ptr<cv::cuda::GpuMat> labeledMask;
    std::vector<cv::Rect> componentRects;
    int componentCount = 0;
};

struct MarkerChainOutput {
    std::vector<cv::Point3d> markerPoints3D;
    cv::Matx33d R;
    cv::Vec3d T;
    bool success = false;
};

} // namespace calib
