/**
 * @file calib_types.h
 * @brief 标定流程通用类型定义
 */

#pragma once

#include <cstdint>
#include <opencv2/core.hpp>
#include <stdexcept>

#include "quality_flag.h"

namespace calib {

struct TemperatureData {
    float temperature;
    uint64_t timestamp;
};

struct StereoCalibration {
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Mat R;
    cv::Mat P;
    cv::Mat Q;
    cv::Mat R1;
    cv::Mat R2;
    cv::Mat P1;
    cv::Mat P2;
    cv::Size imageSize;

    void validate() const {
        if (!cameraMatrix.empty()) {
            if (cameraMatrix.size() != cv::Size(3, 3))
                throw std::invalid_argument("StereoCalibration::cameraMatrix must be 3x3");
            if (cameraMatrix.type() != CV_64F && cameraMatrix.type() != CV_32F)
                throw std::invalid_argument("StereoCalibration::cameraMatrix must be CV_32F or CV_64F");
        }
        if (!distCoeffs.empty()) {
            if (distCoeffs.total() < 4 || distCoeffs.total() > 8)
                throw std::invalid_argument("StereoCalibration::distCoeffs must have 4~8 elements");
            if (distCoeffs.type() != CV_64F && distCoeffs.type() != CV_32F)
                throw std::invalid_argument("StereoCalibration::distCoeffs must be CV_32F or CV_64F");
        }
        if (!R.empty()) {
            if (R.size() != cv::Size(3, 3))
                throw std::invalid_argument("StereoCalibration::R must be 3x3");
        }
        if (!P.empty()) {
            if (P.rows != 3 || P.cols != 4)
                throw std::invalid_argument("StereoCalibration::P must be 3x4");
        }
        if (!Q.empty()) {
            if (Q.size() != cv::Size(4, 4))
                throw std::invalid_argument("StereoCalibration::Q must be 4x4");
        }
        if (!R1.empty()) {
            if (R1.size() != cv::Size(3, 3))
                throw std::invalid_argument("StereoCalibration::R1 must be 3x3");
        }
        if (!R2.empty()) {
            if (R2.size() != cv::Size(3, 3))
                throw std::invalid_argument("StereoCalibration::R2 must be 3x3");
        }
        if (!P1.empty()) {
            if (P1.rows != 3 || P1.cols != 4)
                throw std::invalid_argument("StereoCalibration::P1 must be 3x4");
        }
        if (!P2.empty()) {
            if (P2.rows != 3 || P2.cols != 4)
                throw std::invalid_argument("StereoCalibration::P2 must be 3x4");
        }
    }

    bool empty() const {
        return cameraMatrix.empty() && distCoeffs.empty()
            && R.empty() && P.empty() && Q.empty()
            && R1.empty() && R2.empty() && P1.empty() && P2.empty();
    }

    cv::Matx33d stereoK() const {
        if (P.empty())
            throw std::runtime_error("StereoCalibration::stereoK() requires P to be set");
        cv::Matx33d K;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                K(i, j) = P.at<double>(i, j);
        return K;
    }

    cv::Matx33d stereoR() const {
        if (R.empty())
            throw std::runtime_error("StereoCalibration::stereoR() requires R to be set");
        cv::Matx33d Rm;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                Rm(i, j) = R.at<double>(i, j);
        return Rm;
    }

    cv::Matx33d getR1() const {
        cv::Matx33d m;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m(i, j) = R1.at<double>(i, j);
        return m;
    }

    cv::Matx33d getR2() const {
        cv::Matx33d m;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m(i, j) = R2.at<double>(i, j);
        return m;
    }

    cv::Matx34d getP1() const {
        cv::Matx34d m;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                m(i, j) = P1.at<double>(i, j);
        return m;
    }

    cv::Matx34d getP2() const {
        cv::Matx34d m;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                m(i, j) = P2.at<double>(i, j);
        return m;
    }
};

struct EdgePoint {
    double x = 0.0;
    double y = 0.0;
    double angle = 0.0;
    double amplitude = 0.0;
    int pixelX = 0;
    int pixelY = 0;
};

} // namespace calib
