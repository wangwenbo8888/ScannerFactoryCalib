#pragma once

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <stdexcept>
#include <type_traits>

namespace calib {

inline void checkRequiredField(const nlohmann::json& j, const std::string& field, const std::string& context) {
    if (!j.contains(field)) {
        throw std::invalid_argument(context + ": missing required field '" + field + "'");
    }
}

template<typename T>
T getRequired(const nlohmann::json& j, const std::string& field, const std::string& context) {
    checkRequiredField(j, field, context);
    try {
        return j.at(field).get<T>();
    }
    catch (const nlohmann::json::type_error&) {
        throw std::invalid_argument(
            context + ": field '" + field + "' type mismatch, expected " +
            (std::is_same_v<T, int> ? "integer" :
             std::is_same_v<T, double> ? "number" :
             std::is_same_v<T, bool> ? "boolean" :
             std::is_same_v<T, std::string> ? "string" : "value"));
    }
}

inline nlohmann::json matToJson(const cv::Mat& mat) {
    nlohmann::json rows = nlohmann::json::array();
    for (int r = 0; r < mat.rows; ++r) {
        nlohmann::json row = nlohmann::json::array();
        for (int c = 0; c < mat.cols; ++c) {
            row.push_back(mat.at<double>(r, c));
        }
        rows.push_back(row);
    }
    return rows;
}

inline cv::Mat jsonToMat(const nlohmann::json& j, int rows, int cols) {
    cv::Mat mat(rows, cols, CV_64F);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            mat.at<double>(r, c) = j[r][c].get<double>();
        }
    }
    return mat;
}

inline cv::Mat jsonToMatAuto(const nlohmann::json& j) {
    int rows = static_cast<int>(j.size());
    int cols = static_cast<int>(j[0].size());
    cv::Mat mat(rows, cols, CV_64F);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            mat.at<double>(r, c) = j[r][c].get<double>();
        }
    }
    return mat;
}

} // namespace calib
