#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <optional>

namespace fc {

// ============================================================================
// LaserCalibConfig —— laser/config.json 解析结果
// 字段对齐 PlaneMapTempTableParams / PlaneMapParams / UndistortPointsParams
// 等算子参数。温度字段缺省从 handoff 继承，config.json 可覆盖。
// ============================================================================
struct LaserCalibConfig {
    // plane_map 网格/深度
    // gridStep 默认 8：0.5 会使候选缓冲达数百 GB（GPU 越界崩溃，见守卫）；
    //   8px 网格 + 64 深度采样 ≈ 每线 5 万像素，14 线约 0.7GB，安全且满足映射表分辨率
    float gridStep     = 8.0f;
    float depthMin     = 20.0f;
    float depthMax     = 500.0f;
    int   depthSamples = 64;
    float epipolarStep = 0.5f;

    // 通用 GPU 设备
    int deviceId = 0;

    // mask_extract 前端参数（喂 4-1；实拍激光线宽仅 2~6px，
    // 默认 erodeSize=5 会把线整条腐蚀掉，必须按 config.json 配置）
    int maskThreshold    = 60;
    int maskErodeSize    = 1;
    int maskDilateSize   = 3;
    int maskMinArea      = 100;
    int maskMaxArea      = 100000;

    // laser_match 视差范围（喂 4-7；近距目标实际视差可达 600~1400px，
    // 默认 max_disparity=500 会拒掉全部匹配 → matched=0）
    float matchMinDisparity = 0.0f;
    float matchMaxDisparity = 2000.0f;

    // epipolar_interp 相邻点对约束（喂 4-6；默认 max_x_diff=1/max_y_span=2
    // 是为垂直激光线设计的，左斜/右斜 45° 线相邻行 dx≈1 会全被拒）
    float interpStep      = 0.5f;
    float interpMaxXDiff  = 10.0f;
    float interpMaxYSpan  = 6.0f;

    // 激光线编号（喂 VirtualPixelGenerator/plane_map）
    // 缺省空：运行期由 pose_optimize 后实际出现的线号决定
    std::vector<int> lineIds;

    // 温度（缺省与 handoff 一致；config.json 可覆盖）
    double cte           = 23.6e-6;
    double referenceTemp = 25.0;
    double tempRangeMin  = -10.0;
    double tempRangeMax  = 10.0;
    double tempStep      = 0.2;

    // 立体矫正（喂 PlaneMapTempTableParams.flags/alpha）
    double rectifyAlpha = 0.0;
    int    rectifyFlags = 1;

    static LaserCalibConfig fromJson(const std::string& path);
};

// ============================================================================
// CameraCalibHandoff —— 解析模块1 输出的 camera_calib.json
// 字段映射（见 module1/calib_io.cpp::buildCameraCalibJson + 各算子 toJson）:
//   cameraMatrixL/R ← intrinsic.left/right.camera_matrix
//   distCoeffsL/R   ← intrinsic.left/right.dist_coeffs
//   R, T            ← extrinsic.R, extrinsic.T
//   R1, R2, P1, P2, Q ← rectify.R1, R2, P1, P2, Q
//   imageSize        ← imageSize ([W, H])
//   referenceTemp, cte, tempRangeMin/Max/Step ← 顶层字段
// ============================================================================
struct CameraCalibHandoff {
    cv::Mat cameraMatrixL, distCoeffsL;
    cv::Mat cameraMatrixR, distCoeffsR;
    cv::Mat R, T;
    cv::Mat R1, R2, P1, P2, Q;
    cv::Size imageSize;

    double referenceTemp = 25.0;
    double cte           = 23.6e-6;
    double tempRangeMin  = -10.0;
    double tempRangeMax  = 10.0;
    double tempStep      = 0.2;

    // JSON 顶层 schema 字段（诊断/版本校验用）
    std::string schema;

    bool empty() const { return cameraMatrixL.empty() && cameraMatrixR.empty() && R.empty(); }
};

std::optional<CameraCalibHandoff> loadCameraCalibHandoff(const std::string& path);

// 一致性校验：cte / tempRange / tempStep / imageSize 必须一致
// 不一致时填充 why 并返回 false
bool validateHandoffConsistency(const LaserCalibConfig& cfg,
                                const CameraCalibHandoff& h,
                                std::string& why);

// ============================================================================
// PoseFrame / LaserInput —— laser 数据目录扫描
// 目录布局（设计稿 §5.3）:
//   <dir>/config.json
//   <dir>/camera_calib.json   ← handoff
//   <dir>/pose_00/L_tube0.png + R_tube0.png  [+ L_tube1.png + R_tube1.png ...]
//   <dir>/pose_01/...
// ============================================================================
struct PoseFrame {
    cv::Mat leftGray;
    cv::Mat rightGray;
};

struct LaserInput {
    LaserCalibConfig config;
    CameraCalibHandoff handoff;
    std::vector<std::string> poseDirs;             // 姿态目录名（诊断用）
    std::vector<std::vector<PoseFrame>> poseFrames;  // [pose][tube]
};

std::optional<LaserInput> loadLaserInput(const std::string& dir);

// 写 JSON 到磁盘（模块2 独立实现，避免与 module1 的 fc::writeJson 重定义）
bool writeLaserJson(const std::string& path, const nlohmann::json& j);

} // namespace fc
