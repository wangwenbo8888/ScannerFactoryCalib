#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <optional>

#include "stereo_rectify_temp_table/stereo_rectify_temp_table_cpu.h"

namespace fc {

// ============================================================================
// LaserCalibConfig —— laser/config.json 解析结果
// 字段对齐 PlaneMapTempTableParams / PlaneMapParams / UndistortPointsParams
// 等算子参数。温度字段缺省从 handoff 继承，config.json 可覆盖。
// ============================================================================
struct LaserCalibConfig {
    // plane_map 网格/深度
    float gridStep     = 0.5f;
    float depthMin     = 100.0f;
    float depthMax     = 5000.0f;
    int   depthSamples = 200;
    float epipolarStep = 0.5f;

    // 通用 GPU 设备
    int deviceId = 0;

    // 激光线编号（喂 VirtualPixelGenerator/plane_map）
    // 缺省空：运行期由 pose_optimize 后实际出现的线号决定
    std::vector<int> lineIds;

    // 温度（缺省与 handoff 一致；config.json 可覆盖）
    double cte           = 23.6e-6;
    double referenceTemp = 25.0;
    double tempRangeMin  = -15.0;
    double tempRangeMax  = 15.0;
    double tempStep      = 0.5;

    // 立体矫正（喂 PlaneMapTempTableParams.flags/alpha）
    double rectifyAlpha = 0.0;
    int    rectifyFlags = 1;

    // 激光线编号扫描方向（喂 LaserLabelParams）:
    //   0=竖切中心列(默认, 水平多线形态), 1=横切中心行(斜线形态)
    int labelScanDirection = 0;
    int labelCenterRowOffset = 0;

    // F7 漂移解并发线程数（顶层键 f7Threads）
    //   0=自动（键缺省）: min(逻辑核-2, 8); 1=串行回退; 键值 <=0 解析时归一为 1
    int f7Threads = 0;

    // CMTT 生成器档级并行线程数（顶层键 cmttThreads）
    //   0=自动（键缺省）: min(硬件线程-2, 16); >0=精确值; <=-1 由生成器视为 1
    //   （线程数无关确定性: 任意取值产物逐字节一致）
    int cmttThreads = 0;

    static LaserCalibConfig fromJson(const std::string& path);
};

// ============================================================================
// LaserOpParams —— 算子层可调参数（2026-10-04 主库化）
// 编译默认＝产线定稿值（等价改造）；主库 laser_calib_params.json 与
// 数据集 config.json 均可逐叶覆盖（同 schema 同解析器 applyLaserParamsJson）。
// 不外置（红线）: CMTT rowStep/depthMin/depthMax（随标定走）; 4-6 mode=Labeled。
// ============================================================================
struct LaserOpParams {
    // 4-1 mask（2026-08-25/26 fcstepdump 六轮扫参定稿）
    int maskThreshold       = 50;
    int maskErodeSize       = 1;
    int maskLaserDilateSize = 19;
    int maskPostErodeSize   = 13;
    // 4-2 ccl
    int cclMinArea   = 0;
    int cclTopXCount = 27;
    // 4-3 label
    int labelRealLineTolerance = 10;
    // 4-4 steger（deviceId 走 LaserCalibConfig::deviceId）
    float stegerSigma         = 1.5f;
    int   stegerKernelSize    = 0;
    float stegerLowThreshold  = 2.0f;
    float stegerHighThreshold = 0.0f;
    int   stegerMaxLabels     = 256;
    // 4-7 match：视差上界 = |f·Tx| / depthMin × 本系数
    double matchDisparityMarginFactor = 1.05;
    // PJC（对齐 ProjectorJointCalibParams 字段）
    int    pjcMaxIterations        = 100;
    double pjcConvergenceThreshold = 1e-8;
    int    pjcMinPoses             = 5;
    int    pjcMinPointsPerPose     = 50;
    double pjcPlaneFitInlierThresh = 0.6;
    bool   pjcEnableTiming         = false;
    double pjcLambda0              = 1.0;
    double pjcLambdaDecay          = 0.95;
    double pjcHuberToCauchyThresh  = 1.0;
    double pjcCauchyToL2Thresh     = 0.3;
    double pjcTopologyEpsilon      = 1e-3;
    int    pjcCurveDegree          = 3;
    bool   pjcUseCeres             = false;
    bool   pjcUseBlockJtJ          = true;
    double pjcAnomalyRmsThreshold  = 0.15;
    bool   pjcRobustEnabled        = true;
    double pjcHuberDelta0          = 1.0;
    int    pjcIrlsMaxRounds        = 2;
    bool   pjcPoseWeightEnabled    = true;
    // PJC 验收门禁阈值（2026-10-04 外置；默认＝CLI 原硬编码 AcceptanceCfg）
    double pjcAccA1MaxLineRms      = 0.5;
    double pjcAccA2MaxP95          = 1.5;
    double pjcAccA3MaxOutlierRatio = 0.02;
    double pjcAccA5MaxLineDrift    = 1.0;
    double pjcAccA6MaxPoseDrift    = 2.0;
    double pjcAccA7MaxBasinSpread  = 0.5;
    // 4-14 CMTT（rowStep/depth 继承标定，不在此）
    double cmttTempHalfRange = 15.0;
    // clamped 档占比告警阈（超出仅 warn，不改 verdict；默认 5%）
    double cmttClampedWarnRatio = 0.05;
};

// 统一解析器：对一份 JSON 文档做 present-key 覆盖（主库与数据集 config 共用）
void applyLaserParamsJson(const nlohmann::json& j,
                          LaserCalibConfig& c, LaserOpParams& ops);

// ============================================================================
// CameraCalibHandoff —— 解析模块1 输出的 camera_calib.json
// 字段映射（见 module1/calib_io.cpp::buildCameraCalibJson + 各算子 toJson）:
//   cameraMatrixL/R ← intrinsic.left/right.camera_matrix
//   distCoeffsL/R   ← intrinsic.left/right.dist_coeffs
//   R, T            ← extrinsic.R, extrinsic.T
//   R1, R2, P1, P2, Q ← rectify.R1, R2, P1, P2, Q
//   imageSize        ← imageSize ([W, H])
//   referenceTemp, cte, tempRangeMin/Max/Step ← 顶层字段
//   rectifyTempTable ← stereoRectifyTempTable (可选整节, 缺节时 haveRectifyTempTable=false)
// ============================================================================
struct CameraCalibHandoff {
    cv::Mat cameraMatrixL, distCoeffsL;
    cv::Mat cameraMatrixR, distCoeffsR;
    cv::Mat R, T;
    cv::Mat R1, R2, P1, P2, Q;
    cv::Size imageSize;

    double referenceTemp = 25.0;
    double cte           = 23.6e-6;
    double tempRangeMin  = -15.0;
    double tempRangeMax  = 15.0;
    double tempStep      = 0.5;

    // JSON 顶层 schema 字段（诊断/版本校验用）
    std::string schema;

    // 模块1 立体矫正温度表整表〔CMTT 4-14 源①〕；缺节时 false
    calib::StereoRectifyTempTableResult rectifyTempTable;
    bool haveRectifyTempTable = false;

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
    LaserOpParams ops;
    CameraCalibHandoff handoff;
    std::vector<std::string> poseDirs;             // 姿态目录名（诊断用）
    std::vector<std::vector<PoseFrame>> poseFrames;  // [pose][tube]
};

// baseCfg/baseOps：上层（主库）合并结果；数据集 config.json 的键覆盖之（缺文件保留 base）
std::optional<LaserInput> loadLaserInput(const std::string& dir,
                                         const LaserCalibConfig* baseCfg = nullptr,
                                         const LaserOpParams* baseOps = nullptr);

// 注：模块1 侧同名功能叫 writeJson——本模块故意命名 writeLaserJson，
// 避免两模块 calib_io.obj 同符号（GUI 同时链接 fc1_io+fc2_io 时 LNK2005）。
bool writeLaserJson(const std::string& path, const nlohmann::json& j);

} // namespace fc
