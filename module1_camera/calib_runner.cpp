// =============================================================================
// calib_runner.cpp —— 把 camera_calib_cli.cpp 的 main() 主体抽出为分步函数
// =============================================================================

#include "calib_runner.h"
#include "chessboard_corner.h"

#include <spdlog/spdlog.h>

#include <sstream>
#include <utility>

namespace fc {

namespace {

// 默认 spdlog 输出 → cb.onLog；cb 为空时走 spdlog 默认 sink
void log(const CalibCallbacks& cb, const std::string& msg) {
    if (cb.onLog) cb.onLog(msg);
    else          spdlog::info("{}", msg);
}
void progress(const CalibCallbacks& cb, int step, int total, const std::string& msg) {
    if (cb.onProgress) cb.onProgress(step, total, msg);
}
bool cancelled(const CalibCallbacks& cb) {
    return cb.shouldCancel && cb.shouldCancel();
}

// ---- 把 CameraCalibConfig 转成各算子 params（来自 CLI 文件，保持等价） ----
calib::IntrinsicCalibParams makeIntrinParams(const CameraCalibConfig& c) {
    calib::IntrinsicCalibParams p;
    p.chessboard_width = c.chessWidth;
    p.chessboard_height = c.chessHeight;
    p.square_size_mm = c.squareSizeMm;
    p.image_width = c.imageWidth;
    p.image_height = c.imageHeight;
    p.use_calibrateCameraRO = c.useCalibrateCameraRO;
    p.calib_flags = c.intrinsicFlags;
    p.reproj_error_threshold = c.reprojErrorThreshold;
    p.temperature_coeff = c.plateTempCoeff;
    p.plate_temp = c.plateTemp;
    return p;
}

calib::ExtrinsicCalibCpuParams makeExtrinParams(const CameraCalibConfig& c,
    const std::vector<std::vector<cv::Point2f>>& lpts,
    const std::vector<std::vector<cv::Point2f>>& rpts)
{
    calib::ExtrinsicCalibCpuParams p;
    p.leftPointsPerView = lpts;
    p.rightPointsPerView = rpts;
    // 物点：与 IntrinsicCalibCPU::generateObjectPoints 同一约定（行主序 + 温度膨胀修正），
    // 保证内/外参尺度一致。validate() 要求 objectPoints 数 == 每视角角点数。
    double actualSize = c.squareSizeMm * (1.0 + c.plateTempCoeff * (c.plateTemp - 20.0));
    p.objectPoints.reserve(static_cast<size_t>(c.chessWidth) * c.chessHeight);
    for (int i = 0; i < c.chessHeight; ++i)
        for (int j = 0; j < c.chessWidth; ++j)
            p.objectPoints.emplace_back(
                static_cast<float>(j * actualSize),
                static_cast<float>(i * actualSize),
                0.0f);
    p.imageSize = cv::Size(c.imageWidth, c.imageHeight);
    p.patternSize = cv::Size(c.chessWidth, c.chessHeight);
    p.squareSize = static_cast<float>(c.squareSizeMm);
    p.maxReprojError = c.reprojErrorThreshold * 100.0;
    p.minViewCount = 4;
    return p;
}

}  // namespace

// ---------------------------------------------------------------------------
// Step 1: 角点提取
// ---------------------------------------------------------------------------
CornerExtractionResult extractCorners(const CameraInput& input,
                                       const CalibCallbacks& cb) {
    CornerExtractionResult out;
    out.totalFrames = static_cast<int>(input.frames.size());
    const auto& cfg = input.config;

    ChessboardCornerParams cp;
    cp.patternSize = cv::Size(cfg.chessWidth, cfg.chessHeight);

    log(cb, "[step 1/5] 提取棋盘角点...");
    for (size_t i = 0; i < input.frames.size(); ++i) {
        if (cancelled(cb)) { out.message = "cancelled"; return out; }
        ChessboardCornerResult rl, rr;
        if (!extractChessboardCorners(input.frames[i].leftGray, cp, rl) ||
            !extractChessboardCorners(input.frames[i].rightGray, cp, rr))
        {
            std::ostringstream os;
            os << "[warn] frame " << i << ": corner extraction failed, skip";
            log(cb, os.str());
            continue;
        }
        normalizeLRCornerOrder(rl, rr);
        out.leftPoints.push_back(std::move(rl.corners));
        out.rightPoints.push_back(std::move(rr.corners));
    }
    out.validFrames = static_cast<int>(out.leftPoints.size());

    if (out.validFrames < 4) {
        std::ostringstream os;
        os << "[error] too few valid frames: " << out.validFrames;
        log(cb, os.str());
        out.message = "too few valid frames";
        return out;
    }
    std::ostringstream os;
    os << "[step 1/5] OK, valid frames = " << out.validFrames
       << " / " << out.totalFrames;
    log(cb, os.str());
    progress(cb, 1, 5, os.str());
    out.success = true;
    return out;
}

// ---------------------------------------------------------------------------
// Step 2: 内参
// ---------------------------------------------------------------------------
calib::IntrinsicCalibResult calibrateIntrinsic(
    const CameraCalibConfig& cfg,
    const CornerExtractionResult& corners,
    const CalibCallbacks& cb)
{
    log(cb, "[step 2/5] 内参标定...");
    calib::IntrinsicCalibCPU intrin(makeIntrinParams(cfg));
    calib::IntrinsicCalibResult res;
    if (!intrin.Execute(corners.leftPoints, corners.rightPoints, res) || !res.success) {
        std::ostringstream os;
        os << "[error] intrinsic failed: " << res.message;
        log(cb, os.str());
        return res;
    }
    std::ostringstream os;
    os << "[step 2/5] OK, reproj_mean=" << res.reproj_error_mean;
    log(cb, os.str());
    progress(cb, 2, 5, os.str());
    return res;
}

// ---------------------------------------------------------------------------
// Step 3: 外参
// ---------------------------------------------------------------------------
calib::ExtrinsicCalibCpuResult calibrateExtrinsic(
    const CameraCalibConfig& cfg,
    const CornerExtractionResult& corners,
    const calib::IntrinsicCalibResult& intrin,
    const CalibCallbacks& cb)
{
    log(cb, "[step 3/5] 外参标定...");
    auto ep = makeExtrinParams(cfg, corners.leftPoints, corners.rightPoints);
    calib::ExtrinsicCalibCpu extrin(ep);
    auto res = extrin.Execute(
        intrin.left.camera_matrix, intrin.left.dist_coeffs,
        intrin.right.camera_matrix, intrin.right.dist_coeffs);
    if (!res.success) {
        std::ostringstream os;
        os << "[error] extrinsic failed: " << res.message;
        log(cb, os.str());
        return res;
    }
    std::ostringstream os;
    os << "[step 3/5] OK, stereo_rms=" << res.stereoReprojError
       << " epipolar_mean=" << res.epipolarErrorMean;
    log(cb, os.str());
    progress(cb, 3, 5, os.str());
    return res;
}

// ---------------------------------------------------------------------------
// Step 4: 立体矫正
// ---------------------------------------------------------------------------
calib::StereoRectifyCpuResult stereoRectify(
    const CameraCalibConfig& cfg,
    const calib::IntrinsicCalibResult& intrin,
    const calib::ExtrinsicCalibCpuResult& extrin,
    const CalibCallbacks& cb)
{
    log(cb, "[step 4/5] 立体矫正...");
    calib::StereoRectifyCpuParams rp;
    rp.cameraMatrixL = intrin.left.camera_matrix;
    rp.distCoeffsL   = intrin.left.dist_coeffs;
    rp.cameraMatrixR = intrin.right.camera_matrix;
    rp.distCoeffsR   = intrin.right.dist_coeffs;
    rp.imageSize     = cv::Size(cfg.imageWidth, cfg.imageHeight);
    rp.R = extrin.R; rp.T = extrin.T;
    rp.alpha = cfg.rectifyAlpha; rp.flags = cfg.rectifyFlags;
    calib::StereoRectifyCpu rectify(rp);
    auto res = rectify.Execute();
    if (!res.success) {
        std::ostringstream os;
        os << "[error] rectify failed: " << res.message;
        log(cb, os.str());
        return res;
    }
    log(cb, "[step 4/5] OK");
    progress(cb, 4, 5, "rectify OK");
    return res;
}

// ---------------------------------------------------------------------------
// Step 5: 温度表
// ---------------------------------------------------------------------------
TempTableSet buildTempTables(
    const CameraCalibConfig& cfg,
    const calib::IntrinsicCalibResult& intrin,
    const calib::ExtrinsicCalibCpuResult& extrin,
    const CalibCallbacks& cb)
{
    TempTableSet out;
    log(cb, "[step 5/5] 温度补偿表...");

    calib::CameraIntrinsics cL{intrin.left.camera_matrix.at<double>(0,0),
                               intrin.left.camera_matrix.at<double>(1,1),
                               intrin.left.camera_matrix.at<double>(0,2),
                               intrin.left.camera_matrix.at<double>(1,2),
                               cfg.referenceTemp};
    calib::CameraIntrinsics cR{intrin.right.camera_matrix.at<double>(0,0),
                               intrin.right.camera_matrix.at<double>(1,1),
                               intrin.right.camera_matrix.at<double>(0,2),
                               intrin.right.camera_matrix.at<double>(1,2),
                               cfg.referenceTemp};
    calib::IntrinsicCompensateCPUParams icp;
    icp.cte = cfg.cte; icp.tempStep = cfg.tempStep;
    icp.tempRangeMin = cfg.tempRangeMin; icp.tempRangeMax = cfg.tempRangeMax;
    calib::IntrinsicCompensateCPU icomp(icp);
    out.intrinL = icomp.Execute(cL);
    out.intrinR = icomp.Execute(cR);

    calib::CameraExtrinsics ce; ce.referenceTemp = cfg.referenceTemp;
    for (int i = 0; i < 3; ++i) ce.T[i] = extrin.T.at<double>(i);
    for (int i = 0; i < 9; ++i) ce.R[i] = extrin.R.at<double>(i/3, i%3);
    calib::ExtrinsicCompensateCPUParams ecp;
    ecp.cte = cfg.cte; ecp.tempStep = cfg.tempStep;
    ecp.tempRangeMin = cfg.tempRangeMin; ecp.tempRangeMax = cfg.tempRangeMax;
    calib::ExtrinsicCompensateCPU ecomp(ecp);
    out.extrin = ecomp.Execute(ce);

    calib::StereoRectifyTempTableParams strp;
    strp.cameraMatrixL = intrin.left.camera_matrix;
    strp.distCoeffsL   = intrin.left.dist_coeffs;
    strp.cameraMatrixR = intrin.right.camera_matrix;
    strp.distCoeffsR   = intrin.right.dist_coeffs;
    strp.imageSize     = cv::Size(cfg.imageWidth, cfg.imageHeight);
    strp.R = extrin.R; strp.T = extrin.T;
    strp.referenceTemp = cfg.referenceTemp; strp.cte = cfg.cte;
    strp.tempStep = cfg.tempStep;
    strp.tempRangeMin = cfg.tempRangeMin; strp.tempRangeMax = cfg.tempRangeMax;
    strp.alpha = cfg.rectifyAlpha; strp.flags = cfg.rectifyFlags;
    calib::StereoRectifyTempTableCpu strtab(strp);
    out.rectify = strtab.Execute();

    log(cb, "[step 5/5] OK");
    progress(cb, 5, 5, "temp tables OK");
    out.success = true;
    return out;
}

// ---------------------------------------------------------------------------
// 全流程
// ---------------------------------------------------------------------------
bool runCameraCalib(const std::string& inputDir,
                    const std::string& outputPath,
                    const CalibCallbacks& cb) {
    log(cb, std::string("--- 加载输入: ") + inputDir);
    auto input = loadCameraInput(inputDir);
    if (!input) { log(cb, "[error] load input failed"); return false; }
    const auto& cfg = input->config;

    auto corners = extractCorners(*input, cb);
    if (!corners.success) return false;

    auto intrin = calibrateIntrinsic(cfg, corners, cb);
    if (!intrin.success) return false;

    auto extrin = calibrateExtrinsic(cfg, corners, intrin, cb);
    if (!extrin.success) return false;

    auto rect = stereoRectify(cfg, intrin, extrin, cb);
    if (!rect.success) return false;

    auto tabs = buildTempTables(cfg, intrin, extrin, cb);
    if (!tabs.success) return false;

    auto j = buildCameraCalibJson(cfg, intrin, extrin, rect,
                                  tabs.intrinL, tabs.intrinR,
                                  tabs.extrin, tabs.rectify);
    if (!writeJson(outputPath, j)) {
        log(cb, std::string("[error] write json failed: ") + outputPath);
        return false;
    }
    log(cb, std::string("--- camera_calib done -> ") + outputPath);

    std::string processPath = deriveProcessPath(outputPath);
    if (writeJson(processPath, buildCameraCalibProcessJson(cfg, intrin, extrin))) {
        log(cb, std::string("--- process data -> ") + processPath);
    } else {
        log(cb, std::string("[warn] process json write failed: ") + processPath);
    }
    return true;
}

}  // namespace fc
