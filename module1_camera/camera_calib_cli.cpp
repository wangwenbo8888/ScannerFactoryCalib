#include "calib_io.h"
#include "chessboard_corner.h"

#include "intrinsic_calib_cpu.h"
#include "extrinsic_calib_cpu.h"
#include "stereo_rectify_cpu.h"
#include "intrinsic_compensate_cpu.h"
#include "extrinsic_compensate_cpu.h"
#include "stereo_rectify_temp_table_cpu.h"

#include <spdlog/spdlog.h>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace fc;
using namespace calib;

namespace {

IntrinsicCalibParams makeIntrinParams(const CameraCalibConfig& c) {
    IntrinsicCalibParams p;
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

// 注意: ExtrinsicCalibCpuParams 不持有 cameraMatrixL/distCoeffsL/R —— 这些通过
// Execute(KL,DL,KR,DR) 重载在调用点直接传入（见 step 3）。本函数只填观测点+板参数。
ExtrinsicCalibCpuParams makeExtrinParams(const CameraCalibConfig& c,
    const std::vector<std::vector<cv::Point2f>>& lpts,
    const std::vector<std::vector<cv::Point2f>>& rpts)
{
    ExtrinsicCalibCpuParams p;
    p.leftPointsPerView = lpts;
    p.rightPointsPerView = rpts;
    p.imageSize = cv::Size(c.imageWidth, c.imageHeight);
    p.patternSize = cv::Size(c.chessWidth, c.chessHeight);
    p.squareSize = static_cast<float>(c.squareSizeMm);
    p.maxReprojError = c.reprojErrorThreshold * 100.0;
    p.minViewCount = 4;
    return p;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: camera_calib <input_dir> [output_json]\n"
                  << "  input_dir 含 config.json + left/ + right/ + temps.txt\n";
        return 2;
    }
    std::string inDir = argv[1];
    std::string outPath = argc >= 3 ? argv[2] : "camera_calib.json";

    auto input = loadCameraInput(inDir);
    if (!input) { spdlog::error("load input failed"); return 1; }
    const auto& cfg = input->config;

    // 1. 逐帧提取棋盘角点
    ChessboardCornerParams cp;
    cp.patternSize = cv::Size(cfg.chessWidth, cfg.chessHeight);
    std::vector<std::vector<cv::Point2f>> lpts, rpts;
    for (size_t i = 0; i < input->frames.size(); ++i) {
        ChessboardCornerResult rl, rr;
        if (!extractChessboardCorners(input->frames[i].leftGray, cp, rl) ||
            !extractChessboardCorners(input->frames[i].rightGray, cp, rr))
        {
            spdlog::warn("frame {}: corner extraction failed, skip", i);
            continue;
        }
        normalizeLRCornerOrder(rl, rr);
        lpts.push_back(std::move(rl.corners));
        rpts.push_back(std::move(rr.corners));
    }
    if (lpts.size() < 4) { spdlog::error("too few valid frames: {}", lpts.size()); return 1; }

    // 2. 内参
    IntrinsicCalibCPU intrin(makeIntrinParams(cfg));
    IntrinsicCalibResult intrinRes;
    if (!intrin.Execute(lpts, rpts, intrinRes) || !intrinRes.success) {
        spdlog::error("intrinsic calib failed: {}", intrinRes.message); return 1;
    }
    spdlog::info("intrinsic OK, reproj_mean={}", intrinRes.reproj_error_mean);

    // 3. 外参
    ExtrinsicCalibCpuParams ep = makeExtrinParams(cfg, lpts, rpts);
    ExtrinsicCalibCpu extrin(ep);
    ExtrinsicCalibCpuResult extrinRes = extrin.Execute(
        intrinRes.left.camera_matrix, intrinRes.left.dist_coeffs,
        intrinRes.right.camera_matrix, intrinRes.right.dist_coeffs);
    if (!extrinRes.success) { spdlog::error("extrinsic failed: {}", extrinRes.message); return 1; }

    // 4. 立体矫正
    StereoRectifyCpuParams rp;
    rp.cameraMatrixL = intrinRes.left.camera_matrix;
    rp.distCoeffsL   = intrinRes.left.dist_coeffs;
    rp.cameraMatrixR = intrinRes.right.camera_matrix;
    rp.distCoeffsR   = intrinRes.right.dist_coeffs;
    rp.imageSize     = cv::Size(cfg.imageWidth, cfg.imageHeight);
    rp.R = extrinRes.R; rp.T = extrinRes.T;
    rp.alpha = cfg.rectifyAlpha; rp.flags = cfg.rectifyFlags;
    StereoRectifyCpu rectify(rp);
    StereoRectifyCpuResult rectifyRes = rectify.Execute();
    if (!rectifyRes.success) { spdlog::error("rectify failed: {}", rectifyRes.message); return 1; }

    // 5. 三张温度表
    CameraIntrinsics cL{intrinRes.left.camera_matrix.at<double>(0,0),
                        intrinRes.left.camera_matrix.at<double>(1,1),
                        intrinRes.left.camera_matrix.at<double>(0,2),
                        intrinRes.left.camera_matrix.at<double>(1,2),
                        cfg.referenceTemp};
    CameraIntrinsics cR{intrinRes.right.camera_matrix.at<double>(0,0),
                        intrinRes.right.camera_matrix.at<double>(1,1),
                        intrinRes.right.camera_matrix.at<double>(0,2),
                        intrinRes.right.camera_matrix.at<double>(1,2),
                        cfg.referenceTemp};
    IntrinsicCompensateCPUParams icp; icp.cte=cfg.cte; icp.tempStep=cfg.tempStep;
    icp.tempRangeMin=cfg.tempRangeMin; icp.tempRangeMax=cfg.tempRangeMax;
    IntrinsicCompensateCPU icomp(icp);
    auto tableL = icomp.Execute(cL);
    auto tableR = icomp.Execute(cR);

    CameraExtrinsics ce; ce.referenceTemp = cfg.referenceTemp;
    for (int i=0;i<3;++i) ce.T[i]=extrinRes.T.at<double>(i);
    for (int i=0;i<9;++i) ce.R[i]=extrinRes.R.at<double>(i/3,i%3);
    ExtrinsicCompensateCPUParams ecp; ecp.cte=cfg.cte; ecp.tempStep=cfg.tempStep;
    ecp.tempRangeMin=cfg.tempRangeMin; ecp.tempRangeMax=cfg.tempRangeMax;
    ExtrinsicCompensateCPU ecomp(ecp);
    auto tableE = ecomp.Execute(ce);

    StereoRectifyTempTableParams strp;
    strp.cameraMatrixL=rp.cameraMatrixL; strp.distCoeffsL=rp.distCoeffsL;
    strp.cameraMatrixR=rp.cameraMatrixR; strp.distCoeffsR=rp.distCoeffsR;
    strp.imageSize=rp.imageSize; strp.R=rp.R; strp.T=rp.T;
    strp.referenceTemp=cfg.referenceTemp; strp.cte=cfg.cte;
    strp.tempStep=cfg.tempStep; strp.tempRangeMin=cfg.tempRangeMin; strp.tempRangeMax=cfg.tempRangeMax;
    strp.alpha=cfg.rectifyAlpha; strp.flags=cfg.rectifyFlags;
    StereoRectifyTempTableCpu strtab(strp);
    auto tableR2 = strtab.Execute();

    // 6. 写交接文件
    auto j = buildCameraCalibJson(cfg, intrinRes, extrinRes, rectifyRes,
                                  tableL, tableR, tableE, tableR2);
    if (!writeJson(outPath, j)) return 1;
    spdlog::info("camera_calib done -> {}", outPath);
    return 0;
}
