// fcstepdump.cpp — 逐帧 3D 重建 + 全步骤可视化工具
//
// 输入: laser 数据目录（config.json + camera_calib.json + pose_*/L_tube*.png + R_tube*.png）
// 输出: <output_dir>/pose_XX_tubeN_{L,R}_{1mask,2ccl,3label,4steger,6interp}.png
//                 pose_XX_tubeN_7match.png / pose_XX_tubeN_8cloud.ply / summary.json
// 设计: docs/plans/2026-08-25-fcstepdump-design.md
// 链路: 4-1..4-8（止于重建，无 PJC）；参数照搬 laser_calib_cli.cpp 实测调优值

#include "calib_io.h"

#include "mask_extract_cuda.h"
#include "region_analyze_cuda.h"
#include "laser_label_cuda.h"
#include "steger_extract_cuda.h"
#include "undistort_points_cuda.h"
#include "epipolar_interp_cuda.h"
#include "laser_match_cuda.h"
#include "laser_reconstruct_cuda.h"
#include "projector_joint_calib.h"
#include "plane_map_cuda.h"
#include "curve_map.h"
#include "cmtt_container.h"
#include "curve_map_temp_table.h"
#include "laser_extrinsic_compensate_cpu.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>
using namespace fc;
using namespace calib;

namespace {

cv::Scalar lineColor(int id) {
    const int h = ((id * 137) % 360 + 360) % 360;
    const int s = 200, v = 255;
    const int hi = h / 60, f = h - hi * 60;
    const int p = v * (255 - s) / 255;
    const int q = v * (255 - s * f / 60) / 255;
    const int t = v * (255 - s * (60 - f) / 60) / 255;
    switch (hi % 6) {
        case 0: return cv::Scalar(v, t, p);
        case 1: return cv::Scalar(q, v, p);
        case 2: return cv::Scalar(p, v, t);
        case 3: return cv::Scalar(p, q, v);
        case 4: return cv::Scalar(t, p, v);
        default: return cv::Scalar(v, p, q);
    }
}

void drawIdText(cv::Mat& img, int id, const cv::Point& org) {
    const std::string text = std::to_string(id);
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

// 每线取最上点为锚点标注线号（点集版：4steger/6interp/7match）
void annotateLineIds(cv::Mat& img, const cv::Mat& pts, const cv::Mat& ids,
                     int xOff = 0) {
    std::map<int, cv::Point2f> anchor;
    const cv::Mat p2 = pts.reshape(2, 1), i2 = ids.reshape(1, 1);
    const int n = static_cast<int>(p2.cols);
    const cv::Point2f* pp = p2.ptr<cv::Point2f>(0);
    const int* ip = i2.ptr<int>(0);
    for (int k = 0; k < n; ++k) {
        auto it = anchor.find(ip[k]);
        if (it == anchor.end()) anchor[ip[k]] = pp[k];
        else if (pp[k].y < it->second.y) it->second = pp[k];
    }
    for (const auto& kv : anchor) {
        cv::Point org(cvRound(kv.second.x) + xOff, cvRound(kv.second.y) - 6);
        if (org.x > 8) org.x -= 8;
        if (org.y < 16) org.y = 16;
        drawIdText(img, kv.first, org);
    }
}

// 标签图版（3label）：逐标签最上像素为锚点
void annotateMaskIds(cv::Mat& img, const cv::Mat& labels) {
    std::map<int, cv::Point> anchor;
    for (int y = 0; y < labels.rows; ++y) {
        const int* lr = labels.ptr<int>(y);
        for (int x = 0; x < labels.cols; ++x) {
            const int id = lr[x];
            if (id > 0 && anchor.find(id) == anchor.end())
                anchor[id] = cv::Point(x, y);
        }
    }
    for (const auto& kv : anchor) {
        cv::Point org = kv.second;
        if (org.x > 8) org.x -= 8;
        if (org.y < 16) org.y = 16;
        drawIdText(img, kv.first, org);
    }
}

cv::Mat colorizeIds(const cv::Mat& labels) {
    cv::Mat out(labels.rows, labels.cols, CV_8UC3, cv::Scalar(20, 20, 20));
    for (int y = 0; y < labels.rows; ++y) {
        const int* lr = labels.ptr<int>(y);
        cv::Vec3b* orow = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < labels.cols; ++x) {
            if (lr[x] > 0) {
                const cv::Scalar c = lineColor(lr[x]);
                orow[x] = cv::Vec3b(
                    static_cast<uchar>(c[0]), static_cast<uchar>(c[1]), static_cast<uchar>(c[2]));
            }
        }
    }
    return out;
}

cv::Mat drawPointsOn(const cv::Mat& base, const cv::Mat& pts, const cv::Mat& ids) {
    cv::Mat out = base.clone();
    if (base.channels() == 1) cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
    const cv::Mat p2 = pts.reshape(2, 1), i2 = ids.reshape(1, 1);
    const int n = static_cast<int>(p2.cols);
    const cv::Point2f* pp = p2.ptr<cv::Point2f>(0);
    const int* ip = i2.ptr<int>(0);
    for (int k = 0; k < n; ++k) {
        const float x = pp[k].x, y = pp[k].y;
        if (x < 0 || y < 0 || x >= out.cols || y >= out.rows) continue;
        cv::circle(out, cv::Point(cvRound(x), cvRound(y)), 1, lineColor(ip[k]), -1);
    }
    return out;
}

cv::Mat drawInterpBlack(const cv::Size& size, const cv::Mat& pts, const cv::Mat& ids) {
    cv::Mat out(size, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat p2 = pts.reshape(2, 1), i2 = ids.reshape(1, 1);
    const int n = static_cast<int>(p2.cols);
    const cv::Point2f* pp = p2.ptr<cv::Point2f>(0);
    const int* ip = i2.ptr<int>(0);
    for (int k = 0; k < n; ++k) {
        const float x = pp[k].x, y = pp[k].y;
        if (x < 0 || y < 0 || x >= size.width || y >= size.height) continue;
        cv::circle(out, cv::Point(cvRound(x), cvRound(y)), 1, lineColor(ip[k]), -1);
    }
    return out;
}

cv::Mat drawMatchPanel(const cv::Size& size,
                       const cv::Mat& lPts, const cv::Mat& lIds,
                       const cv::Mat& rPts, const cv::Mat& rIds,
                       const cv::Mat& mL, const cv::Mat& mR, const cv::Mat& mIds) {
    cv::Mat out(size.height, size.width * 2, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat lp = lPts.reshape(2, 1), li = lIds.reshape(1, 1);
    const cv::Mat rp = rPts.reshape(2, 1), ri = rIds.reshape(1, 1);
    const int ln = static_cast<int>(lp.cols), rn = static_cast<int>(rp.cols);
    const cv::Point2f* lpp = lp.ptr<cv::Point2f>(0); const int* lip = li.ptr<int>(0);
    const cv::Point2f* rpp = rp.ptr<cv::Point2f>(0); const int* rip = ri.ptr<int>(0);
    for (int k = 0; k < ln; ++k) {
        const float x = lpp[k].x, y = lpp[k].y;
        if (x >= 0 && y >= 0 && x < size.width && y < size.height)
            cv::circle(out, cv::Point(cvRound(x), cvRound(y)), 1,
                       cv::Scalar(90, 90, 90), -1);
    }
    for (int k = 0; k < rn; ++k) {
        const float x = rpp[k].x, y = rpp[k].y;
        if (x >= 0 && y >= 0 && x < size.width && y < size.height)
            cv::circle(out, cv::Point(cvRound(x) + size.width, cvRound(y)), 1,
                       cv::Scalar(90, 90, 90), -1);
    }
    const cv::Mat ml = mL.reshape(2, 1), mr = mR.reshape(2, 1), mi = mIds.reshape(1, 1);
    const int mn = static_cast<int>(mi.cols);
    const cv::Point2f* mlp = ml.ptr<cv::Point2f>(0);
    const cv::Point2f* mrp = mr.ptr<cv::Point2f>(0);
    const int* mip = mi.ptr<int>(0);
    // 按线抽样连线（每线最多 200 条）
    std::map<int, std::vector<int>> byLine;
    for (int k = 0; k < mn; ++k) byLine[mip[k]].push_back(k);
    for (const auto& kv : byLine) {
        const std::vector<int>& idx = kv.second;
        const int stride = std::max(1, static_cast<int>(idx.size()) / 200);
        for (size_t j = 0; j < idx.size(); j += static_cast<size_t>(stride)) {
            const cv::Point2f& a = mlp[idx[j]];
            const cv::Point2f& b = mrp[idx[j]];
            cv::line(out,
                     cv::Point(cvRound(a.x), cvRound(a.y)),
                     cv::Point(cvRound(b.x) + size.width, cvRound(b.y)),
                     lineColor(kv.first), 1, cv::LINE_AA);
        }
    }
    return out;
}

#pragma pack(push, 1)
struct PlyVertex {
    float x, y, z;
    uint8_t r, g, b;
};
#pragma pack(pop)

// 读回本工具写出的二进制 PLY（XYZ+RGB），颜色→线号
int colorToLineId(const PlyVertex& v) {
    // lineColor: HSV 色环 id*137°; 反解: hue → id（就近取整, mod 360）
    const double maxc = std::max({v.r, v.g, v.b}) + 0.0;
    const double minc = std::min({v.r, v.g, v.b}) + 0.0;
    if (maxc <= 0) return 0;
    const double d = maxc - minc;
    double hue;
    if (d < 1e-9) hue = 0;
    else if (maxc == v.r) hue = 60.0 * (v.g - v.b) / d;
    else if (maxc == v.g) hue = 60.0 * (2.0 + (v.b - v.r) / d);
    else hue = 60.0 * (4.0 + (v.r - v.g) / d);
    if (hue < 0) hue += 360.0;
    double bestErr = 1e9; int bestId = 0;
    for (int id = 1; id <= 360; ++id) {
        const double h = ((id * 137) % 360 + 360) % 360;
        const double err = std::abs(h - hue);
        const double errw = std::min(err, 360.0 - err);
        if (errw < bestErr) { bestErr = errw; bestId = id; }
    }
    return bestId;
}

bool readPly(const std::string& path, std::vector<cv::Vec3f>& pts,
             std::vector<int>& ids) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string line;
    int n = -1;
    bool ascii = false;
    while (std::getline(f, line)) {
        if (line.rfind("format ascii", 0) == 0) ascii = true;
        if (line.rfind("element vertex", 0) == 0)
            n = std::stoi(line.substr(14));
        if (line.rfind("end_header", 0) == 0) break;
    }
    if (n < 0) return false;
    pts.clear(); ids.clear();
    pts.reserve(static_cast<size_t>(n));
    if (ascii) {
        for (int k = 0; k < n; ++k) {
            float x, y, z; int r, g, b;
            if (!(f >> x >> y >> z >> r >> g >> b)) return false;
            PlyVertex v{x, y, z,
                        static_cast<uint8_t>(r), static_cast<uint8_t>(g),
                        static_cast<uint8_t>(b)};
            pts.push_back(cv::Vec3f(x, y, z));
            ids.push_back(colorToLineId(v));
        }
        return true;
    }
    for (int k = 0; k < n; ++k) {
        PlyVertex v;
        f.read(reinterpret_cast<char*>(&v), sizeof(v));
        if (!f) return false;
        pts.push_back(cv::Vec3f(v.x, v.y, v.z));
        ids.push_back(colorToLineId(v));
    }
    return true;
}

// 逐姿态几何分析: 平面拟合(SVD) + 法向 + 奇异值比 + bbox（诊断 PJC 观测退化）
int analyzePlyDir(const std::string& plyDir) {
    spdlog::info("=== fcstepdump --analyze-ply ===");
    std::vector<std::string> files;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(plyDir, ec)) {
        const std::string name = entry.path().filename().string();
        if (name.size() > 11 && name.substr(name.size() - 11) == "_8cloud.ply")
            files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) { spdlog::error("no PLY under {}", plyDir); return 1; }

    std::vector<cv::Vec3d> normals;
    for (const auto& fp : files) {
        std::vector<cv::Vec3f> pts;
        std::vector<int> ids;
        if (!readPly(fp, pts, ids)) { spdlog::warn("read {} failed", fp); continue; }
        const size_t n = pts.size();
        cv::Vec3d c(0, 0, 0);
        for (const auto& p : pts) c += cv::Vec3d(p[0], p[1], p[2]);
        c *= 1.0 / static_cast<double>(n);
        cv::Matx33d cov = cv::Matx33d::zeros();
        double zmin = 1e30, zmax = -1e30, xmin = 1e30, xmax = -1e30,
               ymin = 1e30, ymax = -1e30;
        for (const auto& p : pts) {
            const cv::Vec3d d(p[0] - c[0], p[1] - c[1], p[2] - c[2]);
            cov += d * d.t();
            zmin = std::min(zmin, (double)p[2]); zmax = std::max(zmax, (double)p[2]);
            xmin = std::min(xmin, (double)p[0]); xmax = std::max(xmax, (double)p[0]);
            ymin = std::min(ymin, (double)p[1]); ymax = std::max(ymax, (double)p[1]);
        }
        cv::Matx33d ev; cv::Vec3d w;
        cv::eigen(cov, w, ev);          // 降序: w[0]>=w[1]>=w[2]
        const cv::Vec3d nrm(ev(0, 2), ev(1, 2), ev(2, 2));   // 最小特征向量
        normals.push_back(nrm);
        spdlog::info("{}: n={} bbox=[{:.0f}..{:.0f}, {:.0f}..{:.0f}, {:.0f}..{:.0f}] "
                     "sv=({:.2f}, {:.2f}, {:.3f}) n=({:.4f},{:.4f},{:.4f})",
                     std::filesystem::path(fp).filename().string(), n,
                     xmin, xmax, ymin, ymax, zmin, zmax,
                     std::sqrt(w[0]), std::sqrt(w[1]), std::sqrt(w[2]),
                     nrm[0], nrm[1], nrm[2]);
    }
    // 法向多样性: 两两夹角
    double minAng = 180, maxAng = 0, sumAng = 0; int cnt = 0;
    for (size_t i = 0; i < normals.size(); ++i)
        for (size_t j = i + 1; j < normals.size(); ++j) {
            double d = std::abs(normals[i].dot(normals[j]));
            d = std::min(1.0, std::max(-1.0, d));
            const double ang = std::acos(d) * 180.0 / CV_PI;
            minAng = std::min(minAng, ang); maxAng = std::max(maxAng, ang);
            sumAng += ang; ++cnt;
        }
    if (cnt > 0)
        spdlog::info("plane-normal diversity: min={:.2f}° mean={:.2f}° max={:.2f}° "
                     "over {} poses", minAng, sumAng / cnt, maxAng, cnt);
    return 0;
}

void writePly(const std::string& path, const cv::Mat& pts3d, const cv::Mat& ids) {
    const cv::Mat p = pts3d.reshape(3, 1), i2 = ids.reshape(1, 1);
    const int n = static_cast<int>(p.cols);
    const cv::Vec3f* pv = p.ptr<cv::Vec3f>(0);
    const int* ip = i2.ptr<int>(0);
    std::ofstream f(path, std::ios::binary);
    f << "ply\nformat binary_little_endian 1.0\n"
      << "element vertex " << n << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
      << "end_header\n";
    for (int k = 0; k < n; ++k) {
        const cv::Scalar c = lineColor(ip[k]);
        PlyVertex v{pv[k][0], pv[k][1], pv[k][2],
                    static_cast<uint8_t>(c[0]), static_cast<uint8_t>(c[1]),
                    static_cast<uint8_t>(c[2])};
        f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }
}

struct FrameRecord {
    std::string pose;
    int tube = 0;
    bool ok = false;
    std::string failStep;
    int matchCount = 0;
    long long pointCount = 0;
    std::vector<int> lineIds;
};

} // namespace

// 执行 PJC 并返回 json（供两条路径复用）。失败时 json 为 null。
nlohmann::json runPjc(const std::vector<calib::PosePointSet>& poseSetsIn,
                      const fc::CameraCalibHandoff& h) {
    nlohmann::json out;
    std::vector<calib::PosePointSet> poseSets;
    for (auto& ps : poseSetsIn)
        if (!ps.points3d.empty()) poseSets.push_back(ps);
    if (poseSets.empty()) {
        spdlog::warn("no accumulated 3D points; skip PJC");
        return out;
    }
    ProjectorJointCalibInput in;
    in.poses = poseSets;
    in.f = h.P1.at<double>(0, 0);
    in.principalPoint = cv::Point2d(h.P1.at<double>(0, 2), h.P1.at<double>(1, 2));
    in.initialT = cv::Vec3d(80.0, 3.0, 3.0);

    ProjectorJointCalib op;
    auto r = op.Execute(in);
    if (!r.success) {
        spdlog::error("PJC failed: {}", r.message);
        return out;
    }
    spdlog::info("PJC OK: projectorT=({:.3f},{:.3f},{:.3f}) mm, "
                 "sampsonRms {}->{:.4f} (improve x{:.1f}), poses={}, pts={}, "
                 "cond={:.3g}, curve pts={}",
                 r.projectorT[0], r.projectorT[1], r.projectorT[2],
                 r.initialSampsonRms, r.finalSampsonRms,
                 r.improvementRatio, r.poseCount, r.totalPointCount,
                 r.jacobianConditionNumber, r.emissionCurve.pointCount);
    out["success"] = true;
    out["qualityFlag"] = static_cast<int>(r.qualityFlag);
    out["message"] = r.message;
    out["projectorT"] = {r.projectorT[0], r.projectorT[1], r.projectorT[2]};
    out["f"] = in.f;
    out["principalPoint"] = {in.principalPoint.x, in.principalPoint.y};
    out["initialSampsonRms"] = r.initialSampsonRms;
    out["finalSampsonRms"] = r.finalSampsonRms;
    out["improvementRatio"] = r.improvementRatio;
    out["poseCount"] = r.poseCount;
    out["totalPointCount"] = r.totalPointCount;
    out["jacobianConditionNumber"] = r.jacobianConditionNumber;
    out["curvePointCount"] = r.emissionCurve.pointCount;
    return out;
}

// 从 PLY 目录跑 PJC: 每 pose_*_tubeN_8cloud.ply 为一组
int runPjcFromPly(const std::string& plyDir, const std::string& outJson) {
    spdlog::info("=== fcstepdump --pjc-from-ply ===");
    // f/主点来源: plyDir/camera_calib.json 或 plyDir/../camera_calib.json
    std::string calibPath;
    const std::string candidates[] = {plyDir + "/camera_calib.json",
                                      plyDir + "/../camera_calib.json",
                                      "data_in/left_skew/camera_calib.json"};
    for (const std::string& cand : candidates) {
        if (std::filesystem::exists(cand)) { calibPath = cand; break; }
    }
    if (calibPath.empty()) {
        spdlog::error("camera_calib.json not found near {}", plyDir);
        return 1;
    }
    auto h = fc::loadCameraCalibHandoff(calibPath);
    if (!h) {
        spdlog::error("load handoff failed: {}", calibPath);
        return 1;
    }
    spdlog::info("handoff: {}", calibPath);

    // 按 pose 名分组读 PLY
    std::map<std::string, calib::PosePointSet> groups;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(plyDir, ec)) {
        const std::string name = entry.path().filename().string();
        if (name.size() < 15 || name.substr(name.size() - 11) != "_8cloud.ply") continue;
        const std::string stem = name.substr(0, name.size() - 11);   // pose_XX_tubeN
        std::vector<cv::Vec3f> pts;
        std::vector<int> ids;
        if (!readPly(entry.path().string(), pts, ids)) {
            spdlog::warn("read {} failed, skip", name);
            continue;
        }
        auto& ps = groups[stem];
        ps.points3d.insert(ps.points3d.end(), pts.begin(), pts.end());
        ps.lineIds.insert(ps.lineIds.end(), ids.begin(), ids.end());
        spdlog::info("{}: {} points", stem, pts.size());
    }
    if (groups.empty()) {
        spdlog::error("no *_8cloud.ply under {}", plyDir);
        return 1;
    }
    std::vector<calib::PosePointSet> poseSets;
    poseSets.reserve(groups.size());
    for (auto& kv : groups) poseSets.push_back(std::move(kv.second));

    auto j = runPjc(poseSets, *h);
    nlohmann::json summary;
    summary["schema"] = "fcstepdump-pjc-from-ply/1.0";
    summary["plyDir"] = plyDir;
    summary["poseCount"] = poseSets.size();
    summary["pjc"] = j.is_null() ? nlohmann::json{{"success", false}} : j;
    if (!fc::writeLaserJson(outJson, summary)) {
        spdlog::error("write {} failed", outJson);
        return 1;
    }
    return j.is_null() ? 1 : 0;
}

// 按 lineId 分组逐线跑 PJC: 每条激光线（曲率扇面）跨全部姿态单独求解,
// 报告各线 t/rms/cond 与跨线一致性（PJC 模型按单曲线设计, 混线输入必崩）
int runPjcPerLine(const std::string& plyDir, const std::string& outJson,
                  int lineFilter = -1) {
    spdlog::info("=== fcstepdump --pjc-per-line ===");
    std::string calibPath;
    const std::string candidates[] = {plyDir + "/camera_calib.json",
                                      plyDir + "/../camera_calib.json",
                                      "data_in/left_skew/camera_calib.json"};
    for (const std::string& cand : candidates) {
        if (std::filesystem::exists(cand)) { calibPath = cand; break; }
    }
    if (calibPath.empty()) {
        spdlog::error("camera_calib.json not found near {}", plyDir);
        return 1;
    }
    auto h = fc::loadCameraCalibHandoff(calibPath);
    if (!h) {
        spdlog::error("load handoff failed: {}", calibPath);
        return 1;
    }

    // 读全部 PLY → [pose][lineId] 分桶
    std::vector<std::map<int, calib::PosePointSet>> perPose;
    std::error_code ec;
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(plyDir, ec)) {
        const std::string name = entry.path().filename().string();
        if (name.size() > 11 && name.substr(name.size() - 11) == "_8cloud.ply")
            files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());
    for (const auto& fp : files) {
        std::vector<cv::Vec3f> pts;
        std::vector<int> ids;
        if (!readPly(fp, pts, ids)) { spdlog::warn("read {} failed", fp); continue; }
        perPose.emplace_back();
        auto& buckets = perPose.back();
        for (size_t k = 0; k < pts.size(); ++k) {
            auto& ps = buckets[ids[k]];
            ps.points3d.push_back(pts[k]);
            ps.lineIds.push_back(ids[k]);
        }
        std::map<int, size_t> hist;
        for (int id : ids) ++hist[id];
        std::string hs;
        for (const auto& kv : hist)
            hs += std::to_string(kv.first) + ":" + std::to_string(kv.second) + " ";
        spdlog::info("{}: {} pts, ids: {}", std::filesystem::path(fp).filename().string(),
                     pts.size(), hs);
    }
    if (perPose.empty()) {
        spdlog::error("no PLY under {}", plyDir);
        return 1;
    }
    // 汇总全部线号
    std::set<int> allLines;
    for (const auto& pm : perPose)
        for (const auto& kv : pm) allLines.insert(kv.first);

    // 逐线: 跨姿态收集该线点集 → runPjc
    nlohmann::json lines = nlohmann::json::array();
    int okLines = 0;
    cv::Vec3d tSum(0, 0, 0);
    for (int lid : allLines) {
        if (lineFilter > 0 && lid != lineFilter) continue;
        std::vector<calib::PosePointSet> poseSets;
        for (const auto& pm : perPose) {
            auto it = pm.find(lid);
            if (it != pm.end() && !it->second.points3d.empty())
                poseSets.push_back(it->second);
        }
        nlohmann::json j;
        j["lineId"] = lid;
        j["poseCount"] = poseSets.size();
        if (poseSets.size() < 5) {   // minPoses 门禁, 跳过噪声色桶
            lines.push_back(std::move(j));
            continue;
        }
        auto r = runPjc(poseSets, *h);
        if (!r.is_null()) {
            ++okLines;
            const auto& tv = r["projectorT"];
            tSum += cv::Vec3d(tv[0].get<double>(), tv[1].get<double>(),
                              tv[2].get<double>());
        }
        if (!r.is_null()) j["pjc"] = r;
        lines.push_back(std::move(j));
    }

    // 稳健共识: 取 rms<1.0 且 cond<1e15 的线, 各分量中位数; R1ᵀ 反变换回原始左相机系
    if (okLines > 0) {
        const cv::Vec3d tMean = tSum * (1.0 / okLines);
        (void)tMean;
        std::vector<double> xs, ys, zs;
        for (const auto& j : lines) {
            if (!j.contains("pjc")) continue;
            const auto& p = j["pjc"];
            if (!p.contains("finalSampsonRms") || !p.contains("jacobianConditionNumber"))
                continue;
            if (p["finalSampsonRms"].get<double>() > 1.0) continue;
            if (p["jacobianConditionNumber"].get<double>() > 1e15) continue;
            const auto& tv = p["projectorT"];
            xs.push_back(tv[0].get<double>());
            ys.push_back(tv[1].get<double>());
            zs.push_back(tv[2].get<double>());
        }
        if (!xs.empty()) {
            std::sort(xs.begin(), xs.end());
            std::sort(ys.begin(), ys.end());
            std::sort(zs.begin(), zs.end());
            const cv::Vec3d tMed(xs[xs.size() / 2], ys[ys.size() / 2],
                                 zs[zs.size() / 2]);
            // 矫正系 → 原始左相机系: t_orig = R1ᵀ · t_rect
            const cv::Matx33d R1 = h->R1;
            const cv::Vec3d tOrig = R1.t() * tMed;
            spdlog::info("per-line median consensus over {}/{} lines "
                         "(rms<1.0, cond<1e15): tRect=({:.2f},{:.2f},{:.2f}) mm; "
                         "tOrig=R1^T*tRect=({:.2f},{:.2f},{:.2f}) mm "
                         "|t|={:.2f} mm",
                         xs.size(), okLines, tMed[0], tMed[1], tMed[2],
                         tOrig[0], tOrig[1], tOrig[2],
                         std::sqrt(tOrig.dot(tOrig)));
        }
    }
    nlohmann::json summary;
    summary["schema"] = "fcstepdump-pjc-per-line/1.0";
    summary["plyDir"] = plyDir;
    summary["lines"] = std::move(lines);
    if (!fc::writeLaserJson(outJson, summary)) {
        spdlog::error("write {} failed", outJson);
        return 1;
    }
    return okLines > 0 ? 0 : 1;
}

// 多线联合 PJC: 按线分组 → ExecuteMultiLine（共享 t + 逐线曲线）
int runPjcMultiline(const std::string& plyDir, const std::string& outJson,
                    const cv::Vec3d& t0Override = cv::Vec3d(
                        std::numeric_limits<double>::quiet_NaN(),
                        0.0, 0.0)) {
    spdlog::info("=== fcstepdump --pjc-multiline ===");
    std::string calibPath;
    const std::string candidates[] = {plyDir + "/camera_calib.json",
                                      plyDir + "/../camera_calib.json",
                                      "data_in/left_skew/camera_calib.json"};
    for (const std::string& cand : candidates) {
        if (std::filesystem::exists(cand)) { calibPath = cand; break; }
    }
    if (calibPath.empty()) {
        spdlog::error("camera_calib.json not found near {}", plyDir);
        return 1;
    }
    auto h = fc::loadCameraCalibHandoff(calibPath);
    if (!h) {
        spdlog::error("load handoff failed: {}", calibPath);
        return 1;
    }

    std::vector<std::map<int, calib::PosePointSet>> perPose;
    std::error_code ec;
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(plyDir, ec)) {
        const std::string name = entry.path().filename().string();
        if (name.size() > 11 && name.substr(name.size() - 11) == "_8cloud.ply")
            files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());
    for (const auto& fp : files) {
        std::vector<cv::Vec3f> pts;
        std::vector<int> ids;
        if (!readPly(fp, pts, ids)) continue;
        perPose.emplace_back();
        auto& buckets = perPose.back();
        for (size_t k = 0; k < pts.size(); ++k) {
            auto& ps = buckets[ids[k]];
            ps.points3d.push_back(pts[k]);
            ps.lineIds.push_back(ids[k]);
        }
    }
    if (perPose.empty()) {
        spdlog::error("no PLY under {}", plyDir);
        return 1;
    }
    std::set<int> allLines;
    for (const auto& pm : perPose)
        for (const auto& kv : pm) allLines.insert(kv.first);

    calib::MultiLineInput in;
    in.lines.resize(allLines.size());
    size_t li = 0;
    std::vector<int> lineIds;
    for (int lid : allLines) {
        for (const auto& pm : perPose) {
            auto it = pm.find(lid);
            if (it != pm.end() && !it->second.points3d.empty())
                in.lines[li].push_back(it->second);
        }
        lineIds.push_back(lid);
        ++li;
    }
    in.f = h->P1.at<double>(0, 0);
    in.principalPoint = cv::Point2d(h->P1.at<double>(0, 2), h->P1.at<double>(1, 2));
    calib::ProjectorJointCalibParams op_params;          // 默认参数（阶段1/1.5 共用）
    op_params.enableTiming = true;                       // 诊断工具: 相位计时常开
    // 两阶段初始化: 默认先用逐线 PJC 的中位数共识做初值（多线联合代价面非凸,
    // 机械初值/任意初值会落入劣质盆地, 实测共识初值盆地的 rms 优 3 倍）;
    // t0 覆盖时跳过（盆地探测用）
    {
        const cv::Mat R1m = h->R1;
        cv::Matx33d R1 = R1m;
        if (std::isnan(t0Override[0])) {
            in.initialT = R1 * cv::Vec3d(80.0, 3.0, 3.0);   // 兜底
            // —— 阶段1: 逐线求解取中位数（rms<1 且 cond<1e15 的线）——
            std::vector<double> xs, ys, zs;
            calib::ProjectorJointCalib warmOp;
            int warmLines = 0;
            for (size_t li = 0; li < in.lines.size(); ++li) {
                if (in.lines[li].size() < 5) continue;
                calib::ProjectorJointCalibInput si;
                si.poses = in.lines[li];
                si.f = in.f;
                si.principalPoint = in.principalPoint;
                si.initialT = R1 * cv::Vec3d(80.0, 3.0, 3.0);
                auto sr = warmOp.Execute(si);
                if (!sr.success) continue;
                if (sr.finalSampsonRms > 1.0) continue;
                if (sr.jacobianConditionNumber > 1e15) continue;
                xs.push_back(sr.projectorT[0]);
                ys.push_back(sr.projectorT[1]);
                zs.push_back(sr.projectorT[2]);
                ++warmLines;
            }
            if (xs.size() >= 3) {
                std::sort(xs.begin(), xs.end());
                std::sort(ys.begin(), ys.end());
                std::sort(zs.begin(), zs.end());
                const cv::Vec3d tMed(xs[xs.size() / 2], ys[ys.size() / 2],
                                     zs[zs.size() / 2]);
                in.initialT = tMed;
                spdlog::info("stage-1 per-line init: median of {} lines "
                             "t0=({:.1f},{:.1f},{:.1f})",
                             warmLines, tMed[0], tMed[1], tMed[2]);
                // 阶段1.5 盆地探测: 共识点 vs 收缩点(×0.35 向零收缩, 实测最优解
                // 比逐线共识更靠近原点)。各跑 15 迭代短多线, 低 rms 者胜出。
                auto shortRun = [&](const cv::Vec3d& t0) {
                    calib::ProjectorJointCalibParams sp = op_params;
                    sp.maxIterations = 15;
                    sp.enableTiming = true;
                    calib::ProjectorJointCalib sop(sp);
                    calib::MultiLineInput si = in;
                    si.initialT = t0;
                    auto sr = sop.ExecuteMultiLine(si);
                    return sr.success ? sr.finalSampsonRms : 1e30;
                };
                const double r1 = shortRun(tMed);
                // 探测点2: z 向上拉 60mm（实测最优盆在 z≈+10, 逐线共识 z 系统性偏负
                // ——单线 t_z 弱观测的已知缺陷, 联合解在其上方）
                const cv::Vec3d tZlift(tMed[0], tMed[1], tMed[2] + 60.0);
                const double r2 = shortRun(tZlift);
                if (r2 < r1) {
                    in.initialT = tZlift;
                    spdlog::info("stage-1.5 basin probe: median rms={:.2f} "
                                 "z+60 rms={:.2f} -> pick z-lift ({:.1f},{:.1f},{:.1f})",
                                 r1, r2, tZlift[0], tZlift[1], tZlift[2]);
                } else {
                    spdlog::info("stage-1.5 basin probe: median rms={:.2f} "
                                 "z+60 rms={:.2f} -> keep median", r1, r2);
                }
            } else {
                spdlog::warn("stage-1 init failed ({} usable lines), "
                             "fallback to R1*(80,3,3)", xs.size());
            }
        } else {
            in.initialT = t0Override;
        }
    }
    spdlog::info("multiline: {} lines, initialT=({:.1f},{:.1f},{:.1f}) (two-stage)",
                 in.lines.size(), in.initialT[0], in.initialT[1], in.initialT[2]);

    calib::ProjectorJointCalib op(op_params);   // 全解共用 op_params（含 enableTiming）
    auto r = op.ExecuteMultiLine(in);
    if (!r.success) {
        spdlog::error("multi-line PJC failed: {}", r.message);
        return 1;
    }
    // 矫正系 → 原始左相机系
    const cv::Mat R1m = h->R1;
    const cv::Matx33d R1 = R1m;
    const cv::Vec3d tOrig = R1.t() * r.projectorT;
    spdlog::info("MULTI-LINE PJC: tRect=({:.3f},{:.3f},{:.3f}) mm -> "
                 "tOrig=({:.3f},{:.3f},{:.3f}) mm |t|={:.2f}, "
                 "rms {}->{:.4f}, lines={}, poses={}, pts={}, cond={:.3g}",
                 r.projectorT[0], r.projectorT[1], r.projectorT[2],
                 tOrig[0], tOrig[1], tOrig[2],
                 std::sqrt(tOrig.dot(tOrig)),
                 r.initialSampsonRms, r.finalSampsonRms,
                 r.lineCount, r.poseCount, r.totalPointCount,
                 r.jacobianConditionNumber);

    nlohmann::json summary;
    summary["schema"] = "fcstepdump-pjc-multiline/1.0";
    summary["plyDir"] = plyDir;
    nlohmann::json pj;
    pj["success"] = true;
    pj["qualityFlag"] = static_cast<int>(r.qualityFlag);
    pj["message"] = r.message;
    pj["projectorT_rect"] = {r.projectorT[0], r.projectorT[1], r.projectorT[2]};
    pj["projectorT_orig"] = {tOrig[0], tOrig[1], tOrig[2]};
    pj["initialSampsonRms"] = r.initialSampsonRms;
    pj["finalSampsonRms"] = r.finalSampsonRms;
    pj["improvementRatio"] = r.improvementRatio;
    pj["lineCount"] = r.lineCount;
    pj["poseCount"] = r.poseCount;
    pj["totalPointCount"] = r.totalPointCount;
    pj["jacobianConditionNumber"] = r.jacobianConditionNumber;
    nlohmann::json curves = nlohmann::json::array();
    for (size_t k = 0; k < r.emissionCurves.size(); ++k) {
        nlohmann::json cj;
        cj["inputLineIdx"] = r.usedLineIdx[k];
        cj["lineId"] = lineIds.empty() ? r.usedLineIdx[k]
                                       : lineIds[static_cast<size_t>(r.usedLineIdx[k])];
        const auto& c = r.emissionCurves[k];
        cj["coeffs"] = {c.coeffs[0], c.coeffs[1], c.coeffs[2],
                        c.coeffs[3], c.coeffs[4], c.coeffs[5]};
        cj["discriminant"] = c.discriminant;
        cj["pointCount"] = c.pointCount;
        curves.push_back(std::move(cj));
    }
    pj["curves"] = std::move(curves);
    summary["pjc"] = std::move(pj);
    if (!fc::writeLaserJson(outJson, summary)) {
        spdlog::error("write {} failed", outJson);
        return 1;
    }
    return 0;
}

// F5: CMOS 视图 —— 25 条发射曲线叠画（曲线本体 + 0.7 行栅格交点标记）
// 输入 laser_calib.json (pjc 节), 输出 PNG。像面尺寸取 principalPoint×2。
int drawCmosView(const std::string& calibJson, const std::string& outPng) {
    std::ifstream f(calibJson);
    if (!f) { spdlog::error("cannot open {}", calibJson); return 1; }
    nlohmann::json j = nlohmann::json::parse(f);

    if (!j.contains("pjc") || !j["pjc"].contains("emissionCurves")) {
        spdlog::error("no pjc.emissionCurves in {}", calibJson);
        return 1;
    }
    const auto& pjc = j["pjc"];
    const double cx = pjc["principalPoint"][0].get<double>();
    const double cy = pjc["principalPoint"][1].get<double>();
    const double fpx = pjc["f"].get<double>();
    const double rowStep = pjc.value("epipolarRowStep", 0.7);
    const int W = static_cast<int>(cx * 2);
    const int H = static_cast<int>(cy * 2);

    cv::Mat img(H, W, CV_8UC3, cv::Scalar(15, 15, 15));
    // 0.7 行栅格（淡线, 每 50 行标注）
    for (double y = 0; y < H; y += rowStep * 50) {
        cv::line(img, cv::Point(0, cvRound(y)), cv::Point(W - 1, cvRound(y)),
                 cv::Scalar(45, 45, 45), 1);
    }
    // 主点十字
    cv::line(img, cv::Point(cvRound(cx) - 20, cvRound(cy)),
             cv::Point(cvRound(cx) + 20, cvRound(cy)), cv::Scalar(90, 90, 90), 1);
    cv::line(img, cv::Point(cvRound(cx), cvRound(cy) - 20),
             cv::Point(cvRound(cx), cvRound(cy) + 20), cv::Scalar(90, 90, 90), 1);

    // 25 条曲线: 沿 u 扫描, 解 F(u,v)=0 取当前线附近的 v 根（牛顿迭代）
    int drawn = 0;
    for (const auto& cj : pjc["emissionCurves"]) {
        const int lid = cj["lineId"].get<int>();
        const auto C = cj["coeffs"];
        const double A = C[0], B = C[1], Cc = C[2], D = C[3], E = C[4], F0 = C[5];
        auto evalF = [&](double u, double v) {
            return A*u*u + B*u*v + Cc*v*v + D*u + E*v + F0;
        };
        auto evalFv = [&](double u, double v) {
            return B*u + 2.0*Cc*v + E;
        };
        const cv::Scalar col = lineColor(lid);
        const cv::Scalar colDim(col[0] * 0.55, col[1] * 0.55, col[2] * 0.55);
        // 从像面中心 u 出发向两侧跟踪 v 根（牛顿, 步进 1px; 曲线近水平对 v 解良态）
        // 起点策略: 先试中心 u; 无根则全像面找 |F| 最小点作起点
        auto findStart = [&](double& uS, double& vS) {
            // 1) 中心列种子
            double bestAbs = 1e30, bestV = -1;
            for (double vs = 0; vs < H; vs += 1.0) {
                const double fv = evalF(static_cast<double>(cx), vs);
                if (std::fabs(fv) < bestAbs) { bestAbs = std::fabs(fv); bestV = vs; }
            }
            if (bestV >= 0 && bestAbs < 1e-6) { uS = cx; vS = bestV; return true; }
            // 2) 粗网格全扫 (u×v 步 8px) 找 |F| 极小
            double gBest = 1e30, gU = -1, gV = -1;
            for (double uq = 0; uq < W; uq += 8.0)
                for (double vq = 0; vq < H; vq += 8.0) {
                    const double fv = evalF(uq, vq);
                    if (std::fabs(fv) < gBest) { gBest = std::fabs(fv); gU = uq; gV = vq; }
                }
            if (gU < 0 || gBest > 1e-3) return false;
            uS = gU; vS = gV;
            return true;
        };
        std::vector<cv::Point> rightSide, leftSide;
        {
            double uS = 0, vS = 0;
            if (!findStart(uS, vS)) continue;
            auto trackDir2 = [&](double uStart, double vStart, double duSign,
                                std::vector<cv::Point>& out) {
                double v = vStart;
                for (double u = uStart; u >= 0 && u < W; u += duSign) {
                    for (int it = 0; it < 10; ++it) {
                        const double fv = evalF(u, v);
                        const double dfv = evalFv(u, v);
                        if (std::fabs(dfv) < 1e-12) break;
                        const double step = fv / dfv;
                        v -= (std::fabs(step) > 50.0) ? 50.0 * (step > 0 ? 1 : -1) : step;
                    }
                    if (v < -20 || v > H + 20) break;
                    if (std::fabs(evalF(u, v)) > 1e-3) break;
                    out.emplace_back(cvRound(u), cvRound(v));
                }
            };
            trackDir2(uS, vS, +1.0, rightSide);
            trackDir2(uS, vS, -1.0, leftSide);
        }
        std::vector<cv::Point> ptsFull;
        ptsFull.insert(ptsFull.end(), leftSide.rbegin(), leftSide.rend());
        ptsFull.insert(ptsFull.end(), rightSide.begin(), rightSide.end());
        std::vector<cv::Point> ptsRow;
        for (const auto& p : ptsFull) {
            const double vv = p.y;
            const double row = std::round(vv / rowStep) * rowStep;
            if (std::fabs(vv - row) < 0.2)
                ptsRow.emplace_back(p.x, cvRound(row));
        }
        if (ptsFull.size() < 2) continue;
        // 平滑连线（曲线）
        for (size_t k = 1; k < ptsFull.size(); ++k)
            if (std::abs(ptsFull[k].y - ptsFull[k-1].y) < 30)
                cv::line(img, ptsFull[k-1], ptsFull[k], colDim, 1, cv::LINE_AA);
        // 行交点标记
        for (const auto& p : ptsRow)
            cv::circle(img, p, 1, col, -1, cv::LINE_AA);
        // 线号标注（每线最左端）
        cv::putText(img, std::to_string(lid), ptsFull.front() + cv::Point(4, -4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(img, std::to_string(lid), ptsFull.front() + cv::Point(4, -4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        ++drawn;
    }
    // 标注
    cv::putText(img, "virtual CMOS " + std::to_string(W) + "x" + std::to_string(H)
                 + "  f=" + std::to_string(static_cast<int>(fpx))
                 + "  rowStep=" + std::to_string(rowStep).substr(0, 3)
                 + "  curves=" + std::to_string(drawn),
                 cv::Point(12, 22), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                 cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(img, "virtual CMOS " + std::to_string(W) + "x" + std::to_string(H)
                 + "  f=" + std::to_string(static_cast<int>(fpx))
                 + "  rowStep=" + std::to_string(rowStep).substr(0, 3)
                 + "  curves=" + std::to_string(drawn),
                 cv::Point(12, 22), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                 cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    if (!cv::imwrite(outPng, img)) {
        spdlog::error("write {} failed", outPng);
        return 1;
    }
    spdlog::info("CMOS view: {} curves -> {}", drawn, outPng);
    return 0;
}

// R1 坐标系定案实验:
// 矫正系真点 P_rect（在光心 tRect 射线上）→ kernel 公式 uL = P1·(R1·P_rect)/z
// vs ground truth uL_true = P1·P_rect/z。若 kernel 公式对 → R1 多乘是 bug;
// 若错 → kernel 期望原始系输入（喂 tRect 前应逆旋转）。
int runR1Audit(const std::string& calibJson) {
    std::ifstream f(calibJson);
    if (!f) { spdlog::error("cannot open {}", calibJson); return 1; }
    nlohmann::json j = nlohmann::json::parse(f);
    auto h = fc::loadCameraCalibHandoff("data_in/left_skew/camera_calib.json");
    if (!h) { spdlog::error("load handoff failed"); return 1; }

    const double fx = j["pjc"]["f"].get<double>();
    const cv::Point2d pp(j["pjc"]["principalPoint"][0].get<double>(),
                         j["pjc"]["principalPoint"][1].get<double>());
    const cv::Vec3d tRect(j["pjc"]["projectorT"][0].get<double>(),
                          j["pjc"]["projectorT"][1].get<double>(),
                          j["pjc"]["projectorT"][2].get<double>());
    cv::Matx33d R1(h->R1);
    cv::Matx34d P1(h->P1);

    spdlog::info("=== R1 audit ===");
    spdlog::info("tRect=({:.3f},{:.3f},{:.3f})  R1 yaw≈{:.2f}°  P1[0,0]={:.1f}",
                 tRect[0], tRect[1], tRect[2],
                 std::asin(std::min(1.0, std::max(-1.0, -R1(0, 2)))) * 180 / CV_PI,
                 P1(0, 0));

    // 构造: 虚拟 CMOS 若干像点 → 矫正系射线 → 深度 t 上的真点 P_rect
    // ground truth: 矫正系点投到矫正像 = P1·P_rect（R1 不参与）
    // kernel 公式:   uL = P1·(R1·P_rect) —— 平移部分用 tRect（当前 CLI 喂法）
    double maxErrKernel = 0.0, maxErrAlt = 0.0;
    for (double u = 200; u <= 1800; u += 200) {
        for (double v = 200; v <= 1300; v += 300) {
            for (double depth = 150; depth <= 500; depth += 100) {
                // 矫正系射线方向
                const cv::Vec3d d((u - pp.x) / fx, (v - pp.y) / fx, 1.0);
                const cv::Vec3d P = tRect + depth * d;   // 矫正系真点
                // ground truth: P1 直投
                const cv::Vec3d pG = P1 * cv::Vec4d(P[0], P[1], P[2], 1.0);
                const double uTrue = pG[0] / pG[2], vTrue = pG[1] / pG[2];
                // kernel 公式: P1·(R1·P)
                const cv::Vec3d Pk = R1 * P;
                const cv::Vec3d pK = P1 * cv::Vec4d(Pk[0], Pk[1], Pk[2], 1.0);
                const double uKern = pK[0] / pK[2], vKern = pK[1] / pK[2];
                const double eK = std::hypot(uKern - uTrue, vKern - vTrue);
                maxErrKernel = std::max(maxErrKernel, eK);
                // 备选假设: kernel 期望 P 在原始系 → 喂 tOrig=R1ᵀ·tRect、d_orig=R1ᵀ·d
                // 则 P_orig = tOrig + depth·d_orig, R1·P_orig = R1·tOrig+depth·d = tRect+depth·d=P ✓
                // 该假设下 kernel 输出 == ground truth（恒等）, 误差 0
                maxErrAlt = 0.0;
            }
        }
    }
    spdlog::info("假设A(现 CLI 喂法, P 在矫正系, kernel 再乘 R1): 最大像素误差 = {:.1f} px",
                 maxErrKernel);
    spdlog::info("假设B(喂 R1ᵀ·tRect, P 在原始系, kernel 的 R1 负责 rectify): 误差 = 0 px (恒等)");
    spdlog::info("结论: kernel 期望的射线原点是 {}——{}",
                 maxErrKernel > 1.0 ? "原始左相机系 (R1ᵀ·tRect)" : "矫正系 (tRect 直喂)",
                 maxErrKernel > 1.0
                     ? "当前 CLI 直接喂 tRect 是 BUG, 需改为喂 R1ᵀ·tRect 且虚拟像素射线方向也需在原始系表达"
                     : "现喂法正确");
    return 0;
}

// 方案Y 修复对照实验: 同一虚拟像素集, 喂「矫正系直喂(旧,错)」vs「R1ᵀ翻译(新,对)」
// 各跑一次 4-12, 对比 totalPairs 与首条目坐标——错位 488px 时大量条目出 FOV, pairs 骤降。
int runPlaneMapAblation(const std::string& calibJson) {
    std::ifstream f(calibJson);
    if (!f) { spdlog::error("cannot open {}", calibJson); return 1; }
    nlohmann::json j = nlohmann::json::parse(f);
    auto h = fc::loadCameraCalibHandoff("data_in/left_skew/camera_calib.json");
    if (!h) { spdlog::error("load handoff failed"); return 1; }

    const cv::Vec3d tRect(j["pjc"]["projectorT"][0].get<double>(),
                          j["pjc"]["projectorT"][1].get<double>(),
                          j["pjc"]["projectorT"][2].get<double>());
    const double fx = j["pjc"]["f"].get<double>();
    const cv::Point2d pp(j["pjc"]["principalPoint"][0].get<double>(),
                         j["pjc"]["principalPoint"][1].get<double>());
    cv::Mat R1cv = h->R1;
    const cv::Matx33d R1 = R1cv;
    cv::Mat R2cv = h->R2;
    const cv::Matx33d R2 = R2cv;
    cv::Matx34d P1(h->P1), P2(h->P2);

    calib::StereoCalibration sc;
    sc.R1 = h->R1; sc.R2 = h->R2; sc.P1 = h->P1; sc.P2 = h->P2;
    sc.imageSize = h->imageSize;

    // 虚拟像素: 简化网格 25 线 × 每 32px 一点
    std::vector<cv::Vec3f> pixels;
    for (int lid = 1; lid <= 25; ++lid)
        for (double u = 100; u < 1900; u += 32)
            for (double v = 100; v < 1400; v += 64)
                pixels.push_back(cv::Vec3f((float)u, (float)v, (float)lid));
    cv::Mat h_px(1, static_cast<int>(pixels.size()), CV_32FC3,
                 const_cast<cv::Vec3f*>(pixels.data()));
    cv::cuda::GpuMat d_px(h_px);

    cv::Matx33d vK(fx, 0, pp.x, 0, fx, pp.y, 0, 0, 1);

    calib::PlaneMapParams pm;
    calib::PlaneMapCuda op(pm);
    cv::cuda::Stream stream;

    spdlog::info("=== plane_map ablation: {} virtual pixels ===", pixels.size());
    // A: 旧喂法（矫正系直喂, 已证错 488px）
    PlaneMapResult rA, rB;
    {
        rA = op.Execute(d_px, vK, cv::Matx33d::eye(), tRect, sc, stream);
        spdlog::info("A 旧喂法(vR=I, vT=tRect):  success={} pairs={}{}",
                     rA.success, rA.totalPairs,
                     rA.success ? "" : " msg=" + rA.message);
    }
    // B: 方案Y（R1ᵀ 翻译）
    {
        rB = op.Execute(d_px, vK, R1, R1.t() * tRect, sc, stream);
        spdlog::info("B 方案Y(vR=R1, vT=R1ᵀtRect): success={} pairs={}{}",
                     rB.success, rB.totalPairs,
                     rB.success ? "" : " msg=" + rB.message);
    }
    // 坐标级对比: 下载两表, 同 (uL,vL) 量化格上的 uR 差异分布
    if (rA.success && rB.success && rA.d_left_to_right && rB.d_left_to_right) {
        cv::Mat a, b;
        rA.d_left_to_right->download(a);
        rB.d_left_to_right->download(b);
        cv::Mat a4 = a.reshape(4, 1), b4 = b.reshape(4, 1);
        const int n = std::min(a4.cols, b4.cols);
        // 简化统计: 逐条目直接比（两表同 pixels 输入同序生成, 条目序可能因 FOV 裁剪不同——
        // 改为统计各自 uR 均值/范围, 差异应显著）
        auto stats = [&](const cv::Mat& m) {
            double s = 0, mn = 1e30, mx = -1e30; int cnt = 0;
            for (int k = 0; k < m.cols; ++k) {
                const cv::Vec4f& v = m.at<cv::Vec4f>(0, k);
                s += v[2]; mn = std::min(mn, (double)v[2]); mx = std::max(mx, (double)v[2]);
                ++cnt;
            }
            return std::make_tuple(cnt, s / std::max(1, cnt), mn, mx);
        };
        auto [na, ma, mna, mxa] = stats(a4);
        auto [nb, mb, mnb, mxb] = stats(b4);
        spdlog::info("uR stats  A: n={} mean={:.1f} range=[{:.0f},{:.0f}]", na, ma, mna, mxa);
        spdlog::info("uR stats  B: n={} mean={:.1f} range=[{:.0f},{:.0f}]", nb, mb, mnb, mxb);
        spdlog::info("两喂法表内容{}（uR 均值差 {:.1f} px）",
                     std::fabs(ma - mb) < 1.0 && na == nb ? "相同——翻译无效果?!"
                         : "不同——翻译改变了表（预期, 修复生效）",
                     std::fabs(ma - mb));
    }
    return 0;
}

// T5: curve_map 实数据生成＋B1 闭环验收（表值 vs 解析直算, 抽样 N 点）
int runCurveMap(const std::string& calibJson, const std::string& outBin) {
    std::ifstream f(calibJson);
    if (!f) { spdlog::error("cannot open {}", calibJson); return 1; }
    nlohmann::json j = nlohmann::json::parse(f);
    auto h = fc::loadCameraCalibHandoff("data_in/left_skew/camera_calib.json");
    if (!h) { spdlog::error("load handoff failed"); return 1; }

    // 输入组装（全部矫正系直连）
    calib::CurveMapInput in;
    for (const auto& cj : j["pjc"]["emissionCurves"]) {
        calib::ImplicitCurve c;
        for (int k = 0; k < 6; ++k)
            c.coeffs[k] = cj["coeffs"][k].get<double>();
        in.curves.push_back(c);
    }
    in.f = j["pjc"]["f"].get<double>();
    in.principalPoint = cv::Point2d(j["pjc"]["principalPoint"][0].get<double>(),
                                    j["pjc"]["principalPoint"][1].get<double>());
    // 基线 B = |P2(0,3)|/f
    cv::Matx34d P2(h->P2);
    in.baseline = std::fabs(P2(0, 3)) / in.f;
    in.imageSize = h->imageSize;
    in.virtualT = cv::Vec3d(j["pjc"]["projectorT"][0].get<double>(),
                            j["pjc"]["projectorT"][1].get<double>(),
                            j["pjc"]["projectorT"][2].get<double>());
    // ROI: validRoiLeft 实读
    {
        std::ifstream hf("data_in/left_skew/camera_calib.json");
        nlohmann::json hj = nlohmann::json::parse(hf);
        const auto& vr = hj["rectify"]["validRoiLeft"];
        in.roi = cv::Rect(vr["x"].get<int>(), vr["y"].get<int>(),
                          vr["w"].get<int>(), vr["h"].get<int>());
    }
    spdlog::info("curve_map: {} curves, f={:.1f}, B={:.2f}mm, t=({:.1f},{:.1f},{:.1f}), "
                 "ROI={}x{}+{}+{}", in.curves.size(), in.f, in.baseline,
                 in.virtualT(0), in.virtualT(1), in.virtualT(2),
                 in.roi.width, in.roi.height, in.roi.x, in.roi.y);

    calib::CurveMapParams prm;
    prm.epipolarRowStep = j["pjc"].value("epipolarRowStep", 0.7f);
    calib::CurveMapGenerator gen(prm);
    auto r = gen.Generate(in);
    if (!r.success) { spdlog::error("curve_map failed: {}", r.message); return 1; }
    spdlog::info("curve_map OK: entries={}, rows={} bytes={} ({:.1f}MB) flag={} msg={}",
                 r.entryCount, r.rowCountWithData, r.tableBytesSize,
                 r.tableBytesSize / 1048576.0, static_cast<int>(r.qualityFlag),
                 r.message);

    // 落盘
    {
        std::ofstream of(outBin, std::ios::binary);
        of.write(reinterpret_cast<const char*>(r.tableBytes.data()),
                 static_cast<std::streamsize>(r.tableBytes.size()));
        if (!of) { spdlog::error("write {} failed", outBin); return 1; }
    }
    spdlog::info("table -> {}", outBin);

    // B1 闭环: 载入→随机抽样→表值 vs 解析直算（同测试口径的独立重推）
    calib::CurveMapTable tbl;
    std::string err;
    if (!tbl.Load(r.tableBytes.data(), r.tableBytes.size(), err)) {
        spdlog::error("reload failed: {}", err);
        return 1;
    }
    const double cx = in.principalPoint.x, cy = in.principalPoint.y;
    const double tx = in.virtualT(0), ty = in.virtualT(1), tz = in.virtualT(2);
    const double Ex = in.f * tx / tz + cx;
    const double Ey = in.f * ty / tz + cy;
    const double rowStep = tbl.rowStep();
    // 行采样域 = 栅格行号（v = row·rowStep 落在 ROI 内）——不是像素 y！
    const int rowLo = static_cast<int>(std::floor(in.roi.y / rowStep));
    const int rowHi = static_cast<int>(std::ceil((in.roi.y + in.roi.height) / rowStep));
    std::mt19937 rng(2026);
    int checked = 0, mismatch = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const int row = std::uniform_int_distribution<int>(rowLo, rowHi - 1)(rng);
        const double v = row * rowStep;
        const int k = std::uniform_int_distribution<int>(
            0, static_cast<int>(in.roi.width / rowStep) - 1)(rng);
        const double uLpx = in.roi.x + k * rowStep;
        const int uLg = static_cast<int>(std::lround(uLpx / rowStep));
        calib::CurveMapCand cands[64];
        const int n = tbl.Lookup(row, uLg, cands, 64);
        for (size_t li = 0; li < in.curves.size(); ++li) {
            // 独立解析真值: 参数化极线（过 p 与 E）×二次曲线 精确求交（同单测口径）
            const auto& c = in.curves[li];
            const double du = Ex - uLpx, dv = Ey - v;
            const double A = c.coeffs[0], Bc = c.coeffs[1], C = c.coeffs[2];
            const double D = c.coeffs[3], E2 = c.coeffs[4], F0 = c.coeffs[5];
            const double qa = A*du*du + Bc*du*dv + C*dv*dv;
            const double qb = 2*A*uLpx*du + Bc*(uLpx*dv + v*du) + 2*C*v*dv + D*du + E2*dv;
            const double qc2 = A*uLpx*uLpx + Bc*uLpx*v + C*v*v + D*uLpx + E2*v + F0;
            double roots[2];
            int nr = 0;
            if (std::fabs(qa) < 1e-18) {
                if (std::fabs(qb) > 1e-18) roots[nr++] = -qc2 / qb;
            } else {
                const double disc = qb*qb - 4*qa*qc2;
                if (disc >= 0) {
                    roots[nr++] = (-qb + std::sqrt(disc)) / (2*qa);
                    roots[nr++] = (-qb - std::sqrt(disc)) / (2*qa);
                }
            }
            int gtU[2];
            int nGT = 0;
            for (int rr = 0; rr < nr; ++rr) {
                const double qu = uLpx + roots[rr] * du;
                const double qv = v + roots[rr] * dv;
                if (qu < 0 || qu >= in.imageSize.width ||
                    qv < 0 || qv >= in.imageSize.height) continue;
                const double qcx = qu - cx;
                const double den = qcx - (uLpx - cx);
                if (std::fabs(den) < 1e-9) continue;
                const double Z = (qcx * tz - in.f * tx) / den;
                if (Z < 100.0 || Z > 700.0) continue;
                const double uRpx = uLpx - in.f * in.baseline / Z;
                if (uRpx < 0 || uRpx >= in.imageSize.width) continue;
                gtU[nGT++] = static_cast<int>(std::lround(uRpx / rowStep));
            }
            if (nGT == 2 && gtU[0] == gtU[1]) nGT = 1;
            int tblU[2];
            int nTbl = 0;
            for (int q = 0; q < n && nTbl < 2; ++q)
                if (cands[q].lid == li + 1) tblU[nTbl++] = cands[q].uR;
            if (nGT == 2 && tblU[0] == tblU[1]) nTbl = 1;
            if (nGT == 0 && nTbl == 0) continue;
            ++checked;
            // 集合比较（顺序无关, ±2 格容差）
            bool ok = (nGT == nTbl);
            if (ok) {
                for (int g1 = 0; g1 < nGT && ok; ++g1) {
                    bool matched = false;
                    for (int g2 = 0; g2 < nTbl; ++g2)
                        if (std::abs(gtU[g1] - (int)tblU[g2]) <= 2) {
                            matched = true; break;
                        }
                    if (!matched) ok = false;
                }
            }
            if (!ok) {
                ++mismatch;
                if (mismatch <= 6)
                    spdlog::info("MISMATCH#{} row={} k={} uLpx={:.2f} uLg={} line={} "
                                 "nGT={} gt=[{},{}] nTbl={} tbl=[{},{}] v={:.3f}",
                                 mismatch, row, k, uLpx, uLg, li + 1,
                                 nGT, nGT > 0 ? gtU[0] : -1, nGT > 1 ? gtU[1] : -1,
                                 nTbl, nTbl > 0 ? (int)tblU[0] : -1,
                                 nTbl > 1 ? (int)tblU[1] : -1, v);
            }
        }
    }
    const double rate = checked > 0 ? 100.0 * (checked - mismatch) / checked : 0.0;
    spdlog::info("B1 closed-loop: checked={} mismatch={} rate={:.3f}% {}",
                 checked, mismatch, rate,
                 rate >= 99.5 ? "PASS" : "FAIL");
    return rate >= 99.5 ? 0 : 1;
}

// --cmtt-gen: 冻结标定 JSON → CMTT sidecar 真规模再生成（照 runCurveMap 模式）。
// 三源: ① handoff stereoRectifyTempTable（camera_calib.json）② pjc.emissionCurves
//       ③ laserExtrinsicTempTable.virtualToLeft（温档参数全部来自两表插值）。
// rowStep = json pjc.epipolarRowStep; depth 照 4-14 口径（cfg.depthMin/Max）。
// GPU 非确定免疫: 输入全取冻结 json → 同 json 三档 tierThreads 再生成 sha256
// 必须逐字节相等（生成链纯 CPU, 线程数无关确定性）。
// 注意: handoff 硬编码 data_in/left_skew/camera_calib.json（CWD 相对——与
//       runCurveMap 同款行为, 须自工程根目录起跑）。
int runCmttGen(const std::string& calibJson, const std::string& outBin, int tierThreads) {
    std::ifstream f(calibJson);
    if (!f) { spdlog::error("cannot open {}", calibJson); return 1; }
    nlohmann::json j = nlohmann::json::parse(f);
    auto h = fc::loadCameraCalibHandoff("data_in/left_skew/camera_calib.json");
    if (!h) { spdlog::error("load handoff failed"); return 1; }
    if (!h->haveRectifyTempTable) {
        spdlog::error("handoff has no stereoRectifyTempTable (4-14 source #1)");
        return 1;
    }

    // 源③: laserExtrinsicTempTable.virtualToLeft → LaserExtrinsicCompensateCPUResult
    //      （TempParamInterpolator 只消费 leftResult.table 的 temperature/T）
    calib::LaserExtrinsicCompensateCPUResult laserExtrin;
    try {
        const auto& lt = j.at("laserExtrinsicTempTable");
        laserExtrin.success = true;
        laserExtrin.referenceTemp = lt.at("referenceTemperature").get<double>();
        if (lt.contains("cte")) laserExtrin.cte = lt.at("cte").get<double>();
        auto& tab = laserExtrin.leftResult.table;
        for (const auto& e : lt.at("virtualToLeft").at("table")) {
            calib::ExtrinsicCompensatedEntry entry;
            entry.temperature = e.at("temperature").get<double>();
            for (int i = 0; i < 3; ++i)
                entry.T[i] = e.at("T").at(i).get<double>();
            tab.push_back(entry);
        }
        laserExtrin.leftResult.success = true;
        laserExtrin.leftResult.referenceTemp = laserExtrin.referenceTemp;
        std::stable_sort(tab.begin(), tab.end(),
                         [](const calib::ExtrinsicCompensatedEntry& a,
                            const calib::ExtrinsicCompensatedEntry& b) {
                             return a.temperature < b.temperature;
                         });
    } catch (const std::exception& e) {
        spdlog::error("laserExtrinsicTempTable parse failed: {}", e.what());
        return 1;
    }
    if (laserExtrin.leftResult.table.empty()) {
        spdlog::error("laserExtrinsicTempTable.virtualToLeft.table empty");
        return 1;
    }

    // 源②: pjc 发射曲线
    std::vector<calib::ImplicitCurve> curves;
    try {
        for (const auto& cj : j.at("pjc").at("emissionCurves")) {
            calib::ImplicitCurve c;
            for (int k = 0; k < 6; ++k)
                c.coeffs[k] = cj.at("coeffs").at(k).get<double>();
            curves.push_back(c);
        }
    } catch (const std::exception& e) {
        spdlog::error("pjc.emissionCurves parse failed: {}", e.what());
        return 1;
    }

    calib::CurveMapTempTableGenParams gp;
    gp.referenceTemp = laserExtrin.referenceTemp;
    gp.tempHalfRange = 15.f;                       // ±15°C（4-14 同款, 设计硬规则 #5）
    gp.tempStep      = 0.5f;
    gp.rowStep       = j.at("pjc").value("epipolarRowStep", 0.7f);
    gp.depthMin      = 100.0f;                     // 4-14 口径: cfg.depthMin/Max
    gp.depthMax      = 5000.0f;                    // （left_skew config.json 同值）
    gp.tierThreads   = tierThreads;

    spdlog::info("cmtt-gen: curves={} rowStep={:.2f} depth=[{:.0f},{:.0f}] refTemp={} "
                 "tierThreads={}", curves.size(), gp.rowStep, gp.depthMin, gp.depthMax,
                 gp.referenceTemp, gp.tierThreads);

    const auto t0 = std::chrono::steady_clock::now();
    calib::CurveMapTempTableResult r;
    try {
        calib::CurveMapTempTableGenerator gen(gp);
        r = gen.Generate(curves, h->rectifyTempTable, laserExtrin, h->imageSize);
    } catch (const std::exception& e) {
        spdlog::error("cmtt-gen exception: {}", e.what());
        return 1;
    } catch (...) {
        spdlog::error("cmtt-gen unknown exception");
        return 1;
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    if (!r.success) {
        spdlog::error("cmtt-gen failed (ref-tier red line): {}", r.message);
        return 1;
    }

    {
        std::ofstream of(outBin, std::ios::binary);
        of.write(reinterpret_cast<const char*>(r.sidecarBytes.data()),
                 static_cast<std::streamsize>(r.sidecarBytes.size()));
        if (!of) { spdlog::error("write {} failed", outBin); return 1; }
    }
    spdlog::info("cmtt-gen OK: tiers={} ok={} tierThreads={} elapsed={:.0f}ms "
                 "bytes={} ({:.1f}MB) -> {}",
                 r.tiers.size(), r.okCount(), gp.tierThreads, ms,
                 r.sidecarBytes.size(), r.sidecarBytes.size() / 1048576.0, outBin);
    return 0;
}

// --cmtt: CMTT sidecar 容器转储——容器头（tierCount/温度 X10/rowStep/depth/冻结栅格/
//         gridHash）＋锚定说明（tempBase＝参考温度非首档温, 参考档 i_ref 居中,
//         tierTemp(i)=tempBase+(i-i_ref)·tempStep）＋逐档行（锚定档温/entryCount/
//         flags: R=参考 M=missing C=clamped/tierBytes 主动 CRC 验证, missing 档标 '-'）
//         ＋汇总; 容器坏（打不开/load 拒绝/无参考档锚/载荷 CRC 失败）→ 返回 1
int runCmttDump(const std::string& sidecar, const std::string& outFile) {
    std::vector<uint8_t> raw;
    {
        std::ifstream f(sidecar, std::ios::binary | std::ios::ate);
        if (!f) { spdlog::error("cannot open sidecar '{}'", sidecar); return 1; }
        const std::streamoff n = f.tellg();
        if (n <= 0) { spdlog::error("sidecar '{}' empty or size query failed", sidecar); return 1; }
        raw.resize(static_cast<size_t>(n));
        f.seekg(0, std::ios::beg);
        if (!f.read(reinterpret_cast<char*>(raw.data()),
                    static_cast<std::streamsize>(raw.size()))) {
            spdlog::error("sidecar '{}' short read", sidecar);
            return 1;
        }
    }

    CmttReader reader;
    if (!reader.load(raw.data(), raw.size())) {
        spdlog::error("CMTT container load rejected (magic/version/index bounds)");
        return 1;
    }
    const CmttHeader& hd = reader.header();
    const double baseC = hd.tempBaseX10 / 10.0;   // 参考温度（非首档温, cmtt_container.h:37）
    const double stepC = hd.tempStepX10 / 10.0;
    // 档温锚定: 参考档 i_ref（flags R）一次扫出（isReference, O(N)·N≤1024 无害），
    // tierTemp(i) = tempBase + (i - i_ref)·tempStep——与逐档索引 temperatureX10 同网格
    // （cmtt_container.h:57; nearestTier 即按该逐档字段比较, cmtt_container.cpp:172-189）；
    // 无参考档 → 锚丢失, 显示 ERROR 并拒绝转储（生成器红线保证理论不可达）。
    long long iRef = -1;
    for (size_t i = 0; i < reader.tierCount(); ++i)
        if (reader.isReference(i)) { iRef = static_cast<long long>(i); break; }
    if (iRef < 0) {
        spdlog::error("no reference tier (anchor lost): tier temps unavailable, dump aborted");
        return 1;
    }

    std::ostringstream os;
    os << "== CMTT container: " << sidecar << " ==\n"
       << "tierCount : " << reader.tierCount() << "\n"
       << "tempBase  : X10=" << hd.tempBaseX10 << " -> " << baseC << " C (参考温度, 非首档温)\n"
       << "tempStep  : X10=" << hd.tempStepX10 << " -> " << stepC << " C\n"
       << "rowStep   : " << hd.rowStep << "\n"
       << "depth     : [" << hd.depthMin << ", " << hd.depthMax << "]\n"
       << "frozenRoi : x=" << hd.roiX << " y=" << hd.roiY
       << " w=" << hd.roiW << " h=" << hd.roiH << "\n"
       << "rows      : min=" << hd.rowMin << " count=" << hd.rowCount << "\n"
       << "gridHash  : 0x" << std::hex << hd.gridHash << std::dec << "\n"
       << "anchor    : tier[" << iRef << "] T=" << std::fixed << std::setprecision(1)
       << baseC << "C (R 档; tierTemp(i)=tempBase+(i-" << iRef << ")*tempStep)\n";

    size_t nOk = 0, nMissing = 0, nClamped = 0, nCrcFail = 0;
    for (size_t i = 0; i < reader.tierCount(); ++i) {
        const bool missing = reader.isMissing(i);
        std::string flags;
        if (reader.isReference(i)) flags += "R";
        if (missing) flags += "M";
        if (reader.isClamped(i)) flags += "C";
        if (flags.empty()) flags = "-";
        const char* crc = "-";
        if (!missing) {
            if (reader.tierBytes(i)) { crc = "OK"; ++nOk; }
            else { crc = "CRC-FAIL"; ++nCrcFail; }
        }
        if (missing) ++nMissing;
        if (reader.isClamped(i)) ++nClamped;
        os << "tier[" << std::setw(3) << i << "] T=" << std::fixed
           << std::setprecision(1) << std::setw(6)
           << baseC + (static_cast<double>(i) - static_cast<double>(iRef)) * stepC
           << "C entries=" << std::setw(8) << reader.entryCount(i)
           << " flags=" << std::setw(3) << flags << " crc=" << crc << "\n";
    }
    os << "summary   : ok=" << nOk << " missing=" << nMissing
       << " clamped=" << nClamped << " crcFail=" << nCrcFail << "\n";

    spdlog::info("{}", os.str());
    if (!outFile.empty()) {
        std::ofstream of(outFile);
        if (!of) { spdlog::error("write '{}' failed", outFile); return 1; }
        of << os.str();
        spdlog::info("dump -> {}", outFile);
    }
    return nCrcFail == 0 ? 0 : 1;
}

// mask_bench.cpp 提供
int runMaskBench(const std::string& inDir, const std::string& outDir, int iters);

// interp_bench.cu 提供
int runInterpBench(const std::string& inDir, const std::string& outDir, int iters);

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--mask-bench") {
        return runMaskBench(argv[2], "mask_bench_out", 20);
    }
    if (argc == 5 && std::string(argv[1]) == "--mask-bench") {
        return runMaskBench(argv[2], argv[3], std::stoi(argv[4]));
    }
    if (argc == 3 && std::string(argv[1]) == "--interp-bench") {
        return runInterpBench(argv[2], "interp_bench_out", 20);
    }
    if (argc == 5 && std::string(argv[1]) == "--interp-bench") {
        return runInterpBench(argv[2], argv[3], std::stoi(argv[4]));
    }
    if (argc == 3 && std::string(argv[1]) == "--curve-map") {
        return runCurveMap(argv[2], "curve_map_table.bin");
    }
    if (argc == 4 && std::string(argv[1]) == "--curve-map") {
        return runCurveMap(argv[2], argv[3]);
    }
    if (argc == 3 && std::string(argv[1]) == "--cmtt") {
        return runCmttDump(argv[2], std::string());
    }
    if (argc == 4 && std::string(argv[1]) == "--cmtt") {
        return runCmttDump(argv[2], argv[3]);
    }
    if (argc == 4 && std::string(argv[1]) == "--cmtt-gen") {
        return runCmttGen(argv[2], argv[3], 0);
    }
    if (argc == 5 && std::string(argv[1]) == "--cmtt-gen") {
        return runCmttGen(argv[2], argv[3], std::stoi(argv[4]));
    }
    if (argc == 3 && std::string(argv[1]) == "--r1-audit") {
        return runR1Audit(argv[2]);
    }
    if (argc == 3 && std::string(argv[1]) == "--pm-ablation") {
        return runPlaneMapAblation(argv[2]);
    }
    if (argc == 4 && std::string(argv[1]) == "--pjc-cmos") {
        return drawCmosView(argv[2], argv[3]);
    }
    if (argc == 3 && std::string(argv[1]) == "--analyze-ply") {
        return analyzePlyDir(argv[2]);
    }
    if (argc == 4 && std::string(argv[1]) == "--pjc-per-line") {
        const int filter = (argc >= 5) ? std::stoi(argv[4]) : -1;
        return runPjcPerLine(argv[2], argv[3], filter);
    }
    if (argc == 4 && std::string(argv[1]) == "--pjc-multiline") {
        return runPjcMultiline(argv[2], argv[3]);
    }
    if (argc == 7 && std::string(argv[1]) == "--pjc-multiline") {
        // 盆地探测: 额外传初值 t0
        return runPjcMultiline(argv[2], argv[3],
                               cv::Vec3d(std::stod(argv[4]),
                                         std::stod(argv[5]),
                                         std::stod(argv[6])));
    }
    if (argc == 4 && std::string(argv[1]) == "--pjc-from-ply") {
        return runPjcFromPly(argv[2], argv[3]);
    }
    if (argc < 3) {
        std::cerr << "usage: fcstepdump <input_dir> <output_dir>\n"
                  << "       fcstepdump --pjc-from-ply <ply_dir> <output_summary.json>\n"
                  << "       fcstepdump --cmtt <sidecar.bin> [out.txt]\n"
                  << "       fcstepdump --cmtt-gen <calib_json> <out.bin> [tierThreads]\n"
                  << "  input_dir  含 config.json + camera_calib.json + pose_*/L_tube*.png\n"
                  << "  output_dir 产物目录（自动创建）\n"
                  << "  ply_dir    本工具输出的 *_8cloud.ply 目录（需同源 camera_calib.json\n"
                  << "             供 f/主点; 于 ply_dir 上一级或 ply_dir 查找）\n";
        return 2;
    }
    const std::string inDir = argv[1];
    const std::string outDir = argv[2];

    spdlog::info("=== fcstepdump ===");

    try {

    std::error_code ec;
    std::filesystem::create_directories(outDir, ec);
    if (ec) {
        spdlog::error("create output dir failed: {}", ec.message());
        return 1;
    }

    auto input = loadLaserInput(inDir);
    if (!input) {
        spdlog::error("load laser input failed");
        return 1;
    }
    const auto& cfg = input->config;
    const auto& h = input->handoff;

    std::string why;
    if (!validateHandoffConsistency(cfg, h, why)) {
        spdlog::error("handoff inconsistent: {}", why);
        return 1;
    }
    spdlog::info("poses={}, imageSize={}x{}", input->poseFrames.size(),
                 h.imageSize.width, h.imageSize.height);

    cv::cuda::Stream stream;

    MaskExtractParams maskParams;
    maskParams.threshold = 50;
    maskParams.erodeSize = 1;
    maskParams.laserDilateSize = 19;
    maskParams.postErodeSize = 13;
    MaskExtractCUDA maskL(maskParams), maskR(maskParams);

    RegionAnalyzerParams cclParams;
    cclParams.deviceId = cfg.deviceId;
    cclParams.minArea = 0;
    cclParams.topXCount = 27;
    RegionAnalyzerCUDA cclL(cclParams), cclR(cclParams);

    LaserLabelParams labelParams;
    labelParams.deviceId = cfg.deviceId;
    labelParams.scanDirection = cfg.labelScanDirection;
    labelParams.realLineTolerance = 10;
    labelParams.centerRowOffset = cfg.labelCenterRowOffset;
    LaserLabelerCUDA labelL(labelParams), labelR(labelParams);

    StegerParams stegerParams;
    stegerParams.deviceId = cfg.deviceId;
    StegerExtractorCUDA stegerL(stegerParams), stegerR(stegerParams);

    cv::Mat P1_3x3 = h.P1.clone(); if (P1_3x3.cols > 3) P1_3x3.at<double>(0, 3) = 0.0;
    cv::Mat P2_3x3 = h.P2.clone(); if (P2_3x3.cols > 3) P2_3x3.at<double>(0, 3) = 0.0;
    UndistortPointsParams undistL;
    undistL.cameraMatrix = h.cameraMatrixL;
    undistL.distCoeffs   = h.distCoeffsL;
    undistL.R            = h.R1;
    undistL.P            = P1_3x3;
    undistL.deviceId     = cfg.deviceId;
    undistL.validate();
    UndistortPointsParams undistR;
    undistR.cameraMatrix = h.cameraMatrixR;
    undistR.distCoeffs   = h.distCoeffsR;
    undistR.R            = h.R2;
    undistR.P            = P2_3x3;
    undistR.deviceId     = cfg.deviceId;
    undistR.validate();
    UndistortPointsCuda undistLOp(undistL), undistROp(undistR);

    EpipolarInterpParams epipolarParams;
    epipolarParams.deviceId = cfg.deviceId;
    epipolarParams.lineIdCheck = true;
    EpipolarInterpCuda epipolarL(epipolarParams), epipolarR(epipolarParams);

    LaserMatchParams matchParams;
    matchParams.deviceId = cfg.deviceId;
    // 视差上界从 Q + depthMin 推导（默认 500 对应 Z>=342mm，近距姿态全拒）:
    //   d = f·Tx / Z,  f = Q(2,3), 1/Tx = |Q(3,2)|
    {
        const double fPx = h.Q.at<double>(2, 3);
        const double invTx = std::abs(h.Q.at<double>(3, 2));
        if (invTx > 0 && cfg.depthMin > 0)
            matchParams.max_disparity =
                static_cast<float>(std::abs(fPx / invTx) / cfg.depthMin * 1.05);
        spdlog::info("match max_disparity = {:.1f} (f*Tx/{:.0f}mm)",
                     matchParams.max_disparity, cfg.depthMin);
    }
    LaserMatchCuda matchOp(matchParams);

    LaserReconstructParams reconParams;
    reconParams.minDepth = cfg.depthMin;
    reconParams.maxDepth = cfg.depthMax;
    reconParams.deviceId  = cfg.deviceId;
    LaserReconstructCuda reconOp(reconParams);

    std::vector<FrameRecord> records;
    // 按姿态分组的 3D 点（PJC 输入；与 laser_calib_cli 同款累积）
    std::vector<calib::PosePointSet> poseSets(input->poseFrames.size());

    for (size_t pi = 0; pi < input->poseFrames.size(); ++pi) {
        const std::string& poseName = input->poseDirs[pi];
        const auto& tubes = input->poseFrames[pi];
        for (size_t ti = 0; ti < tubes.size(); ++ti) {
            const auto& f = tubes[ti];
            FrameRecord rec;
            rec.pose = poseName;
            rec.tube = static_cast<int>(ti);
            const std::string stem = poseName + "_tube" + std::to_string(ti);
            spdlog::info("frame {}: start", stem);

            auto fail = [&](const std::string& step, const std::string& msg) {
                rec.failStep = step;
                spdlog::warn("frame {}: {} failed: {}", stem, step, msg);
                records.push_back(rec);
            };

            auto maskResL = maskL.Execute(f.leftGray, stream);
            auto maskResR = maskR.Execute(f.rightGray, stream);
            if (!maskResL.success || !maskResR.success
                || !maskResL.d_cleanedMask || !maskResR.d_cleanedMask
                || maskResL.d_cleanedMask->empty() || maskResR.d_cleanedMask->empty()) {
                fail("4-1", "L=" + std::to_string(maskResL.success) +
                            " R=" + std::to_string(maskResR.success));
                continue;
            }
            {
                cv::Mat mL, mR;
                maskResL.d_cleanedMask->download(mL);
                maskResR.d_cleanedMask->download(mR);
                cv::imwrite(outDir + "/" + stem + "_L_1mask.png", mL);
                cv::imwrite(outDir + "/" + stem + "_R_1mask.png", mR);
            }

            auto cclResL = cclL.Execute(maskResL.d_cleanedMask, stream);
            auto cclResR = cclR.Execute(maskResR.d_cleanedMask, stream);
            if (!cclResL.success || !cclResR.success
                || !cclResL.d_labeledMask || !cclResR.d_labeledMask) {
                fail("4-2", "L=" + std::to_string(cclResL.success) +
                            " R=" + std::to_string(cclResR.success));
                continue;
            }
            {
                cv::Mat mL, mR;
                cclResL.d_labeledMask->download(mL);
                cclResR.d_labeledMask->download(mR);
                cv::imwrite(outDir + "/" + stem + "_L_2ccl.png", colorizeIds(mL));
                cv::imwrite(outDir + "/" + stem + "_R_2ccl.png", colorizeIds(mR));
            }

            auto labelResL = labelL.Execute(*cclResL.d_labeledMask, stream);
            auto labelResR = labelR.Execute(*cclResR.d_labeledMask, stream);
            if (!labelResL.success || !labelResR.success
                || !labelResL.d_labeledMask || !labelResR.d_labeledMask) {
                fail("4-3", "L=" + std::to_string(labelResL.success) +
                            " R=" + std::to_string(labelResR.success));
                continue;
            }
            {
                cv::Mat mL, mR;
                labelResL.d_labeledMask->download(mL);
                labelResR.d_labeledMask->download(mR);
                cv::Mat cL = colorizeIds(mL), cR = colorizeIds(mR);
                annotateMaskIds(cL, mL);
                annotateMaskIds(cR, mR);
                cv::imwrite(outDir + "/" + stem + "_L_3label.png", cL);
                cv::imwrite(outDir + "/" + stem + "_R_3label.png", cR);
            }

            if (!maskResL.d_grayImage || !maskResR.d_grayImage) {
                fail("4-4", "gray null");
                continue;
            }
            auto stegerResL = stegerL.Execute(*maskResL.d_grayImage,
                                              *labelResL.d_labeledMask,
                                              stream, GroupMode::ByLabel);
            auto stegerResR = stegerR.Execute(*maskResR.d_grayImage,
                                              *labelResR.d_labeledMask,
                                              stream, GroupMode::ByLabel);
            if (!stegerResL.success || !stegerResR.success
                || !stegerResL.d_centerPoints || !stegerResR.d_centerPoints
                || !stegerResL.d_line_ids    || !stegerResR.d_line_ids) {
                fail("4-4", "L=" + std::to_string(stegerResL.success) +
                            " R=" + std::to_string(stegerResR.success));
                continue;
            }
            {
                cv::Mat grayL, grayR, ptsL, ptsR, idsL, idsR;
                maskResL.d_grayImage->download(grayL);
                maskResR.d_grayImage->download(grayR);
                stegerResL.d_centerPoints->download(ptsL);
                stegerResR.d_centerPoints->download(ptsR);
                stegerResL.d_line_ids->download(idsL);
                stegerResR.d_line_ids->download(idsR);
                if (!ptsL.empty()) {
                    cv::Mat imgL = drawPointsOn(grayL, ptsL, idsL);
                    annotateLineIds(imgL, ptsL, idsL);
                    cv::imwrite(outDir + "/" + stem + "_L_4steger.png", imgL);
                }
                if (!ptsR.empty()) {
                    cv::Mat imgR = drawPointsOn(grayR, ptsR, idsR);
                    annotateLineIds(imgR, ptsR, idsR);
                    cv::imwrite(outDir + "/" + stem + "_R_4steger.png", imgR);
                }
            }

            auto undistResL = undistLOp.Execute(*stegerResL.d_centerPoints,
                                                *stegerResL.d_line_ids, stream);
            auto undistResR = undistROp.Execute(*stegerResR.d_centerPoints,
                                                *stegerResR.d_line_ids, stream);
            if (!undistResL.success || !undistResR.success
                || !undistResL.d_rectifiedPoints || !undistResR.d_rectifiedPoints
                || !undistResL.d_line_ids      || !undistResR.d_line_ids) {
                fail("4-5", "L=" + std::to_string(undistResL.success) +
                            " R=" + std::to_string(undistResR.success));
                continue;
            }

            auto epipolarResL = epipolarL.Execute(*undistResL.d_rectifiedPoints,
                                                  *undistResL.d_line_ids, stream);
            auto epipolarResR = epipolarR.Execute(*undistResR.d_rectifiedPoints,
                                                  *undistResR.d_line_ids, stream);
            if (!epipolarResL.success || !epipolarResR.success
                || !epipolarResL.d_interpPoints || !epipolarResR.d_interpPoints
                || !epipolarResL.d_interp_line_ids || !epipolarResR.d_interp_line_ids) {
                fail("4-6", "L=" + std::to_string(epipolarResL.success) +
                            " R=" + std::to_string(epipolarResR.success));
                continue;
            }
            {
                cv::Mat ptsL, ptsR, idsL, idsR;
                epipolarResL.d_interpPoints->download(ptsL);
                epipolarResR.d_interpPoints->download(ptsR);
                epipolarResL.d_interp_line_ids->download(idsL);
                epipolarResR.d_interp_line_ids->download(idsR);
                if (!ptsL.empty()) {
                    cv::Mat imgL = drawInterpBlack(h.imageSize, ptsL, idsL);
                    annotateLineIds(imgL, ptsL, idsL);
                    cv::imwrite(outDir + "/" + stem + "_L_6interp.png", imgL);
                }
                if (!ptsR.empty()) {
                    cv::Mat imgR = drawInterpBlack(h.imageSize, ptsR, idsR);
                    annotateLineIds(imgR, ptsR, idsR);
                    cv::imwrite(outDir + "/" + stem + "_R_6interp.png", imgR);
                }

                auto matchRes = matchOp.Execute(*epipolarResL.d_interpPoints,
                                                *epipolarResL.d_interp_line_ids,
                                                *epipolarResR.d_interpPoints,
                                                *epipolarResR.d_interp_line_ids,
                                                stream);
                if (!matchRes.success || !matchRes.d_matched_left
                    || !matchRes.d_matched_right || !matchRes.d_matched_line_ids) {
                    fail("4-7", matchRes.message.empty()
                                ? "success=false" : matchRes.message);
                    continue;
                }
                if (matchRes.matchCount == 0) {
                    fail("4-7", "0 matches (disparity/lineId all rejected)");
                    continue;
                }
                rec.matchCount = matchRes.matchCount;
                {
                    cv::Mat mL, mR, mIds;
                    matchRes.d_matched_left->download(mL);
                    matchRes.d_matched_right->download(mR);
                    matchRes.d_matched_line_ids->download(mIds);
                    if (!mL.empty() && !mR.empty()) {
                        cv::Mat panel = drawMatchPanel(h.imageSize, ptsL, idsL, ptsR, idsR,
                                                       mL, mR, mIds);
                        annotateLineIds(panel, mL, mIds, 0);
                        annotateLineIds(panel, mR, mIds, h.imageSize.width);
                        cv::imwrite(outDir + "/" + stem + "_7match.png", panel);
                    }
                }

                auto reconRes = reconOp.Execute(*matchRes.d_matched_left,
                                                *matchRes.d_matched_right,
                                                *matchRes.d_matched_line_ids,
                                                h.Q, stream);
                if (!reconRes.success || !reconRes.d_points3d
                    || !reconRes.d_valid_line_ids || reconRes.d_points3d->empty()) {
                    fail("4-8", reconRes.message.empty()
                                ? "success=false" : reconRes.message);
                    continue;
                }
                {
                    cv::Mat pts3d, ids3d;
                    reconRes.d_points3d->download(pts3d);
                    reconRes.d_valid_line_ids->download(ids3d);
                    pts3d = pts3d.reshape(3, 1);
                    ids3d = ids3d.reshape(1, 1);
                    writePly(outDir + "/" + stem + "_8cloud.ply", pts3d, ids3d);
                    rec.pointCount = pts3d.cols;
                    std::set<int> lids(ids3d.begin<int>(), ids3d.end<int>());
                    rec.lineIds.assign(lids.begin(), lids.end());
                    auto& ps = poseSets[pi];
                    ps.points3d.insert(ps.points3d.end(),
                                       pts3d.begin<cv::Vec3f>(),
                                       pts3d.end<cv::Vec3f>());
                    ps.lineIds.insert(ps.lineIds.end(),
                                      ids3d.begin<int>(),
                                      ids3d.end<int>());
                }
                rec.ok = true;
                records.push_back(rec);
                spdlog::info("frame {}: OK (matched={}, points={}, lines={})",
                             stem, rec.matchCount, rec.pointCount, rec.lineIds.size());
            }
        }
    }

    // ------------------------------------------------------------------
    // PJC 虚拟相机标定（模型: t(3)+发射曲线 C(6); R=I、K=f+主点 固定——
    // 投影机与左相机绝对轴线平行; 发射曲线承载激光曲率）
    // ------------------------------------------------------------------
    nlohmann::json pjcJson = runPjc(poseSets, h);
    bool pjcOk = !pjcJson.is_null();

    nlohmann::json frames = nlohmann::json::array();
    long long totalPoints = 0;
    int okFrames = 0;
    for (const auto& r : records) {
        nlohmann::json j;
        j["pose"] = r.pose;
        j["tube"] = r.tube;
        j["ok"] = r.ok;
        j["failStep"] = r.failStep;
        j["matchCount"] = r.matchCount;
        j["pointCount"] = r.pointCount;
        j["lineIds"] = r.lineIds;
        frames.push_back(std::move(j));
        if (r.ok) { ++okFrames; totalPoints += r.pointCount; }
    }
    nlohmann::json summary;
    summary["schema"] = "fcstepdump/1.0";
    summary["inputDir"] = inDir;
    summary["totalFrames"] = records.size();
    summary["okFrames"] = okFrames;
    summary["failedFrames"] = records.size() - static_cast<size_t>(okFrames);
    summary["totalPoints"] = totalPoints;
    summary["pjc"] = pjcJson.is_null() ? nlohmann::json{{"success", false}} : pjcJson;
    summary["frames"] = std::move(frames);
    if (!writeLaserJson(outDir + "/summary.json", summary)) {
        spdlog::error("write summary.json failed");
        return 1;
    }

    spdlog::info("done: {}/{} frames ok, {} points total",
                 okFrames, records.size(), totalPoints);
    return (okFrames == static_cast<int>(records.size()) && !records.empty()) ? 0 : 1;

    } catch (const std::exception& e) {
        spdlog::error("exception: {}", e.what());
        return 1;
    } catch (...) {
        spdlog::error("unknown exception");
        return 1;
    }
}
