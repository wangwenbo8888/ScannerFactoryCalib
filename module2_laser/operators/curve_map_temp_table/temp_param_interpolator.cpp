// temp_param_interpolator.cpp — 温度参数插值器实现
//
// 设计: docs/plans/2026-09-01-curve-map-temp-table-design.md（Task 3）

#include "temp_param_interpolator.h"
#include "common/calib_logging.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace calib {

CALIB_DEFINE_LOG_TAG(12, TempParamInterpolator);

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态（构造期校验/排序/参考节点定位, at()/atReference() 纯 const 读）
// 并发策略: 多线程可并发调用 at()/atReference()
// ==============================

namespace {

/// 定位结果: exact（节点命中）或 clamped（越界）→ lo==hi 直取（w 无意义）;
/// 否则 [lo,hi] 为包夹区间, w 为线性插值权重。
/// 节点命中用精确相等——成立依据: 调用方与表生成方以相同算式（ref + k×step）
/// 计算温度时 double 逐位相同, 与步距是否为有限小数无关。
struct Bracket {
    size_t lo = 0, hi = 0;
    double w = 0.0;
    bool exact = false;
    bool clamped = false;
};

template <typename NodeT>
Bracket bracket(const std::vector<NodeT>& nodes, double t) {
    if (std::isnan(t)) {   // NaN 与一切比较为 false → 二分路径会越界, 入口显式拒绝
        CALIB_LOG_ERROR("temperature is NaN");
        throw std::invalid_argument("TempParamInterpolator: at() temperature must not be NaN");
    }
    Bracket b;
    if (t <= nodes.front().temp) {
        b.lo = b.hi = 0;
        b.exact = (t == nodes.front().temp);
        b.clamped = !b.exact;
        return b;
    }
    if (t >= nodes.back().temp) {
        b.lo = b.hi = nodes.size() - 1;
        b.exact = (t == nodes.back().temp);
        b.clamped = !b.exact;
        return b;
    }
    const auto hiIt = std::upper_bound(
        nodes.begin(), nodes.end(), t,
        [](double v, const NodeT& n) { return v < n.temp; });
    const size_t hi = static_cast<size_t>(hiIt - nodes.begin());
    const size_t lo = hi - 1;
    if (nodes[lo].temp == t) {
        b.lo = b.hi = lo;
        b.exact = true;
        return b;
    }
    b.lo = lo;
    b.hi = hi;
    b.w = (t - nodes[lo].temp) / (nodes[hi].temp - nodes[lo].temp);
    return b;
}

} // namespace

TempParamInterpolator::TempParamInterpolator(const StereoRectifyTempTableResult& rectify,
                                             const LaserExtrinsicCompensateCPUResult& laserExtrin) {
    if (rectify.table.empty()) {
        CALIB_LOG_ERROR("stereo rectify temp table empty");
        throw std::invalid_argument("TempParamInterpolator: stereo rectify temp table is empty");
    }
    const auto& v2l = laserExtrin.leftResult.table;
    if (v2l.empty()) {
        CALIB_LOG_ERROR("laser virtual-to-left temp table empty");
        throw std::invalid_argument("TempParamInterpolator: laser virtual-to-left temp table is empty");
    }

    // 两表参考温度必须一致（atReference() 的 f/pp/B 与 virtualT 须出自同一参考档）
    if (rectify.referenceTemp != laserExtrin.referenceTemp) {
        CALIB_LOG_ERROR("reference temp mismatch: rectify {}C vs laser {}C",
                        rectify.referenceTemp, laserExtrin.referenceTemp);
        throw std::invalid_argument(
            "TempParamInterpolator: reference temp mismatch (rectify " +
            std::to_string(rectify.referenceTemp) + "C, laser " +
            std::to_string(laserExtrin.referenceTemp) + "C)");
    }

    // 硬规则 #1/#4 前提: 参考节点在两表各自可定位（排序前在原始表上定位）
    referenceTemp_ = rectify.referenceTemp;
    bool rectRefFound = false;
    for (const auto& e : rectify.table) {
        if (e.temperature == rectify.referenceTemp) {
            refRoi_ = e.validRoiLeft;
            rectRefFound = true;
            break;
        }
    }
    bool laserRefFound = false;
    for (const auto& e : v2l) {
        if (e.temperature == laserExtrin.referenceTemp) {
            laserRefFound = true;
            break;
        }
    }
    if (!rectRefFound) {
        CALIB_LOG_ERROR("reference temp {}C not found in stereo rectify table", rectify.referenceTemp);
        throw std::invalid_argument(
            "TempParamInterpolator: reference temp " +
            std::to_string(rectify.referenceTemp) + "C not found in stereo rectify table");
    }
    if (!laserRefFound) {
        CALIB_LOG_ERROR("reference temp {}C not found in laser v2l table", laserExtrin.referenceTemp);
        throw std::invalid_argument(
            "TempParamInterpolator: reference temp " +
            std::to_string(laserExtrin.referenceTemp) + "C not found in laser virtual-to-left table");
    }

    // 派生节点（fcscan 配方: f=P1(0,0), pp=(P1(0,2),P1(1,2)), B=|P2(0,3)|/f）
    rect_.reserve(rectify.table.size());
    for (const auto& e : rectify.table) {
        if (e.P1.empty() || e.P1.rows < 2 || e.P1.cols < 3 || e.P1.type() != CV_64F ||
            e.P2.empty() || e.P2.rows < 1 || e.P2.cols < 4 || e.P2.type() != CV_64F) {
            CALIB_LOG_ERROR("entry {}C: P1/P2 missing or not CV_64F", e.temperature);
            throw std::invalid_argument("TempParamInterpolator: entry P1/P2 must be CV_64F (>=2x3 / >=1x4)");
        }
        RectNode n;
        n.temp = e.temperature;
        n.f = e.P1.at<double>(0, 0);
        n.pp = cv::Point2d(e.P1.at<double>(0, 2), e.P1.at<double>(1, 2));
        if (!(n.f > 0.0)) {
            CALIB_LOG_ERROR("entry {}C: P1(0,0)={} not positive", e.temperature, n.f);
            throw std::invalid_argument(
                "TempParamInterpolator: P1(0,0) must be > 0 (got " + std::to_string(n.f) +
                ") at " + std::to_string(e.temperature) + "C");
        }
        n.baseline = std::fabs(e.P2.at<double>(0, 3)) / n.f;
        rect_.push_back(n);
    }

    laser_.reserve(v2l.size());
    for (const auto& e : v2l) {
        LaserNode n;
        n.temp = e.temperature;
        n.virtualT = cv::Vec3d(e.T[0], e.T[1], e.T[2]);
        laser_.push_back(n);
    }

    // 构造期排序（查询路径纯二分）
    std::sort(rect_.begin(), rect_.end(),
              [](const RectNode& a, const RectNode& b) { return a.temp < b.temp; });
    std::sort(laser_.begin(), laser_.end(),
              [](const LaserNode& a, const LaserNode& b) { return a.temp < b.temp; });

    const auto rr = std::find_if(rect_.begin(), rect_.end(),
                                 [this](const RectNode& n) { return n.temp == referenceTemp_; });
    rectRef_ = static_cast<size_t>(rr - rect_.begin());
    const auto lr = std::find_if(laser_.begin(), laser_.end(),
                                 [ref = laserExtrin.referenceTemp](const LaserNode& n) {
                                     return n.temp == ref;
                                 });
    laserRef_ = static_cast<size_t>(lr - laser_.begin());
}

InterpParams TempParamInterpolator::at(double tempC) const {
    InterpParams p;
    p.roi = refRoi_;   // 硬规则 #1: 先冻结 ROI, 后续任何分支不再触碰

    const Bracket br = bracket(rect_, tempC);
    const Bracket bl = bracket(laser_, tempC);
    p.clamped = br.clamped || bl.clamped;

    if (br.exact || br.clamped) {
        // 节点命中/越界 clamp → 直取（零浮点运算, 逐位等于表项派生值）
        const RectNode& n = rect_[br.lo];
        p.f = n.f;
        p.principalPoint = n.pp;
        p.baseline = n.baseline;
    } else {
        const RectNode& a = rect_[br.lo];
        const RectNode& b = rect_[br.hi];
        p.f = a.f + (b.f - a.f) * br.w;
        p.principalPoint.x = a.pp.x + (b.pp.x - a.pp.x) * br.w;
        p.principalPoint.y = a.pp.y + (b.pp.y - a.pp.y) * br.w;
        p.baseline = a.baseline + (b.baseline - a.baseline) * br.w;
    }

    if (bl.exact || bl.clamped) {
        p.virtualT = laser_[bl.lo].virtualT;
    } else {
        const LaserNode& a = laser_[bl.lo];
        const LaserNode& b = laser_[bl.hi];
        for (int i = 0; i < 3; ++i)
            p.virtualT[i] = a.virtualT[i] + (b.virtualT[i] - a.virtualT[i]) * bl.w;
    }
    return p;
}

InterpParams TempParamInterpolator::atReference() const {
    // 硬规则 #4: 零插值——直取参考节点字段, 不经过 at() 的任何插值/clamp 分支
    InterpParams p;
    const RectNode& r = rect_[rectRef_];
    const LaserNode& v = laser_[laserRef_];
    p.f = r.f;
    p.principalPoint = r.pp;
    p.baseline = r.baseline;
    p.roi = refRoi_;
    p.virtualT = v.virtualT;
    p.clamped = false;
    return p;
}

} // namespace calib
