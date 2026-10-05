// temp_param_interpolator.h — 温度参数插值器（curve_map 温度表生成链 Task 3）
//
// 设计: docs/plans/2026-09-01-curve-map-temp-table-design.md（硬规则 #1/#4/#5）
// 派生配方与 fcscan_e2e.cpp:106-133 的 CurveMapInput 组装一致:
//   f = P1(0,0); pp = (P1(0,2), P1(1,2)); B = |P2(0,3)|/f; virtualT = v2l 补偿 T
// 硬规则 #1: roi 恒取参考节点 validRoiLeft, 任何温度（含越界 clamp）永不插值
// 硬规则 #4: atReference() 零插值直取参考节点——独立于 at() 的代码路径,
//            供 Task 4 参考档与 fcstepdump 金标准逐位对拍（浮点路径不同即不可能逐位同）
// 越界: clamp 到端点档直取值＋clamped=true（源表越界检查前置由生成器负责, 此处仅兜底）

#pragma once

#include <opencv2/core.hpp>
#include <cstddef>
#include <vector>
#include "common/scanner_api.h"
#include "stereo_rectify_temp_table_cpu.h"
#include "laser_extrinsic_compensate_cpu.h"

namespace calib {

/// 某温度下的 CurveMapInput 温度相关参数（派生值）
struct InterpParams {
    double f = 0.0;                 ///< 焦距（P1(0,0)）
    cv::Point2d principalPoint;     ///< 主点（(P1(0,2), P1(1,2))）
    double baseline = 0.0;          ///< 基线（|P2(0,3)|/f）
    cv::Rect roi;                   ///< 恒冻结参考档 validRoiLeft（硬规则 #1）
    cv::Vec3d virtualT;             ///< 虚拟相机光心平移（v2l 补偿表 T）
    bool clamped = false;           ///< 查询温度越界, 参数为端点档直取
};

/// 两源表 → 逐温度插值器（构造期校验/排序/参考节点定位, 查询纯 const）
class SCANNER_API TempParamInterpolator {
public:
    static constexpr const char* kLogTag = "12b-TempParamInterpolator";

    /// @param rectify     模块1 立体矫正温度表（f/pp/B/roi 源）
    /// @param laserExtrin 模块2 激光外参补偿表（取 virtual→left 组作 virtualT 源）
    /// @throws std::invalid_argument 空表 / P1·P2 缺失或非 CV_64F / f≤0 /
    ///                              参考温度节点缺失 / 两表参考温度不一致
    TempParamInterpolator(const StereoRectifyTempTableResult& rectify,
                          const LaserExtrinsicCompensateCPUResult& laserExtrin);

    /// 精确节点→直取; 非节点→f/pp/B/virtualT 线性插值; 越界→端点档直取＋clamped=true。
    /// roi 在所有分支恒为参考节点 validRoiLeft（硬规则 #1）。
    /// @throws std::invalid_argument tempC 为 NaN
    InterpParams at(double tempC) const;

    /// 硬规则 #4: 零插值直取参考节点（独立代码路径, 不过任何插值/clamp 分支）
    InterpParams atReference() const;

    double referenceTemp() const { return referenceTemp_; }
    size_t nodeCount() const { return rect_.size(); }

private:
    struct RectNode {                ///< 每档矫正派生量
        double temp = 0.0;
        double f = 0.0;
        cv::Point2d pp;
        double baseline = 0.0;
    };
    struct LaserNode {               ///< 每档 v2l 补偿 T
        double temp = 0.0;
        cv::Vec3d virtualT;
    };

    std::vector<RectNode> rect_;     ///< 构造期按温度升序
    std::vector<LaserNode> laser_;   ///< 构造期按温度升序
    cv::Rect refRoi_;                ///< 参考节点 validRoiLeft（冻结）
    size_t rectRef_ = 0;             ///< 参考节点索引（排序后）
    size_t laserRef_ = 0;
    double referenceTemp_ = 0.0;
};

} // namespace calib
