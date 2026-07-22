/**
 * @file calib_result_types.h
 * @brief 标定 → 扫描 跨应用数据契约类型
 *
 * 定义标定产物（生产者：calibration/）与扫描（消费者：scanning/）之间
 * 共用的类型契约。本文件只定义类型与纯查询方法（lookup），不含任何
 * 生成逻辑；产物的生成归 calibration/，交付机制（磁盘文件 / 启动加载 /
 * 进程内指针）由应用层决定。
 *
 * 设计依据：docs/plans/2026-06-22-scanner-refactor-design.md §4.6
 */

#pragma once

#include <map>
#include <opencv2/core.hpp>

namespace calib {

// ============================================================
// 立体矫正参数（单温度点）
// 标定 biaoding/05 生产，扫描 marker/06（双目去畸变）+ laser/05（去畸变）消费
// ============================================================

struct StereoRectifyParams {
    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect validRoi;

    bool empty() const {
        return R1.empty() && R2.empty() && P1.empty() && P2.empty() && Q.empty();
    }
};

struct StereoRectifyTempTable {
    double referenceTemperature = 0.0;
    double tempStep = 0.2;                       // °C
    std::map<double, StereoRectifyParams> table; // 温度 → 参数

    bool empty() const { return table.empty(); }

    // 按温度查表：返回温度最接近 tempC 的条目；表为空时返回 nullptr（不抛异常，
    // 对齐算子规范 §9：跨模块数据契约不得用异常传递错误）
    const StereoRectifyParams* lookup(double tempC) const {
        if (table.empty())
            return nullptr;

        auto it = table.lower_bound(tempC);
        if (it == table.end()) {
            // tempC 高于所有键 → 取最大键
            return &table.rbegin()->second;
        }
        if (it == table.begin()) {
            // tempC 低于或等于最小键 → 取最小键
            return &it->second;
        }
        // 在 (prev, it) 之间取更接近者
        auto prev = std::prev(it);
        if ((tempC - prev->first) <= (it->first - tempC))
            return &prev->second;
        return &it->second;
    }
};

// ============================================================
// 激光面映射（单温度点）
// 标定 jiguangbiaoding/13 生产，扫描 scanning/laser/laser_match_scan 消费
// ============================================================

struct LaserPlaneMap {
    double temperature = 0.0;
    // 逐网格左→右极线映射表。布局契约：N×4 CV_32FC1，每行 = [xL, yL, uR, lineId]
    //   - xL, yL : 左相机网格点坐标
    //   - uR     : 对应右相机极线预测横坐标
    //   - lineId : 激光线编号（int 存为 float）
    // 消费者（scanning/LaserMatchScanCuda）经 SetTempTable 注入只读引用。
    // 注：当前生产者（calibration/PlaneMapTempTable）尚未按此契约产出 leftToRightMap，
    // 仅 JSON 路径（mapData: [[xL,yL,uR,lineId],...]）可用，属待集成的已知缺口。
    cv::Mat leftToRightMap;
    cv::Mat rightUMap;
    int totalPairs = 0;

    bool empty() const {
        return leftToRightMap.empty() && rightUMap.empty();
    }
};

struct LaserPlaneMapTempTable {
    double referenceTemperature = 0.0;
    double tempStep = 0.2;
    std::map<double, LaserPlaneMap> table;      // 温度 → 映射表

    bool empty() const { return table.empty(); }

    // 按温度查表：返回温度最接近 tempC 的条目；表为空时返回 nullptr（不抛异常）
    const LaserPlaneMap* lookup(double tempC) const {
        if (table.empty())
            return nullptr;

        auto it = table.lower_bound(tempC);
        if (it == table.end()) {
            return &table.rbegin()->second;
        }
        if (it == table.begin()) {
            return &it->second;
        }
        auto prev = std::prev(it);
        if ((tempC - prev->first) <= (it->first - tempC))
            return &prev->second;
        return &it->second;
    }
};

} // namespace calib
