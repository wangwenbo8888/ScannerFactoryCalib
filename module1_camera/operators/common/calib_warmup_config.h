/**
 * @file calib_warmup_config.h
 * @brief 标定流程预热配置结构体
 */

#pragma once

#include <cstdint>

namespace calib {

/**
 * @brief 预热配置结构体
 * 
 * 统一的预热配置，用于各算子的 warmup() 方法
 */
struct WarmupConfig {
    int rows = 0;           ///< 图像行数
    int cols = 0;           ///< 图像列数
    int maxPointCount = 0;  ///< 最大点数（点云类算子使用）

    WarmupConfig() = default;
    
    WarmupConfig(int r, int c, int mpc = 0) 
        : rows(r), cols(c), maxPointCount(mpc) {}

    /**
     * @brief 为图像类算子创建配置
     */
    static WarmupConfig forImage(int rows, int cols) {
        return WarmupConfig(rows, cols, 0);
    }

    /**
     * @brief 为点云类算子创建配置
     */
    static WarmupConfig forPointCloud(int maxPointCount) {
        return WarmupConfig(0, 0, maxPointCount);
    }

    /**
     * @brief 为混合类算子创建配置（如图像+点云）
     */
    static WarmupConfig forHybrid(int rows, int cols, int maxPointCount) {
        return WarmupConfig(rows, cols, maxPointCount);
    }
};

} // namespace calib
