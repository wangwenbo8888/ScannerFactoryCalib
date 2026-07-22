/**
 * @file mask_extract_cuda.h
 * @brief 激光掩膜区域提取算�?- 公开头文件（�?C++，不�?CUDA 类型�? *
 * 所属流程：标定流程1�?/ 标定流程2❺（共用�? * 平台：GPU（OpenCV CUDA API�? *
 * 功能：从灰度图像中提取激光线区域的二值掩�? * 本算子是 GPU 激光处理链入口：输入为 cv::Mat（Host 端灰度图），
 * 内部执行 Host→Device 上传（upload �?GpuMat），后续算子全程 GpuMat 传递�? * 左右相机各持独立实例�?6L / 06R），各自上传各自图像�? *
 * 精度容差档次：档次②（整像素/几何类）
 * - CPU vs CUDA 误差 < 0.1px（或整像素一致性）
 * - �?SM 架构�?CUDA 互比差异 < 0.1px
 */

#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

// 前向声明（隔�?CUDA 类型�?namespace cv { namespace cuda { class GpuMat; class Stream; } }
struct WarmupConfig;

/**
 * @brief 激光掩膜提取参�? */
struct MaskExtractParams {
    int threshold = 80;          ///< 二值化阈�?[0, 255]
    int erodeSize = 5;           ///< 腐蚀核大小（奇数，去噪）
    int laserDilateSize = 3;     ///< 激光膨胀核大小（奇数，恢复形状）
    int minArea = 100;           ///< 最小区域面积阈值
    int maxArea = 100000;        ///< 最大区域面积阈值
    /**
    * @brief 参数合法性校�? 
    * @throws std::invalid_argument 参数不合�?     */
    void validate() const {
        if (threshold < 0 || threshold > 255)
            throw std::invalid_argument("MaskExtractParams::threshold must be [0, 255]");
        if (erodeSize < 1 || erodeSize % 2 == 0)
            throw std::invalid_argument("MaskExtractParams::erodeSize must be positive odd");
        if (laserDilateSize < 1 || laserDilateSize % 2 == 0)
            throw std::invalid_argument("MaskExtractParams::laserDilateSize must be positive odd");
        if (minArea < 0)
            throw std::invalid_argument("MaskExtractParams::minArea must be >= 0");
        if (maxArea <= minArea)
            throw std::invalid_argument("MaskExtractParams::maxArea must be > minArea");
    }

    /**
     * @brief 序列化为 JSON
     */
    nlohmann::json toJson() const {
        return {
            {"threshold", threshold},
            {"erodeSize", erodeSize},
            {"laserDilateSize", laserDilateSize},
            {"minArea", minArea},
            {"maxArea", maxArea}
        };
    }

    /**
     * @brief �?JSON 反序列化
     * @throws std::invalid_argument 参数不合�?     */
    static MaskExtractParams fromJson(const nlohmann::json& j) {
        MaskExtractParams p;
        if (j.contains("threshold")) p.threshold = j.at("threshold").get<int>();
        if (j.contains("erodeSize")) p.erodeSize = j.at("erodeSize").get<int>();
        if (j.contains("laserDilateSize")) p.laserDilateSize = j.at("laserDilateSize").get<int>();
        if (j.contains("minArea")) p.minArea = j.at("minArea").get<int>();
        if (j.contains("maxArea")) p.maxArea = j.at("maxArea").get<int>();
        p.validate();
        return p;
    }
};

/**
 * @brief 激光掩膜提取结�? * 
 * 结果结构体包含：
 * - success: 处理是否成功（二值快速分支）
 * - message: 错误或辅助信息（success=false 时保证非空）
 * - qualityFlag: 质量标记（success=true 时有效，三值精细报告）
 * - d_laserMask: GPU 激光区域二值掩膜（d_ 前缀表示 device 端数据）
 * - d_cleanedMask: GPU 面积过滤后的掩膜（d_ 前缀表示 device 端数据）
 * 
 * 设计决策�? * - 使用 shared_ptr<GpuMat> 传�?GPU 数据所有权
 * - 理由：头文件不含 CUDA 类型定义（pImpl 隔离），shared_ptr 允许跨编译单元传�? */
struct MaskExtractResult {
    bool success = false;                                         ///< 处理是否成功
    std::string message;                                          ///< 错误或辅助信息
    QualityFlag qualityFlag = QualityFlag::Normal;                ///< 质量标记

    std::shared_ptr<cv::cuda::GpuMat> d_grayImage;      ///< GPU 灰度图像（Host→Device 上传结果，供下游算子复用）
    std::shared_ptr<cv::cuda::GpuMat> d_laserMask;      ///< GPU 激光区域二值掩膜（device 端）
    std::shared_ptr<cv::cuda::GpuMat> d_cleanedMask;    ///< GPU 面积过滤后的掩膜（device 端）

    MaskExtractResult() = default;
    ~MaskExtractResult() = default;

    // 移动语义
    MaskExtractResult(MaskExtractResult&&) = default;
    MaskExtractResult& operator=(MaskExtractResult&&) = default;

    // 禁止拷贝（GPU 资源独占�?    MaskExtractResult(const MaskExtractResult&) = delete;
    MaskExtractResult& operator=(const MaskExtractResult&) = delete;
};

/**
 * @brief 激光掩膜提取算子（CUDA 实现�? *
 * 使用 pImpl 模式隔离 CUDA 类型，公开头文件不含任�?CUDA 依赖�? * 
 * 线程安全约束�? * - 非线程安全：不同线程不得同时调用 Execute() �?setParams()
 * - 单线程流水线中应在帧间隙调用 setParams()
 * - 多实例并发场景各实例独立持有参数无需加锁
 * - Debug 模式下维�?inProcess_ 原子变量进行断言检�? * - Release 模式下并发调用为未定义行为（数据竞争�? */
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: Impl 仅持有每调用重置的 GPU 暂存缓冲；SetParams 缓存阈值等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API MaskExtractCUDA {
public:
    static constexpr const char* kLogTag = "06-MaskExtractCUDA";

    /**
    * @brief 构造函�? 
    * @param params 激光掩膜提取参�? 
    * @throws std::invalid_argument 参数校验失败
     */
    explicit MaskExtractCUDA(const MaskExtractParams& params = {});

    ~MaskExtractCUDA();

    // 禁止拷贝
    MaskExtractCUDA(const MaskExtractCUDA&) = delete;
    MaskExtractCUDA& operator=(const MaskExtractCUDA&) = delete;

    /**
     * @brief 销毁所有预分配资源（GPU 缓冲等）
     *
     * 必须 idempotent（多次调用安全、无副作用）�?     * 析构前必须调�?Destroy()�?     */
    void Destroy();

    /**
     * @brief 提取激光掩膜（主接口）
     *
     * 输入 Host 端灰度图像（CV_8UC1），内部 upload �?GpuMat�?     * 执行 GPU 形态学流水线（二值化 �?腐蚀 �?膨胀 �?可选面积过滤）�?     * 返回 GPU 端激光掩膜�?     *
    * @param grayImage Host 端灰度图像（CV_8UC1�? 
    * @param stream CUDA 流引用（流水线调用必须显式传入）
     * @return MaskExtractResult 包含 GPU 掩膜和质量信�?     */
    MaskExtractResult Execute(const cv::Mat& grayImage,
                              cv::cuda::Stream& stream);

    /**
     * @brief 提取激光掩膜（使用默认空流，向后兼容）
     *
     * 便利重载，仅用于单算子独立测试；流水线调用必须使�?Execute(gray, stream)�?     */
    MaskExtractResult Execute(const cv::Mat& grayImage);

    /**
     * @brief 预热 GPU 资源
     * @param rows 图像行数
     * @param cols 图像列数
     *
     * 预分�?GPU 缓冲区，首次 Execute() 调用无冷启动延迟�?     * @throws std::runtime_error 显存分配失败
     */
    void Warmup(int rows, int cols);

    /**
     * @brief 使用统一配置预热
     * @param config 预热配置（来�?common/calib_warmup_config.h�?     */
    void Warmup(const WarmupConfig& config);

    /**
    * @brief 动态更新参�? 
    * @param params 新参�? 
    * @throws std::invalid_argument 参数校验失败
     *
    * @note 调用后已�?GPU 缓冲区按新参数尺寸自动调�? 
    * @warning 不得�?Execute() 并发调用
     */
    void SetParams(const MaskExtractParams& params);

    /**
     * @brief 获取当前参数
     */
    const MaskExtractParams& GetParams() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getMaskExtractCUDAInfo();

} // namespace calib