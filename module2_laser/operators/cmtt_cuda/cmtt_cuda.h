// cmtt_cuda.h — curve_map_temp_table 61 档生成器 CUDA 试验版（tests/cmtt_cuda/）
//
// 定位: 性能原型——复刻 operators/curve_map_temp_table 的编排流（参考档先行→冻结基准
//       →逐档插值/重标→生成→冻结断言→CmttBuilder 拼装），仅把每档的
//       CurveMapGenerator 换成 CurveMapCuda（GPU 求交）。
// 复用: TempParamInterpolator / rescaleCurve / cmttTierGridMatches / CmttBuilder
//       均为现有算子公开 API，本目录不改任何现有算子。
// 差异声明: 档级串行（GPU 内部已高度并行）；产物 sidecar 结构与生产一致，
//           字节级不保证与 CPU 版逐位相同（浮点边界），见 curve_map_cuda.h。

#pragma once

#include "curve_map_cuda.h"
#include "curve_map_temp_table.h"

namespace calib {

class CmttCudaGenerator {
public:
    static constexpr const char* kLogTag = "XD-CmttCuda";

    explicit CmttCudaGenerator(const CurveMapTempTableGenParams& params);
    CmttCudaGenerator(const CmttCudaGenerator&) = delete;
    CmttCudaGenerator& operator=(const CmttCudaGenerator&) = delete;

    /// 与 CurveMapTempTableGenerator::Generate 同契约/同返回结构
    CurveMapTempTableResult Generate(
        const std::vector<ImplicitCurve>& curves,
        const StereoRectifyTempTableResult& rectifyTable,
        const LaserExtrinsicCompensateCPUResult& laserExtrinTable,
        const cv::Size& imageSize);

private:
    CurveMapTempTableGenParams params_;
};

} // namespace calib
