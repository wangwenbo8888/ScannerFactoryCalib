// curve_map_cuda.h — curve_map 单档生成器 CUDA 试验版（tests/cmtt_cuda/）
//
// 定位: 性能原型——复刻 operators/curve_map 的 Generate 几何与 E.2 序列化格式,
//       求交热区(GPU kernel + thrust 排序)搬到 GPU, 打包/序列化留在 CPU(复刻逻辑)。
// 复用: 输入/输出/参数类型直接用 calib::CurveMapInput/Result/Params(原算子头),
//       不修改任何现有算子。
// 等价性目标: 结构等价(同 entryCount/行分组/可被 CurveMapTable::Load 解析),
//       非逐位等价(FMA/舍入边界可能有极少量条目差异, 见 -fmad=false 缓解)。
// 确定性: 逐条目独立计算 + 唯一键全序排序, 无原子顺序依赖, 同架构逐次运行可复现。

#pragma once

#include "curve_map.h"

#include <utility>

namespace calib {

class CurveMapCuda {
public:
    static constexpr const char* kLogTag = "XC-CurveMapCuda";

    explicit CurveMapCuda(const CurveMapParams& params = {});
    CurveMapCuda(const CurveMapCuda&) = delete;
    CurveMapCuda& operator=(const CurveMapCuda&) = delete;

    /// 与 CurveMapGenerator::Generate 同契约（tableBytes 格式 E.2 一致）
    CurveMapResult Generate(const CurveMapInput& input);

    void SetParams(const CurveMapParams& params);
    const CurveMapParams& GetParams() const { return params_; }

    // ---- 基线景深窗（2026-10-09；语义别名，见下） ----
    // 基线景深定义：矫正系 z 坐标。平面1=立体矫正后左右相机两光轴所在平面(y=0)；
    // 景深面⊥平面1 且∥基线(X 轴) ⟺ z=const 平面；景深面到基线(z=0)的投影距离=z。
    // 求交内核的 Z=(qc·tz−f·tx)/(q_u−uL) 即该量（推导见
    // docs/plans/2026-10-09-curve-map-cuda-baseline-depth-design.md §2），
    // 故本组接口与 CurveMapParams.depthMin/depthMax 是同一真相源的显式命名。
    /// 设置基线景深窗 [d1,d2]（只对该范围内的激光线条目进行计算）。
    /// 校验：值有限（NaN/Inf 拒绝——validate() 比较式对 NaN 静默放行）且
    /// 0 < d1 < d2（按窄化后 float 值）；违例抛 std::invalid_argument，参数保持原值。
    void SetBaselineDepthWindow(double d1, double d2);
    /// 回读基线景深窗（从 float 字段扩回 double）。
    std::pair<double, double> GetBaselineDepthWindow() const;

private:
    CurveMapParams params_;
};

} // namespace calib
