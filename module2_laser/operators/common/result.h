/**
 * @file result.h
 * @brief 统一错误状态契约（算子规范 §9）
 *
 * 所有算子 *Result 结构体均含 success / message / qualityFlag 三字段。
 * 本文件提供语义化的 ok()/fail() 工厂与 makeFail() 便捷构造，
 * 供算子返回失败结果时统一使用，取代跨边界抛异常。
 */

#pragma once

#include <string>
#include <utility>

#include "quality_flag.h"

namespace calib {

// 统一错误状态。算子 Result 结构体可聚合或对照此结构体保持字段一致。
struct ResultStatus {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    static ResultStatus ok() {
        return ResultStatus{true, std::string{}, QualityFlag::Normal};
    }

    static ResultStatus fail(std::string msg,
                             QualityFlag flag = QualityFlag::Warning) {
        return ResultStatus{false, std::move(msg), flag};
    }
};

// 便捷工厂：为任意含 success/message/qualityFlag 字段的算子 Result 构造失败结果。
// 仅依赖字段名约定（鸭子类型），无需公共基类，避免侵入既有 40+ Result 结构体。
template <class ResultT>
inline ResultT makeFail(std::string msg,
                        QualityFlag flag = QualityFlag::Warning) {
    ResultT r;
    r.success = false;
    r.message = std::move(msg);
    r.qualityFlag = flag;
    return r;
}

} // namespace calib
