/**
 * @file calib_logging.h
 * @brief 标定流程日志宏定义
 */

#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/format.h>  // for fmt::format if needed
#include <string>

// 日志标签定义宏
#define CALIB_DEFINE_LOG_TAG(module_id, name) \
    static constexpr const char* kLogTag = #name;

// 日志输出宏（自动附加算子标签）
#define CALIB_LOG_DEBUG(...)    SPDLOG_DEBUG("[{}] {}", kLogTag, fmt::format(__VA_ARGS__))
#define CALIB_LOG_INFO(...)     SPDLOG_INFO("[{}] {}", kLogTag, fmt::format(__VA_ARGS__))
#define CALIB_LOG_WARN(...)     SPDLOG_WARN("[{}] {}", kLogTag, fmt::format(__VA_ARGS__))
#define CALIB_LOG_ERROR(...)    SPDLOG_ERROR("[{}] {}", kLogTag, fmt::format(__VA_ARGS__))

// 简化版本（用于不需要格式化的日志）
#define CALIB_LOG_DEBUG_SIMPLE(msg) SPDLOG_DEBUG("[{}] {}", kLogTag, msg)
#define CALIB_LOG_INFO_SIMPLE(msg)  SPDLOG_INFO("[{}] {}", kLogTag, msg)
#define CALIB_LOG_WARN_SIMPLE(msg)  SPDLOG_WARN("[{}] {}", kLogTag, msg)
#define CALIB_LOG_ERROR_SIMPLE(msg) SPDLOG_ERROR("[{}] {}", kLogTag, msg)
