#include "SpdlogBridge.h"

#include <QMetaObject>
#include <algorithm>

namespace fc::gui {

SpdlogBridge::SpdlogBridge(QObject* parent) : QObject(parent) {
    // 复用 spdlog 默认 formatter；这里不做额外 pattern（业务 logger 已格式化）
}

void SpdlogBridge::sink_it_(const spdlog::details::log_msg& msg) {
    spdlog::memory_buf_t formatted;
    base_sink<std::mutex>::formatter_->format(msg, formatted);
    QString line = QString::fromUtf8(formatted.data(),
                                      static_cast<int>(formatted.size()))
                       .trimmed();

    // 关键修复：此刻仍持有 base_sink 的 mutex_（非递归）。但 invokeMethod
    // (QueuedConnection) 仅向本对象所属线程 (GUI) 的事件队列 push 一个事件，
    // 不直接调用 onLogLine —— 因此不会与本锁重入，也不会在持锁期间触碰任何
    // QWidget。onLogLine 稍后在 GUI 线程执行时，锁早已释放，且此时再 emit
    // logLine 给 globalLog_ 全程在 GUI 线程内，生命周期由 Qt 正常管理。
    QMetaObject::invokeMethod(this, "onLogLine", Qt::QueuedConnection,
                              Q_ARG(QString, line));
}

void SpdlogBridge::flush_() { /* no-op */ }

void SpdlogBridge::uninstall() {
    // 从 spdlog default logger 的 sinks 中摘除自己，杜绝后续 sink_it_ 调用。
    auto logger = spdlog::default_logger();
    if (!logger) return;
    auto& sinks = logger->sinks();
    sinks.erase(std::remove_if(sinks.begin(), sinks.end(),
        [this](const std::shared_ptr<spdlog::sinks::sink>& s) {
            return s.get() == this;
        }), sinks.end());
}

}  // namespace fc::gui
