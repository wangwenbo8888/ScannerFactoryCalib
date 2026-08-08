#include "SpdlogBridge.h"

#include <QMetaObject>
#include <algorithm>
#include <chrono>
#include <thread>

namespace fc::gui {

SpdlogBridge::SpdlogBridge(QObject* parent) : QObject(parent) {
    // 复用 spdlog 默认 formatter；这里不做额外 pattern（业务 logger 已格式化）
}

void SpdlogBridge::sink_it_(const spdlog::details::log_msg& msg) {
    // 析构防护：uninstall 标记 destroyed_ 后，不再 invokeMethod(this)，
    // 避免 GUI 关闭析构期间标定线程的 sink_it_ 访问悬空 this。
    sinkInFlight_.fetch_add(1, std::memory_order_acq_rel);
    struct Guard {
        std::atomic<int>& c;
        ~Guard() { c.fetch_sub(1, std::memory_order_acq_rel); }
    } guard{sinkInFlight_};
    if (destroyed_.load(std::memory_order_acquire)) return;

    spdlog::memory_buf_t formatted;
    base_sink<std::mutex>::formatter_->format(msg, formatted);
    QString line = QString::fromUtf8(formatted.data(),
                                      static_cast<int>(formatted.size()))
                       .trimmed();

    // invokeMethod(QueuedConnection) 仅向 GUI 线程事件队列 push 一个事件，
    // 不直接调用 onLogLine，不会与 base_sink 的锁重入。
    QMetaObject::invokeMethod(this, "onLogLine", Qt::QueuedConnection,
                              Q_ARG(QString, line));
}

void SpdlogBridge::flush_() { /* no-op */ }

void SpdlogBridge::uninstall() {
    // 1. 先标记 destroyed_：新进入的 sink_it_ 检查后直接 return，不再 invokeMethod
    destroyed_.store(true, std::memory_order_release);
    // 2. 等待已进入的 in-flight sink_it_ 退出（本调用返回后保证不再有 sink_it_ 访问 this）
    for (int i = 0; i < 200; ++i) {
        if (sinkInFlight_.load(std::memory_order_acquire) == 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    // 3. 从 spdlog default logger 的 sinks 中摘除自己，杜绝后续 sink_it_ 调用。
    auto logger = spdlog::default_logger();
    if (!logger) return;
    auto& sinks = logger->sinks();
    sinks.erase(std::remove_if(sinks.begin(), sinks.end(),
        [this](const std::shared_ptr<spdlog::sinks::sink>& s) {
            return s.get() == this;
        }), sinks.end());
}

}  // namespace fc::gui
