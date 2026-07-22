#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/base_sink.h>

#include <QObject>
#include <QString>
#include <memory>
#include <vector>

namespace fc::gui {

// spdlog 自定义 sink：把所有 level 日志安全转发到 GUI 线程的日志控件。
//
// 线程安全设计（修复原版 Qt5Widgets.dll 0xC0000005 读 0x8 崩溃）：
//   - sink_it_ 会被 spdlog 从【任意线程】调用：GUI 线程、CalibWorker 工作线程、
//     甚至相机 SDK 内部线程。
//   - 旧实现直接 `emit logLine()`，发生在 base_sink 的【非递归】mutex 持有期间：
//       * 同线程 (DirectConnection) 会重入死锁；
//       * 跨线程 + 控件析构时序错乱时，信号投递到已删除的 QPlainTextEdit，
//         读其 d_ptr(nullptr) → Qt5Widgets 内 0xC0000005 读 0x8。
//   - 新实现：sink_it_ 只做格式化，随后 QMetaObject::invokeMethod(this,
//     "onLogLine", QueuedConnection) 把工作【投递回 SpdlogBridge 所属线程】
//     (GUI 线程)。QueuedConnection 仅向事件队列 push 一个事件，不直接调用，
//     故不会与 base_sink 的锁重入；onLogLine 稍后在 GUI 线程执行时锁早已释放。
//   - 退出/析构前必须调 uninstall()，把本 sink 从 default logger 移除，
//     避免静态析构期间 spdlog 继续往已析构对象投递。
//   - SpdlogBridge 生命周期 >= MainWindow（spdlog default logger 持 shared_ptr
//     直到静态析构），故 invokeMethod 期间 this 必然有效。
class SpdlogBridge
    : public QObject,
      public spdlog::sinks::base_sink<std::mutex> {
    Q_OBJECT
public:
    explicit SpdlogBridge(QObject* parent = nullptr);

    // 从 spdlog default logger 的 sinks 中移除自己（主窗口析构前调用一次）。
    void uninstall();

signals:
    // 已格式化的一行日志（仅在 GUI 线程被 emit，下游可放心 DirectConnection）。
    void logLine(QString line);

private slots:
    // 由 sink_it_ 通过 QueuedConnection 投递到 GUI 线程执行；这里再 emit logLine。
    void onLogLine(QString line) { emit logLine(line); }

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override;
    void flush_() override;
};

}  // namespace fc::gui
