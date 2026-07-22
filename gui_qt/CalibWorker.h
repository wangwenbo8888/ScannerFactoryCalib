#pragma once

#include <QObject>
#include <QString>
#include <QFuture>
#include <QFutureWatcher>
#include <memory>

namespace fc::gui {

// 在工作线程里调用 fc::runCameraCalib 或 fc::runLaserCalib，
// 把进度/日志/完成 通过 Qt 信号发到 UI 线程。
class CalibWorker : public QObject {
    Q_OBJECT
public:
    enum class Mode { Camera, Laser };

    explicit CalibWorker(QObject* parent = nullptr);
    ~CalibWorker() override;

    void startCamera(const QString& inputDir, const QString& outputPath);
    void startLaser(const QString& inputDir, const QString& outputPath);

    bool isRunning() const;

signals:
    // 来自业务库的进度回调
    void progress(int step, int totalSteps, QString msg);
    // 来自 spdlog（已格式化）
    void logLine(QString line);
    // 完成时（成功/失败都发）；success=true 表示业务函数返回 true
    void done(bool success, QString summary);

private:
    void run_(Mode mode, const QString& inputDir, const QString& outputPath);

    QFutureWatcher<bool>* watcher_ = nullptr;
};

}  // namespace fc::gui
