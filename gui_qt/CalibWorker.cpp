#include "CalibWorker.h"

#include "calib_runner.h"   // fc::runCameraCalib（module1）
// 模块2 的 calib_runner.h 同名，但因为我们只引用 fc::runLaserCalib 符号，
// 而 module2 的 .h 也声明了同名 namespace 函数，所以直接前向声明即可。
namespace fc { bool runLaserCalib(const std::string&, const std::string&,
                                  const CalibCallbacks&); }

#include <QtConcurrent>
#include <spdlog/spdlog.h>

namespace fc::gui {

CalibWorker::CalibWorker(QObject* parent) : QObject(parent) {
    watcher_ = new QFutureWatcher<bool>(this);
    connect(watcher_, &QFutureWatcher<bool>::finished, this, [this]() {
        bool ok = watcher_->result();
        emit done(ok, ok ? QStringLiteral("成功")
                         : QStringLiteral("失败（详见日志）"));
    });
}

CalibWorker::~CalibWorker() = default;

void CalibWorker::startCamera(const QString& inputDir, const QString& outputPath) {
    run_(Mode::Camera, inputDir, outputPath);
}

void CalibWorker::startLaser(const QString& inputDir, const QString& outputPath) {
    run_(Mode::Laser, inputDir, outputPath);
}

bool CalibWorker::isRunning() const { return watcher_->isRunning(); }

void CalibWorker::wait() {
    if (watcher_ && watcher_->isRunning()) {
        watcher_->waitForFinished();
    }
}

void CalibWorker::run_(Mode mode, const QString& inputDir, const QString& outputPath) {
    if (watcher_->isRunning()) {
        emit logLine(QStringLiteral("[warn] 已有任务在跑，忽略新请求"));
        return;
    }
    std::string inDir  = inputDir.toStdString();
    std::string outPath = outputPath.toStdString();
    CalibCallbacks cb;
    cb.onProgress = [this](int s, int t, const std::string& m) {
        emit progress(s, t, QString::fromUtf8(m.data(), static_cast<int>(m.size())));
    };
    cb.onLog = [this](const std::string& line) {
        emit logLine(QString::fromUtf8(line.data(), static_cast<int>(line.size())));
    };
    cb.shouldCancel = []() { return false; };

    auto fn = [mode, inDir, outPath, cb]() -> bool {
        try {
            if (mode == Mode::Camera) {
                return fc::runCameraCalib(inDir, outPath, cb);
            } else {
                return fc::runLaserCalib(inDir, outPath, cb);
            }
        } catch (const std::exception& e) {
            spdlog::error("CalibWorker exception: {}", e.what());
            return false;
        }
    };

    emit logLine(QStringLiteral("--- 启动 %1 ---").arg(
        mode == Mode::Camera ? QStringLiteral("相机标定") : QStringLiteral("激光标定")));
    watcher_->setFuture(QtConcurrent::run(fn));
}

}  // namespace fc::gui
