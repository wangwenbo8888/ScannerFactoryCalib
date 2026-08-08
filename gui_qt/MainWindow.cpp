#include "MainWindow.h"
#include "AcquisitionTab.h"
#include "CameraCalibTab.h"
#include "LaserCalibTab.h"
#include "SpdlogBridge.h"

#include <QApplication>
#include <QDockWidget>
#include <QPointer>
#include <QMenuBar>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QStatusBar>
#include <QTabWidget>
#include <QStatusBar>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace fc::gui {

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(QStringLiteral("factory_calib — 厂家标定工具（统一版）"));
    resize(1400, 900);

    buildCentral();
    buildLogDock();
    buildMenu();
    installSpdlogBridge();

    statusBar()->showMessage(QStringLiteral("就绪"));
}

MainWindow::~MainWindow() {
    // 必须在子控件 (globalLog_) 析构之前，把 SpdlogBridge 从 spdlog default
    // logger 摘除。否则其它线程的 spdlog 日志会在析构期间继续投递到即将失效的
    // globalLog_，引发 Qt5Widgets 内读空 d_ptr 的崩溃。
    if (spdlogBridge_) spdlogBridge_->uninstall();
    // 断开各 tab→MainWindow 的 statusMessage 连接，避免 ~QMainWindow 删除
    // 子 tab 时残留信号触发 fwd 访问已析构的 statusBar()（0xC0000005 读 0x8 根因）。
    if (acquisitionTab_) acquisitionTab_->disconnect(this);
    if (cameraTab_)      cameraTab_->disconnect(this);
    if (laserTab_)       laserTab_->disconnect(this);
}

void MainWindow::buildCentral() {
    tabs_ = new QTabWidget(this);
    acquisitionTab_ = new AcquisitionTab;
    cameraTab_      = new CameraCalibTab;
    laserTab_       = new LaserCalibTab;

    tabs_->addTab(acquisitionTab_, QStringLiteral("① 图像采集"));
    tabs_->addTab(cameraTab_,      QStringLiteral("② 相机标定"));
    tabs_->addTab(laserTab_,       QStringLiteral("③ 激光线标定"));
    setCentralWidget(tabs_);

    // 用 QPointer 捕获 this：MainWindow 析构后 QPointer 自动置 null，
    // 避免析构期间残留的 statusMessage 事件触发 fwd 访问已析构的 statusBar()。
    auto fwd = [self = QPointer<MainWindow>(this)](QString m) {
        if (self) self->statusBar()->showMessage(m, 5000);
    };
    connect(acquisitionTab_, &AcquisitionTab::statusMessage, this, fwd);
    connect(cameraTab_,      &CameraCalibTab::statusMessage, this, fwd);
    connect(laserTab_,       &LaserCalibTab::statusMessage, this, fwd);
}

void MainWindow::buildLogDock() {
    logDock_ = new QDockWidget(QStringLiteral("全局日志（spdlog）"), this);
    logDock_->setAllowedAreas(Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);
    globalLog_ = new QPlainTextEdit;
    globalLog_->setReadOnly(true);
    globalLog_->setMaximumBlockCount(20000);
    globalLog_->setStyleSheet(
        "QPlainTextEdit { font-family: Consolas, 'Courier New', monospace;"
        " font-size: 9pt; background: #1a1a1a; color: #cccccc; }");
    logDock_->setWidget(globalLog_);
    addDockWidget(Qt::BottomDockWidgetArea, logDock_);
}

void MainWindow::installSpdlogBridge() {
    spdlogBridge_ = std::make_shared<SpdlogBridge>(this);
    // 业务库用 spdlog::default_logger()，把 SpdlogBridge 和文件 sink 都挂上去
    spdlog::default_logger()->sinks().push_back(spdlogBridge_);
    try {
        auto fileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
            "E:/workfold/factory_calib/factory_calib.log", true);  // 绝对路径 + 每次启动覆盖
        spdlog::default_logger()->sinks().push_back(fileSink);
    } catch (...) {}
    connect(spdlogBridge_.get(), &SpdlogBridge::logLine,
            globalLog_, &QPlainTextEdit::appendPlainText);
}

void MainWindow::buildMenu() {
    auto* fileMenu = menuBar()->addMenu(QStringLiteral("文件(&F)"));
    fileMenu->addAction(QStringLiteral("退出(&X)"), qApp, &QApplication::quit,
                        QKeySequence(QStringLiteral("Ctrl+Q")));

    auto* viewMenu = menuBar()->addMenu(QStringLiteral("视图(&V)"));
    viewMenu->addAction(logDock_->toggleViewAction());

    auto* helpMenu = menuBar()->addMenu(QStringLiteral("帮助(&H)"));
    helpMenu->addAction(QStringLiteral("关于(&A)..."), this, &MainWindow::onAbout,
                        QKeySequence(QStringLiteral("F1")));
}

void MainWindow::onAbout() {
    QMessageBox::about(
        this,
        QStringLiteral("关于 factory_calib"),
        QStringLiteral(
            "<h3>factory_calib GUI</h3>"
            "<p>采集 / 相机标定 / 激光线标定 统一界面。</p>"
            "<p>直接链接 fc1_ops/fc1_io/fc2_ops/fc2_io 业务库，"
            "所有步骤在工作线程内执行；日志通过 spdlog 自定义 sink 实时显示。</p>"
            "<p style='color:#888;'>Qt %1 / OpenCV 4.13 / CUDA 12.6</p>")
            .arg(QString::fromUtf8(qVersion())));
}

}  // namespace fc::gui
