#include "MainWindow.h"
#include "AcquisitionTab.h"
#include "CameraCalibTab.h"
#include "LaserCalibTab.h"
#include "SpdlogBridge.h"

#include <QApplication>
#include <QCloseEvent>
#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QDockWidget>
#include <QFile>
#include <QFileInfo>
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

    // 标定启动 → 自动停止扫描仪（电机/激光/预览），避免标定期间设备继续运转
    connect(cameraTab_, &CameraCalibTab::calibrationStarted,
            acquisitionTab_, &AcquisitionTab::stopScannerIfRunning);
    connect(laserTab_,  &LaserCalibTab::calibrationStarted,
            acquisitionTab_, &AcquisitionTab::stopScannerIfRunning);
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

void MainWindow::closeEvent(QCloseEvent* event) {
    // 关闭窗口时备份并清理 data_in 图像（QApplication::quit 不经过这里，
    // 菜单退出已改为 close() 以保证走本钩子）
    backupDataIn();
    QMainWindow::closeEvent(event);
}

void MainWindow::backupDataIn() {
    // data_bak 与 data_in 同级（main() 已把 cwd 设为工程根）
    const QString dataIn = QDir::current().filePath(QStringLiteral("data_in"));
    if (!QDir(dataIn).exists()) return;

    // 只备份 camera/ 与 laser/ 下的图像文件（png/jpg/bmp），保留相对路径结构
    const QStringList imgFilters = { QStringLiteral("*.png"),
                                     QStringLiteral("*.jpg"),
                                     QStringLiteral("*.bmp") };
    struct Item { QString abs; QString rel; };
    QVector<Item> items;
    QDirIterator it(dataIn, imgFilters, QDir::Files, QDirIterator::Subdirectories);
    const QDir inDir(dataIn);
    while (it.hasNext()) {
        it.next();
        const QString rel = inDir.relativeFilePath(it.filePath());
        if (!rel.startsWith(QStringLiteral("camera/"))
            && !rel.startsWith(QStringLiteral("laser/"))) {
            continue;
        }
        items.append({ it.filePath(), rel });
    }
    if (items.isEmpty()) return;  // 无图像 → 不建空备份、不清除、不打扰

    const QString stamp = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss"));
    const QString destRoot = QDir::current().filePath(
        QStringLiteral("data_bak/") + stamp);

    // 逐个复制；任何失败 → 中止且绝不清除 data_in（保数据优先）
    for (const auto& im : items) {
        const QString dest = destRoot + QLatin1Char('/') + im.rel;
        QDir().mkpath(QFileInfo(dest).absolutePath());
        if (!QFile::copy(im.abs, dest)
            || QFileInfo(dest).size() != QFileInfo(im.abs).size()) {
            spdlog::error("[backup] copy failed: {} -> {}",
                          im.abs.toStdString(), dest.toStdString());
            QMessageBox::warning(
                this, QStringLiteral("备份失败"),
                QStringLiteral("备份失败，data_in 原始数据已保留。\n文件: %1").arg(im.rel));
            return;
        }
    }

    // 备份全部成功 → 删除 data_in 中图像，目录结构原样保留
    for (const auto& im : items) QFile::remove(im.abs);

    spdlog::info("[backup] {} images -> {} ; data_in images cleared",
                 items.size(), destRoot.toStdString());
    QMessageBox::information(
        this, QStringLiteral("备份完成"),
        QStringLiteral("已备份 %1 张图像到\n%2\n\ndata_in 中的图像已清除（目录结构保留）。")
            .arg(items.size())
            .arg(QDir::toNativeSeparators(destRoot)));
}

void MainWindow::buildMenu() {
    auto* fileMenu = menuBar()->addMenu(QStringLiteral("文件(&F)"));
    // 用 close() 而非 QApplication::quit()：close 会触发 closeEvent，
    // 从而执行关闭时的 data_in 备份清理（quit 直接退出事件循环，绕过 closeEvent）
    fileMenu->addAction(QStringLiteral("退出(&X)"), this, &MainWindow::close,
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
