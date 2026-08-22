#pragma once

#include <QMainWindow>
#include <memory>

class QTabWidget;
class QStatusBar;
class QPlainTextEdit;
class QDockWidget;
class QCloseEvent;

namespace fc::gui {

class AcquisitionTab;
class CameraCalibTab;
class LaserCalibTab;
class SpdlogBridge;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;  // 先卸载 spdlog sink，再让子控件析构

protected:
    void closeEvent(QCloseEvent* event) override;

private slots:
    void onAbout();

private:
    void buildMenu();
    void buildCentral();
    void buildLogDock();
    void installSpdlogBridge();
    void backupDataIn();  // 关闭时备份 data_in 图像到 data_bak/<时间戳>/，成功后清除

    QTabWidget* tabs_ = nullptr;
    AcquisitionTab*  acquisitionTab_ = nullptr;
    CameraCalibTab*  cameraTab_      = nullptr;
    LaserCalibTab*   laserTab_       = nullptr;

    QDockWidget*   logDock_   = nullptr;
    QPlainTextEdit* globalLog_ = nullptr;
    std::shared_ptr<SpdlogBridge> spdlogBridge_;
};

}  // namespace fc::gui
