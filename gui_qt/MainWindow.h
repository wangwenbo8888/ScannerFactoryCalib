#pragma once

#include <QMainWindow>
#include <memory>

class QTabWidget;
class QStatusBar;
class QPlainTextEdit;
class QDockWidget;

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

private slots:
    void onAbout();

private:
    void buildMenu();
    void buildCentral();
    void buildLogDock();
    void installSpdlogBridge();

    QTabWidget* tabs_ = nullptr;
    AcquisitionTab*  acquisitionTab_ = nullptr;
    CameraCalibTab*  cameraTab_      = nullptr;
    LaserCalibTab*   laserTab_       = nullptr;

    QDockWidget*   logDock_   = nullptr;
    QPlainTextEdit* globalLog_ = nullptr;
    std::shared_ptr<SpdlogBridge> spdlogBridge_;
};

}  // namespace fc::gui
