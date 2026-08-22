#pragma once

#include <QWidget>

class QLineEdit;
class QPushButton;
class QProgressBar;
class QPlainTextEdit;
class QLabel;

namespace fc::gui {

class ResultView;
class CalibWorker;

// 激光线标定 tab：调用 fc::runLaserCalib（依赖 data_in/laser/camera_calib.json）。
class LaserCalibTab : public QWidget {
    Q_OBJECT
public:
    explicit LaserCalibTab(QWidget* parent = nullptr);
    ~LaserCalibTab() override;

signals:
    void statusMessage(QString msg);
    void calibrationStarted();   // 标定启动（MainWindow 接线→自动停止扫描仪）

private slots:
    void onBrowseInput();
    void onBrowseOutput();
    void onRunAll();
    void onCopyHandoff();   // 把 data_out/camera_calib.json 拷到 data_in/laser/camera_calib.json
    void onWorkerDone(bool ok, QString summary);

private:
    void appendLog(const QString& msg, const QString& color = "#d4d4d4");
    void setRunningUI(bool running);

    QLineEdit* inputEdit_   = nullptr;
    QLineEdit* outputEdit_  = nullptr;
    QPushButton* inBtn_   = nullptr;
    QPushButton* outBtn_  = nullptr;
    QPushButton* runBtn_  = nullptr;
    QPushButton* copyBtn_ = nullptr;
    QProgressBar* progress_ = nullptr;
    QPlainTextEdit* log_ = nullptr;
    ResultView* resultView_ = nullptr;
    CalibWorker* worker_ = nullptr;
};

}  // namespace fc::gui
