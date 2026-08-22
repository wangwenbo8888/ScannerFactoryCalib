#pragma once

#include <QWidget>
#include <optional>
#include <memory>

#include "calib_io.h"
#include "calib_runner.h"

class QLineEdit;
class QPushButton;
class QProgressBar;
class QPlainTextEdit;
class QLabel;
class QComboBox;
class QCheckBox;

namespace fc::gui {

class PreviewWidget;
class CalibWorker;

// 相机标定 tab：含 4 个独立步骤 + 全流程，保留中间结果供下一步用。
// 「使用旧数据」勾选框：不勾＝本次采集（data_in/camera），勾＝手动选旧数据文件夹。
class CameraCalibTab : public QWidget {
    Q_OBJECT
public:
    explicit CameraCalibTab(QWidget* parent = nullptr);
    ~CameraCalibTab() override;

signals:
    void statusMessage(QString msg);
    void calibrationStarted();   // 标定启动（MainWindow 接线→自动停止扫描仪）

private slots:
    void onBrowseInput();
    void onBrowseOutput();
    void onToggleOldData(bool on);   // 勾选「使用旧数据」→ 启用手动选目录
    void onPickOldDataDir();         // 选择旧数据文件夹（应含 left/ right/ 子目录）
    void onRunStep(int step);     // step ∈ {1,2,3,4} 对应 内参/外参/立体/温度表
    void onRunAll();              // 全流程
    void onPreviewFirstPair();    // 把首帧棋盘叠加显示
    void onWorkerProgress(int s, int t, QString msg);
    void onWorkerLog(QString line);
    void onWorkerDone(bool ok, QString summary);

private:
    void appendLog(const QString& msg, const QString& color = "#d4d4d4");
    void refreshStepLabels();
    void setRunningUI(bool running);

    // —— 输入输出 ——
    QLineEdit* inputEdit_  = nullptr;
    QLineEdit* outputEdit_ = nullptr;
    QPushButton* inBtn_  = nullptr;
    QPushButton* outBtn_ = nullptr;

    // —— 旧数据选择（不勾＝本次采集 data_in/camera，勾＝手动选文件夹）——
    QCheckBox*    oldDataChk_ = nullptr;
    QPushButton*  oldDataBtn_ = nullptr;

    // —— 步骤按钮 + 状态 ——
    QPushButton* step1Btn_ = nullptr;   // 棋盘格
    QPushButton* step2Btn_ = nullptr;   // 内参
    QPushButton* step3Btn_ = nullptr;   // 外参
    QPushButton* step4Btn_ = nullptr;   // 立体矫正 + 温度表
    QPushButton* runAllBtn_ = nullptr;
    QLabel* step1Lbl_ = nullptr;
    QLabel* step2Lbl_ = nullptr;
    QLabel* step3Lbl_ = nullptr;
    QLabel* step4Lbl_ = nullptr;

    QProgressBar* progress_ = nullptr;
    QPlainTextEdit* log_ = nullptr;

    // —— 预览 ——
    PreviewWidget* preview1_ = nullptr;  // 棋盘叠加左
    PreviewWidget* preview2_ = nullptr;  // 棋盘叠加右
    QPushButton* previewBtn_ = nullptr;

    // —— 结果摘要 ——
    QLabel* resultLbl_ = nullptr;

    // —— 会话状态（按步骤累积）——
    std::optional<CameraInput> input_;
    std::optional<CornerExtractionResult> corners_;
    std::optional<calib::IntrinsicCalibResult> intrin_;
    std::optional<calib::ExtrinsicCalibCpuResult> extrin_;
    std::optional<calib::StereoRectifyCpuResult> rectify_;

    CalibWorker* worker_ = nullptr;
};

}  // namespace fc::gui
