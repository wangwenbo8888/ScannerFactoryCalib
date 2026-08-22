#pragma once

#include <QWidget>

class QLineEdit;
class QPushButton;
class QProgressBar;
class QPlainTextEdit;
class QLabel;
class QComboBox;
class QCheckBox;

namespace fc::gui {

class ResultView;
class CalibWorker;

// 激光线标定 tab：按四种激光线类型（左斜/右斜/精细/深孔）分别标定，
// 输入 data_in/laser/<type>/，输出 data_out/laser_calib_<type>.json（四个独立文件）。
// 调用 fc::runLaserCalib（依赖 <输入目录>/camera_calib.json）。
class LaserCalibTab : public QWidget {
    Q_OBJECT
public:
    explicit LaserCalibTab(QWidget* parent = nullptr);
    ~LaserCalibTab() override;

signals:
    void statusMessage(QString msg);
    void calibrationStarted();   // 标定启动（MainWindow 接线→自动停止扫描仪）

private slots:
    void onTypeChanged();        // 切换激光线类型 → 更新输入/输出路径与按钮文案
    void onToggleOldData(bool on);   // 勾选「使用旧数据」→ 启用手动选目录
    void onPickOldDataDir();         // 选择旧数据文件夹（应含 pose_* 子目录）
    void onBrowseInput();
    void onBrowseOutput();
    void onRunAll();
    void onCopyHandoff();   // 把 data_out/camera_calib.json 拷到 <输入>/camera_calib.json（并补 config.json）
    void onWorkerDone(bool ok, QString summary);

private:
    void appendLog(const QString& msg, const QString& color = "#d4d4d4");
    void setRunningUI(bool running);
    // 确保 inDir 下有 camera_calib.json + config.json（缺则自动补；overwrite=true 强制刷新）
    bool ensureHandoffFiles(const QString& inDir, bool overwrite);
    // 把用户所选目录（备份根/laser 一级/精确类型目录均可）解析为当前类型数据目录；
    // 成功后记住 laser 根（oldDataLaserRoot_）供切类型自动重定位
    bool resolveOldData(const QString& picked);

    QLineEdit* inputEdit_   = nullptr;
    QLineEdit* outputEdit_  = nullptr;
    QComboBox* typeCbx_     = nullptr;   // 激光线类型：左斜/右斜/精细/深孔
    QCheckBox* oldDataChk_  = nullptr;   // 勾选＝用旧数据标定（手动选目录）；默认本次采集
    QPushButton* oldDataBtn_ = nullptr;  // 选择旧数据文件夹（勾选后启用）
    QPushButton* inBtn_   = nullptr;
    QPushButton* outBtn_  = nullptr;
    QPushButton* runBtn_  = nullptr;
    QPushButton* copyBtn_ = nullptr;
    QProgressBar* progress_ = nullptr;
    QPlainTextEdit* log_ = nullptr;
    ResultView* resultView_ = nullptr;
    CalibWorker* worker_ = nullptr;
    QString oldDataLaserRoot_;   // 旧数据模式的 laser 根目录（空＝未解析/精确类型目录）
};

}  // namespace fc::gui
