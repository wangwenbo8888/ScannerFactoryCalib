#pragma once

#include <QWidget>
#include <QImage>
#include <memory>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>
#include <opencv2/core.hpp>

class QComboBox;
class QSpinBox;
class QPushButton;
class QLabel;
class QDoubleSpinBox;
class QLineEdit;
class QTimer;
class QCheckBox;
class QSlider;

namespace fc::gui {

class PreviewWidget;
class StereoCameraRig;
class ScannerControl;
class SerialMonitorDialog;

// 图像采集 tab：双相机点击抓拍 + 显示 + 保存。
// 不做实时预览（用户眼睛反应不过来），仅在用户点击"抓拍"时取一帧。
class AcquisitionTab : public QWidget {
    Q_OBJECT
public:
    explicit AcquisitionTab(QWidget* parent = nullptr);
    ~AcquisitionTab() override;

signals:
    void statusMessage(QString msg);
    // 照搬 LeadScanK2：左右配对后 emit updateImages(QImage,QImage)，QueuedConnection 到 onUpdateImages
    void updateImages(QImage left, QImage right);
    // 串口监视：上位机↓发送 / 下位机↑返回帧（ScannerControl onTx/onRx 转发）
    void serialTx(QString frame);
    void serialRx(QString frame);

private slots:
    void onRefreshDevices();
    void onOpenClose();
    void onFreezeFrame();        // 预览当前帧：界面停在当前帧（后台 lastLeft_/Right 继续刷新）
    void onTogglePreview(bool on);  // 连续预览开关（on=刷新界面，off=停在当前帧）
    void onToggleRecord(bool on);   // 连续存储开关：on=开始整帧连续存盘（独立文件夹）；off=停止收尾
    void onSaveCamera();         // 把当前【显示】帧保存到 data_in/camera/left,N.png + right,N.png
    void onSaveLaser();          // 保存到 data_in/laser/pose_NN/L_tube*.png + R_tube*.png
    void onExportContrast();     // 左右对比度导出到扫描软件 config/camera.json（读改写合并）
    void onBrowseLeftFolder();
    void onBrowseRightFolder();
    void onSourceTypeChanged();

    // 扫描仪硬件串口控制（仿 LeadScanK2/LEADSCANSeries）
    void onRefreshComPorts();
    void onOpenCloseScanner();   // 打开/关闭扫描仪串口
    void onStartScanner();       // 发 N10 启动扫描仪 + 自动开始实时预览
    void onStopScanner();        // 发 N11 停止扫描仪 + 停止预览
    void onUpdateImages(QImage left, QImage right);  // 照搬 LeadScanK2 onUpdateImages

public:
    // 照搬 LeadScanK2 setLeftImage/setRightImage：由相机回调线程调用，配对后 emit updateImages
    void setLeftImage(cv::Mat& left, uint64_t id);
    void setRightImage(cv::Mat& right, uint64_t id);

    // 标定启动时由 MainWindow 调用：扫描仪在运行则自动停止（N11 + 停预览/采集）
    void stopScannerIfRunning();

private:
    void appendLog(const QString& msg);
    void updateSnapCount();
    // 启动时扫描已有数据目录，采集计数从最大编号续起（防跨会话覆盖旧图）
    void initSnapCounters();
    void setDeviceOpenUI(bool opened);
    // 按用户勾选的「左/右 180°」复选框旋转图像（GUI 层处理，与 deviceId 解耦）
    void applyGuiRotation(cv::Mat& leftImg, cv::Mat& rightImg);
    // 左右对比度滑条 → 相机源（Galaxy 逐帧 LUT）＋QSettings 持久化（261002）
    void applyContrastFromUi();

    // —— 连续存储（独立写盘线程，全分辨率左右成对帧）——
    void stopRecord();             // 停止存储（幂等；join 留给下次 start 与析构）
    void enqueueRecordPair(cv::Mat left, cv::Mat right);  // 回调线程：已克隆的私有整帧入队
    void recordWriterLoop();       // 写盘线程主循环

    // 开机自动流程：开串口 → 启动扫描仪 → 开设备 → 进实时预览
    void autoStart();

    // —— 源选择 ——
    QComboBox*   sourceTypeCbx_ = nullptr;
    QPushButton* refreshBtn_    = nullptr;
    QLabel*      deviceListLbl_ = nullptr;
    QSpinBox*    leftDevSpin_   = nullptr;
    QSpinBox*    rightDevSpin_  = nullptr;
    QLineEdit*   leftFolderEdit_  = nullptr;
    QLineEdit*   rightFolderEdit_ = nullptr;
    QPushButton* leftBrowseBtn_   = nullptr;
    QPushButton* rightBrowseBtn_  = nullptr;

    // —— 控制 ——
    QPushButton* openBtn_    = nullptr;
    QPushButton* captureBtn_ = nullptr;       // 「预览当前帧」：界面停在当前帧
    QPushButton* previewBtn_ = nullptr;       // 「连续预览」checkable 开关
    QDoubleSpinBox* exposureSpin_ = nullptr;
    QDoubleSpinBox* gainSpin_     = nullptr;
    QCheckBox*  leftRotateChk_    = nullptr;  // 左相机 180° 旋转
    QCheckBox*  rightRotateChk_   = nullptr;  // 右相机 180° 旋转
    // —— 左右分置对比度（261002）——软件 LUT（-50..100，0=不调整），启动读配置文件
    QSlider*    contrastLSlider_  = nullptr;  // 左相机对比度
    QSlider*    contrastRSlider_  = nullptr;  // 右相机对比度
    QLabel*     contrastLValLbl_  = nullptr;
    QLabel*     contrastRValLbl_  = nullptr;
    QPushButton* exportContrastBtn_ = nullptr;  // 导出对比度到扫描软件 camera.json
    bool        previewing_       = false;   // 扫描仪启动后自动进入预览态
    bool        scannerRunning_   = false;   // 扫描仪电机/激光是否已启动（N10 已发）

    // —— 连续存储 ——
    QPushButton* recordBtn_       = nullptr; // 「⏺ 连续存储」checkable（勾选=绿色=存储中）
    std::atomic<bool> recording_  = {false}; // 存储开关（相机回调线程读）
    std::thread           recordWriter_;     // 后台写盘线程
    std::mutex            recordMtx_;
    std::condition_variable recordCv_;
    std::deque<std::pair<cv::Mat, cv::Mat>> recordQueue_;  // 待写盘左右成对帧
    QString               recordDir_;        // 本次存储目录（data_in/continuous/<开始时间>）
    std::atomic<int>      recordWritten_ = {0};  // 已写盘对数（writer 写，心跳/UI 读）
    int                   recordDropped_ = 0;  // 队列满丢弃对数（recordMtx_ 保护）
    QTimer*               recordHeartbeat_  = nullptr;  // 1s 心跳：进度回报+断流报警（UI 线程）
    int64_t               lastFedPairs_    = 0;  // 心跳上轮 pairCount_ 快照（断流检测）
    int                   starvedTicks_    = 0;  // 连续无新配对帧的秒数
    int                   heartbeatTicks_  = 0;  // 心跳累计 tick（每 5s 回报一次进度）

    // —— 扫描仪硬件控制（串口）——
    QComboBox*   comPortCbx_       = nullptr;
    QPushButton* comRefreshBtn_    = nullptr;
    QPushButton* scannerOpenBtn_   = nullptr;   // 打开/关闭扫描仪串口
    QPushButton* scannerStartBtn_  = nullptr;   // 启动扫描仪 (N10)
    QPushButton* scannerStopBtn_   = nullptr;   // 停止扫描仪 (N11)
    QComboBox*   scannerModeCbx_   = nullptr;   // 0=仅补光(相机标定) 1=激光+补光(激光标定)
    QSlider*     freqSlider_       = nullptr;   // 电机频率
    QSlider*     bgLightSlider_    = nullptr;   // 背光强度
    QSlider*     laserSlider_      = nullptr;   // 激光强度
    QLabel*      freqValLbl_       = nullptr;
    QLabel*      bgValLbl_         = nullptr;
    QLabel*      laserValLbl_      = nullptr;
    QLabel*      tempLbl_          = nullptr;   // 下位机 4 路温度（G02@1Hz，随串口常显）
    std::unique_ptr<ScannerControl> scanner_;

    // —— 串口通讯监视（开机自动弹出的 TX/RX 弹窗）——
    SerialMonitorDialog* monitor_       = nullptr;
    QPushButton*         serialMonitorBtn_ = nullptr;

    // —— 显示 ——
    PreviewWidget* leftPreview_  = nullptr;
    PreviewWidget* rightPreview_ = nullptr;
    QLabel* leftInfoLbl_  = nullptr;
    QLabel* rightInfoLbl_ = nullptr;

    // —— 保存 ——
    QPushButton* saveCameraBtn_ = nullptr;
    QPushButton* saveLaserBtn_  = nullptr;
    QLabel*     snapCountLbl_   = nullptr;

    std::unique_ptr<StereoCameraRig> rig_;
    // 中心 ROI（261005）：exe 同目录 factory_calib_gui.ini [acquisition] roiWidth/
    // roiHeight 读入；0=满幅。开设备时下发给相机源（Galaxy 硬件 AOI），改尺寸只改 ini 不改代码
    int roiW_ = 1688;
    int roiH_ = 1400;
    cv::Mat lastLeft_;          // 后台持续刷新的最新帧（相机回调写入）
    cv::Mat lastRight_;
    cv::Mat frozenLeft_;        // 当前【显示】帧：连续预览时随 lastLeft_ 更新；点「预览当前帧」后冻结
    cv::Mat frozenRight_;       // 保存图像存这个，保证存的就是用户看到的那一帧
    int cameraSnapshotIdx_ = 0;
    int laserPoseIdx_[4] = {0, 0, 0, 0};   // 四种激光线类型各自独立的 pose 计数

    // 照搬 LeadScanK2：左右帧配对用
    cv::Mat* m_pLeft = nullptr;
    cv::Mat* m_pRight = nullptr;
    uint64_t m_iLeftId = 0;
    uint64_t m_iRightId = 0;
    uint64_t m_iIndex = 0;

    // 配对改为轮次制（2026-09-27 修复「连续存储有时 0 张」）：左右各有未消费新帧即成
    // 一对（后到者触发），不再用 frameID 严格相等。旧 == 门禁在单边多一帧（布防间隙
    // 抢到脉冲/ok=0 失败帧占号/单边丢帧）后 ID 永久错位 → 整场 0 配对且无任何报错。
    std::mutex pairMtx_;                    // 保护 m_pLeft/Right 换帧 + fresh 标志 + 配对克隆
    bool leftFresh_  = false;               // 左有未消费新帧（pairMtx_）
    bool rightFresh_ = false;               // 右有未消费新帧（pairMtx_）
    std::atomic<int64_t> pairCount_ = {0};  // 累计配对对数（录制断流报警/帧率统计用）

    // —— 帧率统计（相机回调累计计数，UI 每秒差分上屏；与预览/录制状态无关）——
    QTimer*   fpsTimer_ = nullptr;              // 1s 差分定时器（设备打开期间常开）
    QLabel*   fpsLbl_   = nullptr;              // 保存行小标签：fps L/R/pair
    std::atomic<int64_t> leftFrames_  = {0};    // 左相机累计帧数（左回调线程递增）
    std::atomic<int64_t> rightFrames_ = {0};    // 右相机累计帧数（右回调线程递增）
    int64_t   lastLeftFrames_  = 0;             // 上秒快照（仅 UI 线程）
    int64_t   lastRightFrames_ = 0;
    int64_t   lastPairCount_   = 0;

    // —— 丢帧检测（frameID 跳号 = 中间帧在相机缓冲/USB 传输环节真丢失）——
    std::atomic<int64_t> leftDropped_  = {0};   // 左丢帧累计（左回调线程写）
    std::atomic<int64_t> rightDropped_ = {0};   // 右丢帧累计（右回调线程写）
    int64_t   leftLastId_  = -1;                // 左最近 frameID（仅左回调线程；<0=待基线）
    int64_t   rightLastId_ = -1;                // 右最近 frameID（仅右回调线程）
    int64_t   lastLeftDropped_  = 0;            // 上秒丢帧快照（仅 UI 线程）
    int64_t   lastRightDropped_ = 0;

    int64_t   lastPreviewNs_ = 0;               // 上次预览 emit 时刻 ns（pairMtx_；限频用）
    QString   leftBaseInfo_;                    // 左信息栏基础文案（设备名+分辨率），fps 追加在后
    QString   rightBaseInfo_;
};

}  // namespace fc::gui
