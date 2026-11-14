#include "AcquisitionTab.h"
#include "PreviewWidget.h"
#include "SerialMonitorDialog.h"
#include "gui_common.h"
#include "camera/ICameraSource.h"
#include "camera/StereoCameraRig.h"
#include "camera/GalaxyCameraSource.h"
#include "scanner/ScannerControl.h"
#include <map>

#include <QComboBox>
#include <QDateTime>
#include <QDebug>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QCoreApplication>
#include <QThread>
#include <QPixmap>
#include <QPushButton>
#include <QSettings>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QSaveFile>
#include <QFile>
#include <fstream>
#include <QCheckBox>
#include <QSerialPortInfo>
#include <QSlider>
#include <QSpinBox>
#include <QTimer>
#include <QVBoxLayout>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>   // 261114 findChessboardCorners/drawChessboardCorners（棋盘格实时判定）
// 261108 目标线数全局缓存（analyzeShot 为 static 函数访问不到成员；spin 变更时同步）
static std::atomic<int> judgeExpectGlobal{25};
static std::atomic<int> scanDirectionGlobal{1};   // 261114 游程宽度检查用（0=竖切 1=横切）

#include <spdlog/spdlog.h>
#include <QTransform>
#include <atomic>
#include <chrono>
#include <memory>

namespace fc::gui {

namespace {
// 连续存储待写队列上限（左右成对帧）。写盘跟不上时丢新帧，防止内存无限增长
constexpr size_t kRecordQueueCap = 120;
// 预览 emit 最小间隔（ns，≈30fps）：120fps 全量 QPixmap 转换+setPixmap 会吃满 UI
// 线程；录制不受限频影响
constexpr int64_t kPreviewMinIntervalNs = 33'000'000;
}

AcquisitionTab::AcquisitionTab(QWidget* parent) : QWidget(parent) {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(8, 8, 8, 8);
    root->setSpacing(6);

    // —— 源选择 ——
    auto* srcRow = new QHBoxLayout;
    srcRow->addWidget(new QLabel(QStringLiteral("源类型:")));
    sourceTypeCbx_ = new QComboBox;
    for (const auto& t : availableSourceTypes()) {
        sourceTypeCbx_->addItem(QString::fromUtf8(t.data(), (int)t.size()));
    }
    // 默认选 galaxy（扫描仪相机）：开机即查询到真相机（onRefreshDevices 会枚举 galaxy）
    int galaxyIdx = sourceTypeCbx_->findText(QStringLiteral("galaxy"));
    if (galaxyIdx >= 0) sourceTypeCbx_->setCurrentIndex(galaxyIdx);
    srcRow->addWidget(sourceTypeCbx_);
    refreshBtn_ = new QPushButton(QStringLiteral("刷新设备列表"));
    srcRow->addWidget(refreshBtn_);
    deviceListLbl_ = new QLabel(QStringLiteral("（未刷新）"));
    deviceListLbl_->setStyleSheet("color:#666;");
    deviceListLbl_->setWordWrap(true);
    srcRow->addWidget(deviceListLbl_, 1);
    root->addLayout(srcRow);

    // —— 左/右源配置 ——
    auto* leftRow = new QHBoxLayout;
    leftRow->addWidget(new QLabel(QStringLiteral("左源 ID:")));
    leftDevSpin_ = new QSpinBox;
    leftDevSpin_->setRange(0, 15); leftDevSpin_->setValue(0);
    leftRow->addWidget(leftDevSpin_);
    leftRow->addWidget(new QLabel(QStringLiteral("左源文件夹(sim):")));
    leftFolderEdit_ = new QLineEdit(QStringLiteral("D:/CameraCalib"));
    leftRow->addWidget(leftFolderEdit_, 1);
    leftBrowseBtn_ = new QPushButton(QStringLiteral("..."));
    leftRow->addWidget(leftBrowseBtn_);
    root->addLayout(leftRow);

    auto* rightRow = new QHBoxLayout;
    rightRow->addWidget(new QLabel(QStringLiteral("右源 ID:")));
    rightDevSpin_ = new QSpinBox;
    rightDevSpin_->setRange(0, 15); rightDevSpin_->setValue(1);
    rightRow->addWidget(rightDevSpin_);
    rightRow->addWidget(new QLabel(QStringLiteral("右源文件夹(sim):")));
    rightFolderEdit_ = new QLineEdit(QStringLiteral("D:/CameraCalib"));
    rightRow->addWidget(rightFolderEdit_, 1);
    rightBrowseBtn_ = new QPushButton(QStringLiteral("..."));
    rightRow->addWidget(rightBrowseBtn_);
    root->addLayout(rightRow);

    // —— 控制按钮 ——
    auto* ctrlRow = new QHBoxLayout;
    openBtn_    = new QPushButton(QStringLiteral("打开设备"));
    captureBtn_ = new QPushButton(QStringLiteral("⏸ 预览当前帧"));
    captureBtn_->setEnabled(false);
    QFont bf = captureBtn_->font(); bf.setBold(true); captureBtn_->setFont(bf);
    previewBtn_ = new QPushButton(QStringLiteral("▶ 连续预览"));
    previewBtn_->setCheckable(true);
    previewBtn_->setChecked(true);   // 默认连续预览
    previewBtn_->setEnabled(false);
    recordBtn_ = new QPushButton(QStringLiteral("⏺ 连续存储"));
    recordBtn_->setCheckable(true);
    recordBtn_->setEnabled(false);
    // 勾选=存储中（绿色），再点一次停止并收尾落盘
    recordBtn_->setStyleSheet(
        QStringLiteral("QPushButton:checked { background-color: #2fa84f; color: white; }"));
    saveCameraBtn_ = new QPushButton(QStringLiteral("💾 保存到 data_in/camera"));
    saveLaserBtn_  = new QPushButton(QStringLiteral("💾 保存到 data_in/laser/pose_NN"));
    saveCameraBtn_->setEnabled(false);
    saveLaserBtn_->setEnabled(false);
    exposureSpin_ = new QDoubleSpinBox;
    exposureSpin_->setRange(10.0, 1e7); exposureSpin_->setValue(2000.0);
    exposureSpin_->setSuffix(QStringLiteral(" μs"));
    exposureSpin_->setDecimals(0);
    gainSpin_ = new QDoubleSpinBox;
    gainSpin_->setRange(0.0, 30.0); gainSpin_->setValue(0.0);
    gainSpin_->setSuffix(QStringLiteral(" dB"));
    gainSpin_->setDecimals(2);
    leftRotateChk_  = new QCheckBox(QStringLiteral("左 180°"));
    rightRotateChk_ = new QCheckBox(QStringLiteral("右 180°"));
    leftRotateChk_->setChecked(false);   // 默认与参考 CameraControl 一致：仅右图旋转
    rightRotateChk_->setChecked(true);
    ctrlRow->addWidget(openBtn_);
    ctrlRow->addWidget(captureBtn_);
    ctrlRow->addWidget(previewBtn_);
    ctrlRow->addWidget(recordBtn_);
    ctrlRow->addWidget(new QLabel(QStringLiteral("曝光:")));
    ctrlRow->addWidget(exposureSpin_);
    ctrlRow->addWidget(new QLabel(QStringLiteral("增益:")));
    ctrlRow->addWidget(gainSpin_);
    ctrlRow->addWidget(leftRotateChk_);
    ctrlRow->addWidget(rightRotateChk_);
    ctrlRow->addStretch(1);

    // —— 扫描仪硬件控制（串口 + 启停）——
    // 仿 LeadScanK2/series/LEADSCANSeries.ui: pushButton_Start_Scanner / Stop_Scanner / horizontalSlider_*
    auto* scanRow1 = new QHBoxLayout;
    scanRow1->addWidget(new QLabel(QStringLiteral("扫描仪 COM:")));
    comPortCbx_ = new QComboBox;
    scanRow1->addWidget(comPortCbx_);
    comRefreshBtn_ = new QPushButton(QStringLiteral("刷新"));
    scanRow1->addWidget(comRefreshBtn_);
    scannerOpenBtn_ = new QPushButton(QStringLiteral("打开串口"));
    scanRow1->addWidget(scannerOpenBtn_);
    scanRow1->addWidget(new QLabel(QStringLiteral("模式:")));
    scannerModeCbx_ = new QComboBox;
    scannerModeCbx_->addItem(QStringLiteral("仅标志点（相机标定）"), 0);
    scannerModeCbx_->addItem(QStringLiteral("标志点+左斜激光线"), 1);
    scannerModeCbx_->addItem(QStringLiteral("标志点+右斜激光线"), 2);
    scannerModeCbx_->addItem(QStringLiteral("标志点+精细激光线"), 3);
    scannerModeCbx_->addItem(QStringLiteral("标志点+深孔激光线"), 4);
    scannerModeCbx_->addItem(QStringLiteral("标志点+中竖激光线(备用25线)"), 5);
    scannerModeCbx_->setCurrentIndex(1);  // 默认激光模式：实时预览看激光线变换
    scanRow1->addWidget(scannerModeCbx_);
    scannerStartBtn_ = new QPushButton(QStringLiteral("▶ 启动扫描仪"));
    scannerStartBtn_->setEnabled(false);
    scannerStopBtn_  = new QPushButton(QStringLiteral("■ 停止扫描仪"));
    scannerStopBtn_->setEnabled(false);
    scanRow1->addWidget(scannerStartBtn_);
    scanRow1->addWidget(scannerStopBtn_);
    serialMonitorBtn_ = new QPushButton(QStringLiteral("🖥 串口监视"));
    scanRow1->addWidget(serialMonitorBtn_);
    scanRow1->addStretch(1);
    root->addLayout(scanRow1);

    auto* scanRow2 = new QHBoxLayout;
    scanRow2->addWidget(new QLabel(QStringLiteral("频率:")));
    freqSlider_ = new QSlider(Qt::Horizontal);
    freqSlider_->setRange(0, 200); freqSlider_->setValue(50);
    freqValLbl_ = new QLabel(QStringLiteral("50"));
    freqValLbl_->setMinimumWidth(32);
    scanRow2->addWidget(freqSlider_, 1);
    scanRow2->addWidget(freqValLbl_);

    scanRow2->addWidget(new QLabel(QStringLiteral("  背光:")));
    bgLightSlider_ = new QSlider(Qt::Horizontal);
    bgLightSlider_->setRange(0, 100); bgLightSlider_->setValue(40);
    bgValLbl_ = new QLabel(QStringLiteral("40"));
    bgValLbl_->setMinimumWidth(32);
    scanRow2->addWidget(bgLightSlider_, 1);
    scanRow2->addWidget(bgValLbl_);

    scanRow2->addWidget(new QLabel(QStringLiteral("  激光:")));
    laserSlider_ = new QSlider(Qt::Horizontal);
    laserSlider_->setRange(0, 100); laserSlider_->setValue(60);
    laserValLbl_ = new QLabel(QStringLiteral("60"));
    laserValLbl_->setMinimumWidth(32);
    scanRow2->addWidget(laserSlider_, 1);
    scanRow2->addWidget(laserValLbl_);

    // 下位机 4 路温度（G02@1Hz）：随串口打开常显，与扫描启停无关
    tempLbl_ = new QLabel(QStringLiteral("温度 --"));
    tempLbl_->setStyleSheet("color:#888;");
    tempLbl_->setToolTip(QStringLiteral(
        "下位机 G02 温度回传（N12 T1000 = 1Hz）。A/B/C/D 四路传感器位置语义见下位机资料；"
        "串口关闭后停止刷新。"));
    scanRow2->addWidget(tempLbl_);
    root->addLayout(scanRow2);

    // 打开/关闭设备 + 曝光/增益/旋转 行移至滑动条下方
    root->addLayout(ctrlRow);

    // —— 左右分置对比度（261002 新增）——软件 LUT 逐帧变换（Galaxy 源内实现），
    // 与曝光/增益（SDK 硬件参数）不同层；左/右独立调节，改即存配置文件
    auto* contrastRow = new QHBoxLayout;
    contrastRow->addWidget(new QLabel(QStringLiteral("对比度 左:")));
    contrastLSlider_ = new QSlider(Qt::Horizontal);
    contrastLSlider_->setRange(-50, 100);
    contrastLSlider_->setValue(0);
    contrastLSlider_->setToolTip(QStringLiteral(
        "左相机软件对比度：output=(v−128)×factor+128，factor=1+值/100；"
        "范围 -50~100（0.5x~2.0x），0=不调整。改动即存配置文件，启动自动读回"));
    contrastLValLbl_ = new QLabel(QStringLiteral("0"));
    contrastLValLbl_->setMinimumWidth(32);
    contrastRow->addWidget(contrastLSlider_, 1);
    contrastRow->addWidget(contrastLValLbl_);
    contrastRow->addWidget(new QLabel(QStringLiteral("  右:")));
    contrastRSlider_ = new QSlider(Qt::Horizontal);
    contrastRSlider_->setRange(-50, 100);
    contrastRSlider_->setValue(0);
    contrastRSlider_->setToolTip(QStringLiteral(
        "右相机软件对比度：output=(v−128)×factor+128，factor=1+值/100；"
        "范围 -50~100（0.5x~2.0x），0=不调整。改动即存配置文件，启动自动读回"));
    contrastRValLbl_ = new QLabel(QStringLiteral("0"));
    contrastRValLbl_->setMinimumWidth(32);
    contrastRow->addWidget(contrastRSlider_, 1);
    contrastRow->addWidget(contrastRValLbl_);
    // 导出：把调好的左右对比度写进扫描软件（scan_demo）的 config/camera.json——
    // 扫描软件每次启动读该文件取 camera.contrastLeft/contrastRight（AppContext）
    exportContrastBtn_ = new QPushButton(QStringLiteral("📤 导出到扫描软件"));
    exportContrastBtn_->setToolTip(QStringLiteral(
        "把当前左右对比度写入 camera.json（读-改-写，保留其它装机口径键，只更新 "
        "camera.contrastLeft/contrastRight）。扫描软件每次启动读其 exe 目录 "
        "config\\camera.json 自动应用。\n"
        "同机：直接选扫描软件的 config\\camera.json；\n"
        "跨机（标定机≠扫描机）：选 U 盘/共享目录存一份，拷到扫描机扫描软件 "
        "config\\ 下替换即可（扫描软件构建只播种不覆盖，不会冲掉此文件）。"
        "路径会记住，第二次起一键导出"));
    contrastRow->addWidget(exportContrastBtn_);
    root->addLayout(contrastRow);

    // —— 双预览（点采集后才显示）——
    auto* previewRow = new QHBoxLayout;
    auto* leftCol = new QVBoxLayout;
    leftCol->addWidget(new QLabel(QStringLiteral("左相机")));
    leftPreview_ = new PreviewWidget;
    leftPreview_->setPlaceholder(QStringLiteral("（点击「采集一帧」后显示）"));
    leftInfoLbl_ = new QLabel(QStringLiteral("-"));
    leftInfoLbl_->setStyleSheet("color:#888;");
    leftCol->addWidget(leftPreview_, 1);
    leftCol->addWidget(leftInfoLbl_);
    auto* rightCol = new QVBoxLayout;
    rightCol->addWidget(new QLabel(QStringLiteral("右相机")));
    rightPreview_ = new PreviewWidget;
    rightPreview_->setPlaceholder(QStringLiteral("（点击「采集一帧」后显示）"));
    rightInfoLbl_ = new QLabel(QStringLiteral("-"));
    rightInfoLbl_->setStyleSheet("color:#888;");
    rightCol->addWidget(rightPreview_, 1);
    rightCol->addWidget(rightInfoLbl_);
    previewRow->addLayout(leftCol, 1);
    previewRow->addLayout(rightCol, 1);
    // —— 261108 采集参数实时判定列（右侧）——
    // 三指标（261108 两批左斜数据标定的窗口）：过曝 2.8~5.0%（右斜成功批
    // 3.89%）、游程>30px=0（斜线正常 8~22）、中部行线数=目标±1；双眼全过
    // 才显示「可以采集」。调亮度档时看此面板，绿灯后再正式采集。
    auto* judgeCol = new QVBoxLayout;
    // 261114 判定启用开关（面板最顶上；取消勾选＝跳过分析，面板显示「已禁用」）
    judgeEnabledCheck_ = new QCheckBox(QStringLiteral("启用实时判定"));
    judgeEnabledCheck_->setChecked(true);
    judgeEnabledCheck_->setToolTip(QStringLiteral(
        "取消勾选则跳过实时判定分析（预览不受影响，面板显示「已禁用」）。\n"
        "模式0（相机标定）＝棋盘格判定；激光模式＝线质量判定。"));
    judgeCol->addWidget(judgeEnabledCheck_);
    auto* judgeTitleRow = new QHBoxLayout;
    judgeTitleRow->addWidget(new QLabel(QStringLiteral("采集参数判定")));
    judgeExpectSpin_ = new QSpinBox;
    judgeExpectSpin_->setRange(1, 40);
    judgeExpectSpin_->setValue(25);
    judgeExpectSpin_->setToolTip(QStringLiteral(
        "目标线数：左斜/右斜=25，精细=7。与 4-3 门禁 expectedLineCount 同值"));
    judgeTitleRow->addWidget(new QLabel(QStringLiteral("目标线数")));
    judgeTitleRow->addWidget(judgeExpectSpin_);
    autoTuneBtn_ = new QPushButton(QStringLiteral("自动调节"));
    autoTuneBtn_->setCheckable(true);
    autoTuneBtn_->setToolTip(QStringLiteral(
        "开启后按判定自动调 补光/激光 档（每次 1 步、800ms 节流、绿灯自停）。\n"
        "规则：背景>25 降补光 / 过曝>5.5% 降激光 / 过曝<2.5%且线暗 升激光。\n"
        "手动滑杆随时可覆盖（关自动后）。"));
    judgeTitleRow->addWidget(autoTuneBtn_);
    judgeTitleRow->addStretch(1);
    judgeCol->addLayout(judgeTitleRow);
    judgeLbl_ = new QLabel(QStringLiteral(
        "<p style='color:#888;'>（采集预览后实时判定）</p>"
        "<p>①过曝占比 目标 2.8~5.0%</p>"
        "<p>②游程>30px 目标 0（无粘连）</p>"
        "<p>③线数 = 目标±1（双眼）</p>"));
    judgeLbl_->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    judgeLbl_->setWordWrap(true);
    judgeLbl_->setMinimumWidth(220);
    judgeLbl_->setStyleSheet(
        "background:#fafafa; border:1px solid #d0d0d0; border-radius:4px; padding:8px;");
    judgeCol->addWidget(judgeLbl_, 1);
    previewRow->addLayout(judgeCol, 0);
    root->addLayout(previewRow, 4);

    // —— 保存（用户确认后才落盘）——
    auto* saveRow = new QHBoxLayout;
    saveRow->addWidget(saveCameraBtn_);
    saveRow->addWidget(saveLaserBtn_);
    saveRow->addStretch(1);
    fpsLbl_ = new QLabel(QStringLiteral("fps L:- R:- pair:- | 丢帧 L:-(?) R:-(?)"));
    fpsLbl_->setStyleSheet("color:#888;");
    fpsLbl_->setToolTip(QStringLiteral(
        "丢帧 = frameID 跳变检测到的真丢失（相机缓冲/USB 传输环节）。格式：每秒(累计)。\n"
        "判读：丢帧>0 而 fps 低 → 传输/带宽丢帧；丢帧=0 而 fps<设定 → 触发侧没给够脉冲。"));
    saveRow->addWidget(fpsLbl_);
    snapCountLbl_ = new QLabel(QStringLiteral("相机: 0 张    激光: 0 pose"));
    saveRow->addWidget(snapCountLbl_);
    root->addLayout(saveRow);

    // —— 信号 ——
    connect(refreshBtn_,     &QPushButton::clicked,     this, &AcquisitionTab::onRefreshDevices);
    connect(openBtn_,        &QPushButton::clicked,     this, &AcquisitionTab::onOpenClose);
    connect(captureBtn_,     &QPushButton::clicked,     this, &AcquisitionTab::onFreezeFrame);
    connect(previewBtn_,     &QPushButton::toggled,     this, &AcquisitionTab::onTogglePreview);
    connect(recordBtn_,      &QPushButton::toggled,     this, &AcquisitionTab::onToggleRecord);
    connect(saveCameraBtn_,  &QPushButton::clicked,     this, &AcquisitionTab::onSaveCamera);
    connect(saveLaserBtn_,   &QPushButton::clicked,     this, &AcquisitionTab::onSaveLaser);
    connect(leftBrowseBtn_,  &QPushButton::clicked,     this, &AcquisitionTab::onBrowseLeftFolder);
    connect(rightBrowseBtn_, &QPushButton::clicked,     this, &AcquisitionTab::onBrowseRightFolder);
    connect(sourceTypeCbx_,  QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &AcquisitionTab::onSourceTypeChanged);

    // 扫描仪串口信号
    connect(comRefreshBtn_,   &QPushButton::clicked, this, &AcquisitionTab::onRefreshComPorts);
    connect(scannerOpenBtn_,  &QPushButton::clicked, this, &AcquisitionTab::onOpenCloseScanner);
    connect(scannerStartBtn_, &QPushButton::clicked, this, &AcquisitionTab::onStartScanner);
    connect(scannerStopBtn_,  &QPushButton::clicked, this, &AcquisitionTab::onStopScanner);
    connect(serialMonitorBtn_, &QPushButton::clicked, this, [this] {
        monitor_->show();
        monitor_->raise();
        monitor_->activateWindow();
    });
    // 照搬 LeadScanK2：updateImages 配对双张，Qt::QueuedConnection
    connect(this, SIGNAL(updateImages(QImage,QImage)), this, SLOT(onUpdateImages(QImage,QImage)), Qt::QueuedConnection);
    // 261108 采集参数实时判定（右侧面板）
    connect(this, SIGNAL(updateJudge(QString)), this, SLOT(onUpdateJudge(QString)), Qt::QueuedConnection);
    connect(this, SIGNAL(autoTuneRequest(int,int,double,int)), this, SLOT(onAutoTune(int,int,double,int)), Qt::QueuedConnection);
    if (autoTuneBtn_)
        connect(autoTuneBtn_, &QPushButton::toggled, this, [this](bool on) {
            autoTuneOn_.store(on);
            autoTuneBtn_->setText(on ? QStringLiteral("调节中…(点停)")
                                      : QStringLiteral("自动调节"));
        });
    connect(judgeExpectSpin_, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) {
        judgeExpect_.store(v, std::memory_order_relaxed);
        judgeExpectGlobal.store(v, std::memory_order_relaxed);
    });
    // 261114 判定启用开关 → 回调线程安全缓存（照搬 judgeExpect_ 同步模式）
    connect(judgeEnabledCheck_, &QCheckBox::toggled, this, [this](bool on) {
        judgeEnabled_.store(on, std::memory_order_relaxed);
    });
    connect(freqSlider_,      &QSlider::valueChanged, this, [this](int v){
        freqValLbl_->setText(QString::number(v));
    });
    connect(bgLightSlider_,   &QSlider::valueChanged, this, [this](int v){
        bgValLbl_->setText(QString::number(v));
    });
    connect(laserSlider_,     &QSlider::valueChanged, this, [this](int v){
        laserValLbl_->setText(QString::number(v));
    });
    // 模式切换 → 激光保存按钮按类型更新文案/可用性（相机标定模式下禁用）
    connect(scannerModeCbx_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
        int mode = scannerModeCbx_->currentData().toInt();
        laserMode_.store(mode, std::memory_order_relaxed);   // 261114 判定分流缓存（0=棋盘格）
        scanDirectionGlobal.store(mode == 1 || mode == 2 ? 1 : 0, std::memory_order_relaxed); // 261114 游程方向
        if (mode >= 1 && mode <= kLaserTypeCount) {
            saveLaserBtn_->setEnabled(true);
            saveLaserBtn_->setText(QStringLiteral("💾 保存到 data_in/laser/%1/pose_NN")
                                       .arg(QString::fromLatin1(kLaserTypes[mode - 1].key)));
        } else {
            saveLaserBtn_->setEnabled(false);
            saveLaserBtn_->setText(QStringLiteral("💾 保存激光帧（请先选激光线类型）"));
        }
    });
    // 初始化按钮文案（不触发上面的信号）；261114 模式缓存同步一次（默认0=相机标定）
    laserMode_.store(scannerModeCbx_->currentData().toInt(), std::memory_order_relaxed);
    if (scannerModeCbx_->currentData().toInt() >= 1) {
        saveLaserBtn_->setText(QStringLiteral("💾 保存到 data_in/laser/%1/pose_NN")
                                   .arg(QString::fromLatin1(
                                       kLaserTypes[scannerModeCbx_->currentData().toInt() - 1].key)));
    }

    // 曝光/增益实时下发：用户一改就写相机（缓存也同步更新，
    // 下次 startAcquisition 按参考顺序重新应用同一值）
    connect(exposureSpin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double us) {
        if (!rig_ || !rig_->isOpen()) return;
        if (auto* L = rig_->left())
            if (L->capability().exposureUs) L->setExposureUs(us);
        if (auto* R = rig_->right())
            if (R->capability().exposureUs) R->setExposureUs(us);
        emit statusMessage(QStringLiteral("曝光已下发: %1 μs").arg(us, 0, 'f', 0));
    });
    connect(gainSpin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double db) {
        if (!rig_ || !rig_->isOpen()) return;
        if (auto* L = rig_->left())
            if (L->capability().gainDb) L->setGainDb(db);
        if (auto* R = rig_->right())
            if (R->capability().gainDb) R->setGainDb(db);
        emit statusMessage(QStringLiteral("增益已下发: %1 dB").arg(db, 0, 'f', 2));
    });

    // 左右分置对比度（261002）：改动即应用到相机源（Galaxy 逐帧 LUT，直显直存
    // 均为变换后图像）＋写配置文件；启动时下方读回。设备未开时只记账，开设备
    // 时 onOpenClose 统一下发
    connect(contrastLSlider_, &QSlider::valueChanged, this, [this](int v) {
        contrastLValLbl_->setText(QString::number(v));
        applyContrastFromUi();
    });
    connect(contrastRSlider_, &QSlider::valueChanged, this, [this](int v) {
        contrastRValLbl_->setText(QString::number(v));
        applyContrastFromUi();
    });
    // 导出左右对比度到扫描软件 camera.json（261002：工厂标定调好 → 扫描软件用）
    connect(exportContrastBtn_, &QPushButton::clicked, this, &AcquisitionTab::onExportContrast);
    // 启动读配置文件（261002 用户口径：每次启动读左右相机对比度）——setValue
    // 触发 valueChanged → 滑条/相机/配置文件三处对齐
    {
        QSettings s(QCoreApplication::applicationDirPath() + QStringLiteral("/factory_calib_gui.ini"),
                    QSettings::IniFormat);
        contrastLSlider_->setValue(qBound(-50, s.value("acquisition/contrastLeft", 0).toInt(), 100));
        contrastRSlider_->setValue(qBound(-50, s.value("acquisition/contrastRight", 0).toInt(), 100));
        // 中心 ROI（261005）：默认 1688×1400（用户口径），ini 可改；0=满幅。
        // 开设备时 onOpenClose 下发，存图/预览/标定全链自动跟随
        roiW_ = s.value("acquisition/roiWidth", 1688).toInt();
        roiH_ = s.value("acquisition/roiHeight", 1400).toInt();
        if (roiW_ <= 0 || roiH_ <= 0) { roiW_ = 0; roiH_ = 0; }
    }

    // 连续存储心跳（UI 线程，1s）：每 5s 回报「已写/待写/丢弃」；3 秒无配对帧报警——
    // 旧版配对死锁/相机不出图时整场 0 张且静默，用户无从得知（2026-09-27 修复）
    recordHeartbeat_ = new QTimer(this);
    recordHeartbeat_->setInterval(1000);
    connect(recordHeartbeat_, &QTimer::timeout, this, [this] {
        if (!recording_.load()) { recordHeartbeat_->stop(); return; }
        ++heartbeatTicks_;
        const int64_t fed = pairCount_.load();
        if (fed != lastFedPairs_) {
            lastFedPairs_ = fed;
            starvedTicks_ = 0;
        } else {
            ++starvedTicks_;
            if (starvedTicks_ == 3 || starvedTicks_ % 10 == 0) {
                emit statusMessage(QStringLiteral(
                    "⚠ 连续存储：%1 秒未收到左右配对帧（相机未出图/左右未配对），"
                    "本次可能存不到图，建议「■ 停止扫描仪」后重新启动").arg(starvedTicks_));
            }
        }
        if (heartbeatTicks_ % 5 == 0) {
            size_t queued = 0; int dropped = 0;
            {
                std::lock_guard<std::mutex> lk(recordMtx_);
                queued = recordQueue_.size();
                dropped = recordDropped_;
            }
            emit statusMessage(QStringLiteral("⏺ 录制中: 已写 %1 对 | 待写 %2 | 丢弃 %3")
                                   .arg(recordWritten_.load()).arg(queued).arg(dropped));
        }
    });

    // 帧率统计（1s 差分）：相机回调累计计数，UI 线程差分上屏。常开于设备打开期间
    // （含未启动扫描仪/预览冻结/录制中），L/R 单机帧率一眼分辨「单边不出图」，
    // pair 为左右配对速率（配对死锁时 L/R 有值而 pair=0，一目了然）
    fpsTimer_ = new QTimer(this);
    fpsTimer_->setInterval(1000);
    connect(fpsTimer_, &QTimer::timeout, this, [this] {
        const int64_t l = leftFrames_.load();
        const int64_t r = rightFrames_.load();
        const int64_t p = pairCount_.load();
        const int64_t ld = leftDropped_.load();
        const int64_t rd = rightDropped_.load();
        const int64_t lf = l - lastLeftFrames_;
        const int64_t rf = r - lastRightFrames_;
        const int64_t pf = p - lastPairCount_;
        fpsLbl_->setText(QStringLiteral("fps L:%1 R:%2 pair:%3 | 丢帧 L:%4(%5) R:%6(%7)")
                             .arg(lf).arg(rf).arg(pf)
                             .arg(ld - lastLeftDropped_).arg(ld)
                             .arg(rd - lastRightDropped_).arg(rd));
        if (!leftBaseInfo_.isEmpty())
            leftInfoLbl_->setText(leftBaseInfo_ + QStringLiteral("  |  %1 fps").arg(lf));
        if (!rightBaseInfo_.isEmpty())
            rightInfoLbl_->setText(rightBaseInfo_ + QStringLiteral("  |  %1 fps").arg(rf));
        lastLeftFrames_ = l;
        lastRightFrames_ = r;
        lastPairCount_ = p;
        lastLeftDropped_ = ld;
        lastRightDropped_ = rd;
    });

    rig_ = std::make_unique<StereoCameraRig>();
    scanner_ = std::make_unique<ScannerControl>();
    m_pLeft = new cv::Mat();
    m_pRight = new cv::Mat();
    m_iLeftId = 0;
    m_iRightId = 0;
    m_iIndex = 0;
    // 用回调代替 Qt signal（避开 ScannerControl 的 MOC 依赖问题）
    scanner_->onStatus = [this](const QString& msg) { emit statusMessage(msg); };
    // 串口监视：下行/上行帧转发为 Qt 信号（SerialMonitorDialog 显示）
    scanner_->onTx = [this](const QString& frame) { emit serialTx(frame); };
    scanner_->onRx = [this](const QString& frame) { emit serialRx(frame); };
    // G02 温度（主线程事件循环回调，直接刷标签；1Hz 打一条日志留档）
    scanner_->onTemp = [this](float a, float b, float c, float d) {
        tempLbl_->setText(QStringLiteral("温度 A:%1 B:%2 C:%3 D:%4 °C")
                              .arg(a, 0, 'f', 1).arg(b, 0, 'f', 1)
                              .arg(c, 0, 'f', 1).arg(d, 0, 'f', 1));
        spdlog::info("[Scanner] G02 温度 A:{:.1f} B:{:.1f} C:{:.1f} D:{:.1f}", a, b, c, d);
    };

    // 串口通讯监视弹窗：随软件打开自动弹出（非模态，不挡主界面操作），
    // 关掉后可点「🖥 串口监视」重开
    monitor_ = new SerialMonitorDialog(this);
    connect(this, &AcquisitionTab::serialTx, monitor_, &SerialMonitorDialog::appendTx);
    connect(this, &AcquisitionTab::serialRx, monitor_, &SerialMonitorDialog::appendRx);

    onRefreshDevices();
    onRefreshComPorts();
    initSnapCounters();
    updateSnapCount();

    // 开机自动：开串口 → 开设备（注册帧回调）。
    // 扫描仪启动留给用户点（onStartScanner 里发 N10 + 自动进预览）。
    // 延迟到事件循环启动后执行，避免在构造函数里阻塞 UI
    QTimer::singleShot(0, this, [this] {
        monitor_->show();   // 串口监视弹窗随软件打开自动弹出
        autoStart();
    });
}

AcquisitionTab::~AcquisitionTab() {
    // 先收尾连续存储：停入队 → 等写盘线程排空退出（关窗时允许等待），再拆相机/串口
    stopRecord();
    if (recordWriter_.joinable()) recordWriter_.join();
    // 退出前先停扫描仪（电机/激光）——无论哪种退出路径（关闭窗口/菜单退出/quit），
    // Qt 对象树析构都会走到这里；串口未开或未启动则静默跳过，不发多余 N11
    if (scanner_ && scanner_->isOpen() && scannerRunning_) {
        scanner_->stop();
        scannerRunning_ = false;
    }
    if (rig_) {
        // 顺序关键（修复关闭 GUI 时 Qt5Widgets 0xC0000005 读 0x8 崩溃）：
        //   1. 先清回调 → 新 dispatchFrame 拷贝到空 cb，不再执行捕获 this 的 lambda
        //   2. stopAcquisition → 内部等待 in-flight dispatchFrame 退出（见 GalaxyCameraSource）
        //   3. disconnect → 断开 updateImages，杜绝残留 QueuedConnection 事件
        //   4. close → 释放 SDK 资源
        rig_->setLeftFrameCallback(nullptr);
        rig_->setRightFrameCallback(nullptr);
        if (rig_->isAcquiring()) rig_->stopAcquisition();
        disconnect(this, &AcquisitionTab::updateImages, this, &AcquisitionTab::onUpdateImages);
        rig_->close();
    }
    delete m_pLeft;  m_pLeft = nullptr;   // 修复原裸指针泄漏
    delete m_pRight; m_pRight = nullptr;
}

void AcquisitionTab::onSourceTypeChanged() {
    deviceListLbl_->setText(QStringLiteral("（源类型已改，请刷新）"));
    if (rig_->isOpen()) onOpenClose();  // 自动关闭已打开的旧源
}

void AcquisitionTab::onRefreshDevices() {
    QString type = sourceTypeCbx_->currentText();
    auto names = ICameraSource::enumerateDevices(type.toStdString());
    QStringList qlist;
    for (const auto& n : names) qlist << QString::fromUtf8(n.data(), (int)n.size());
    deviceListLbl_->setText(qlist.isEmpty() ? QStringLiteral("（无设备）")
                                            : qlist.join(QStringLiteral("  |  ")));
}

void AcquisitionTab::onBrowseLeftFolder() {
    QString d = QFileDialog::getExistingDirectory(this,
        QStringLiteral("左源图像文件夹"), leftFolderEdit_->text());
    if (!d.isEmpty()) leftFolderEdit_->setText(d);
}

void AcquisitionTab::onBrowseRightFolder() {
    QString d = QFileDialog::getExistingDirectory(this,
        QStringLiteral("右源图像文件夹"), rightFolderEdit_->text());
    if (!d.isEmpty()) rightFolderEdit_->setText(d);
}

// 左右对比度（261002）：滑条 → 相机源（Galaxy 源内逐帧 LUT；simulated/hikvision
// 默认 no-op）＋配置文件持久化（exe 同目录 factory_calib_gui.ini）。设备未开时
// 仅落盘，开设备时 onOpenClose 会按滑条当前值统一下发
void AcquisitionTab::applyContrastFromUi() {
    const int l = contrastLSlider_->value();
    const int r = contrastRSlider_->value();
    if (rig_ && rig_->isOpen()) {
        if (auto* L = rig_->left())  L->setContrast(l);
        if (auto* R = rig_->right()) R->setContrast(r);
    }
    QSettings s(QCoreApplication::applicationDirPath() + QStringLiteral("/factory_calib_gui.ini"),
                QSettings::IniFormat);
    s.setValue("acquisition/contrastLeft", l);
    s.setValue("acquisition/contrastRight", r);
}

// 导出左右对比度到扫描软件（scan_demo）config/camera.json（261002）：扫描软件每次
// 启动读该文件（AppContext），取 camera.contrastLeft/contrastRight 两键。读-改-写
// 合并——保留 deviceIndex/rotate180/trigger 等其它装机口径键；文件不存在则新建
// 最小结构；路径记入本 GUI 配置，第二次起默认定位同一文件
void AcquisitionTab::onExportContrast() {
    const int l = contrastLSlider_->value();
    const int r = contrastRSlider_->value();
    QSettings ini(QCoreApplication::applicationDirPath() + QStringLiteral("/factory_calib_gui.ini"),
                  QSettings::IniFormat);
    const QString last = ini.value("scanSoftware/cameraJson").toString();
    const QString path = QFileDialog::getSaveFileName(
        this, QStringLiteral("选择扫描软件 camera.json（合并写入对比度）"),
        last.isEmpty() ? QStringLiteral("camera.json") : last,
        QStringLiteral("JSON (*.json)"));
    if (path.isEmpty()) return;   // 用户取消

    // 读现有文件（存在且合法才合并；否则问一次是否新建最小结构）
    QJsonObject root;
    bool merged = false;
    {
        QFile f(path);
        if (f.open(QIODevice::ReadOnly)) {
            QJsonParseError err{};
            const QJsonDocument doc = QJsonDocument::fromJson(f.readAll(), &err);
            if (err.error == QJsonParseError::NoError && doc.isObject()) {
                root = doc.object();
                merged = true;
            }
        }
    }
    if (!merged && QFile::exists(path)) {
        if (QMessageBox::question(this, QStringLiteral("camera.json"),
                QStringLiteral("目标文件不是有效 JSON——覆盖为标准装机口径模板（含当前对比度）？")) !=
            QMessageBox::Yes)
            return;
    }
    if (!merged) {
        // 全键模板（261002 定版：跨机整文件替换安全）——键集对齐扫描软件出厂基线
        // config/camera.json（缺省键由扫描软件 AppContext 按内置默认补齐，行为一致）
        QJsonObject cam;
        cam[QStringLiteral("deviceIndexLeft")]  = 0;
        cam[QStringLiteral("deviceIndexRight")] = 1;
        cam[QStringLiteral("rotateRight180")]   = true;
        cam[QStringLiteral("triggerSource")]    = QStringLiteral("Line2");
        cam[QStringLiteral("previewFps")]       = 10;
        cam[QStringLiteral("pairStrictFrameId")] = true;
        root[QStringLiteral("camera")] = cam;
        root[QStringLiteral("_说明")] = QStringLiteral(
            "相机装机口径配置（工厂标定导出 261002）：contrastLeft/contrastRight＝厂家"
            "标定工位调定的左右相机对比度（扫描软件逐帧 cv::LUT 变换；0=直通；正增强/"
            "负减弱，域 [-100,100]）。整文件拷贝到扫描软件 exe 目录 config\\ 下替换即"
            "生效（启动时读取）；扫描软件构建只播种不覆盖，不会冲掉本文件。");
    }

    QJsonObject cam = root.value(QStringLiteral("camera")).toObject();
    cam[QStringLiteral("contrastLeft")] = l;
    cam[QStringLiteral("contrastRight")] = r;
    root[QStringLiteral("camera")] = cam;

    QSaveFile out(path);
    if (!out.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        emit statusMessage(QStringLiteral("导出失败：无法写 %1").arg(path));
        return;
    }
    out.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    if (!out.commit()) {
        emit statusMessage(QStringLiteral("导出失败：写入 %1 出错").arg(path));
        return;
    }
    ini.setValue("scanSoftware/cameraJson", path);   // 记住路径，下次一键导出
    emit statusMessage(QStringLiteral(
        "对比度已导出到扫描软件：L=%1 R=%2 → %3（扫描软件下次启动生效）")
        .arg(l).arg(r).arg(QDir::toNativeSeparators(path)));
}

void AcquisitionTab::onOpenClose() {
    if (rig_->isOpen()) {
        rig_->close();
        setDeviceOpenUI(false);
        lastLeft_.release();
        lastRight_.release();
        leftPreview_->clearImage();
        rightPreview_->clearImage();
        leftInfoLbl_->setText(QStringLiteral("-"));
        rightInfoLbl_->setText(QStringLiteral("-"));
        emit statusMessage(QStringLiteral("设备已关闭"));
        return;
    }

    QString type = sourceTypeCbx_->currentText();
    QString leftFolder  = leftFolderEdit_->text();
    QString rightFolder = rightFolderEdit_->text();
    if (!rig_->configure(type.toStdString(),
                          leftDevSpin_->value(), rightDevSpin_->value(),
                          leftFolder.toStdString(), rightFolder.toStdString())) {
        emit statusMessage(QStringLiteral("源创建失败"));
        return;
    }
    if (!rig_->open()) {
        emit statusMessage(QStringLiteral("设备 open 失败"));
        return;
    }
    // 中心 ROI（261005）：设备已开未采集，立即下发硬件 AOI；此后 info 栏/预览/存图
    // 均为 ROI 后尺寸（simulated 等不支持 ROI 的源为 no-op，保持原尺寸）
    if (auto* L = rig_->left())  L->setCenterRoi(roiW_, roiH_);
    if (auto* R = rig_->right()) R->setCenterRoi(roiW_, roiH_);
    // 应用曝光/增益（旋转统一在 GUI 层按预览框勾选处理，与 deviceId 解耦）
    if (auto* L = rig_->left()) {
        if (L->capability().exposureUs) L->setExposureUs(exposureSpin_->value());
        if (L->capability().gainDb)     L->setGainDb(gainSpin_->value());
        L->setContrast(contrastLSlider_->value());   // 左右分置对比度（261002，软件 LUT）
        leftBaseInfo_ = QString::fromStdString(L->displayName()) +
            QStringLiteral("  %1×%2").arg(L->width()).arg(L->height());
        leftInfoLbl_->setText(leftBaseInfo_);
    }
    if (auto* R = rig_->right()) {
        if (R->capability().exposureUs) R->setExposureUs(exposureSpin_->value());
        if (R->capability().gainDb)     R->setGainDb(gainSpin_->value());
        R->setContrast(contrastRSlider_->value());   // 左右分置对比度（261002，软件 LUT）
        rightBaseInfo_ = QString::fromStdString(R->displayName()) +
            QStringLiteral("  %1×%2").arg(R->width()).arg(R->height());
        rightInfoLbl_->setText(rightBaseInfo_);
    }
    // 照搬 LeadScanK2：回调线程调 setLeftImage/setRightImage（内部配对后 emit updateImages）
    rig_->setLeftFrameCallback([this](const CameraFrame& f) {
        ++leftFrames_;   // 帧率统计：本机累计
        // 丢帧检测：frameID 跳号 = 中间帧在相机缓冲/USB 传输环节丢失（ID 回退=流重开，重置基线）
        if (leftLastId_ >= 0 && f.frameIndex > leftLastId_ + 1)
            leftDropped_ += f.frameIndex - leftLastId_ - 1;
        leftLastId_ = f.frameIndex;
        cv::Mat img = f.image;
        if (leftRotateChk_ && leftRotateChk_->isChecked() && !img.empty()) {
            cv::rotate(img, img, cv::ROTATE_180);
        }
        lastLeft_ = img;
        setLeftImage(img, static_cast<uint64_t>(f.frameIndex));
    });
    rig_->setRightFrameCallback([this](const CameraFrame& f) {
        ++rightFrames_;   // 帧率统计：本机累计
        // 丢帧检测：同左
        if (rightLastId_ >= 0 && f.frameIndex > rightLastId_ + 1)
            rightDropped_ += f.frameIndex - rightLastId_ - 1;
        rightLastId_ = f.frameIndex;
        cv::Mat img = f.image;
        if (rightRotateChk_ && rightRotateChk_->isChecked() && !img.empty()) {
            cv::rotate(img, img, cv::ROTATE_180);
        }
        lastRight_ = img;
        setRightImage(img, static_cast<uint64_t>(f.frameIndex));
    });

    setDeviceOpenUI(true);
    emit statusMessage(QStringLiteral("设备已打开，点「▶ 启动扫描仪」开始实时预览"));
}

void AcquisitionTab::setDeviceOpenUI(bool opened) {
    openBtn_->setText(opened ? QStringLiteral("关闭设备") : QStringLiteral("打开设备"));
    captureBtn_->setEnabled(opened);
    previewBtn_->setEnabled(opened);
    recordBtn_->setEnabled(opened);
    sourceTypeCbx_->setEnabled(!opened);
    leftDevSpin_->setEnabled(!opened);
    rightDevSpin_->setEnabled(!opened);
    leftFolderEdit_->setEnabled(!opened);
    rightFolderEdit_->setEnabled(!opened);
    refreshBtn_->setEnabled(!opened);
    if (opened) {
        // 帧率统计开表：快照对齐当前计数（首个 1s 差分从现在起算，不吃历史累计）
        lastLeftFrames_ = leftFrames_.load();
        lastRightFrames_ = rightFrames_.load();
        lastPairCount_ = pairCount_.load();
        // 丢帧检测开表：累计清零、frameID 基线待定（首帧建立基线）
        leftDropped_ = 0;  rightDropped_ = 0;
        leftLastId_ = -1;  rightLastId_ = -1;
        lastLeftDropped_ = 0;  lastRightDropped_ = 0;
        fpsTimer_->start();
    } else {
        fpsTimer_->stop();
        fpsLbl_->setText(QStringLiteral("fps L:- R:- pair:- | 丢帧 L:-(?) R:-(?)"));
        stopRecord();  // 设备关闭：连续存储随停（幂等）
        if (recordBtn_->isChecked()) recordBtn_->setChecked(false);
        saveCameraBtn_->setEnabled(false);
        saveLaserBtn_->setEnabled(false);
    }
}

void AcquisitionTab::onFreezeFrame() {
    // 预览当前帧：界面停在当前帧，后台 lastLeft_/Right 仍持续刷新；「保存图像」存这一帧
    previewing_ = false;
    previewBtn_->setChecked(false);
    emit statusMessage(QStringLiteral("已停在当前帧（后台仍刷新）。点「▶ 连续预览」恢复刷新"));
}

void AcquisitionTab::onTogglePreview(bool on) {
    previewing_ = on;
    if (on) emit statusMessage(QStringLiteral("连续预览（界面实时刷新）"));
}

// ============================================================================
// 连续存储：全分辨率左右成对帧 → 独立写盘线程落盘
//   点一下开始（按钮变绿），再点一下停止；与预览状态互不影响（冻结预览也在存）
//   目录 data_in/continuous/<开始时间yyyyMMdd_HHmmss>/{left,right}/NNNNNN.png
// ============================================================================
void AcquisitionTab::onToggleRecord(bool on) {
    if (on) {
        if (recording_) return;
        // 上一轮 writer 若还在排空收尾，此处 join（通常早已退出，瞬时返回）
        if (recordWriter_.joinable()) recordWriter_.join();
        recordDir_ = QStringLiteral("data_in/continuous/") +
            QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss"));
        QDir().mkpath(recordDir_ + QStringLiteral("/left"));
        QDir().mkpath(recordDir_ + QStringLiteral("/right"));
        {
            std::lock_guard<std::mutex> lk(recordMtx_);
            recordQueue_.clear();
        }
        recordWritten_ = 0;
        recordDropped_ = 0;
        recording_ = true;
        recordWriter_ = std::thread([this] { recordWriterLoop(); });
        // 心跳：进度回报 + 断流报警（基线取当前 pairCount_，断流按差值判定）
        lastFedPairs_ = pairCount_.load();
        starvedTicks_ = 0;
        heartbeatTicks_ = 0;
        recordHeartbeat_->start();
        emit statusMessage(QStringLiteral("⏺ 连续存储已开始: ") + recordDir_);
    } else {
        stopRecord();
    }
}

void AcquisitionTab::stopRecord() {
    if (!recording_.exchange(false)) return;   // 未在存储/已停止（幂等）
    recordHeartbeat_->stop();
    recordCv_.notify_all();
    // 不在 UI 线程 join：剩余队列（至多 120 对全分辨率 PNG）写完可能数秒，会卡死界面。
    // writer 线程排空后经 QueuedConnection 回报汇总；join 留给下一次 start（瞬时）与析构。
    size_t pending = 0;
    {
        std::lock_guard<std::mutex> lk(recordMtx_);
        pending = recordQueue_.size();
    }
    if (pending > 0) {
        emit statusMessage(QStringLiteral("停止连续存储：后台正在写完剩余 %1 对…").arg(pending));
    }
}

// 相机回调线程调用（配对成功处）：入队已克隆的私有整帧。
// 克隆在 pairMtx_ 内完成（见 setLeft/RightImage），此处只入队/满队丢弃，
// 不再读 m_pLeft/m_pRight——顺带消除旧版「入队 clone 与对侧换帧并发」的竞态。
void AcquisitionTab::enqueueRecordPair(cv::Mat left, cv::Mat right) {
    if (!recording_) return;
    {
        std::lock_guard<std::mutex> lk(recordMtx_);
        if (recordQueue_.size() >= kRecordQueueCap) {
            ++recordDropped_;   // 写盘跟不上：丢新帧保内存，停止时汇总提示
            return;
        }
        recordQueue_.emplace_back(std::move(left), std::move(right));
    }
    recordCv_.notify_one();
}

// 后台写盘线程：PNG 低压缩（level 1）换写盘速度；停止后排空队列再退出
void AcquisitionTab::recordWriterLoop() {
    const std::vector<int> pngFast = {cv::IMWRITE_PNG_COMPRESSION, 1};
    while (true) {
        std::pair<cv::Mat, cv::Mat> item;
        {
            std::unique_lock<std::mutex> lk(recordMtx_);
            recordCv_.wait(lk, [this] { return !recording_ || !recordQueue_.empty(); });
            if (recordQueue_.empty()) {
                if (!recording_) break;   // 已停止且队列排空
                continue;
            }
            item = std::move(recordQueue_.front());
            recordQueue_.pop_front();
        }
        const QString n = QString::number(recordWritten_.load()).rightJustified(6, '0');
        cv::imwrite((recordDir_ + QStringLiteral("/left/")  + n + QStringLiteral(".png")).toStdString(),
                    item.first, pngFast);
        cv::imwrite((recordDir_ + QStringLiteral("/right/") + n + QStringLiteral(".png")).toStdString(),
                    item.second, pngFast);
        ++recordWritten_;
    }
    // 排空收尾：汇总经 singleShot 投递回 UI 线程（context=this，对象已销毁则自动丢弃；
    // Qt5.9 无 invokeMethod functor+connectionType 重载，故用 singleShot 等价实现）
    const int written = recordWritten_.load();
    int dropped = 0;
    {
        std::lock_guard<std::mutex> lk(recordMtx_);
        dropped = recordDropped_;
    }
    const QString dir = recordDir_;
    QTimer::singleShot(0, this, [this, written, dropped, dir] {
        emit statusMessage(QStringLiteral("⏹ 连续存储结束: %1（已写 %2 对，丢弃 %3 对）")
                               .arg(dir).arg(written).arg(dropped));
    });
}

void AcquisitionTab::onSaveCamera() {
    if (frozenLeft_.empty() || frozenRight_.empty()) {
        emit statusMessage(QStringLiteral("尚无显示帧可保存"));
        return;
    }
    QString base = QStringLiteral("data_in/camera");
    QDir().mkpath(base + "/left");
    QDir().mkpath(base + "/right");
    QString n = QString::number(cameraSnapshotIdx_).rightJustified(3, '0');
    QString lp = base + "/left/"  + n + ".png";
    QString rp = base + "/right/" + n + ".png";
    cv::imwrite(lp.toStdString(), frozenLeft_);
    cv::imwrite(rp.toStdString(), frozenRight_);
    ++cameraSnapshotIdx_;
    updateSnapCount();
    emit statusMessage(QStringLiteral("已保存（当前显示帧）: ") + lp + " / " + rp);
}

void AcquisitionTab::onSaveLaser() {
    if (frozenLeft_.empty() || frozenRight_.empty()) {
        emit statusMessage(QStringLiteral("尚无显示帧可保存"));
        return;
    }
    int mode = scannerModeCbx_->currentData().toInt();
    if (mode < 1 || mode > kLaserTypeCount) {
        emit statusMessage(QStringLiteral("当前是相机标定模式：请先在「模式」中选择激光线类型"));
        return;
    }
    const int ti = mode - 1;
    QString base = laserTypeDir(ti) + QStringLiteral("/pose_") +
        QString::number(laserPoseIdx_[ti]).rightJustified(2, '0');
    QDir().mkpath(base);
    cv::imwrite((base + "/L_tube0.png").toStdString(), frozenLeft_);
    cv::imwrite((base + "/R_tube0.png").toStdString(), frozenRight_);
    QFile f(base + "/temp.txt");
    if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        f.write("ref_temp 25.0\n");
    }
    ++laserPoseIdx_[ti];
    updateSnapCount();
    emit statusMessage(QStringLiteral("[%1] 已保存: ").arg(laserTypeName(ti)) + base);
}

void AcquisitionTab::initSnapCounters() {
    // 相机：data_in/camera/left/NNN.png 取最大编号+1
    QDir camDir(QStringLiteral("data_in/camera/left"));
    for (const QString& fn : camDir.entryList({QStringLiteral("*.png")}, QDir::Files)) {
        bool ok = false;
        int n = QFileInfo(fn).completeBaseName().toInt(&ok);
        if (ok && n + 1 > cameraSnapshotIdx_) cameraSnapshotIdx_ = n + 1;
    }
    // 四类激光：data_in/laser/<key>/pose_NN 取最大编号+1
    for (int i = 0; i < kLaserTypeCount; ++i) {
        QDir d(laserTypeDir(i));
        for (const QString& dn :
             d.entryList({QStringLiteral("pose_*")}, QDir::Dirs | QDir::NoDotAndDotDot)) {
            bool ok = false;
            int n = dn.mid(QString("pose_").size()).toInt(&ok);
            if (ok && n + 1 > laserPoseIdx_[i]) laserPoseIdx_[i] = n + 1;
        }
    }
}

void AcquisitionTab::updateSnapCount() {
    snapCountLbl_->setText(QString(QStringLiteral("相机: %1 张    激光pose — 左斜:%2 右斜:%3 精细:%4 深孔:%5"))
                              .arg(cameraSnapshotIdx_)
                              .arg(laserPoseIdx_[0]).arg(laserPoseIdx_[1])
                              .arg(laserPoseIdx_[2]).arg(laserPoseIdx_[3]));
}

void AcquisitionTab::appendLog(const QString&) {
    // 全局日志走 spdlog sink → MainWindow dock
}

// ============================================================================
// GUI 层 180° 旋转（按预览框勾选，不管 deviceId 与物理相机的对应）
// 用户在 UI 上勾哪个框，就旋转哪个预览的图
// ============================================================================
void AcquisitionTab::applyGuiRotation(cv::Mat& leftImg, cv::Mat& rightImg) {
    if (leftRotateChk_->isChecked() && !leftImg.empty()) {
        cv::Mat rotated;
        cv::rotate(leftImg, rotated, cv::ROTATE_180);
        leftImg = rotated;
    }
    if (rightRotateChk_->isChecked() && !rightImg.empty()) {
        cv::Mat rotated;
        cv::rotate(rightImg, rotated, cv::ROTATE_180);
        rightImg = rotated;
    }
}

// ============================================================================
// 扫描仪硬件串口控制（仿 LeadScanK2/series/LEADSCANSeries）
// ============================================================================

void AcquisitionTab::onRefreshComPorts() {
    comPortCbx_->clear();
    for (const auto& name : ScannerControl::availablePorts()) {
        comPortCbx_->addItem(name);
    }
    if (comPortCbx_->count() == 0) {
        emit statusMessage(QStringLiteral("未发现任何 COM 口"));
    }
}

void AcquisitionTab::onOpenCloseScanner() {
    if (!scanner_) return;
    if (scanner_->isOpen()) {
        scanner_->close();
        scannerOpenBtn_->setText(QStringLiteral("打开串口"));
        scannerStartBtn_->setEnabled(false);
        scannerStopBtn_->setEnabled(false);
        comPortCbx_->setEnabled(true);
        comRefreshBtn_->setEnabled(true);
        tempLbl_->setText(QStringLiteral("温度 --"));
        return;
    }
    QString port = comPortCbx_->currentText();
    if (port.isEmpty()) {
        emit statusMessage(QStringLiteral("请先选择 COM 口"));
        return;
    }
    if (scanner_->open(port)) {
        scannerOpenBtn_->setText(QStringLiteral("关闭串口"));
        scannerStartBtn_->setEnabled(true);
        scannerStopBtn_->setEnabled(true);
        comPortCbx_->setEnabled(false);
        comRefreshBtn_->setEnabled(false);
        // 开口即开温度回传：N12 T1000（1Hz G02 四路）。<5ms 会压爆 115200 串口
        // （协议 D9 结论）；显示用途 1Hz 足够，且与扫描启停无关、常显
        if (!scanner_->setTempReportPeriod(1000)) {
            emit statusMessage(QStringLiteral("警告：温度回传周期设置失败（N12 T1000）"));
        }
    }
}

void AcquisitionTab::onStartScanner() {
    if (!scanner_ || !scanner_->isOpen()) {
        emit statusMessage(QStringLiteral("请先打开扫描仪串口"));
        return;
    }
    // 照搬 LeadScanK2 on_pushButton_Start_Scanner_Clicked：直接 start（N10），不先 stop。
    // 先 stop(N11)+start(N10) 连续发会导致 MCU 偶发不启动（第一次稳定、后续偶发的根因）。
    // 如需停再启，由用户分别点「停止」「启动」（秒级间隔），不在一次点击里连发。

    ScannerParams p;
    p.freq       = freqSlider_->value();
    p.background = bgLightSlider_->value();
    // 模式 → N10 五管开关（261105 固件管语义定版：T=左斜、V=右斜、C=深孔、
    //   D=精细、E=备用中竖25线；261002 临时机口径「T=精细/V=左斜/C=右斜/
    //   D=占位」作废）：
    //   0=仅标志点 → 五管全关（仅补光，相机标定）
    //   1=左斜 → 仅 T1；2=右斜 → 仅 V1；3=精细 → 仅 D1；4=深孔 → 仅 C1；
    //   5=中竖 → 仅 E1（备用25线）
    // 协议语义：已开启的管轮流点亮，单开一管即固定该激光线
    int mode = scannerModeCbx_->currentData().toInt();
    p.laser = laserSlider_->value();
    switch (mode) {
        case 1:  p.tubeT = 1; break;   // 左斜 → T 管（261105 语义）
        case 2:  p.tubeV = 1; break;   // 右斜 → V 管
        case 3:  p.tubeD = 1; break;   // 精细 → D 管
        case 4:  p.tubeC = 1; break;   // 深孔 → C 管
        case 5:  p.tubeE = 1; break;   // 中竖 → E 管（备用25线）
        default: p.laser = 0; break;   // 仅标志点：激光关（五管全 0）
    }
    // 布防顺序关键（2026-09-27 修复）：先布防相机、后发 N10。
    // 旧顺序先发 N10 电机立刻转，相机左右布防有数~数十 ms 间隙（含两次 OpenStream），
    // 间隙内来一个触发脉冲只有一侧拍到 → 左右 frameID 从此永久差 1 → 配对死锁
    // （连续存储整场 0 张的根因之一）。现在脉冲到来时两相机必然都已就绪。
    bool camReady = true;
    if (rig_ && rig_->isOpen()) {
        if (rig_->isAcquiring()) rig_->stopAcquisition();  // 防重复 StartGrab/Register
        camReady = rig_->startAcquisition();               // 两相机全部 arm + AcquisitionStart
    }

    // 照搬 LeadScanK2 on_pushButton_Start_Scanner_Clicked：直接 start（N10），不先 stop。
    // 先 stop(N11)+start(N10) 连续发会导致 MCU 偶发不启动（第一次稳定、后续偶发的根因）。
    // 如需停再启，由用户分别点「停止」「启动」（秒级间隔），不在一次点击里连发。
    scanner_->start(p);
    scannerRunning_ = true;

    if (camReady) {
        previewing_ = true;
        previewBtn_->setChecked(true);   // 默认连续预览
        emit statusMessage(QStringLiteral("实时预览已开启（连续预览，回调驱动刷新）"));
    } else if (rig_ && rig_->isOpen()) {
        emit statusMessage(QStringLiteral("警告：相机启动采集失败，无法预览"));
    }

    if (mode == 0) {
        emit statusMessage(QStringLiteral("[相机标定模式] 仅标志点 L=0 (激光关)"));
    } else {
        emit statusMessage(QStringLiteral("[%1激光线模式] 激光 L=%2 + 补光 B=%3 "
                                          "(若看不到激光线：调高激光滑块/降背光/增曝光)")
                              .arg(laserTypeName(mode - 1)).arg(p.laser).arg(p.background));
    }
}

void AcquisitionTab::onStopScanner() {
    if (!scanner_) return;
    // 停相机采集（注销 SDK 回调，停止预览）→ 停扫描仪
    previewing_ = false;
    previewBtn_->setChecked(false);
    if (rig_ && rig_->isAcquiring()) rig_->stopAcquisition();
    scanner_->stop();
    scannerRunning_ = false;
}

void AcquisitionTab::stopScannerIfRunning() {
    // 标定启动时调用：未启动/串口未开则什么都不做（避免对未启动的扫描仪盲发 N11）
    if (!scanner_ || !scanner_->isOpen() || !scannerRunning_) return;
    onStopScanner();
    emit statusMessage(QStringLiteral("检测到标定启动，已自动停止扫描仪"));
}

// 照搬 LeadScanK2 convertMattoQImage：Format_Indexed8 + 逐行 memcpy 到 scanLine + 256 灰度颜色表
namespace {
QImage convertMattoQImage(cv::Mat& mat) {
    if (mat.type() == CV_8UC1) {
        QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
        image.setColorCount(256);
        for (int i = 0; i < 256; i++) image.setColor(i, qRgb(i, i, i));
        uchar* pSrc = mat.data;
        for (int row = 0; row < mat.rows; row++) {
            uchar* pDest = image.scanLine(row);
            memcpy(pDest, pSrc, mat.cols);
            pSrc += mat.step;
        }
        return image;
    }
    return QImage();
}

// 261114 BGR 8UC3 → QImage（棋盘角点叠加帧专用）：包装构造 + copy() 深拷贝
// （源 Mat 为回调线程局部对象，跨线程 emit 前必须断开数据指针）
QImage matBgrToQImage(const cv::Mat& mat) {
    if (mat.empty() || mat.type() != CV_8UC3) return QImage();
    QImage img(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step),
               QImage::Format_RGB888);
    return img.copy();
}

// 261108 采集参数判定（口径与 tools/calib_shot_monitor.py 同源）：
// 过曝占比 / 游程>30px 计数 / 中部行线数估计。mask 阈值 120 与标定 config 同口径。
// 261114 二修：加腐蚀（对齐标定 mask erodeSize=3 全分辨率 ≈ 降采样 1px）＋
// 逐行游程宽度硬门（对齐标定 laser_label maxRunWidth=30px 全分辨率 ≈ 降采样 8px）——
// 拍照时提前发现粘连，不再到标定端才被拒帧。
struct ShotMetrics {
    double hotPct = 0.0;
    int wideRuns = 0;
    int lineEst = 0;
    double bg = 0.0;   // 261108 背景灰度（环境光泄漏指标——右斜 PASS 批
                       // 14~19 稳定 vs 左斜 28~55 波动即残差 6.5 主因）
    int maxRunWidthDs = 0; // 261114 降采样域最大游程宽度（px）
    cv::Mat overlay;       // 261114 激光线叠加预览图（BGR 小图；空=不叠加）
};
static ShotMetrics analyzeShot(const cv::Mat& full) {
    ShotMetrics m;
    if (full.empty()) return m;
    // 261108 提速：0.25 降采样（全分辨率 CCL ~40ms 回调线程阻塞＝判定滞后
    // 主因；降采样后 ~3ms。比例类指标（过曝/背景）不变；尺寸门等比 /4、
    // 面积门 /16）
    cv::Mat gray;
    cv::resize(full, gray, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
    const int hot = cv::countNonZero(gray > 235);
    m.hotPct = 100.0 * hot / (gray.rows * gray.cols);
    m.bg = cv::mean(gray)[0];
    cv::Mat bw;
    cv::threshold(gray, bw, 120, 255, cv::THRESH_BINARY);
    // 线数＝CCL 大组件数——用腐蚀前的原图（261114 一修教训：腐蚀后细线
    // 1~3px 被面积门≥32 滤掉→线数骤降误报「线不足」；线数口径维持 v4 终版）
    cv::Mat labels, statsv, centv;
    const int n = cv::connectedComponentsWithStats(bw, labels, statsv, centv, 8);
    std::vector<int> areas;
    for (int i = 1; i < n; ++i) {
        const int area = statsv.at<int>(i, cv::CC_STAT_AREA);
        const int hh = statsv.at<int>(i, cv::CC_STAT_HEIGHT);
        const int ww = statsv.at<int>(i, cv::CC_STAT_WIDTH);
        if (area >= 32 && std::max(hh, ww) >= 13)
            areas.push_back(area);
    }
    std::sort(areas.begin(), areas.end(), std::greater<int>());
    const int expect = judgeExpectGlobal.load(std::memory_order_relaxed);
    if (!areas.empty()) {
        const int nth = areas[std::min<size_t>(areas.size() - 1,
                                               static_cast<size_t>(std::max(1, expect) - 1))];
        const int gate = static_cast<int>(0.3 * nth);   // 261114 二修 0.7→0.3：远近亮度差异致暗线被误杀
        m.lineEst = static_cast<int>(
            std::count_if(areas.begin(), areas.end(),
                          [gate](int a) { return a >= gate; }));
        // 261114 三修 激光线彩色叠加：每条线不同颜色着色其像素——
        // 刷新率限 2Hz（500ms 一次）：每帧重建色图→颜色跳来跳去「闪烁看不清」
        {
            static auto s_lastOvlTime = std::chrono::steady_clock::now();
            const auto now2 = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(
                    now2 - s_lastOvlTime).count() >= 500) {
                s_lastOvlTime = now2;
                // 构建 label→是否计入线数 的映射
                std::map<int, bool> labelIsLine;
                for (int i = 1; i < n; ++i) {
                    const int area = statsv.at<int>(i, cv::CC_STAT_AREA);
                    const int hh = statsv.at<int>(i, cv::CC_STAT_HEIGHT);
                    const int ww = statsv.at<int>(i, cv::CC_STAT_WIDTH);
                    if (area >= 32 && std::max(hh, ww) >= 13)
                        labelIsLine[i] = (area >= gate);
                }
                static const cv::Scalar kPalette[] = {
                    {0,0,255},{0,64,255},{0,128,255},{0,191,255},{0,255,255},
                    {0,255,191},{0,255,128},{0,255,64},{0,255,0},{64,255,0},
                    {128,255,0},{191,255,0},{255,255,0},{255,191,0},{255,128,0},
                    {255,64,0},{255,0,0},{255,0,64},{255,0,128},{255,0,191},
                    {255,0,255},{191,0,255},{128,0,255},{64,0,255},{0,0,255}
                };
                const int palSize = sizeof(kPalette) / sizeof(kPalette[0]);
                cv::cvtColor(gray, m.overlay, cv::COLOR_GRAY2BGR);
                int lineIdx = 0;
                for (const auto& [label, isLine] : labelIsLine) {
                    if (!isLine) continue;
                    const cv::Scalar& color = kPalette[lineIdx % palSize];
                    for (int y = 0; y < labels.rows; ++y) {
                        for (int x = 0; x < labels.cols; ++x) {
                            if (labels.at<int>(y, x) == label) {
                                cv::Vec3b& px = m.overlay.at<cv::Vec3b>(y, x);
                                px[0] = static_cast<uchar>(0.5 * px[0] + 0.5 * color[0]);
                                px[1] = static_cast<uchar>(0.5 * px[1] + 0.5 * color[1]);
                                px[2] = static_cast<uchar>(0.5 * px[2] + 0.5 * color[2]);
                            }
                        }
                    }
                    ++lineIdx;
                }
            }
        }
    }
    // 261114 逐行游程宽度（对齐标定 laser_label maxRunWidth 硬门）：
    // 腐蚀后的二值图 → 中部行（或列，按 scanDirection）切一刀 → 数连续亮像素段
    // 降采样域门限 ≈ 全分辨率 30px ÷ 4 ≈ 8px——超过即粘连
    {
        // 261114 腐蚀仅在游程测量用——分开的线腐蚀后 still 分开（游程小）；
        // 粘连的线腐蚀后仍连通（游程大）→ 暴露粘连。线数计数不受影响。
        cv::Mat bwEroded;
        cv::erode(bw, bwEroded, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
        const int row = scanDirectionGlobal.load(std::memory_order_relaxed) == 1
                            ? bwEroded.rows / 2     // 斜线形态：水平切
                            : -1;                    // 竖线形态：用列切
        if (row >= 0 && row < bwEroded.rows) {
            const uchar* r = bwEroded.ptr<uchar>(row);
            int run = 0;
            for (int x = 0; x < bwEroded.cols; ++x) {
                if (r[x] > 0) { ++run; if (run > m.maxRunWidthDs) m.maxRunWidthDs = run; }
                else run = 0;
            }
        } else if (!bwEroded.empty()) {
            const int col = bwEroded.cols / 2;
            int run = 0;
            for (int y = 0; y < bwEroded.rows; ++y) {
                if (bwEroded.at<uchar>(y, col) > 0) { ++run; if (run > m.maxRunWidthDs) m.maxRunWidthDs = run; }
                else run = 0;
            }
        }
        // 降采样域粘连门限：全分辨率 30px ÷ 4 = 7.5 → 取 8px
        if (m.maxRunWidthDs > 8)
            m.wideRuns = 1;
    }
    return m;
}
}  // namespace

// 261114 相机标定棋盘格实时判定（模式0 专用）：双目角点检出（11×8）＋铺满度＋过曝。
// 相机回调线程执行（与 analyzeShot 同约定：收全分辨率克隆帧，内部 0.25 降采样提速，
// ≈30Hz 限频窗口内每对一次）；报告经 updateJudge（QueuedConnection）送 UI 线程渲染。
// 双目均检出时角点叠加帧经 updateImages 直发并返回 true——调用方跳过本帧常规预览
// emit（两个 Queued 信号按序到达，后发的原图会冲掉先发的叠加）。
bool AcquisitionTab::analyzeChessboard(const cv::Mat& leftFull, const cv::Mat& rightFull) {
    // 棋盘格参数：11×8 内角点（与 CameraCalibTab / data_in camera 标定口径一致）
    const cv::Size boardSize(11, 8);

    struct BoardResult {
        bool found = false;
        double coverage = 0.0;     // 角点包围盒面积占画面比 %
        double overexposed = 0.0;  // 过曝占比 %（>235，与 analyzeShot 同口径）
        cv::Mat overlay;           // 角点叠加画布（BGR 小图；未检出时为空）
    };

    auto analyzeOne = [&](const cv::Mat& full, BoardResult& r) {
        if (full.empty()) return;
        // 0.25 降采样（与 analyzeShot 同款：全分辨率角点检测太慢，回调线程拖不起）
        cv::Mat small;
        cv::resize(full, small, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
        const int hot = cv::countNonZero(small > 235);
        r.overexposed = 100.0 * hot / (small.rows * small.cols);
        // 棋盘格角点检测
        std::vector<cv::Point2f> corners;
        const bool found = cv::findChessboardCorners(small, boardSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        if (found && corners.size() == static_cast<size_t>(boardSize.area())) {
            r.found = true;
            // 铺满度：角点包围盒面积 / 图像面积（比例值，与降采样无关）
            const cv::Rect bbox = cv::boundingRect(corners);
            r.coverage = 100.0 * static_cast<double>(bbox.area())
                       / (small.rows * small.cols);
            // 角点叠加：灰度转 BGR 画彩点（drawChessboardCorners 需 8UC3 输入）
            cv::cvtColor(small, r.overlay, cv::COLOR_GRAY2BGR);
            cv::drawChessboardCorners(r.overlay, boardSize, corners, true);
        }
    };

    BoardResult l, r;
    analyzeOne(leftFull, l);
    analyzeOne(rightFull, r);

    // 判定：双目全检出 ＋ 平均过曝 ≤5%。铺满度为信息项（不同距离标定是正确
    // 做法——远拍铺满度低但解耦焦距/距离，不做门限只显示参考值 261114）
    const bool bothFound = l.found && r.found;
    const double avgCov = (l.coverage + r.coverage) / 2.0;
    const double avgOver = (l.overexposed + r.overexposed) / 2.0;
    const bool overexpOk = avgOver <= 5.0;
    const bool allOk = bothFound && overexpOk;

    // HTML 报告（样式/结构与激光判定报告同款：结论置顶＋逐项指标＋调参建议）
    const QString now = QDateTime::currentDateTime().toString(QStringLiteral("HH:mm:ss"));
    QString report = QString("<span style='color:#888;'>[%1]</span><br>").arg(now);
    report += bothFound ? QStringLiteral("✅ 双目棋盘检出（11×8）")
                        : QStringLiteral("✗ 棋盘未检出");
    report += QString("<br>左: %1 | 右: %2")
                  .arg(l.found ? QStringLiteral("✓") : QStringLiteral("✗"),
                       r.found ? QStringLiteral("✓") : QStringLiteral("✗"));
    if (bothFound)
        report += QString("<br>铺满度: %1%（远/中/近都采——信息项）")
                      .arg(avgCov, 0, 'f', 0);
    report += QString("<br>过曝: %1%2（≤5%）")
                  .arg(avgOver, 0, 'f', 1)
                  .arg(overexpOk ? QStringLiteral("✅") : QStringLiteral("✗"));
    if (allOk) {
        report = QStringLiteral(
            "<p style='color:#1a7f37; font-weight:bold;'>✅ 可以采集（双眼全过，按此档采满标定帧）</p>")
            + report;
    } else {
        QString tip;
        if (!bothFound) tip += QStringLiteral("棋盘未完整检出→确保棋盘完整入画、无遮挡、角度别太斜；");
        if (!overexpOk) tip += QStringLiteral("过曝→降曝光或补光；");
        report = QStringLiteral("<p style='color:#b3261e; font-weight:bold;'>✗ 继续调整</p>")
               + report + QStringLiteral("<br><span style='color:#b3261e;'>%1</span>").arg(tip);
    }
    emit updateJudge(report);

    // 角点叠加预览：双目均检出才发（让用户直观看到检测到了什么；未检出发原图）
    if (bothFound) {
        emit updateImages(matBgrToQImage(l.overlay), matBgrToQImage(r.overlay));
        return true;
    }
    return false;
}

// 左右帧配对（SDK 回调线程调用）：轮次制——左右各有未消费新帧即成一对，后到者触发。
// 2026-09-27 修复：旧版照搬 LeadScanK2 用 frameID 严格相等（m_iLeftId==m_iRightId），
// 单边多一帧（左右布防间隙抢到触发脉冲 / ok=0 失败帧占号 / 单边丢帧）后 ID 永久错位 1，
// == 从此永不成立 → 预览冻结 + 连续存储整场 0 张且无报错。轮次制错位自愈，
// 最坏损失一帧新鲜度。配对瞬间在 pairMtx_ 内克隆私有整帧（录制与预览共用，与对侧
// 换帧互不干扰）。
// 预览 emit 限频 ~30fps（120fps 全量 QPixmap 转换+setPixmap 会吃满 UI 线程）；
// 录制不受限频影响。限频窗口且未录制时整帧克隆直接省掉（回调路径最大开销削减点）。
void AcquisitionTab::setLeftImage(cv::Mat& left, uint64_t id) {
    cv::Mat pairL, pairR;
    bool doPreview = false;
    bool judgeOverlaid = false;   // 261114 棋盘角点叠加帧已直发→跳过本帧常规预览 emit
    {
        std::lock_guard<std::mutex> lk(pairMtx_);
        *m_pLeft = left;
        m_iLeftId = id;
        leftFresh_ = true;
        if (!rightFresh_) return;
        const int64_t nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        doPreview = nowNs - lastPreviewNs_ >= kPreviewMinIntervalNs;
        if (doPreview || recording_.load()) {
            pairL = m_pLeft->clone();
            pairR = m_pRight->clone();
        }
        if (doPreview) lastPreviewNs_ = nowNs;
        leftFresh_ = rightFresh_ = false;
    }
    ++pairCount_;
    ++pairCount_;
    // 261108 采集参数判定：每 3 对（≈10Hz）分析一次（0.25 降采样 ~3ms）
    // 261114 判定分流（置于激光判定之前）：开关关→面板提示已禁用；模式0（相机标定）
    // →棋盘格判定（双目检出时角点叠加帧直发，judgeOverlaid 置位跳过本帧常规预览
    // emit，防后到原图冲掉叠加）；其余＝激光线判定（261108 原逻辑不动）
    if (doPreview && !pairL.empty() && !pairR.empty()
        && !judgeEnabled_.load(std::memory_order_relaxed)) {
        emit updateJudge(QStringLiteral(
            "<p style='color:#888; font-weight:bold;'>实时判定已禁用</p>"));
    } else if (doPreview && !pairL.empty() && !pairR.empty()
               && laserMode_.load(std::memory_order_relaxed) == 0) {
        judgeOverlaid = analyzeChessboard(pairL, pairR);
    } else if (doPreview && !pairL.empty() && !pairR.empty()) {   // 261108 每对分析（3ms 无压力——滞后主因曾是间隔+全图CCL）
        const ShotMetrics mlRaw = analyzeShot(pairL);
        const ShotMetrics mrRaw = analyzeShot(pairR);
        // 两帧滑动平均（调参抖动滤波；首帧直通）
        static ShotMetrics pmL, pmR;
        static bool hasPrev = false;
        ShotMetrics ml = mlRaw, mr = mrRaw;
        if (hasPrev) {
            ml.hotPct = 0.5 * (ml.hotPct + pmL.hotPct); ml.bg = 0.5 * (ml.bg + pmL.bg);
            mr.hotPct = 0.5 * (mr.hotPct + pmR.hotPct); mr.bg = 0.5 * (mr.bg + pmR.bg);
        }
        pmL = mlRaw; pmR = mrRaw; hasPrev = true;
        const int expect = judgeExpect_.load(std::memory_order_relaxed);
        auto oneEye = [&](const char* tag, const ShotMetrics& m) {
            const bool okHot = m.hotPct <= 5.0;   // 261108 下限取消：过曝低无罪（线亮由线数反映），上限防粘连
            // 261114 游程宽度升级为硬门（对齐标定 maxRunWidth=30px 口径）
            const bool okWide = m.maxRunWidthDs <= 8;
            const int lineTol = expect >= 15 ? 5 : 2;   // 261108 自适应：25线±5、7线±2（fine 基线 7~9 实证）
            const bool okLines = std::abs(m.lineEst - expect) <= lineTol;
            const bool okBg = true;   // 261114 背景改信息项——线数+游程是直接指标，背景只是间接风险提示
            QString s = QString("%1眼：过曝%2 %3 背景%4 线数%5 %6 游程%7px %8")
                .arg(tag, QString::number(m.hotPct, 'f', 2) + "%", okHot ? "✓" : "✗")
                .arg(QString::number(m.bg, 'f', 0), okBg ? "✓" : "✗")
                .arg(QString::number(m.lineEst), okLines ? "✓" : "✗")
                .arg(QString::number(m.maxRunWidthDs), okWide ? "✓" : "✗");
            return s;
        };
        const char* now = QDateTime::currentDateTime().toString("HH:mm:ss").toUtf8().constData();
        QString report = QString("<span style='color:#888;'>[%1]</span><br>").arg(now)
                       + oneEye("L", ml) + "<br>" + oneEye("R", mr);
        const bool allL = (ml.hotPct <= 5.0
                           && ml.maxRunWidthDs <= 8
                           && std::abs(ml.lineEst - expect) <= (expect >= 15 ? 5 : 2));
        const bool allR = (mr.hotPct <= 5.0
                           && mr.maxRunWidthDs <= 8
                           && std::abs(mr.lineEst - expect) <= (expect >= 15 ? 5 : 2));
        if (allL && allR) {
            report = "<p style='color:#1a7f37; font-weight:bold;'>✅ 可以采集（双眼全过，按此档采满 pose）</p>" + report;
        } else {
            QString tip;
            for (const ShotMetrics* m : {&ml, &mr}) {
                const char* tag = (m == &ml) ? "L" : "R";
                if (m->lineEst < expect - (expect >= 15 ? 5 : 2)) tip += QString("%1眼线数不足→升亮度/激光档半档；").arg(tag);
                if (m->lineEst > expect + (expect >= 15 ? 5 : 2)) tip += QString("%1眼多线→降补光/激光档；").arg(tag);
                if (m->hotPct > 5.0) tip += QString("%1眼过曝→降激光档；").arg(tag);
                if (m->bg > 25.0) tip += QString("%1眼背景%2高→关补光/挡环境光；").arg(tag).arg(int(m->bg));
            }
            report = QString("<p style='color:#b3261e; font-weight:bold;'>✗ 继续调整</p>")
                   + report + "<br><span style='color:#b3261e;'>" + tip + "</span>";
        }
        emit updateJudge(report);
        // 261114 二修 激光线叠加预览：缓存叠加图，每帧都发（持续显示直到下次
        // 分析刷新）——一修只发一帧被下一帧常规预览冲掉＝闪烁看不清
        {
            static QImage s_ovlL, s_ovlR;
            static bool s_hasOvl = false;
            if (!mlRaw.overlay.empty()) {
                s_ovlL = matBgrToQImage(mlRaw.overlay);
                s_ovlR = s_ovlL;   // 左右同图（当前只分析左图叠加）
                s_hasOvl = true;
            }
            // 模式切走时清缓存
            if (laserMode_.load(std::memory_order_relaxed) == 0
                || !judgeEnabled_.load(std::memory_order_relaxed)) {
                s_hasOvl = false;
            }
            if (s_hasOvl && !s_ovlL.isNull()) {
                emit updateImages(s_ovlL, s_ovlR);
                judgeOverlaid = true;
            }
        }
        // 261108 绿灯→自动停止＋完成提示（一次性任务语义）
        if (autoTuneOn_.load(std::memory_order_relaxed) && (allL && allR)) {
            emit autoTuneRequest(0, 0, 1.0, 0);
        }
        // 261108 V2 分层解耦（曝光对数步进/背光/激光/对比度兜底）：
        //   背景>25：背光有→降背光；否则降曝光；曝光到底→升对比度（压暗背景）
        //   过曝>5.5%：降激光；激光到底→降对比度（压高亮）
        //   过曝<2.5%且线暗：升激光；激光满→升曝光（背景<20）；到顶→升对比度
        //   背景<5：升曝光
        if (autoTuneOn_.load(std::memory_order_relaxed) && !(allL && allR)) {
            const long long nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (nowNs - lastTuneNs_.load() > 800000000LL) {
                const double bgMax = std::max(ml.bg, mr.bg);
                const double bgMin = std::min(ml.bg, mr.bg);
                const double hotMax = std::max(ml.hotPct, mr.hotPct);
                const double hotMin = std::min(ml.hotPct, mr.hotPct);
                const bool linesLow = ml.lineEst < expect - 3 && mr.lineEst < expect - 3;
                const bool bgAt0 = !bgLightSlider_ || bgLightSlider_->value() == 0;
                const bool expAtLo = !exposureSpin_ || exposureSpin_->value() <= 100.0;
                const bool expAtHi = !exposureSpin_ || exposureSpin_->value() >= 1000000.0;
                const bool lsrAt0 = !laserSlider_ || laserSlider_->value() == 0;
                const bool lsrAtHi = !laserSlider_ || laserSlider_->value() == 100;
                int bgD = 0, lsrD = 0, ctrD = 0;
                double expF = 1.0;
                if (bgMax > 25.0) {
                    if (!bgAt0) bgD = -5;
                    else if (!expAtLo) expF = 0.8;
                    else ctrD = +5;          // 对比度兜底：拉伸压暗背景
                } else if (bgMin < 5.0) {
                    if (!expAtHi) expF = 1.25;
                    else if (!lsrAtHi) lsrD = +5;
                } else if (hotMax > 5.5) {
                    if (!lsrAt0) lsrD = -5;
                    else if (!expAtLo) expF = 0.8;
                    else ctrD = -5;          // 压高亮
                } else if (hotMin < 2.5 && linesLow) {
                    if (!lsrAtHi) lsrD = +5;
                    else if (!expAtHi && bgMax < 20.0) expF = 1.25;
                    else ctrD = +5;          // 提线对比
                }
                if (bgD != 0 || lsrD != 0 || expF != 1.0 || ctrD != 0) {
                    lastTuneNs_.store(nowNs);
                    emit autoTuneRequest(bgD, lsrD, expF, ctrD);
                }
            }
        }
    }
    if (doPreview && !judgeOverlaid) {   // 261114 棋盘角点叠加帧已直发则跳过（后到原图会冲掉叠加）
        cv::Mat reducedLeft, reducedRight;
        cv::resize(pairL, reducedLeft, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
        cv::resize(pairR, reducedRight, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
        emit updateImages(QPixmap::fromImage(convertMattoQImage(reducedLeft)).toImage(),
                          QPixmap::fromImage(convertMattoQImage(reducedRight)).toImage());
    }
    if (!pairL.empty()) enqueueRecordPair(std::move(pairL), std::move(pairR));
}

void AcquisitionTab::setRightImage(cv::Mat& right, uint64_t id) {
    cv::Mat pairL, pairR;
    bool doPreview = false;
    {
        std::lock_guard<std::mutex> lk(pairMtx_);
        *m_pRight = right;
        m_iRightId = id;
        rightFresh_ = true;
        if (!leftFresh_) return;
        const int64_t nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        doPreview = nowNs - lastPreviewNs_ >= kPreviewMinIntervalNs;
        if (doPreview || recording_.load()) {
            pairL = m_pLeft->clone();
            pairR = m_pRight->clone();
        }
        if (doPreview) lastPreviewNs_ = nowNs;
        leftFresh_ = rightFresh_ = false;
    }
    ++pairCount_;
    if (doPreview) {
        cv::Mat reducedLeft, reducedRight;
        cv::resize(pairL, reducedLeft, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
        cv::resize(pairR, reducedRight, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
        emit updateImages(QPixmap::fromImage(convertMattoQImage(reducedLeft)).toImage(),
                          QPixmap::fromImage(convertMattoQImage(reducedRight)).toImage());
    }
    if (!pairL.empty()) enqueueRecordPair(std::move(pairL), std::move(pairR));
}

// 照搬 LeadScanK2 onUpdateImages（主线程）：setPixmap
void AcquisitionTab::onUpdateImages(QImage left, QImage right) {
    if (!previewing_) return;  // 停在当前帧：不刷新界面，frozen 保持
    leftPreview_->setPixmap(QPixmap::fromImage(left));
    rightPreview_->setPixmap(QPixmap::fromImage(right));
    // frozen 跟随后台最新帧（=当前显示帧），供「保存图像」存的就是看到的那一帧
    if (!lastLeft_.empty())  frozenLeft_  = lastLeft_.clone();
    if (!lastRight_.empty()) frozenRight_ = lastRight_.clone();
    if (!frozenLeft_.empty()) {
        saveCameraBtn_->setEnabled(true);   // 有显示帧才能保存
        saveLaserBtn_->setEnabled(true);
    }
}

// 261108 采集参数实时判定刷新（回调线程分析 → 此处 UI 线程渲染）
void AcquisitionTab::onUpdateJudge(QString report) {
    if (judgeLbl_) judgeLbl_->setText(report);
}

// 261108 自动调参执行（UI 线程）：滑杆步进 + N10 重发（扫描中参数热更）
void AcquisitionTab::onAutoTune(int bgDelta, int laserDelta, double expFactor, int ctrDelta) {
    if (bgDelta == 0 && laserDelta == 0 && expFactor == 1.0 && ctrDelta == 0) {   // 绿灯停止
        autoTuneOn_.store(false);
        if (autoTuneBtn_) {
            autoTuneBtn_->setChecked(false);
            autoTuneBtn_->setText(QStringLiteral("自动调节"));
        }
        emit statusMessage(QStringLiteral(
            "✅ 自动调节完成——判定全绿，可以开始采集"));
        if (judgeLbl_)
            judgeLbl_->setText(QStringLiteral(
                "<p style='color:#1a7f37; font-weight:bold;'>✅ 调节完成，可以采集</p>")
                + judgeLbl_->text());
        return;
    }
    // 边界检测：方向要求超出滑杆范围→停（提示手动介入）
    const bool expStuck = exposureSpin_
        && ((expFactor < 1.0 && exposureSpin_->value() <= 100.0)
         || (expFactor > 1.0 && exposureSpin_->value() >= 1000000.0));
    const bool bgStuck = bgLightSlider_
        && ((bgDelta < 0 && bgLightSlider_->value() == 0)
         || (bgDelta > 0 && bgLightSlider_->value() == 100));
    const bool lsrStuck = laserSlider_
        && ((laserDelta < 0 && laserSlider_->value() == 0)
         || (laserDelta > 0 && laserSlider_->value() == 100));
    const bool ctrStuck = contrastLSlider_
        && ((ctrDelta > 0 && contrastLSlider_->value() >= contrastLSlider_->maximum())
         || (ctrDelta < 0 && contrastLSlider_->value() <= contrastLSlider_->minimum()));
    if (bgStuck || lsrStuck || expStuck || ctrStuck) {
        autoTuneOn_.store(false);
        if (autoTuneBtn_) {
            autoTuneBtn_->setChecked(false);
            autoTuneBtn_->setText(QStringLiteral("自动调节"));
        }
        emit statusMessage(QStringLiteral(
            "⚠ 自动调节到达档位边界仍未达标——请手动检查（环境光/增益/曝光）"));
        return;
    }
    if (bgDelta != 0 && bgLightSlider_) {
        const int v = qBound(0, bgLightSlider_->value() + bgDelta, 100);
        bgLightSlider_->setValue(v);   // 触发标签刷新
    }
    if (laserDelta != 0 && laserSlider_) {
        const int v = qBound(0, laserSlider_->value() + laserDelta, 100);
        laserSlider_->setValue(v);
    }
    // 重发 N10（参数热更——照抄 onStartScanner 的组包；仅扫描中生效）
    if (scanner_ && scanner_->isOpen() && previewing_) {
        ScannerParams p;
        p.freq = freqSlider_->value();
        p.background = bgLightSlider_->value();
        p.laser = laserSlider_->value();
        const int mode = scannerModeCbx_->currentData().toInt();
        switch (mode) {
            case 1: p.tubeT = 1; break;
            case 2: p.tubeV = 1; break;
            case 3: p.tubeD = 1; break;
            case 4: p.tubeC = 1; break;
            case 5: p.tubeE = 1; break;
            default: p.laser = 0; break;
        }
        scanner_->start(p);
        if (judgeLbl_)
            judgeLbl_->setText(judgeLbl_->text()
                + QStringLiteral("<br><span style='color:#1a73e8;'>⚙ 自动调整中…</span>"));
    }
}

// ============================================================================
// 开机自动流程：开串口 → 开设备（注册帧回调）。
// 扫描仪启动 + 预览 留给用户点「▶ 启动扫描仪」（onStartScanner 内自动进预览）。
// 串口/相机失败时弹窗提示，不静默继续。
// ============================================================================
void AcquisitionTab::autoStart() {
    // 1. 打开扫描仪串口（取下拉框第一个可用 COM 口）
    if (comPortCbx_->count() == 0) {
        QMessageBox::warning(this, QStringLiteral("串口连接失败"),
            QStringLiteral("未发现任何 COM 口，请接好扫描仪串口后重启。"));
        emit statusMessage(QStringLiteral("未发现 COM 口，自动启动中止"));
        return;
    }
    emit statusMessage(QStringLiteral("自动打开串口: %1").arg(comPortCbx_->currentText()));
    onOpenCloseScanner();  // 启动时串口未开 → 执行打开

    if (!scanner_->isOpen()) {
        QMessageBox::warning(this, QStringLiteral("串口连接失败"),
            QStringLiteral("扫描仪串口 %1 打开失败，请检查连接/驱动/是否被占用。")
                .arg(comPortCbx_->currentText()));
        emit statusMessage(QStringLiteral("串口打开失败，自动启动中止"));
        return;
    }

    // 2. 打开设备（galaxy 左0 右1，内部注册帧回调）
    onOpenClose();

    // 3. 扫描仪启动 + 预览 留给用户点「▶ 启动扫描仪」
    if (rig_->isOpen()) {
        emit statusMessage(QStringLiteral("设备已就绪，点「▶ 启动扫描仪」开始实时预览"));
    } else {
        QMessageBox::warning(this, QStringLiteral("相机打开失败"),
            QStringLiteral("扫描仪相机打开失败，请检查相机 USB 连接 / 是否被其他软件占用。"));
    }
}

}  // namespace fc::gui
