#include "AcquisitionTab.h"
#include "PreviewWidget.h"
#include "gui_common.h"
#include "camera/ICameraSource.h"
#include "camera/StereoCameraRig.h"
#include "camera/GalaxyCameraSource.h"
#include "scanner/ScannerControl.h"

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
#include <fstream>
#include <QCheckBox>
#include <QSerialPortInfo>
#include <QSlider>
#include <QSpinBox>
#include <QTimer>
#include <QVBoxLayout>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <QTransform>
#include <atomic>
#include <memory>

namespace fc::gui {

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
    saveCameraBtn_ = new QPushButton(QStringLiteral("💾 保存到 data_in/camera"));
    saveLaserBtn_  = new QPushButton(QStringLiteral("💾 保存到 data_in/laser/pose_NN"));
    saveCameraBtn_->setEnabled(false);
    saveLaserBtn_->setEnabled(false);
    exposureSpin_ = new QDoubleSpinBox;
    exposureSpin_->setRange(10.0, 1e7); exposureSpin_->setValue(10000.0);
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
    scannerModeCbx_->setCurrentIndex(1);  // 默认激光模式：实时预览看激光线变换
    scanRow1->addWidget(scannerModeCbx_);
    scannerStartBtn_ = new QPushButton(QStringLiteral("▶ 启动扫描仪"));
    scannerStartBtn_->setEnabled(false);
    scannerStopBtn_  = new QPushButton(QStringLiteral("■ 停止扫描仪"));
    scannerStopBtn_->setEnabled(false);
    scanRow1->addWidget(scannerStartBtn_);
    scanRow1->addWidget(scannerStopBtn_);
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
    root->addLayout(scanRow2);

    // 打开/关闭设备 + 曝光/增益/旋转 行移至滑动条下方
    root->addLayout(ctrlRow);

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
    root->addLayout(previewRow, 4);

    // —— 保存（用户确认后才落盘）——
    auto* saveRow = new QHBoxLayout;
    saveRow->addWidget(saveCameraBtn_);
    saveRow->addWidget(saveLaserBtn_);
    saveRow->addStretch(1);
    snapCountLbl_ = new QLabel(QStringLiteral("相机: 0 张    激光: 0 pose"));
    saveRow->addWidget(snapCountLbl_);
    root->addLayout(saveRow);

    // —— 信号 ——
    connect(refreshBtn_,     &QPushButton::clicked,     this, &AcquisitionTab::onRefreshDevices);
    connect(openBtn_,        &QPushButton::clicked,     this, &AcquisitionTab::onOpenClose);
    connect(captureBtn_,     &QPushButton::clicked,     this, &AcquisitionTab::onFreezeFrame);
    connect(previewBtn_,     &QPushButton::toggled,     this, &AcquisitionTab::onTogglePreview);
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
    // 照搬 LeadScanK2：updateImages 配对双张，Qt::QueuedConnection
    connect(this, SIGNAL(updateImages(QImage,QImage)), this, SLOT(onUpdateImages(QImage,QImage)), Qt::QueuedConnection);
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
        if (mode >= 1 && mode <= kLaserTypeCount) {
            saveLaserBtn_->setEnabled(true);
            saveLaserBtn_->setText(QStringLiteral("💾 保存到 data_in/laser/%1/pose_NN")
                                       .arg(QString::fromLatin1(kLaserTypes[mode - 1].key)));
        } else {
            saveLaserBtn_->setEnabled(false);
            saveLaserBtn_->setText(QStringLiteral("💾 保存激光帧（请先选激光线类型）"));
        }
    });
    // 初始化按钮文案（不触发上面的信号）
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

    rig_ = std::make_unique<StereoCameraRig>();
    scanner_ = std::make_unique<ScannerControl>();
    m_pLeft = new cv::Mat();
    m_pRight = new cv::Mat();
    m_iLeftId = 0;
    m_iRightId = 0;
    m_iIndex = 0;
    // 用回调代替 Qt signal（避开 ScannerControl 的 MOC 依赖问题）
    scanner_->onStatus = [this](const QString& msg) { emit statusMessage(msg); };

    onRefreshDevices();
    onRefreshComPorts();
    initSnapCounters();
    updateSnapCount();

    // 开机自动：开串口 → 开设备（注册帧回调）。
    // 扫描仪启动留给用户点（onStartScanner 里发 N10 + 自动进预览）。
    // 延迟到事件循环启动后执行，避免在构造函数里阻塞 UI
    QTimer::singleShot(0, this, [this] { autoStart(); });
}

AcquisitionTab::~AcquisitionTab() {
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
    // 应用曝光/增益（旋转统一在 GUI 层按预览框勾选处理，与 deviceId 解耦）
    if (auto* L = rig_->left()) {
        if (L->capability().exposureUs) L->setExposureUs(exposureSpin_->value());
        if (L->capability().gainDb)     L->setGainDb(gainSpin_->value());
        leftInfoLbl_->setText(QString::fromStdString(L->displayName()) +
            QStringLiteral("  %1×%2").arg(L->width()).arg(L->height()));
    }
    if (auto* R = rig_->right()) {
        if (R->capability().exposureUs) R->setExposureUs(exposureSpin_->value());
        if (R->capability().gainDb)     R->setGainDb(gainSpin_->value());
        rightInfoLbl_->setText(QString::fromStdString(R->displayName()) +
            QStringLiteral("  %1×%2").arg(R->width()).arg(R->height()));
    }
    // 照搬 LeadScanK2：回调线程调 setLeftImage/setRightImage（内部配对后 emit updateImages）
    rig_->setLeftFrameCallback([this](const CameraFrame& f) {
        cv::Mat img = f.image;
        if (leftRotateChk_ && leftRotateChk_->isChecked() && !img.empty()) {
            cv::rotate(img, img, cv::ROTATE_180);
        }
        lastLeft_ = img;
        setLeftImage(img, static_cast<uint64_t>(f.frameIndex));
    });
    rig_->setRightFrameCallback([this](const CameraFrame& f) {
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
    sourceTypeCbx_->setEnabled(!opened);
    leftDevSpin_->setEnabled(!opened);
    rightDevSpin_->setEnabled(!opened);
    leftFolderEdit_->setEnabled(!opened);
    rightFolderEdit_->setEnabled(!opened);
    refreshBtn_->setEnabled(!opened);
    if (!opened) {
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
    // 模式 0 = 仅补光（相机标定）→ 激光关（laser=0）
    // 模式 1 = 激光+补光（激光标定）→ 用滑块值
    int mode = scannerModeCbx_->currentData().toInt();
    p.laser = (mode >= 1) ? laserSlider_->value() : 0;   // 四类激光模式协议暂同（下位机未区分）
    scanner_->start(p);
    scannerRunning_ = true;

    // 启动相机采集：注册 SDK 回调，Line2 硬件触发脉冲来即出帧 → 自动进入实时预览。
    // 回调在 SDK 线程 emit frameArrived，queued 投递到主线程 onFrameArrived 刷新。
    if (rig_ && rig_->isOpen()) {
        if (rig_->isAcquiring()) rig_->stopAcquisition();  // 防重复 StartGrab/Register
        if (rig_->startAcquisition()) {
            previewing_ = true;
            previewBtn_->setChecked(true);   // 默认连续预览
            emit statusMessage(QStringLiteral("实时预览已开启（连续预览，回调驱动刷新）"));
        } else {
            emit statusMessage(QStringLiteral("警告：相机启动采集失败，无法预览"));
        }
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
}  // namespace

// 照搬 LeadScanK2 setLeftImage（SDK 回调线程调用）：左右按 id 配对后 emit updateImages
void AcquisitionTab::setLeftImage(cv::Mat& left, uint64_t id) {
    *m_pLeft = left;
    m_iLeftId = id;
    if (m_iLeftId == m_iRightId) {
        cv::Mat reducedLeft, reducedRight;
        cv::resize(*m_pLeft, reducedLeft, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
        cv::resize(*m_pRight, reducedRight, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
        emit updateImages(QPixmap::fromImage(convertMattoQImage(reducedLeft)).toImage(),
                          QPixmap::fromImage(convertMattoQImage(reducedRight)).toImage());
    }
}

// 照搬 LeadScanK2 setRightImage
void AcquisitionTab::setRightImage(cv::Mat& right, uint64_t id) {
    *m_pRight = right;
    m_iRightId = id;
    if (m_iLeftId == m_iRightId) {
        cv::Mat reducedLeft, reducedRight;
        cv::resize(*m_pLeft, reducedLeft, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
        cv::resize(*m_pRight, reducedRight, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
        emit updateImages(QPixmap::fromImage(convertMattoQImage(reducedLeft)).toImage(),
                          QPixmap::fromImage(convertMattoQImage(reducedRight)).toImage());
    }
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
