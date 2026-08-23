#include "LaserCalibTab.h"
#include "CalibWorker.h"
#include "ResultView.h"
#include "gui_common.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QProgressBar>
#include <QPlainTextEdit>
#include <QLabel>
#include <QFileDialog>
#include <QFileInfo>
#include <QDir>
#include <QSettings>
#include <QMessageBox>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QCheckBox>

#include <spdlog/spdlog.h>

namespace fc::gui {

LaserCalibTab::LaserCalibTab(QWidget* parent) : QWidget(parent) {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(8, 8, 8, 8);
    root->setSpacing(6);

    // —— 激光线类型选择（决定输入子目录与输出文件）——
    auto* typeRow = new QHBoxLayout;
    typeRow->addWidget(new QLabel(QStringLiteral("激光线类型:")));
    typeCbx_ = new QComboBox;
    for (int i = 0; i < kLaserTypeCount; ++i)
        typeCbx_->addItem(laserTypeName(i), i);
    typeRow->addWidget(typeCbx_);
    typeRow->addStretch(1);
    root->addLayout(typeRow);

    auto* inRow = new QHBoxLayout;
    inRow->addWidget(new QLabel(QStringLiteral("输入目录:")));
    inputEdit_ = new QLineEdit(laserTypeDir(0));
    inRow->addWidget(inputEdit_, 1);
    inBtn_ = new QPushButton(QStringLiteral("浏览..."));
    inRow->addWidget(inBtn_);
    root->addLayout(inRow);

    // —— 旧数据选择：不勾＝用本次采集（data_in），勾上＝手动选旧数据文件夹 ——
    auto* oldRow = new QHBoxLayout;
    oldDataChk_ = new QCheckBox(QStringLiteral("使用旧数据标定（手动选择文件夹）"));
    oldDataBtn_ = new QPushButton(QStringLiteral("选择旧数据文件夹…"));
    oldDataBtn_->setEnabled(false);   // 未勾选时不可点
    oldRow->addWidget(oldDataChk_);
    oldRow->addWidget(oldDataBtn_);
    oldRow->addStretch(1);
    root->addLayout(oldRow);

    auto* outRow = new QHBoxLayout;
    outRow->addWidget(new QLabel(QStringLiteral("输出 JSON:")));
    outputEdit_ = new QLineEdit(laserTypeOutPath(0));
    outRow->addWidget(outputEdit_, 1);
    outBtn_ = new QPushButton(QStringLiteral("浏览..."));
    outRow->addWidget(outBtn_);
    root->addLayout(outRow);

    auto* ctrlRow = new QHBoxLayout;
    runBtn_ = new QPushButton(QStringLiteral("▶ 运行%1激光线标定").arg(laserTypeName(0)));
    QFont f = runBtn_->font(); f.setBold(true); runBtn_->setFont(f);
    copyBtn_ = new QPushButton(
        QStringLiteral("从相机标定结果导入 camera_calib.json"));
    ctrlRow->addWidget(runBtn_);
    ctrlRow->addWidget(copyBtn_);
    ctrlRow->addStretch(1);
    root->addLayout(ctrlRow);

    progress_ = new QProgressBar;
    progress_->setRange(0, 100);
    progress_->setValue(0);
    root->addWidget(progress_);

    log_ = new QPlainTextEdit;
    log_->setReadOnly(true);
    log_->setMaximumBlockCount(10000);
    log_->setStyleSheet(
        "QPlainTextEdit { font-family: Consolas, 'Courier New', monospace;"
        " font-size: 10pt; background: #1e1e1e; color: #d4d4d4; }");
    root->addWidget(log_, 2);

    resultView_ = new ResultView;
    root->addWidget(resultView_, 1);

    // 持久化（类型优先，路径默认随类型；用户手动 Browse 过的路径也会存）
    QSettings s;
    s.beginGroup("laser_calib_tab");
    int savedType = s.value("type", 0).toInt();
    if (savedType >= 0 && savedType < kLaserTypeCount) typeCbx_->setCurrentIndex(savedType);
    inputEdit_->setText(s.value("input_dir", laserTypeDir(typeCbx_->currentIndex())).toString());
    outputEdit_->setText(s.value("output_path", laserTypeOutPath(typeCbx_->currentIndex())).toString());
    s.endGroup();

    connect(typeCbx_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &LaserCalibTab::onTypeChanged);
    connect(oldDataChk_, &QCheckBox::toggled, this, &LaserCalibTab::onToggleOldData);
    connect(oldDataBtn_, &QPushButton::clicked, this, &LaserCalibTab::onPickOldDataDir);
    connect(inBtn_, &QPushButton::clicked, this, &LaserCalibTab::onBrowseInput);
    connect(outBtn_, &QPushButton::clicked, this, &LaserCalibTab::onBrowseOutput);
    connect(runBtn_, &QPushButton::clicked, this, &LaserCalibTab::onRunAll);
    connect(copyBtn_, &QPushButton::clicked, this, &LaserCalibTab::onCopyHandoff);

    worker_ = new CalibWorker(this);
    connect(worker_, &CalibWorker::logLine, log_, &QPlainTextEdit::appendPlainText);
    connect(worker_, &CalibWorker::progress, this,
            [this](int s, int t, QString msg) {
                if (t > 0) progress_->setValue(static_cast<int>(s * 100.0 / t));
                if (!msg.isEmpty()) appendLog(msg);
            });
    connect(worker_, &CalibWorker::done, this, &LaserCalibTab::onWorkerDone);
}

LaserCalibTab::~LaserCalibTab() {
    // 先等待标定线程完成，避免析构期间标定线程的 spdlog/回调访问已析构对象
    if (worker_) worker_->wait();
    QSettings s;
    s.beginGroup("laser_calib_tab");
    s.setValue("type", typeCbx_->currentIndex());
    s.setValue("input_dir", inputEdit_->text());
    s.setValue("output_path", outputEdit_->text());
    s.endGroup();
}

void LaserCalibTab::onTypeChanged() {
    // 切类型 → 路径/按钮跟随类型；旧数据模式下从 laser 根自动重定位新类型子目录
    const int ti = typeCbx_->currentIndex();
    if (ti < 0 || ti >= kLaserTypeCount) return;
    outputEdit_->setText(laserTypeOutPath(ti));
    if (oldDataChk_ && oldDataChk_->isChecked()) {
        if (!oldDataLaserRoot_.isEmpty()) {
            resolveOldData(oldDataLaserRoot_);   // 有根目录：自动找该类型数据
        } else {
            // 用户选的是精确类型目录：无根可推，仅提示需要重选
            appendLog(QStringLiteral("请重新选择旧数据文件夹（当前为单一类型目录，"
                                     "建议选到 laser 一级或备份根目录）"), "#f39c12");
        }
    } else {
        inputEdit_->setText(laserTypeDir(ti));
    }
    runBtn_->setText(QStringLiteral("▶ 运行%1激光线标定").arg(laserTypeName(ti)));
}

void LaserCalibTab::onToggleOldData(bool on) {
    oldDataBtn_->setEnabled(on);
    if (on) {
        onPickOldDataDir();   // 勾上直接弹文件夹选择
    } else {
        // 取消勾选 → 恢复本次采集的类型默认路径
        inputEdit_->setText(laserTypeDir(typeCbx_->currentIndex()));
    }
}

void LaserCalibTab::onPickOldDataDir() {
    // 用户可选到任意一级：备份根（含 laser/）、laser 一级、或精确类型目录，
    // 软件自动定位到当前类型的数据目录
    QString d = QFileDialog::getExistingDirectory(this,
        QStringLiteral("选择旧数据文件夹（可选中备份根或 laser 目录，"
                       "软件自动定位%1激光线数据）").arg(
            laserTypeName(typeCbx_->currentIndex())),
        QStringLiteral("data_bak"));
    if (d.isEmpty()) return;
    resolveOldData(d);
}

// 把用户所选目录解析成「laser 一级目录」并按当前类型定位数据；
// 返回 false = 各级都找不到该类型 pose_*（弹提示）。成功后更新输入框。
bool LaserCalibTab::resolveOldData(const QString& picked) {
    const int ti = typeCbx_->currentIndex();
    const QString typeKey = QString::fromLatin1(kLaserTypes[ti].key);
    auto hasPose = [](const QString& p) {
        return !QDir(p).entryList({QStringLiteral("pose_*")},
                                  QDir::Dirs | QDir::NoDotAndDotDot).isEmpty();
    };

    // 依次尝试三级：picked/<type> → picked/laser/<type> → picked 本身(已是类型目录)
    QString root;
    if (hasPose(QDir(picked).filePath(typeKey))) {
        root = picked;                       // 选的就是 laser 一级（或备份根含 <type>）
    } else if (hasPose(QDir(picked).filePath(QStringLiteral("laser/") + typeKey))) {
        root = QDir(picked).filePath(QStringLiteral("laser"));   // 选的是备份根
    } else if (hasPose(picked)) {
        // 选的就是精确类型目录：直接用，并尝试推断 laser 根（父目录）供切类型重定位
        inputEdit_->setText(picked);
        QFileInfo fi(picked);
        if (fi.fileName().compare(typeKey, Qt::CaseInsensitive) == 0
            && fi.dir().dirName().compare(QStringLiteral("laser"), Qt::CaseInsensitive) == 0) {
            oldDataLaserRoot_ = fi.absolutePath();   // .../laser
        } else {
            oldDataLaserRoot_.clear();
        }
        appendLog(QStringLiteral("使用旧数据[%1]: %2").arg(laserTypeName(ti),
                      QDir::toNativeSeparators(picked)), "#f39c12");
        return true;
    } else {
        QMessageBox::warning(this, QStringLiteral("未找到该类型数据"),
            QStringLiteral("在以下位置均未找到 %1 激光线的 pose_* 数据:\n"
                           "%2\n%3\n\n请确认旧数据包含该类型，或先切换类型。")
                .arg(laserTypeName(ti),
                     QDir::toNativeSeparators(QDir(picked).filePath(typeKey)),
                     QDir::toNativeSeparators(
                         QDir(picked).filePath(QStringLiteral("laser/") + typeKey))));
        return false;
    }

    oldDataLaserRoot_ = root;                // 记住 laser 根，切类型自动重定位
    QString typed = QDir(root).filePath(typeKey);
    inputEdit_->setText(typed);
    appendLog(QStringLiteral("使用旧数据[%1]: %2").arg(laserTypeName(ti),
                  QDir::toNativeSeparators(typed)), "#f39c12");
    return true;
}

void LaserCalibTab::onBrowseInput() {
    QString d = QFileDialog::getExistingDirectory(this,
        QStringLiteral("选择输入目录"), inputEdit_->text());
    if (!d.isEmpty()) inputEdit_->setText(d);
}

void LaserCalibTab::onBrowseOutput() {
    QString p = QFileDialog::getSaveFileName(this,
        QStringLiteral("选择输出 JSON"), outputEdit_->text(),
        QStringLiteral("JSON (*.json)"));
    if (!p.isEmpty()) outputEdit_->setText(p);
}

bool LaserCalibTab::ensureHandoffFiles(const QString& inDir, bool overwrite) {
    QString dst = QDir(inDir).filePath(QStringLiteral("camera_calib.json"));
    QDir().mkpath(inDir);

    // camera_calib.json（缺失或强制刷新时拷贝）
    if (overwrite || !QFileInfo::exists(dst)) {
        QString src = QStringLiteral("data_out/camera_calib.json");
        if (!QFileInfo::exists(src)) {
            src = QFileDialog::getOpenFileName(this,
                QStringLiteral("选择源 camera_calib.json"), QString(),
                QStringLiteral("JSON (*.json)"));
            if (src.isEmpty()) return false;
        }
        if (QFileInfo::exists(dst) && !QFile::remove(dst)) {
            appendLog(QStringLiteral("无法覆盖旧文件: %1").arg(dst), "#c0392b");
            return false;
        }
        if (!QFile::copy(src, dst)) {
            appendLog(QStringLiteral("拷贝失败 %1 → %2").arg(src, dst), "#c0392b");
            return false;
        }
        appendLog(QStringLiteral("已拷贝: %1 → %2").arg(src, dst), "#27ae60");
    }

    // config.json 缺失时自动生成一份与 handoff 一致的
    // （referenceTemp/cte/tempRange 必须与 handoff 完全一致，否则 validateHandoffConsistency 拒绝；
    //   rectify.flags 只接受 0 或 CALIB_ZERO_DISPARITY=1024）
    QString cfgPath = QDir(inDir).filePath(QStringLiteral("config.json"));
    if (overwrite && QFileInfo::exists(cfgPath)) QFile::remove(cfgPath);
    if (!QFileInfo::exists(cfgPath)) {
        QFile hf(dst);
        double refTemp = 22.5, cte = 23.6e-6, tMin = -10.0, tMax = 10.0, tStep = 0.2;
        if (hf.open(QIODevice::ReadOnly)) {
            QJsonObject o = QJsonDocument::fromJson(hf.readAll()).object();
            if (o.contains("referenceTemp")) refTemp = o["referenceTemp"].toDouble();
            if (o.contains("cte"))           cte    = o["cte"].toDouble();
            if (o.contains("tempRangeMin"))  tMin   = o["tempRangeMin"].toDouble();
            if (o.contains("tempRangeMax"))  tMax   = o["tempRangeMax"].toDouble();
            if (o.contains("tempStep"))      tStep  = o["tempStep"].toDouble();
            hf.close();
        }
        QString cfg = QStringLiteral(
            "{\n"
            "  \"deviceId\": 0,\n"
            "  \"lineIds\": [],\n"
            "  \"mask\": {\n"
            "    \"threshold\": 60,\n"
            "    \"erodeSize\": 1,\n"
            "    \"laserDilateSize\": 3,\n"
            "    \"minArea\": 100,\n"
            "    \"maxArea\": 100000\n"
            "  },\n"
            "  \"temperature\": {\n"
            "    \"cte\": %1,\n"
            "    \"referenceTemp\": %2,\n"
            "    \"tempRangeMin\": %3,\n"
            "    \"tempRangeMax\": %4,\n"
            "    \"tempStep\": %5\n"
            "  },\n"
            "  \"rectify\": { \"alpha\": 0.0, \"flags\": 1024 }\n"
            "}\n")
            .arg(cte).arg(refTemp).arg(tMin).arg(tMax).arg(tStep);
        QFile cf(cfgPath);
        if (cf.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            cf.write(cfg.toUtf8());
            appendLog(QStringLiteral("已生成 config.json（温度参数与 handoff 一致）"), "#27ae60");
        }
    }
    return true;
}

void LaserCalibTab::onCopyHandoff() {
    ensureHandoffFiles(inputEdit_->text().trimmed(), /*overwrite=*/true);
}

void LaserCalibTab::appendLog(const QString& msg, const QString& color) {
    QString html = QStringLiteral("<span style='color:%1;'>&gt; %2</span>")
                       .arg(color, msg.toHtmlEscaped());
    log_->appendHtml(html);
}

void LaserCalibTab::setRunningUI(bool running) {
    runBtn_->setEnabled(!running);
    inputEdit_->setEnabled(!running);
    outputEdit_->setEnabled(!running);
    copyBtn_->setEnabled(!running);
    typeCbx_->setEnabled(!running);
    oldDataChk_->setEnabled(!running);
    oldDataBtn_->setEnabled(!running && oldDataChk_->isChecked());
}

void LaserCalibTab::onRunAll() {
    // —— 数据源选择：默认本次采集（data_in），旧数据需用户主动选 ——
    // 数据源由勾选框决定：未勾＝本次采集（data_in 类型默认路径），勾＝旧数据（自动解析定位）
    QString inDir;
    if (oldDataChk_->isChecked()) {
        const int ti = typeCbx_->currentIndex();
        if (!resolveOldData(inputEdit_->text().trimmed())) {
            appendLog(QStringLiteral("旧数据解析失败：找不到%1激光线数据目录").arg(
                          laserTypeName(ti)), "#c0392b");
            return;
        }
        inDir = inputEdit_->text().trimmed();   // resolve 后的精确类型目录
    } else {
        inDir = laserTypeDir(typeCbx_->currentIndex());   // 强制走本次采集路径，防手动改错
    }

    if (inDir.isEmpty()) { appendLog(QStringLiteral("输入目录为空"), "#c0392b"); return; }
    if (!QFileInfo::exists(inDir)) { appendLog(QStringLiteral("输入目录不存在"), "#c0392b"); return; }

    // 该类型目录下必须有 pose_* 图像（支持用户四类全采完后再逐类标定的流程）
    QDir inD(inDir);
    const auto poseDirs = inD.entryList({QStringLiteral("pose_*")},
                                        QDir::Dirs | QDir::NoDotAndDotDot);
    if (poseDirs.isEmpty()) {
        appendLog(QStringLiteral("输入目录下没有 pose_* 图像目录（%1激光线尚未采集），"
                                 "请先在「① 图像采集」中采集该类型").arg(
                      laserTypeName(typeCbx_->currentIndex())), "#c0392b");
        return;
    }

    // camera_calib.json / config.json 缺失时自动补齐（无需每类手动导入）
    if (!ensureHandoffFiles(inDir, /*overwrite=*/false)) {
        appendLog(QStringLiteral("自动准备 camera_calib.json/config.json 失败"), "#c0392b");
        return;
    }

    QString outPath = outputEdit_->text().trimmed();
    if (outPath.isEmpty()) outPath = laserTypeOutPath(typeCbx_->currentIndex());
    QDir().mkpath(QFileInfo(outPath).absolutePath());

    setRunningUI(true);
    progress_->setValue(5);
    log_->clear();
    emit calibrationStarted();   // 通知 MainWindow：标定开始，必要时自动停止扫描仪
    worker_->startLaser(inDir, outPath);
}

void LaserCalibTab::onWorkerDone(bool ok, QString summary) {
    appendLog(summary, ok ? "#27ae60" : "#c0392b");
    setRunningUI(false);
    progress_->setValue(ok ? 100 : 0);
    emit statusMessage(QStringLiteral("%1激光线标定: %2").arg(
        laserTypeName(typeCbx_->currentIndex()),
        ok ? QStringLiteral("成功") : QStringLiteral("失败")));
    if (ok) {
        QString outPath = outputEdit_->text().trimmed();
        if (QFileInfo::exists(outPath)) resultView_->loadFile(outPath);
    }
}

}  // namespace fc::gui
