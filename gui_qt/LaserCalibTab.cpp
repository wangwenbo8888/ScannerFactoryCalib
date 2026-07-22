#include "LaserCalibTab.h"
#include "CalibWorker.h"
#include "ResultView.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
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

#include <spdlog/spdlog.h>

namespace fc::gui {

LaserCalibTab::LaserCalibTab(QWidget* parent) : QWidget(parent) {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(8, 8, 8, 8);
    root->setSpacing(6);

    auto* inRow = new QHBoxLayout;
    inRow->addWidget(new QLabel(QStringLiteral("输入目录:")));
    inputEdit_ = new QLineEdit(QStringLiteral("data_in/laser"));
    inRow->addWidget(inputEdit_, 1);
    inBtn_ = new QPushButton(QStringLiteral("浏览..."));
    inRow->addWidget(inBtn_);
    root->addLayout(inRow);

    auto* outRow = new QHBoxLayout;
    outRow->addWidget(new QLabel(QStringLiteral("输出 JSON:")));
    outputEdit_ = new QLineEdit(QStringLiteral("data_out/laser_calib.json"));
    outRow->addWidget(outputEdit_, 1);
    outBtn_ = new QPushButton(QStringLiteral("浏览..."));
    outRow->addWidget(outBtn_);
    root->addLayout(outRow);

    auto* ctrlRow = new QHBoxLayout;
    runBtn_ = new QPushButton(QStringLiteral("▶ 运行激光标定"));
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

    // 持久化
    QSettings s;
    s.beginGroup("laser_calib_tab");
    inputEdit_->setText(s.value("input_dir", inputEdit_->text()).toString());
    outputEdit_->setText(s.value("output_path", outputEdit_->text()).toString());
    s.endGroup();

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
    QSettings s;
    s.beginGroup("laser_calib_tab");
    s.setValue("input_dir", inputEdit_->text());
    s.setValue("output_path", outputEdit_->text());
    s.endGroup();
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

void LaserCalibTab::onCopyHandoff() {
    QString src = QStringLiteral("data_out/camera_calib.json");
    if (!QFileInfo::exists(src)) {
        src = QFileDialog::getOpenFileName(this,
            QStringLiteral("选择源 camera_calib.json"), QString(),
            QStringLiteral("JSON (*.json)"));
        if (src.isEmpty()) return;
    }
    QString dst = QDir(inputEdit_->text()).filePath(QStringLiteral("camera_calib.json"));
    if (!QFile::copy(src, dst)) {
        appendLog(QStringLiteral("拷贝失败 %1 → %2").arg(src, dst), "#c0392b");
        return;
    }
    appendLog(QStringLiteral("已拷贝: %1 → %2").arg(src, dst), "#27ae60");
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
}

void LaserCalibTab::onRunAll() {
    QString inDir = inputEdit_->text().trimmed();
    if (inDir.isEmpty()) { appendLog(QStringLiteral("输入目录为空"), "#c0392b"); return; }
    if (!QFileInfo::exists(inDir)) { appendLog(QStringLiteral("输入目录不存在"), "#c0392b"); return; }

    QString outPath = outputEdit_->text().trimmed();
    if (outPath.isEmpty()) outPath = QStringLiteral("data_out/laser_calib.json");
    QDir().mkpath(QFileInfo(outPath).absolutePath());

    setRunningUI(true);
    progress_->setValue(5);
    log_->clear();
    worker_->startLaser(inDir, outPath);
}

void LaserCalibTab::onWorkerDone(bool ok, QString summary) {
    appendLog(summary, ok ? "#27ae60" : "#c0392b");
    setRunningUI(false);
    progress_->setValue(ok ? 100 : 0);
    emit statusMessage(QStringLiteral("激光标定: %1").arg(
        ok ? QStringLiteral("成功") : QStringLiteral("失败")));
    if (ok) {
        QString outPath = outputEdit_->text().trimmed();
        if (QFileInfo::exists(outPath)) resultView_->loadFile(outPath);
    }
}

}  // namespace fc::gui
