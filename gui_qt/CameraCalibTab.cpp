#include "CameraCalibTab.h"
#include "PreviewWidget.h"
#include "CalibWorker.h"
#include "SpdlogBridge.h"

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
#include <QFutureWatcher>
#include <QMessageBox>
#include <QtConcurrent>
#include <QDateTime>
#include <QImage>

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <spdlog/spdlog.h>

namespace fc::gui {

namespace {
QString matToHtml(const cv::Mat& m) {
    if (m.empty()) return QStringLiteral("（空）");
    QString s = QStringLiteral("<table cellspacing=2 cellpadding=2>");
    for (int r = 0; r < m.rows; ++r) {
        s += QStringLiteral("<tr>");
        for (int c = 0; c < m.cols; ++c) {
            double v = m.type() == CV_64F ? m.at<double>(r, c)
                       : m.at<float>(r, c);
            s += QStringLiteral("<td bgcolor='#f8f8f8'>%1</td>")
                     .arg(v, 0, 'f', 4);
        }
        s += QStringLiteral("</tr>");
    }
    s += QStringLiteral("</table>");
    return s;
}
}  // namespace

CameraCalibTab::CameraCalibTab(QWidget* parent) : QWidget(parent) {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(8, 8, 8, 8);
    root->setSpacing(6);

    // —— 输入输出 ——
    auto* inRow = new QHBoxLayout;
    inRow->addWidget(new QLabel(QStringLiteral("输入目录:")));
    inputEdit_ = new QLineEdit(QStringLiteral("data_in/camera"));
    inRow->addWidget(inputEdit_, 1);
    inBtn_ = new QPushButton(QStringLiteral("浏览..."));
    inRow->addWidget(inBtn_);
    root->addLayout(inRow);

    auto* outRow = new QHBoxLayout;
    outRow->addWidget(new QLabel(QStringLiteral("输出 JSON:")));
    outputEdit_ = new QLineEdit(QStringLiteral("data_out/camera_calib.json"));
    outRow->addWidget(outputEdit_, 1);
    outBtn_ = new QPushButton(QStringLiteral("浏览..."));
    outRow->addWidget(outBtn_);
    root->addLayout(outRow);

    // —— 步骤按钮（左竖排）——
    auto* stepsRow = new QHBoxLayout;

    auto* stepsCol = new QVBoxLayout;
    step1Btn_ = new QPushButton(QStringLiteral("① 棋盘格角点"));
    step2Btn_ = new QPushButton(QStringLiteral("② 内参标定"));
    step3Btn_ = new QPushButton(QStringLiteral("③ 外参标定"));
    step4Btn_ = new QPushButton(QStringLiteral("④ 立体矫正+温度表"));
    runAllBtn_ = new QPushButton(QStringLiteral("▶ 一键全流程"));
    QFont f = runAllBtn_->font(); f.setBold(true); runAllBtn_->setFont(f);
    step1Lbl_ = new QLabel(QStringLiteral("状态: 未运行"));
    step2Lbl_ = new QLabel(QStringLiteral("状态: 未运行"));
    step3Lbl_ = new QLabel(QStringLiteral("状态: 未运行"));
    step4Lbl_ = new QLabel(QStringLiteral("状态: 未运行"));
    stepsCol->addWidget(step1Btn_); stepsCol->addWidget(step1Lbl_);
    stepsCol->addWidget(step2Btn_); stepsCol->addWidget(step2Lbl_);
    stepsCol->addWidget(step3Btn_); stepsCol->addWidget(step3Lbl_);
    stepsCol->addWidget(step4Btn_); stepsCol->addWidget(step4Lbl_);
    stepsCol->addStretch(1);
    stepsCol->addWidget(runAllBtn_);
    stepsRow->addLayout(stepsCol, 1);

    // —— 预览（右）——
    auto* prevCol = new QVBoxLayout;
    prevCol->addWidget(new QLabel(QStringLiteral("棋盘格叠加预览（取首帧）:")));
    auto* prevRow = new QHBoxLayout;
    preview1_ = new PreviewWidget;
    preview2_ = new PreviewWidget;
    prevRow->addWidget(preview1_, 1);
    prevRow->addWidget(preview2_, 1);
    prevCol->addLayout(prevRow, 1);
    previewBtn_ = new QPushButton(QStringLiteral("刷新预览"));
    prevCol->addWidget(previewBtn_);
    stepsRow->addLayout(prevCol, 2);
    root->addLayout(stepsRow, 3);

    // —— 进度条 ——
    progress_ = new QProgressBar;
    progress_->setRange(0, 100);
    progress_->setValue(0);
    root->addWidget(progress_);

    // —— 日志 ——
    log_ = new QPlainTextEdit;
    log_->setReadOnly(true);
    log_->setMaximumBlockCount(10000);
    log_->setStyleSheet(
        "QPlainTextEdit { font-family: Consolas, 'Courier New', monospace;"
        " font-size: 10pt; background: #1e1e1e; color: #d4d4d4; }");
    root->addWidget(log_, 2);

    // —— 结果摘要 ——
    resultLbl_ = new QLabel(QStringLiteral("结果: 未生成"));
    resultLbl_->setWordWrap(true);
    resultLbl_->setStyleSheet("background:#f8f8f8; padding:6px;");
    root->addWidget(resultLbl_);

    // —— 信号 ——
    connect(inBtn_, &QPushButton::clicked, this, &CameraCalibTab::onBrowseInput);
    connect(outBtn_, &QPushButton::clicked, this, &CameraCalibTab::onBrowseOutput);
    connect(step1Btn_, &QPushButton::clicked, this, [this]() { onRunStep(1); });
    connect(step2Btn_, &QPushButton::clicked, this, [this]() { onRunStep(2); });
    connect(step3Btn_, &QPushButton::clicked, this, [this]() { onRunStep(3); });
    connect(step4Btn_, &QPushButton::clicked, this, [this]() { onRunStep(4); });
    connect(runAllBtn_, &QPushButton::clicked, this, &CameraCalibTab::onRunAll);
    connect(previewBtn_, &QPushButton::clicked, this, &CameraCalibTab::onPreviewFirstPair);

    // —— 持久化路径 ——
    QSettings s;
    s.beginGroup("camera_calib_tab");
    inputEdit_->setText(s.value("input_dir", inputEdit_->text()).toString());
    outputEdit_->setText(s.value("output_path", outputEdit_->text()).toString());
    s.endGroup();
}

CameraCalibTab::~CameraCalibTab() {
    QSettings s;
    s.beginGroup("camera_calib_tab");
    s.setValue("input_dir", inputEdit_->text());
    s.setValue("output_path", outputEdit_->text());
    s.endGroup();
}

void CameraCalibTab::onBrowseInput() {
    QString d = QFileDialog::getExistingDirectory(this,
        QStringLiteral("选择输入目录"), inputEdit_->text());
    if (!d.isEmpty()) inputEdit_->setText(d);
}

void CameraCalibTab::onBrowseOutput() {
    QString p = QFileDialog::getSaveFileName(this,
        QStringLiteral("选择输出 JSON"), outputEdit_->text(),
        QStringLiteral("JSON (*.json)"));
    if (!p.isEmpty()) outputEdit_->setText(p);
}

void CameraCalibTab::appendLog(const QString& msg, const QString& color) {
    QString html = QStringLiteral("<span style='color:%1;'>&gt; %2</span>")
                       .arg(color, msg.toHtmlEscaped());
    log_->appendHtml(html);
}

void CameraCalibTab::refreshStepLabels() {
    auto fmt = [](bool ok) -> QString {
        return ok ? QStringLiteral("状态: <span style='color:#27ae60;'>✓ 完成</span>")
                  : QStringLiteral("状态: <span style='color:#888;'>未运行</span>");
    };
    step1Lbl_->setText(fmt(corners_ && corners_->success));
    step2Lbl_->setText(fmt(intrin_ && intrin_->success));
    step3Lbl_->setText(fmt(extrin_ && extrin_->success));
    step4Lbl_->setText(fmt(rectify_ && rectify_->success));
}

void CameraCalibTab::setRunningUI(bool running) {
    step1Btn_->setEnabled(!running);
    step2Btn_->setEnabled(!running);
    step3Btn_->setEnabled(!running);
    step4Btn_->setEnabled(!running);
    runAllBtn_->setEnabled(!running);
    inputEdit_->setEnabled(!running);
    outputEdit_->setEnabled(!running);
    progress_->setValue(running ? 5 : 0);
}

void CameraCalibTab::onRunStep(int step) {
    QString inDir = inputEdit_->text().trimmed();
    if (inDir.isEmpty()) { appendLog(QStringLiteral("输入目录为空"), "#c0392b"); return; }
    if (!QFileInfo::exists(inDir)) { appendLog(QStringLiteral("输入目录不存在"), "#c0392b"); return; }

    // 每次重置后续步骤（输入改了或重新跑前置步骤时）
    // 但允许复用前面已完成的步骤
    if (step <= 1) { corners_.reset(); }
    if (step <= 2) { intrin_.reset(); }
    if (step <= 3) { extrin_.reset(); }
    if (step <= 4) { rectify_.reset(); }

    setRunningUI(true);
    appendLog(QStringLiteral("启动步骤 %1...").arg(step));

    // 闭包里捕获需要的值，工作线程内同步执行；GUI 信号走 worker_ 转发
    CalibCallbacks cb;
    cb.onProgress = [this](int s, int t, const std::string& m) {
        QMetaObject::invokeMethod(this, "onWorkerProgress", Qt::QueuedConnection,
            Q_ARG(int, s), Q_ARG(int, t),
            Q_ARG(QString, QString::fromStdString(m)));
    };
    cb.onLog = [this](const std::string& line) {
        QMetaObject::invokeMethod(this, "onWorkerLog", Qt::QueuedConnection,
            Q_ARG(QString, QString::fromStdString(line)));
    };
    cb.shouldCancel = []() { return false; };

    std::string inputDirStr = inDir.toStdString();
    int requestedStep = step;

    // 简化：每次重头跑前置步骤（持久化中间结果在 session_）
    // 工作线程内的步骤累积
    auto futureFn = [this, inputDirStr, requestedStep, cb]() -> bool {
        try {
            // 加载 input（如果还没加载或第一步）
            if (!input_ || requestedStep <= 1) {
                input_ = loadCameraInput(inputDirStr);
                if (!input_) {
                    spdlog::error("[camera tab] 加载输入失败");
                    return false;
                }
                // 重置后续
                corners_.reset();
                intrin_.reset();
                extrin_.reset();
                rectify_.reset();
            }

            if (requestedStep >= 1) {
                if (!corners_ || !corners_->success) {
                    corners_ = extractCorners(*input_, cb);
                    if (!corners_->success) return false;
                }
            }
            if (requestedStep >= 2) {
                if (!intrin_ || !intrin_->success) {
                    intrin_ = calibrateIntrinsic(input_->config, *corners_, cb);
                    if (!intrin_->success) return false;
                }
            }
            if (requestedStep >= 3) {
                if (!extrin_ || !extrin_->success) {
                    extrin_ = calibrateExtrinsic(input_->config, *corners_, *intrin_, cb);
                    if (!extrin_->success) return false;
                }
            }
            if (requestedStep >= 4) {
                if (!rectify_ || !rectify_->success) {
                    rectify_ = stereoRectify(input_->config, *intrin_, *extrin_, cb);
                    if (!rectify_->success) return false;
                }
                auto tabs = buildTempTables(input_->config, *intrin_, *extrin_, cb);
                if (!tabs.success) return false;
                // 写 JSON（结果 + 过程分开）
                auto j = buildCameraCalibJson(input_->config, *intrin_, *extrin_, *rectify_,
                                               tabs.intrinL, tabs.intrinR, tabs.extrin, tabs.rectify);
                std::string outPath = outputEdit_->text().trimmed().toStdString();
                if (outPath.empty()) outPath = "camera_calib.json";
                if (!writeJson(outPath, j)) return false;
                std::string processPath = fc::deriveProcessPath(outPath);
                writeJson(processPath,
                          fc::buildCameraCalibProcessJson(input_->config, *intrin_, *extrin_));
            }
            return true;
        } catch (const std::exception& e) {
            spdlog::error("[camera tab] exception: {}", e.what());
            return false;
        }
    };

    auto* watcher = new QFutureWatcher<bool>(this);
    connect(watcher, &QFutureWatcher<bool>::finished, this, [this, watcher, step]() {
        bool ok = watcher->result();
        QString summary = ok ? QStringLiteral("步骤 %1 成功").arg(step)
                              : QStringLiteral("步骤 %1 失败").arg(step);
        appendLog(summary, ok ? "#27ae60" : "#c0392b");
        if (ok && step >= 4) {
            // 明确提示输出文件位置（writeJson 成功后 GUI 此前无任何路径反馈，易被误认为未保存）
            QString out = outputEdit_->text().trimmed();
            if (out.isEmpty()) out = QStringLiteral("camera_calib.json");
            QString abs = QFileInfo(out).absoluteFilePath();
            appendLog(QStringLiteral("结果已保存: %1").arg(abs), "#27ae60");
            emit statusMessage(QStringLiteral("标定结果已保存: %1").arg(abs));

            // 弹窗汇报标定精度（含阈值判定）
            if (intrin_ && extrin_) {
                const double reprojThr = input_ ? input_->config.reprojErrorThreshold : 0.0;
                const double stereoThr = reprojThr * 100.0;   // 与算子侧 maxReprojError 同式
                const double epipolarThr = 0.05;              // ExtrinsicCalibCpu 默认 maxEpipolarError

                auto item = [](double v, double thr) -> QString {
                    if (thr <= 0) return QStringLiteral("%1").arg(v, 0, 'f', 4);
                    bool good = v <= thr;
                    return QStringLiteral("%1 %2 (阈值 %3)")
                        .arg(v, 0, 'f', 4)
                        .arg(good ? QStringLiteral("✓达标")
                                  : QStringLiteral("⚠超限"))
                        .arg(thr, 0, 'f', 3);
                };

                bool reprojOk = intrin_->reproj_error_mean <= reprojThr;
                bool stereoOk = extrin_->stereoReprojError <= stereoThr;
                bool epipOk = extrin_->epipolarErrorMean <= epipolarThr;
                bool allOk = reprojOk && stereoOk && epipOk;

                QString html = QStringLiteral(
                    "<table cellpadding=3>"
                    "<tr><td>有效帧数</td><td>%1 / %2</td></tr>"
                    "<tr><td>内参重投影均值</td><td>%3</td></tr>"
                    "<tr><td>左相机 RMS</td><td>%4</td></tr>"
                    "<tr><td>右相机 RMS</td><td>%5</td></tr>"
                    "<tr><td>立体标定 RMS</td><td>%6</td></tr>"
                    "<tr><td>极线误差均值</td><td>%7</td></tr>"
                    "</table><br>%8")
                    .arg(intrin_->valid_frames_count)
                    .arg(intrin_->total_frames_input)
                    .arg(item(intrin_->reproj_error_mean, reprojThr))
                    .arg(intrin_->left.rms_error, 0, 'f', 4)
                    .arg(intrin_->right.rms_error, 0, 'f', 4)
                    .arg(item(extrin_->stereoReprojError, stereoThr))
                    .arg(item(extrin_->epipolarErrorMean, epipolarThr))
                    .arg(allOk ? QStringLiteral("<b>标定精度达标</b>")
                               : QStringLiteral("<b style='color:#c0392b;'>部分指标超限，"
                                  "结果已保存但质量降级，建议改善采集条件后重新标定</b>"));

                QMessageBox box(this);
                box.setWindowTitle(allOk ? QStringLiteral("相机标定完成 —— 精度达标")
                                         : QStringLiteral("相机标定完成 —— 部分指标超限"));
                box.setIcon(allOk ? QMessageBox::Information : QMessageBox::Warning);
                box.setTextFormat(Qt::RichText);
                box.setText(html);
                box.setInformativeText(QStringLiteral("结果已保存:\n%1").arg(abs));
                box.setStandardButtons(QMessageBox::Ok);
                box.exec();
            }
        }
        refreshStepLabels();
        setRunningUI(false);
        emit statusMessage(summary);
        if (ok && step >= 2 && intrin_) {
            // 显示内参摘要
            QString html = QStringLiteral(
                "<b>左相机内参</b> RMS=%1<br>"
                "K_L=<br>%2<br>D_L=%3<br><br>"
                "<b>右相机内参</b> RMS=%4<br>"
                "K_R=<br>%5<br>D_R=%6")
                .arg(intrin_->left.rms_error, 0, 'f', 4).arg(matToHtml(intrin_->left.camera_matrix))
                .arg(matToHtml(intrin_->left.dist_coeffs))
                .arg(intrin_->right.rms_error, 0, 'f', 4).arg(matToHtml(intrin_->right.camera_matrix))
                .arg(matToHtml(intrin_->right.dist_coeffs));
            if (extrin_ && extrin_->success) {
                html += QStringLiteral("<br><br><b>外参</b> 立体RMS=%1 epipolar_mean=%2<br>R=<br>%3<br>T=<br>%4")
                            .arg(extrin_->stereoReprojError, 0, 'f', 4)
                            .arg(extrin_->epipolarErrorMean, 0, 'f', 4)
                            .arg(matToHtml(extrin_->R))
                            .arg(matToHtml(extrin_->T));
            }
            if (rectify_ && rectify_->success) {
                html += QStringLiteral("<br><br><b>立体矫正</b> 已生成 R1/R2/P1/P2/Q");
            }
            resultLbl_->setText(html);
        }
        if (ok) onPreviewFirstPair();
        watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run(futureFn));
}

void CameraCalibTab::onRunAll() {
    onRunStep(4);
}

void CameraCalibTab::onWorkerProgress(int s, int t, QString msg) {
    if (t > 0) progress_->setValue(static_cast<int>(s * 100.0 / t));
    if (!msg.isEmpty()) appendLog(QStringLiteral("[%1/%2] %3").arg(s).arg(t).arg(msg));
}

void CameraCalibTab::onWorkerLog(QString line) {
    if (!line.isEmpty()) log_->appendPlainText(line);
}

void CameraCalibTab::onWorkerDone(bool ok, QString summary) {
    appendLog(summary, ok ? "#27ae60" : "#c0392b");
    setRunningUI(false);
    refreshStepLabels();
}

void CameraCalibTab::onPreviewFirstPair() {
    if (!input_ || input_->frames.empty()) return;
    // 棋盘角点叠加：用 cv::drawChessboardCorners
    cv::Mat lColor, rColor;
    cv::cvtColor(input_->frames[0].leftGray, lColor, cv::COLOR_GRAY2BGR);
    cv::cvtColor(input_->frames[0].rightGray, rColor, cv::COLOR_GRAY2BGR);
    if (corners_ && corners_->validFrames > 0) {
        cv::Size psz(input_->config.chessWidth, input_->config.chessHeight);
        cv::drawChessboardCorners(lColor, psz, corners_->leftPoints[0], true);
        cv::drawChessboardCorners(rColor, psz, corners_->rightPoints[0], true);
    }
    preview1_->setImage(lColor);
    preview2_->setImage(rColor);
}

}  // namespace fc::gui
