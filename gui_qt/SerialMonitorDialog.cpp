#include "SerialMonitorDialog.h"

#include <QCheckBox>
#include <QDateTime>
#include <QFont>
#include <QHBoxLayout>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollBar>
#include <QVBoxLayout>

namespace fc::gui {

SerialMonitorDialog::SerialMonitorDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(QStringLiteral("串口通讯监视（↓发送 ↑接收）"));
    resize(600, 400);

    auto* root = new QVBoxLayout(this);

    view_ = new QPlainTextEdit;
    view_->setReadOnly(true);
    QFont mono(QStringLiteral("Consolas"), 9);
    mono.setStyleHint(QFont::Monospace);
    view_->setFont(mono);
    view_->setMaximumBlockCount(2000);   // 环形截断，防长时间运行无限增长
    view_->setPlaceholderText(QStringLiteral(
        "等待串口通讯…\n"
        "↓ 上位机发送（N10/N11/N12/N13）\n"
        "↑ 下位机返回（G01 按键 / G02 温度 / G03 拍照计数）"));
    root->addWidget(view_, 1);

    auto* row = new QHBoxLayout;
    clearBtn_ = new QPushButton(QStringLiteral("清空"));
    autoScrollChk_ = new QCheckBox(QStringLiteral("自动滚动"));
    autoScrollChk_->setChecked(true);
    row->addWidget(clearBtn_);
    row->addWidget(autoScrollChk_);
    row->addStretch(1);
    root->addLayout(row);

    connect(clearBtn_, &QPushButton::clicked, view_, &QPlainTextEdit::clear);
}

void SerialMonitorDialog::appendTx(const QString& frame) {
    appendLine(QStringLiteral("↓"), QStringLiteral("#c0392b"), frame);
}

void SerialMonitorDialog::appendRx(const QString& frame) {
    appendLine(QStringLiteral("↑"), QStringLiteral("#1e8449"), frame);
}

void SerialMonitorDialog::appendLine(const QString& arrow, const QString& color,
                                     const QString& frame) {
    const QString ts = QDateTime::currentDateTime().toString(QStringLiteral("HH:mm:ss.zzz"));
    view_->appendHtml(QStringLiteral("<span style=\"color:%1;\">[%2] %3 %4</span>")
                          .arg(color, ts, arrow, frame.toHtmlEscaped()));
    if (autoScrollChk_->isChecked()) {
        QScrollBar* sb = view_->verticalScrollBar();
        sb->setValue(sb->maximum());
    }
}

}  // namespace fc::gui
