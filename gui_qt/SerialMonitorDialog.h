#pragma once

#include <QDialog>

class QPlainTextEdit;
class QPushButton;
class QCheckBox;

namespace fc::gui {

// 串口通讯监视弹窗：实时显示 上位机↓发送 / 下位机↑返回 的消息内容
// 非模态；软件打开时自动弹出，关掉后可从「🖥 串口监视」按钮重开
class SerialMonitorDialog : public QDialog {
    Q_OBJECT
public:
    explicit SerialMonitorDialog(QWidget* parent = nullptr);

public slots:
    void appendTx(const QString& frame);   // 上位机→下位机（下行 N1x）
    void appendRx(const QString& frame);   // 下位机→上位机（上行 G0x）

private:
    void appendLine(const QString& arrow, const QString& color, const QString& frame);

    QPlainTextEdit* view_          = nullptr;
    QPushButton*    clearBtn_      = nullptr;
    QCheckBox*      autoScrollChk_ = nullptr;
};

}  // namespace fc::gui
