#pragma once

#include <QLabel>
#include <QPixmap>
#include <QImage>
#include <opencv2/core.hpp>

namespace fc::gui {

// 显示 OpenCV Mat 的 widget：自动按 widget 大小缩放，保持纵横比。
// 支持单通道（灰度）和 3 通道（BGR）。
class PreviewWidget : public QLabel {
    Q_OBJECT
public:
    explicit PreviewWidget(QWidget* parent = nullptr);

    // 设置要显示的图像（拷贝；内部转 QPixmap）
    void setImage(const cv::Mat& mat);

    // 直接设 QImage（跨线程 queued 信号投递的帧用此入口；Qt 原生跨线程安全）
    void setQImage(const QImage& img);

    // 占位文字（无图像时显示）
    void setPlaceholder(const QString& text);

    // 清空回到占位
    void clearImage();

protected:
    void resizeEvent(QResizeEvent* e) override;

private:
    void refreshPixmap();

    cv::Mat currentMat_;
    QPixmap currentPix_;
    QString placeholder_ = QStringLiteral("（无图像）");
};

}  // namespace fc::gui
