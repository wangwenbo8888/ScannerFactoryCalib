#include "PreviewWidget.h"

#include <opencv2/imgproc.hpp>
#include <QResizeEvent>
#include <QPainter>
#include <spdlog/spdlog.h>
#include <atomic>

namespace fc::gui {

PreviewWidget::PreviewWidget(QWidget* parent) : QLabel(parent) {
    setMinimumSize(160, 120);  // 仅保留最小尺寸供布局，其余定制去掉（避免与 pixmap 渲染冲突）
}

void PreviewWidget::setImage(const cv::Mat& mat) {
    if (mat.empty()) { clearImage(); return; }
    currentMat_ = mat;

    cv::Mat rgb;
    if (mat.channels() == 1) {
        cv::cvtColor(mat, rgb, cv::COLOR_GRAY2BGR);
    } else {
        rgb = mat.clone();
    }
    cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);

    QImage img(rgb.data, rgb.cols, rgb.rows,
               static_cast<int>(rgb.step), QImage::Format_RGB888);
    currentPix_ = QPixmap::fromImage(img.copy());  // copy 脱离 cv::Mat 生命周期
    refreshPixmap();
}

void PreviewWidget::setPlaceholder(const QString& text) {
    placeholder_ = text;
    if (currentPix_.isNull()) setText(placeholder_);
}

void PreviewWidget::setQImage(const QImage& img) {
    if (img.isNull()) { clearImage(); return; }
    setPixmap(QPixmap::fromImage(img));
    repaint();  // 强制立即重绘（避免偶发不刷新）
}

void PreviewWidget::clearImage() {
    currentMat_.release();
    currentPix_ = QPixmap();
    setText(placeholder_);
}

void PreviewWidget::resizeEvent(QResizeEvent* e) {
    QLabel::resizeEvent(e);
    if (!currentPix_.isNull()) refreshPixmap();
}

void PreviewWidget::refreshPixmap() {
    if (currentPix_.isNull()) return;
    int w = width(), h = height();
    if (w < 2 || h < 2) return;
    setPixmap(currentPix_.scaled(w, h, Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

}  // namespace fc::gui
