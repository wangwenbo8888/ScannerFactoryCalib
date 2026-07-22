#include "gui_common.h"

#include <QFileInfo>
#include <QCoreApplication>
#include <QDir>

namespace fc::gui {

QStringList bytesToLines(const QByteArray& bytes) {
    QString text = QString::fromUtf8(bytes);
    // 保留未完成尾行（不带换行的部分）以便下次拼接
    return text.split('\n', QString::KeepEmptyParts);
}

QString buildChildPathEnv(const QString& exePath) {
    QFileInfo fi(exePath);
    QString exeDir = fi.absolutePath();
    // qEnvironmentVariable 在 Qt 5.10+ 才有；用 QByteArray 版本兼容 5.9
    QString cur = QString::fromLocal8Bit(qgetenv("PATH"));
    // 把 exe 同目录前置（OpenCV DLL 部署在 exe 旁边）
    return exeDir + QDir::listSeparator() + cur;
}

}  // namespace fc::gui
