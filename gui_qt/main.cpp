#include "MainWindow.h"

#include <QApplication>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QStyleFactory>

namespace {
// 探测工程根目录（含 data_in/ 子目录）：依次检查 cwd、exe 同级、exe 父目录。
QString detectProjectRoot() {
    QStringList candidates;
    candidates << QDir::currentPath();
    QString exeDir = QCoreApplication::applicationDirPath();
    candidates << exeDir << QFileInfo(exeDir).absolutePath();
    for (const QString& c : candidates) {
        if (QDir(c).exists(QStringLiteral("data_in"))) return c;
    }
    return QDir::currentPath();  // 退回 cwd，让用户自己 Browse
}
}  // namespace

int main(int argc, char* argv[]) {
    QApplication::setStyle(QStyleFactory::create("Fusion"));
    QApplication app(argc, argv);
    app.setApplicationName("factory_calib_gui");
    app.setOrganizationName("JEAMMWARE");

    QCommandLineParser parser;
    parser.setApplicationDescription(QStringLiteral("factory_calib GUI"));
    QCommandLineOption rootOpt(
        { "r", "root" },
        QStringLiteral("工程根目录（含 data_in/data_out），默认自动探测。"),
        QStringLiteral("dir"));
    parser.addOption(rootOpt);
    parser.addHelpOption();
    parser.process(app);

    QString root;
    if (parser.isSet(rootOpt)) {
        root = parser.value(rootOpt);
    } else {
        root = detectProjectRoot();
    }
    QDir::setCurrent(root);

    fc::gui::MainWindow w;
    w.show();
    return app.exec();
}

