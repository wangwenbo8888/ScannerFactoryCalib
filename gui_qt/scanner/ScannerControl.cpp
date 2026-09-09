// =============================================================================
// ScannerControl.cpp —— 直接移植 LeadScanK2/series/LEADSCANSeries.cpp 串口部分
// =============================================================================

#include "ScannerControl.h"

#include <QSerialPort>
#include <QSerialPortInfo>
#include <spdlog/spdlog.h>
#include <fstream>

namespace fc::gui {

ScannerControl::ScannerControl() {
    port_ = new QSerialPort();
}

ScannerControl::~ScannerControl() {
    close();
    delete port_;
}

QStringList ScannerControl::availablePorts() {
    QStringList names;
    for (const auto& info : QSerialPortInfo::availablePorts()) {
        names << info.portName();
    }
    return names;
}

bool ScannerControl::open(const QString& portName) {
    // 仿 openPort1(): 已开则先关再开
    if (port_->isOpen()) {
        port_->clear();
        port_->close();
    }

    port_->setPortName(portName);
    if (!port_->open(QIODevice::ReadWrite)) {
        spdlog::warn("[Scanner] 打开串口 {} 失败", portName.toStdString());
        notify(QStringLiteral("串口 %1 打开失败").arg(portName));
        return false;
    }

    // 仿参考：115200/8N1/NoParity/OneStop/NoFlowControl
    port_->setBaudRate(QSerialPort::Baud115200, QSerialPort::AllDirections);
    port_->setDataBits(QSerialPort::Data8);
    port_->setFlowControl(QSerialPort::NoFlowControl);
    port_->setParity(QSerialPort::NoParity);
    port_->setStopBits(QSerialPort::OneStop);

    spdlog::info("[Scanner] 串口 {} 已打开 (115200/8N1)", portName.toStdString());
    notify(QStringLiteral("扫描仪串口 %1 已打开").arg(portName));
    return true;
}

void ScannerControl::close() {
    if (port_ && port_->isOpen()) {
        port_->clear();
        port_->close();
        spdlog::info("[Scanner] 串口已关闭");
        notify(QStringLiteral("扫描仪串口已关闭"));
    }
}

bool ScannerControl::isOpen() const {
    return port_ && port_->isOpen();
}

bool ScannerControl::start(const ScannerParams& p) {
    { std::ofstream d("E:/workfold/factory_calib/debug.txt", std::ios::app); d << "scannerStart open=" << isOpen() << " freq=" << p.freq << " laser=" << p.laser
        << " T" << p.tubeT << " V" << p.tubeV << " C" << p.tubeC << " D" << p.tubeD << "\n"; }
    if (!isOpen()) {
        notify(QStringLiteral("请先打开扫描仪串口"));
        return false;
    }
    // 260831 协议七参格式: "N10 H{freq} B{bg} T{t} V{v} C{c} D{d} L{laser};"
    // T/V/C/D 四管开关：已开启者按 T→V→C→D 轮流点亮，单开一管即固定该激光线
    int laserVal = (p.laser > 0) ? p.laser : 0;
    QString cmd = QStringLiteral("N10 H%1 B%2 T%3 V%4 C%5 D%6 L%7;")
                      .arg(p.freq)
                      .arg(p.background)
                      .arg(p.tubeT ? 1 : 0)
                      .arg(p.tubeV ? 1 : 0)
                      .arg(p.tubeC ? 1 : 0)
                      .arg(p.tubeD ? 1 : 0)
                      .arg(laserVal);
    spdlog::info("[Scanner] 启动命令: {}", cmd.toStdString());
    notify(QStringLiteral("发送启动命令: %1").arg(cmd));

    bool ok = sendLine(cmd);
    if (ok) {
        notify(QStringLiteral("扫描仪已启动 (freq=%1 bg=%2 laser=%3 T%4 V%5 C%6 D%7)")
                   .arg(p.freq).arg(p.background).arg(laserVal)
                   .arg(p.tubeT ? 1 : 0).arg(p.tubeV ? 1 : 0)
                   .arg(p.tubeC ? 1 : 0).arg(p.tubeD ? 1 : 0));
    }
    return ok;
}

bool ScannerControl::stop() {
    if (!isOpen()) {
        return false;
    }
    // 仿 on_pushButton_Stop_Scanner_Clicked(): "N11 H0;"
    QString cmd = QStringLiteral("N11 H0;");
    spdlog::info("[Scanner] 停止命令: {}", cmd.toStdString());
    notify(QStringLiteral("发送停止命令: %1").arg(cmd));
    bool ok = sendLine(cmd);
    if (ok) {
        notify(QStringLiteral("扫描仪已停止"));
    }
    return ok;
}

bool ScannerControl::sendLine(const QString& line) {
    if (!port_ || !port_->isOpen()) return false;
    QByteArray data = line.toLocal8Bit();
    qint64 written = port_->write(data);
    if (written != data.size()) {
        spdlog::warn("[Scanner] 写入不完整: 期望 {} 实际 {}", data.size(), written);
        return false;
    }
    return port_->waitForBytesWritten(1000);
}

}  // namespace fc::gui
