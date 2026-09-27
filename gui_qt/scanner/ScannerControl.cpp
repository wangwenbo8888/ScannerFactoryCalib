// =============================================================================
// ScannerControl.cpp —— 直接移植 LeadScanK2/series/LEADSCANSeries.cpp 串口部分
// =============================================================================

#include "ScannerControl.h"

#include <QRegularExpression>
#include <QSerialPort>
#include <QSerialPortInfo>
#include <spdlog/spdlog.h>
#include <fstream>

namespace fc::gui {

ScannerControl::ScannerControl() {
    port_ = new QSerialPort();
    // 上行接收：readyRead → onReadyRead 累积按 ';' 分帧（协议分帧符）。
    // lambda 以 port_ 为 context，port_ 析构连接自动断；不引入 Q_OBJECT/MOC
    QObject::connect(port_, &QSerialPort::readyRead, port_, [this] { onReadyRead(); });
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
        rxBuf_.clear();
        spdlog::info("[Scanner] 串口已关闭");
        notify(QStringLiteral("串口已关闭"));
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

// 设定温度回传周期：260831 协议 "N12 T<ms>;"（G02 周期上报由此触发）
bool ScannerControl::setTempReportPeriod(int ms) {
    if (!isOpen()) return false;
    if (ms < 1) ms = 1;
    if (ms > 1000) ms = 1000;
    QString cmd = QStringLiteral("N12 T%1;").arg(ms);
    spdlog::info("[Scanner] 温度回传周期: {}", cmd.toStdString());
    return sendLine(cmd);
}

bool ScannerControl::sendLine(const QString& line) {
    if (!port_ || !port_->isOpen()) return false;
    QByteArray data = line.toLocal8Bit();
    qint64 written = port_->write(data);
    if (written != data.size()) {
        spdlog::warn("[Scanner] 写入不完整: 期望 {} 实际 {}", data.size(), written);
        return false;
    }
    if (!port_->waitForBytesWritten(1000)) return false;
    if (onTx) onTx(line);   // 串口监视：完整写出后才上报为「已发送」
    return true;
}

// 上行分帧：协议以 ';' 结尾分帧；同时兼容 '\n'（'\r' 按空白剥掉）——实测下位机
// 上行帧尾存在不带 ';' 的可能（2026-09-27 N12 已发但零 RX 帧，待 hex 诊断定位）。
// 缓冲跨包拼半帧，超 4KB 仍无分帧符视为垃圾丢弃（附 hex 预览供协议比对）。
void ScannerControl::onReadyRead() {
    const QByteArray chunk = port_->readAll();
    rxBuf_.append(chunk);
    // 诊断：本包字节里没有任何分帧符 → 打印 hex 预览（真实帧样貌一锤定音；
    // 2026-09-27 加，定位「串口有数据但分帧不符」，问题定位后可删）
    if (!chunk.isEmpty() && chunk.indexOf(';') < 0 && chunk.indexOf('\n') < 0) {
        spdlog::warn("[Scanner] RX 无分帧符字节 hex 预览: [{}]",
                     QString::fromLatin1(rxBuf_.left(48).toHex(' ')).toStdString());
    }
    while (true) {
        // 取最早出现的分帧符（';' 或 '\n'）——混合帧尾时避免跨帧误切
        const int is = rxBuf_.indexOf(';');
        const int in = rxBuf_.indexOf('\n');
        if (is < 0 && in < 0) break;
        const int idx = (is < 0) ? in : ((in < 0) ? is : qMin(is, in));
        const QByteArray frame = rxBuf_.left(idx + 1);
        rxBuf_.remove(0, idx + 1);
        const QString s = QString::fromLatin1(frame).trimmed();
        if (s.isEmpty()) continue;
        spdlog::info("[Scanner] RX: {}", s.toStdString());
        if (s.startsWith(QStringLiteral("G02"))) parseG02(s);
        if (onRx) onRx(s);
    }
    if (rxBuf_.size() > 4096) {
        spdlog::warn("[Scanner] RX 缓冲 {} 字节无分帧符，丢弃。hex 预览: [{}]",
                     rxBuf_.size(),
                     QString::fromLatin1(rxBuf_.left(64).toHex(' ')).toStdString());
        rxBuf_.clear();
    }
}

// G02 温度帧解析：格式 "G02 A25.3 B25.4 C26.0 D24.5;"（0-100，一位小数；
// 字母数字间、字段间空白容忍）。解析失败静默（畸形帧不影响其余流程）。
void ScannerControl::parseG02(const QString& s) {
    static const QRegularExpression re(QStringLiteral(
        "^G02\\s+A(\\d+(?:\\.\\d+)?)\\s+B(\\d+(?:\\.\\d+)?)"
        "\\s+C(\\d+(?:\\.\\d+)?)\\s+D(\\d+(?:\\.\\d+)?)"));
    const auto m = re.match(s);
    if (!m.hasMatch()) {
        spdlog::warn("[Scanner] G02 解析失败: {}", s.toStdString());
        return;
    }
    if (onTemp)
        onTemp(m.captured(1).toFloat(), m.captured(2).toFloat(),
               m.captured(3).toFloat(), m.captured(4).toFloat());
}

}  // namespace fc::gui
