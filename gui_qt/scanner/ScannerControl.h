#pragma once

// =============================================================================
// ScannerControl —— 扫描仪硬件串口控制
//
// 串口参数与启停时序移植 LeadScanK2/series/LEADSCANSeries.cpp:
//   - openPort1()                       115200/8N1/NoParity/OneStop
//   - on_pushButton_Stop_Scanner_Clicked()   发 "N11 H0;"
// N10 命令按《下位机通讯协议说明.md》(260831 版) 七参格式:
//   "N10 H{1-200} B{0-100} T{0/1} V{0/1} C{0/1} D{0/1} L{0-100};"
//   T/V/C/D 为四根激光管开关，已开启者按 T→V→C→D 轮流点亮（单开一管=固定该激光线）
//
// 扫描仪硬件收到 N10 启动后，电机+激光运转，同时在 Line2 上发硬件触发脉冲，
// 此时 GalaxyCameraSource 的相机才能拿到帧（TriggerSource=Line2）。
//
// 注: 不用 Q_OBJECT/signals，避免 AUTOMOC 配置问题；用 std::function 回调。
// =============================================================================

#include <QObject>
#include <QString>
#include <QStringList>
#include <functional>
#include <memory>

class QSerialPort;

namespace fc::gui {

struct ScannerParams {
    int freq       = 50;   // N10 H 拍照频率 (Hz) 1-200
    int background = 40;   // N10 B 补光强度 0-100
    int laser      = 60;   // N10 L 激光管亮度 0-100（四管共用）
    int tubeT      = 0;    // N10 T 激光管1（左斜） 0/1
    int tubeV      = 0;    // N10 V 激光管2（右斜） 0/1
    int tubeC      = 0;    // N10 C 激光管3（精细） 0/1
    int tubeD      = 0;    // N10 D 激光管4（深孔） 0/1
};

class ScannerControl {
public:
    ScannerControl();
    ~ScannerControl();

    // 状态消息回调（替代 Qt signal，避免 MOC 依赖）
    std::function<void(const QString&)> onStatus;

    // 枚举系统所有可用 COM 口
    static QStringList availablePorts();

    // 打开/关闭串口
    bool open(const QString& portName);
    void close();
    bool isOpen() const;

    // 启动扫描仪：按 260831 协议发七参
    //   "N10 H{freq} B{bg} T{t} V{v} C{c} D{d} L{laser};"
    // 激光全关（四管全 0）→ 仅补光模式（相机标定拍标志点）
    bool start(const ScannerParams& p);

    // 停止扫描仪：发 "N11 H0;"
    bool stop();

private:
    bool sendLine(const QString& line);
    void notify(const QString& msg) { if (onStatus) onStatus(msg); }

    QSerialPort* port_ = nullptr;
};

}  // namespace fc::gui
