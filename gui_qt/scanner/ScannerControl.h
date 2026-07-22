#pragma once

// =============================================================================
// ScannerControl —— 扫描仪硬件串口控制
//
// 直接移植 LeadScanK2/series/LEADSCANSeries.cpp 里的:
//   - openPort1()                       115200/8N1/NoParity/OneStop
//   - on_pushButton_Start_Scanner_Clicked()  发 "N10 H{freq} B{bg} T1 V2 L{laser};"
//   - on_pushButton_Stop_Scanner_Clicked()   发 "N11 H0;"
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
    int freq       = 50;   // 电机频率 (Hz)
    int background = 40;   // 背光强度
    int laser      = 60;   // 激光强度
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

    // 启动扫描仪：发 "N10 H{freq} B{bg} T1 V2 L{laser};"
    // 若 laser<=0 → 发 "N10 H{freq} B{bg} T1 V2 L0;"（只补光，激光关）
    bool start(const ScannerParams& p);

    // 停止扫描仪：发 "N11 H0;"
    bool stop();

private:
    bool sendLine(const QString& line);
    void notify(const QString& msg) { if (onStatus) onStatus(msg); }

    QSerialPort* port_ = nullptr;
};

}  // namespace fc::gui
