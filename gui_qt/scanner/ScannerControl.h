#pragma once

// =============================================================================
// ScannerControl —— 扫描仪硬件串口控制
//
// 串口参数与启停时序移植 LeadScanK2/series/LEADSCANSeries.cpp:
//   - openPort1()                       115200/8N1/NoParity/OneStop
//   - on_pushButton_Stop_Scanner_Clicked()   发 "N11 H0;"
// N10 命令七参格式:
//   "N10 H{1-200} B{0-100} T{0/1} V{0/1} C{0/1} D{0/1} L{0-100};"
//   T/V/C/D 为四根激光管开关，已开启者轮流点亮（单开一管=固定该激光线）
//   管语义（261002 临时测试机——扫描仪损坏临时环境）：T=精细线 V=左斜线
//   C=右斜线 D=无对应线〔未接，占位〕；原 260831 管号（T=左斜/V=右斜/
//   C=深孔/D=精细）作废，回正式机须回退
//
// 扫描仪硬件收到 N10 启动后，电机+激光运转，同时在 Line2 上发硬件触发脉冲，
// 此时 GalaxyCameraSource 的相机才能拿到帧（TriggerSource=Line2）。
//
// 上行接收（G01/G02/G03）：readyRead 累积缓冲按 ';' 分帧 → onRx 回调
// （串口监视显示全部帧）。其中 G02（4 路温度）已消费：parseG02 解析后经
// onTemp 回调上报数值（260831 协议: "G02 A<t1> B<t2> C<t3> D<t4>;"）。
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
    int tubeT      = 0;    // N10 T 激光管1（261002 临时测试机：精细线） 0/1
    int tubeV      = 0;    // N10 V 激光管2（261002 临时测试机：左斜线） 0/1
    int tubeC      = 0;    // N10 C 激光管3（261002 临时测试机：右斜线） 0/1
    int tubeD      = 0;    // N10 D 激光管4（261002 临时测试机：无对应线，占位） 0/1
};

class ScannerControl {
public:
    ScannerControl();
    ~ScannerControl();

    // 状态消息回调（替代 Qt signal，避免 MOC 依赖）
    std::function<void(const QString&)> onStatus;
    // 串口监视回调：onTx=下行帧完整写入串口后（sendLine 成功）；onRx=收到上行完整 ';' 帧
    std::function<void(const QString&)> onTx;
    std::function<void(const QString&)> onRx;
    // G02 温度回调：4 路摄氏度（0-100，一位小数；串口线程=主线程事件循环，可直接刷 UI）
    std::function<void(float, float, float, float)> onTemp;

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

    // 设定温度回传周期（260831 协议: "N12 T<ms>;")。ms 1-1000；<5ms 会压爆
    // 115200 串口带宽（协议 D9 结论），显示用途建议 >=500
    bool setTempReportPeriod(int ms);

private:
    bool sendLine(const QString& line);
    void onReadyRead();       // 串口可读：累积缓冲、按 ';' 切帧回调 onRx
    void parseG02(const QString& frame);  // "G02 A.. B.. C.. D..;" → onTemp
    void notify(const QString& msg) { if (onStatus) onStatus(msg); }

    QSerialPort* port_ = nullptr;
    QByteArray rxBuf_;        // 上行接收缓冲（跨 readyRead 拼半帧）
};

}  // namespace fc::gui
