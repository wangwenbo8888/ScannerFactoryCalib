#pragma once
// cmtt_tier_select.h — CMTT sidecar 按温选档（factory 工具层助手，非 09 组件）
//
// 语义: 档温锚定——头 tempBase＝参考温度×10（非首档温，cmtt_container.h:37），真实网格以
//       参考档居中（ref±step·N）；先扫参考档索引 i_ref（isReference），则
//       tierTemp(i) = tempBase + (i - i_ref)·tempStep——与逐档索引 temperatureX10 同网格
//       （cmtt_container.h:57；nearestTier 即按该逐档字段比较，cmtt_container.cpp:172-189）。
//       Reader 无逐档温访问器且 09 组件逐字不可改，故由公开 API 推回；
//       无参考档（理论不可达，生成器红线保证）→ ok=false；
//       选最大 档温<=目标 的非 missing 档（偏低温 tie-break，同容器 nearestTier/主工程口径）；
//       目标以下无非 missing 档 → 升温侧最近非 missing 档（missing 就近跳过: 先低温侧后高温侧）；
//       目标越出档温上界 → 末个非 missing 档（clamp 端档，why 注记）。
// 安全: CmttReader::load 拦整文件级损坏（magic/version/索引边界）；
//       tierBytes 拷出＋逐档 crc32 懒校验，CRC 不过不返回 blob——未验证字节不流向
//       CurveMapTable::Load（内存段重载）。

#include "cmtt_container.h"   // curve_map_temp_table/ 经 fc2_ops include 路径解析（裸名可达）

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace calib {

struct CmttTierChoice {
    bool ok = false;
    std::vector<uint8_t> blob;   // 单档 CMPU 载荷（CurveMapTable::Load 内存段重载可直接吃）
    double tierTemp = 0.0;
    std::string why;             // 失败原因 / 成功时档位＋clamp 注记
};

inline CmttTierChoice selectCmttTier(const std::string& sidecarPath, double targetTempC) {
    CmttTierChoice out;

    // Reader 是非拥有视图——整文件读入本函数栈驻留
    std::vector<uint8_t> raw;
    {
        std::ifstream f(sidecarPath, std::ios::binary | std::ios::ate);
        if (!f) { out.why = "cannot open sidecar '" + sidecarPath + "'"; return out; }
        const std::streamoff n = f.tellg();
        if (n <= 0) { out.why = "sidecar empty or size query failed"; return out; }
        raw.resize(static_cast<size_t>(n));
        f.seekg(0, std::ios::beg);
        if (!f.read(reinterpret_cast<char*>(raw.data()),
                    static_cast<std::streamsize>(raw.size()))) {
            out.why = "sidecar short read";
            return out;
        }
    }

    CmttReader reader;
    if (!reader.load(raw.data(), raw.size())) {
        out.why = "container load rejected (magic/version/index bounds)";
        return out;
    }
    const size_t n = reader.tierCount();
    if (n == 0) { out.why = "sidecar has no tiers"; return out; }

    const double baseC = reader.header().tempBaseX10 / 10.0;   // 参考温度（非首档温）
    const double stepC = reader.header().tempStepX10 / 10.0;
    // 锚定: 参考档索引 i_ref 一次扫出（isReference，O(N)·N≤1024 无害），
    // tierTemp(i) = tempBase + (i - i_ref)·tempStep（网格以参考档居中）；
    // 无参考档 → 锚丢失拒选（生成器红线保证理论不可达）。
    std::optional<size_t> iRef;
    for (size_t i = 0; i < n; ++i)
        if (reader.isReference(i)) { iRef = i; break; }
    if (!iRef) { out.why = "no reference tier (anchor lost)"; return out; }
    const auto tierTempOf = [&](size_t i) {
        return baseC + (static_cast<double>(i) - static_cast<double>(*iRef)) * stepC;
    };

    // 选档（档温随 i 单调升）: 最大 档温<=target 的非 missing 档；
    // 目标低于全部非 missing 档 → 升温侧首个非 missing 档（低端 clamp）。
    std::optional<size_t> pick;
    for (size_t i = 0; i < n; ++i) {
        if (reader.isMissing(i)) continue;
        const double t = tierTempOf(i);
        if (t <= targetTempC) { pick = i; continue; }
        if (!pick) pick = i;
        break;
    }
    if (!pick) { out.why = "all tiers missing"; return out; }
    const size_t idx = *pick;
    const double tierTemp = tierTempOf(idx);

    auto blob = reader.tierBytes(idx);   // 拷出＋逐档 CRC 懒校验（不过 → nullopt）
    if (!blob) {
        out.why = "tier " + std::to_string(idx) + " payload rejected (CRC/missing)";
        return out;
    }

    // clamp 注记: 低端=选中档温高于目标（下方无非 missing 档）；
    //             高端=目标高于选中档温且其上无非 missing 档。
    size_t lastNonMissing = n;
    for (size_t i = n; i-- > 0;)
        if (!reader.isMissing(i)) { lastNonMissing = i; break; }
    std::ostringstream note;
    note << "tier " << idx << " of " << n;
    if (tierTemp > targetTempC)
        note << ", clamped low (target " << targetTempC << " below range)";
    else if (idx == lastNonMissing && targetTempC > tierTemp)
        note << ", clamped high (target " << targetTempC << " above range)";

    out.ok = true;
    out.blob = std::move(*blob);
    out.tierTemp = tierTemp;
    out.why = note.str();
    return out;
}

} // namespace calib
