// cmtt_container.h — CMTT 温度表容器读写器（curve_map 61 档 sidecar）
//
// 设计: docs/plans/2026-09-01-curve-map-temp-table-design.md "sidecar 容器格式"（两阶段契约 ①）
// 格式: 96B 超级头（native 小端, 同 CMPU 惯例）＋ tierCount×32B 索引 ＋ 各档 CMPU blob 拼接。
// 红线: CRC/边界先于一切——load 校验 magic/version/tierCount≤1024/每索引项 offset+length≤size,
//       tierBytes 拷出→逐档 crc32 验证通过才返回（懒校验, 未触碰档不付费）;
//       整文件 sha256 只存标定 JSON（不自指）, 头内无 sha 字段;
//       nearestTier 等距 tie-break 偏低温侧（<=）。

#pragma once

#include <opencv2/core.hpp>
#include <cstdint>
#include <cstddef>
#include <optional>
#include <vector>
#include "common/scanner_api.h"

namespace calib {

/// 魔数 "CMTT"（内存字节序 'C','M','T','T'）
static constexpr uint32_t kCmttMagic = 0x54544D43u;
static constexpr uint32_t kCmttVersion = 1;
/// 档数上限（防 tierCount 字段损坏导致的索引区越界分配/读）
static constexpr uint32_t kCmttMaxTierCount = 1024;

/// 索引项 flags 位定义
static constexpr uint8_t kCmttFlagReference = 0x01;   // bit0 参考档
static constexpr uint8_t kCmttFlagMissing   = 0x02;   // bit1 missing（无载荷）
static constexpr uint8_t kCmttFlagClamped   = 0x04;   // bit2 clamped（源表边缘兜底）

/// CMTT 超级头（96B 固定布局, native 小端）
struct CmttHeader {
    uint32_t magic = kCmttMagic;
    uint32_t version = kCmttVersion;
    uint32_t tierCount = 0;
    int32_t  tempBaseX10 = 0;      // 参考温度×10 定点（如 250 = 25.0C）
    uint16_t tempStepX10 = 0;      // 档距×10（=5 即 0.5C）
    uint16_t reservedU16 = 0;
    float    rowStep = 0.f;
    float    depthMin = 0.f;
    float    depthMax = 0.f;
    int32_t  roiX = 0;             // 冻结栅格: 参考档 ROI
    int32_t  roiY = 0;
    int32_t  roiW = 0;
    int32_t  roiH = 0;
    uint32_t rowMin = 0;
    uint32_t rowCount = 0;
    uint64_t gridHash = 0;         // 冻结栅格身份哈希
    uint8_t  reserved[32] = {};    // 留版本演进/未来 delta 格式位
};
static_assert(sizeof(CmttHeader) == 96, "CMTT header must be fixed 96B (no sha field)");

/// 索引项（32B 固定布局; 指定字段序下 uint64 offset 非自然对齐 → pack(1)）
#pragma pack(push, 1)
struct CmttTierMeta {
    int32_t  temperatureX10 = 0;   // 档温×10 定点
    uint64_t offset = 0;           // 载荷偏移（相对文件头; missing 档 =0）
    uint32_t length = 0;           // 载荷字节数（missing 档 =0）
    uint32_t crc32 = 0;            // 载荷 IEEE crc32（missing 档 =0）
    uint32_t entryCount = 0;
    uint8_t  flags = 0;            // kCmttFlag* 位组合
    uint8_t  pad[7] = {};
};
#pragma pack(pop)
static_assert(sizeof(CmttTierMeta) == 32, "CMTT index entry must be 32B");

/// 生成器: setHeader → 逐档 addTier → build（头＋索引＋载荷, offset 回填）
class SCANNER_API CmttBuilder {
public:
    void setHeader(int32_t tempBaseX10, uint16_t tempStepX10,
                   float rowStep, float depthMin, float depthMax,
                   const cv::Rect& frozenRoi, uint32_t rowMin, uint32_t rowCount,
                   uint64_t gridHash);
    void addTier(int32_t temperatureX10, std::vector<uint8_t> blob,
                 uint32_t entryCount, uint8_t flags);

    /// 组装完整容器字节流。tierCount 以实际 addTier 数为准。
    std::vector<uint8_t> build() const;

private:
    struct TierRec {
        CmttTierMeta meta;
        std::vector<uint8_t> blob;
    };
    CmttHeader header_{};
    std::vector<TierRec> tiers_;
};

/// 读取器: 非拥有视图——load 不拷贝、不保留所有权之外的指针用途,
/// 调用方须保证 data 在 Reader 生命周期内驻留（整块内存/文件映射）。
class SCANNER_API CmttReader {
public:
    static constexpr const char* kLogTag = "12c-CmttContainer";

    /// 校验 magic/version/tierCount≤1024/索引区完整/每项 offset+length≤size。
    /// 任一不过 → 返回 false 并复位, 绝不越界读。
    bool load(const uint8_t* data, size_t size);

    /// 取档载荷: 拷出 → crc32 验证 → 通过才返回。
    /// 任何失败（未 load/越界/missing/crc 不符）返回 nullopt——绝不把未验证字节交给下游。
    std::optional<std::vector<uint8_t>> tierBytes(size_t idx) const;

    /// 就近档: |t−档温| 最小; 等距取低温侧（<=）。未 load/无档 → nullopt。
    std::optional<size_t> nearestTier(double tempC) const;

    size_t tierCount() const { return tiers_.size(); }
    float rowStep() const { return header_.rowStep; }
    uint32_t entryCount(size_t idx) const;
    bool isMissing(size_t idx) const;
    bool isReference(size_t idx) const;
    bool isClamped(size_t idx) const;
    const CmttHeader& header() const { return header_; }

private:
    bool loaded_ = false;
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    CmttHeader header_{};
    std::vector<CmttTierMeta> tiers_;
};

/// 表驱动 IEEE crc32（自包含, 无第三方依赖）
SCANNER_API uint32_t cmttCrc32(const uint8_t* data, size_t n);

} // namespace calib
