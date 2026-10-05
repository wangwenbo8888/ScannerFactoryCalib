// cmtt_container.cpp — CMTT 温度表容器读写器实现
//
// 设计: docs/plans/2026-09-01-curve-map-temp-table-design.md "sidecar 容器格式"

#include "cmtt_container.h"
#include "common/calib_logging.h"

#include <array>
#include <cmath>
#include <cstring>
#include <utility>

namespace calib {

CALIB_DEFINE_LOG_TAG(12, CmttContainer);

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态（Builder 一次性组装; Reader 只读视图）
// 并发策略: 每实例非线程安全; cmttCrc32 纯函数可并发
// ==============================

void CmttBuilder::setHeader(int32_t tempBaseX10, uint16_t tempStepX10,
                            float rowStep, float depthMin, float depthMax,
                            const cv::Rect& frozenRoi, uint32_t rowMin, uint32_t rowCount,
                            uint64_t gridHash) {
    header_ = CmttHeader{};
    header_.magic = kCmttMagic;
    header_.version = kCmttVersion;
    header_.tempBaseX10 = tempBaseX10;
    header_.tempStepX10 = tempStepX10;
    header_.rowStep = rowStep;
    header_.depthMin = depthMin;
    header_.depthMax = depthMax;
    header_.roiX = frozenRoi.x;
    header_.roiY = frozenRoi.y;
    header_.roiW = frozenRoi.width;
    header_.roiH = frozenRoi.height;
    header_.rowMin = rowMin;
    header_.rowCount = rowCount;
    header_.gridHash = gridHash;
}

void CmttBuilder::addTier(int32_t temperatureX10, std::vector<uint8_t> blob,
                          uint32_t entryCount, uint8_t flags) {
    TierRec rec;
    rec.meta.temperatureX10 = temperatureX10;
    rec.meta.entryCount = entryCount;
    rec.meta.flags = flags;
    rec.blob = std::move(blob);
    tiers_.push_back(std::move(rec));
}

std::vector<uint8_t> CmttBuilder::build() const {
    CmttHeader h = header_;
    h.magic = kCmttMagic;
    h.version = kCmttVersion;
    h.tierCount = static_cast<uint32_t>(tiers_.size());

    // 索引: offset 回填（missing 档 length=0/offset=0/crc=0）
    std::vector<CmttTierMeta> idx(tiers_.size());
    size_t cursor = sizeof(CmttHeader) + tiers_.size() * sizeof(CmttTierMeta);
    for (size_t i = 0; i < tiers_.size(); ++i) {
        idx[i] = tiers_[i].meta;
        if (idx[i].flags & kCmttFlagMissing) {
            idx[i].offset = 0;
            idx[i].length = 0;
            idx[i].crc32 = 0;
        } else {
            idx[i].offset = cursor;
            idx[i].length = static_cast<uint32_t>(tiers_[i].blob.size());
            idx[i].crc32 = cmttCrc32(tiers_[i].blob.data(), tiers_[i].blob.size());
            cursor += tiers_[i].blob.size();
        }
    }

    std::vector<uint8_t> out(cursor);
    std::memcpy(out.data(), &h, sizeof(CmttHeader));
    if (!idx.empty())
        std::memcpy(out.data() + sizeof(CmttHeader), idx.data(), idx.size() * sizeof(CmttTierMeta));
    for (size_t i = 0; i < tiers_.size(); ++i) {
        if (!(idx[i].flags & kCmttFlagMissing) && !tiers_[i].blob.empty())
            std::memcpy(out.data() + idx[i].offset, tiers_[i].blob.data(), tiers_[i].blob.size());
    }
    return out;
}

bool CmttReader::load(const uint8_t* data, size_t size) {
    loaded_ = false;
    data_ = nullptr;
    size_ = 0;
    header_ = CmttHeader{};
    tiers_.clear();

    if (!data || size < sizeof(CmttHeader)) {
        spdlog::error("{} load rejected: size {} < header {} or null data",
                      kLogTag, size, sizeof(CmttHeader));
        return false;
    }

    CmttHeader h;
    std::memcpy(&h, data, sizeof(CmttHeader));
    if (h.magic != kCmttMagic || h.version != kCmttVersion) {
        spdlog::error("{} load rejected: bad magic 0x{:08X} or version {}",
                      kLogTag, h.magic, h.version);
        return false;
    }
    if (h.tierCount > kCmttMaxTierCount) {
        spdlog::error("{} load rejected: tierCount {} > max {}",
                      kLogTag, h.tierCount, kCmttMaxTierCount);
        return false;
    }

    const size_t indexOff = sizeof(CmttHeader);
    const size_t indexSize = static_cast<size_t>(h.tierCount) * sizeof(CmttTierMeta);
    if (size < indexOff + indexSize) {
        spdlog::error("{} load rejected: size {} < header+index {}",
                      kLogTag, size, indexOff + indexSize);
        return false;
    }

    std::vector<CmttTierMeta> idx(h.tierCount);
    if (!idx.empty())
        std::memcpy(idx.data(), data + indexOff, indexSize);

    // 硬规则 #7: 边界先于一切——CurveMapTable::Load 对索引值损坏无防御, 容器层必须拦下
    for (size_t i = 0; i < idx.size(); ++i) {
        const CmttTierMeta& t = idx[i];
        if (t.offset > size || t.length > size - t.offset) {
            spdlog::error("{} load rejected: tier {} offset {}+length {} > size {}",
                          kLogTag, i, t.offset, t.length, size);
            return false;
        }
    }

    header_ = h;
    tiers_ = std::move(idx);
    data_ = data;
    size_ = size;
    loaded_ = true;
    return true;
}

std::optional<std::vector<uint8_t>> CmttReader::tierBytes(size_t idx) const {
    if (!loaded_ || idx >= tiers_.size()) {
        spdlog::warn("{} tierBytes: not loaded or idx {} >= tierCount {}",
                     kLogTag, idx, tiers_.size());
        return std::nullopt;
    }
    const CmttTierMeta& t = tiers_[idx];
    if (t.flags & kCmttFlagMissing) {
        spdlog::warn("{} tierBytes: tier {} is missing", kLogTag, idx);
        return std::nullopt;
    }
    if (t.offset > size_ || t.length > size_ - t.offset) {   // 防御性复检
        spdlog::error("{} tierBytes: tier {} offset {}+length {} > size {}",
                      kLogTag, idx, t.offset, t.length, size_);
        return std::nullopt;
    }

    std::vector<uint8_t> out(t.length);
    if (t.length > 0)
        std::memcpy(out.data(), data_ + t.offset, t.length);
    const uint32_t crc = cmttCrc32(out.data(), out.size());
    if (crc != t.crc32) {
        spdlog::error("{} tierBytes: tier {} crc mismatch (stored 0x{:08X}, actual 0x{:08X})",
                      kLogTag, idx, t.crc32, crc);
        return std::nullopt;
    }
    return out;
}

std::optional<size_t> CmttReader::nearestTier(double tempC) const {
    if (!loaded_ || tiers_.empty())
        return std::nullopt;

    const double q = tempC * 10.0;   // 与 temperatureX10 同尺度比较
    size_t best = 0;
    double bestAbs = std::fabs(q - static_cast<double>(tiers_[0].temperatureX10));
    for (size_t i = 1; i < tiers_.size(); ++i) {
        const double a = std::fabs(q - static_cast<double>(tiers_[i].temperatureX10));
        // 等距取低温侧（<=）: 仅严格更近, 或等距且档温更低时替换
        if (a < bestAbs ||
            (a == bestAbs && tiers_[i].temperatureX10 < tiers_[best].temperatureX10)) {
            best = i;
            bestAbs = a;
        }
    }
    return best;
}

uint32_t CmttReader::entryCount(size_t idx) const {
    if (!loaded_ || idx >= tiers_.size())
        return 0;
    return tiers_[idx].entryCount;
}

bool CmttReader::isMissing(size_t idx) const {
    if (!loaded_ || idx >= tiers_.size())
        return false;
    return (tiers_[idx].flags & kCmttFlagMissing) != 0;
}

bool CmttReader::isReference(size_t idx) const {
    if (!loaded_ || idx >= tiers_.size())
        return false;
    return (tiers_[idx].flags & kCmttFlagReference) != 0;
}

bool CmttReader::isClamped(size_t idx) const {
    if (!loaded_ || idx >= tiers_.size())
        return false;
    return (tiers_[idx].flags & kCmttFlagClamped) != 0;
}

uint32_t cmttCrc32(const uint8_t* data, size_t n) {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
        return t;
    }();

    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i)
        crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

} // namespace calib
