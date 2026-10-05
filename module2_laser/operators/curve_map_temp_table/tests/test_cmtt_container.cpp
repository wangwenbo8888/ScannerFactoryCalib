// test_cmtt_container.cpp — CMTT 温度表容器（curve_map 61 档 sidecar）读写器单测
//
// 覆盖: 全字段 roundtrip / 索引损坏拒绝（CRC 与边界先于一切, 防 OOB）/
//       载荷损坏 CRC 拒绝 / 截断拒绝 / nearestTier tie-break 偏低温（<=）/
//       头 96B 固定布局（无 sha 字段——整文件 sha256 只存标定 JSON）。
// 设计: docs/plans/2026-09-01-curve-map-temp-table-design.md "sidecar 容器格式"。

#include <gtest/gtest.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>

#include "../cmtt_container.h"

using namespace calib;

namespace {

// 契约布局常量（测试独立复述, 与实现互为对拍）
constexpr size_t kHeaderSize = 96;
constexpr size_t kIndexEntrySize = 32;

// 标准样例表: 3 档（参考档/正常档/missing 档）, 与设计契约同构
std::vector<uint8_t> buildSampleTable() {
    CmttBuilder b;
    b.setHeader(250 /*tempBaseX10=25.0C*/, 5 /*stepX10=0.5*/,
                0.7f, 100.f, 700.f, cv::Rect(1, 2, 3, 4), 1234, 2194, 0xABCDULL);
    std::vector<uint8_t> blob0(100, 0x11), blob1(200, 0x22), blob2;   // tier2 = missing
    b.addTier(250, blob0, 1000, 0x01 /*bit0 参考档*/);
    b.addTier(255, blob1, 2000, 0x00);
    b.addTier(260, blob2, 0, 0x02 /*bit1 missing*/);
    return b.build();
}

} // namespace

TEST(Cmtt, RoundtripAllFields) {
    auto bytes = buildSampleTable();

    // 布局: 96B 头 + 3x32B 索引 + 载荷(100+200, missing 档无载荷)
    ASSERT_EQ(bytes.size(), kHeaderSize + 3 * kIndexEntrySize + 100 + 200);
    EXPECT_EQ(bytes[0], 'C');
    EXPECT_EQ(bytes[1], 'M');
    EXPECT_EQ(bytes[2], 'T');
    EXPECT_EQ(bytes[3], 'T');

    CmttReader r;
    ASSERT_TRUE(r.load(bytes.data(), bytes.size()));
    EXPECT_EQ(r.tierCount(), 3u);
    EXPECT_FLOAT_EQ(r.rowStep(), 0.7f);

    const CmttHeader& h = r.header();
    EXPECT_EQ(h.version, 1u);
    EXPECT_EQ(h.tierCount, 3u);
    EXPECT_EQ(h.tempBaseX10, 250);
    EXPECT_EQ(h.tempStepX10, 5u);
    EXPECT_FLOAT_EQ(h.depthMin, 100.f);
    EXPECT_FLOAT_EQ(h.depthMax, 700.f);
    EXPECT_EQ(h.roiX, 1);
    EXPECT_EQ(h.roiY, 2);
    EXPECT_EQ(h.roiW, 3);
    EXPECT_EQ(h.roiH, 4);
    EXPECT_EQ(h.rowMin, 1234u);
    EXPECT_EQ(h.rowCount, 2194u);
    EXPECT_EQ(h.gridHash, 0xABCDULL);

    auto v0 = r.tierBytes(0);                       // 触发该档 crc32 校验
    ASSERT_TRUE(v0.has_value());
    ASSERT_EQ(v0->size(), 100u);
    EXPECT_EQ((*v0)[0], 0x11);
    EXPECT_EQ((*v0)[99], 0x11);

    auto v1 = r.tierBytes(1);
    ASSERT_TRUE(v1.has_value());
    EXPECT_EQ(v1->size(), 200u);
    EXPECT_EQ((*v1)[0], 0x22);

    EXPECT_FALSE(r.tierBytes(2).has_value());       // missing 档 → nullopt

    EXPECT_EQ(r.entryCount(0), 1000u);
    EXPECT_EQ(r.entryCount(1), 2000u);
    EXPECT_EQ(r.entryCount(2), 0u);
    EXPECT_TRUE(r.isMissing(2));
    EXPECT_FALSE(r.isMissing(0));
    EXPECT_TRUE(r.isReference(0));
    EXPECT_FALSE(r.isReference(1));
    EXPECT_FALSE(r.isClamped(0));
    EXPECT_FALSE(r.isClamped(1));
}

TEST(Cmtt, CorruptedIndexRejected) {
    // 设计硬规则 #7: CRC/边界先于一切——索引 offset 字段损坏 → load 拒绝, 不得越界读
    auto bytes = buildSampleTable();

    // 档 0 索引项 offset 字段（entry+4 起 8B 小端）最高字节翻转 → offset 巨大越界
    bytes[kHeaderSize + 11] ^= 0x80;
    CmttReader r;
    EXPECT_FALSE(r.load(bytes.data(), bytes.size()));

    // 副作用: load 失败后 reader 不得暴露任何档
    EXPECT_EQ(r.tierCount(), 0u);
    EXPECT_FALSE(r.tierBytes(0).has_value());
}

TEST(Cmtt, CorruptedPayloadRejectedByCrc) {
    auto bytes = buildSampleTable();
    const size_t payloadBase = kHeaderSize + 3 * kIndexEntrySize;

    bytes[payloadBase + 50] ^= 0xFF;                // 翻转档 0 载荷 1 字节
    CmttReader r;
    ASSERT_TRUE(r.load(bytes.data(), bytes.size()));   // 索引完好 → load 通过（懒校验）
    EXPECT_FALSE(r.tierBytes(0).has_value());          // crc 不符 → nullopt, 进程不崩
    auto v1 = r.tierBytes(1);                          // 未触碰档不受影响
    ASSERT_TRUE(v1.has_value());
    EXPECT_EQ(v1->size(), 200u);
}

TEST(Cmtt, TruncatedFileRejected) {
    auto bytes = buildSampleTable();

    CmttReader r;
    {
        std::vector<uint8_t> t(bytes.begin(), bytes.begin() + 50);      // 头都没给全
        EXPECT_FALSE(r.load(t.data(), t.size()));
    }
    {
        std::vector<uint8_t> t(bytes.begin(), bytes.begin() + (kHeaderSize + 3 * kIndexEntrySize - 1));  // 索引缺 1B
        EXPECT_FALSE(r.load(t.data(), t.size()));
    }
    {
        std::vector<uint8_t> t(bytes.begin(), bytes.end() - 1);         // 载荷缺尾 1B → offset+length 越界
        EXPECT_FALSE(r.load(t.data(), t.size()));
    }
    EXPECT_FALSE(r.load(nullptr, 0));
}

TEST(Cmtt, TieBreakLowerTemp) {
    // 设计: 偏低温（<=）——档温 {25.0, 25.5}
    CmttBuilder b;
    b.setHeader(250, 5, 0.7f, 100.f, 700.f, cv::Rect(0, 0, 8, 8), 0, 1, 1ULL);
    std::vector<uint8_t> blob0(10, 0x33), blob1(20, 0x44);
    b.addTier(250, blob0, 10, 0x01);
    b.addTier(255, blob1, 20, 0x00);
    auto bytes = b.build();

    CmttReader r;
    ASSERT_TRUE(r.load(bytes.data(), bytes.size()));
    auto n0 = r.nearestTier(25.24);                 // 距 25.0 更近
    ASSERT_TRUE(n0.has_value());
    EXPECT_EQ(*n0, 0u);
    auto nTie = r.nearestTier(25.25);               // 精确等距 → 偏低温侧（<=）
    ASSERT_TRUE(nTie.has_value());
    EXPECT_EQ(*nTie, 0u);
    auto n1 = r.nearestTier(25.26);                 // 距 25.5 更近
    ASSERT_TRUE(n1.has_value());
    EXPECT_EQ(*n1, 1u);
}

TEST(Cmtt, HeaderNoShaField) {
    // 设计: 整文件 sha256 只存标定 JSON（不自指）, 头为固定 96B 布局（含 32B reserved）
    static_assert(sizeof(CmttHeader) == 96, "header must be fixed 96B");
    static_assert(offsetof(CmttHeader, reserved) == 64, "reserved must be last 32B");
    static_assert(sizeof(CmttTierMeta) == 32, "index entry must be 32B");
    EXPECT_EQ(sizeof(CmttHeader), size_t{96});
    EXPECT_EQ(sizeof(CmttTierMeta), size_t{32});
    EXPECT_EQ(sizeof(CmttHeader::reserved), size_t{32});
}

TEST(Cmtt, Crc32KnownVector) {
    // IEEE crc32 标准校验向量
    const uint8_t v[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
    EXPECT_EQ(cmttCrc32(v, sizeof v), 0xCBF43926u);
    EXPECT_EQ(cmttCrc32(nullptr, 0), 0u);
}

TEST(Cmtt, BadMagicOrVersionOrTierCountRejected) {
    auto bytes = buildSampleTable();

    {
        auto t = bytes; t[0] = 'X';                  // magic
        CmttReader r;
        EXPECT_FALSE(r.load(t.data(), t.size()));
    }
    {
        auto t = bytes; t[4] = 2;                    // version
        CmttReader r;
        EXPECT_FALSE(r.load(t.data(), t.size()));
    }
    {
        auto t = bytes; t[11] = 0x7F;                // tierCount = 0x7F000003 > 1024
        CmttReader r;
        EXPECT_FALSE(r.load(t.data(), t.size()));
    }
}

TEST(Cmtt, OutOfRangeAccessorsSafe) {
    auto bytes = buildSampleTable();
    CmttReader r;
    ASSERT_TRUE(r.load(bytes.data(), bytes.size()));

    EXPECT_FALSE(r.tierBytes(3).has_value());
    EXPECT_EQ(r.entryCount(99), 0u);
    EXPECT_FALSE(r.isMissing(99));
    EXPECT_FALSE(r.isReference(99));
    EXPECT_FALSE(r.isClamped(99));

    CmttReader empty;
    EXPECT_FALSE(empty.nearestTier(25.0).has_value());   // 未 load → nullopt
}

TEST(Cmtt, EmptyTierTableAccepted) {
    // 0 档空索引表: 头即全部 → load 成功, 无档可查
    CmttBuilder b;
    b.setHeader(250, 5, 0.7f, 100.f, 700.f, cv::Rect(0, 0, 8, 8), 0, 1, 1ULL);
    auto bytes = b.build();
    ASSERT_EQ(bytes.size(), kHeaderSize);            // 空索引空载荷
    CmttReader r;
    EXPECT_TRUE(r.load(bytes.data(), bytes.size()));
    EXPECT_EQ(r.tierCount(), size_t{0});
    EXPECT_FALSE(r.nearestTier(25.0).has_value());
    EXPECT_FALSE(r.tierBytes(0).has_value());
}

TEST(Cmtt, SingleTierNearestBothSides) {
    // 单档: 低温侧/命中/高温侧三向都只能落在 0 档
    CmttBuilder b;
    b.setHeader(250, 5, 0.7f, 100.f, 700.f, cv::Rect(0, 0, 8, 8), 0, 1, 1ULL);
    std::vector<uint8_t> blob(16, 0x55);
    b.addTier(250, blob, 16, 0x01);
    auto bytes = b.build();

    CmttReader r;
    ASSERT_TRUE(r.load(bytes.data(), bytes.size()));
    for (double t : {24.9, 25.0, 25.1}) {
        auto n = r.nearestTier(t);
        ASSERT_TRUE(n.has_value());
        EXPECT_EQ(*n, 0u);
    }
}

TEST(Cmtt, CorruptedIndexLengthOrCrcRejected) {
    auto bytes = buildSampleTable();

    {   // length 字段（entry+12..+16）最高字节翻转 → offset+length 越界 → load 拒绝
        auto t = bytes;
        t[kHeaderSize + 15] ^= 0x80;
        CmttReader r;
        EXPECT_FALSE(r.load(t.data(), t.size()));
    }
    {   // crc32 字段（entry+16..+20）翻转 → 懒校验: load 通过, tierBytes(0) nullopt
        auto t = bytes;
        t[kHeaderSize + 16] ^= 0xFF;
        CmttReader r;
        ASSERT_TRUE(r.load(t.data(), t.size()));
        EXPECT_FALSE(r.tierBytes(0).has_value());
        auto v1 = r.tierBytes(1);                    // 未触碰档不受影响
        ASSERT_TRUE(v1.has_value());
        EXPECT_EQ(v1->size(), 200u);
    }
}
