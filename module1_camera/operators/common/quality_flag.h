#pragma once

#include <string>

namespace calib {

enum class QualityFlag {
    Normal = 0,
    Degraded = 1,
    Warning = 2
};

inline const char* qualityFlagToString(QualityFlag f) {
    switch (f) {
    case QualityFlag::Normal: return "Normal";
    case QualityFlag::Degraded: return "Degraded";
    case QualityFlag::Warning: return "Warning";
    default: return "Unknown";
    }
}

inline QualityFlag maxQuality(QualityFlag a, QualityFlag b) {
    return static_cast<int>(a) >= static_cast<int>(b) ? a : b;
}

} // namespace calib
