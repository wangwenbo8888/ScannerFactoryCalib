#pragma once

#define SCANNER_VERSION_MAJOR 1
#define SCANNER_VERSION_MINOR 0
#define SCANNER_VERSION_PATCH 0

namespace calib {

enum class OperatorType {
    CPU,
    CUDA,
    Hybrid
};

struct OperatorInfo {
    const char* name;
    int versionMajor;
    int versionMinor;
    OperatorType type;
};

inline const char* getScannerVersion() {
    return "1.0.0";
}

} // namespace calib
