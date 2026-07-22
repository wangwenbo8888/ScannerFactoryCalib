#include "ICameraSource.h"
#include "SimulatedCameraSource.h"
#include "HikvisionCameraSource.h"
#include "GalaxyCameraSource.h"

#include <spdlog/spdlog.h>

namespace fc::gui {

std::vector<std::string> availableSourceTypes() {
    std::vector<std::string> types;
    types.push_back("simulated");
#ifdef FC_HAVE_MVS
    types.push_back("hikvision");
#endif
#ifdef FC_HAVE_GALAXY
    types.push_back("galaxy");
#endif
    return types;
}

std::unique_ptr<ICameraSource> createCameraSource(
    const std::string& sourceType,
    const std::string& folder)
{
    if (sourceType == "simulated") {
        return std::make_unique<SimulatedCameraSource>(folder);
    }
#ifdef FC_HAVE_MVS
    if (sourceType == "hikvision") {
        return std::make_unique<HikvisionCameraSource>();
    }
#endif
#ifdef FC_HAVE_GALAXY
    if (sourceType == "galaxy") {
        return std::make_unique<GalaxyCameraSource>();
    }
#endif
    spdlog::warn("[createCameraSource] unknown/unsupported source type: {}", sourceType);
    return nullptr;
}

std::vector<std::string> ICameraSource::enumerateDevices(const std::string& sourceType) {
    if (sourceType == "simulated") {
        return { "[0] simulated (folder based)" };
    }
#ifdef FC_HAVE_MVS
    if (sourceType == "hikvision") {
        return HikvisionCameraSource::enumerateDevices();
    }
#endif
#ifdef FC_HAVE_GALAXY
    if (sourceType == "galaxy") {
        return GalaxyCameraSource::enumerateDevices();
    }
#endif
    return {};
}

}  // namespace fc::gui
