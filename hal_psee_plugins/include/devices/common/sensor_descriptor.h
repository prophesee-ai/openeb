#ifndef METAVISION_HAL_SENSOR_DESCRIPTOR_H
#define METAVISION_HAL_SENSOR_DESCRIPTOR_H
#include <functional>
#include "metavision/psee_hw_layer/utils/regmap_data.h"
#include "metavision/hal/utils/device_builder.h"
#include "metavision/hal/utils/device_config.h"
#include "metavision/psee_hw_layer/utils/register_map.h"
#include "metavision/hal/facilities/i_hw_identification.h"


namespace Metavision {

using FacilitySpawnerFunction = std::function<void(DeviceBuilder&, const DeviceConfig&, I_HW_Identification::SensorInfo, std::shared_ptr<RegisterMap>)>;

struct MatchPattern {
    uint32_t addr;
    uint32_t value;
    uint32_t mask;
};

struct SensorDescriptor{
    RegmapElement* regmap;
    size_t size;
    FacilitySpawnerFunction spawn_facilities;
    std::vector<MatchPattern> opt_match_list;
    I_HW_Identification::SensorInfo info;
    std::string encoding_format;
} ;

}
# endif // METAVISION_HAL_SENSOR_DESCRIPTOR_H
