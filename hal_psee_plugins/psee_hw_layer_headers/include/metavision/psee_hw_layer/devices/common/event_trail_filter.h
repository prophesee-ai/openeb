/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A.                                                                                       *
 *                                                                                                                    *
 * Licensed under the Apache License, Version 2.0 (the "License");                                                    *
 * you may not use this file except in compliance with the License.                                                   *
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0                                 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed   *
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                      *
 * See the License for the specific language governing permissions and limitations under the License.                 *
 **********************************************************************************************************************/

#ifndef METAVISION_HAL_EVENT_TRAIL_FILTER_H
#define METAVISION_HAL_EVENT_TRAIL_FILTER_H

#include <string>
#include <map>

#include "metavision/hal/facilities/i_event_trail_filter_module.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_regmap_device.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

namespace Metavision {

class EventTrailFilter : public I_EventTrailFilterModule {
public:
    EventTrailFilter(const std::shared_ptr<TzDeviceWithRegmap> &dev, const I_HW_Identification::SensorInfo &sensor_info,
                     const std::string &sensor_prefix);

    virtual std::set<Type> get_available_types() const override;
    virtual bool enable(bool state) override;
    virtual bool is_enabled() const override;
    virtual bool set_type(Type type) override;
    virtual Type get_type() const override;
    virtual bool set_threshold(uint32_t threshold) override;
    virtual uint32_t get_threshold() const override;
    virtual uint32_t get_max_supported_threshold() const override;
    virtual uint32_t get_min_supported_threshold() const override;

private:
    std::shared_ptr<TzDeviceWithRegmap> dev_;

    static constexpr uint32_t THESHOLD_MS_DEFAULT = 10;
    static constexpr Type FILTERING_TYPE_DEFAULT  = Type::TRAIL;

    std::string sensor_prefix_, stc_prefix_, trail_prefix_;
    uint32_t threshold_ms_{THESHOLD_MS_DEFAULT};
    Type filtering_type_{FILTERING_TYPE_DEFAULT};
    bool enabled_ = false;

    bool is_sensor_saphir;
    bool is_stc_improved;
    std::set<I_EventTrailFilterModule::Type> modes_;

    std::map<const int, std::map<const std::string, const int>> threshold_params_;

    RegisterMap &regmap();
};

} // namespace Metavision

#endif // METAVISION_HAL_EVENT_TRAIL_FILTER_H
