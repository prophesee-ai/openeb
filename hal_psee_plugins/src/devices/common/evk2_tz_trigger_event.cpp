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

#include "devices/common/evk2_tz_trigger_event.h"
#include "utils/register_map.h"

namespace Metavision {

Evk2TzTriggerEvent::Evk2TzTriggerEvent(const std::shared_ptr<RegisterMap> &register_map, const std::string &prefix,
                                       const std::shared_ptr<TzDevice> tzDev) :
    register_map_(register_map), prefix_(prefix), tzDev_(tzDev) {
    for (const auto &id : chan_ids_) {
        disable(static_cast<int>(id));
    }
}

bool Evk2TzTriggerEvent::is_valid_id(uint32_t channel) {
    for (const auto &id : chan_ids_) {
        if (static_cast<uint32_t>(id) == channel) {
            return true;
        }
    }
    return false;
}

bool Evk2TzTriggerEvent::enable(uint32_t channel) {
    bool valid = is_valid_id(channel);

    if (valid) {
        (*register_map_)[prefix_ + "SYSTEM_MONITOR/EXT_TRIGGERS/ENABLE"]["TRIGGER_" + std::to_string(channel)]
            .write_value(1);
    }
    return valid;
}

bool Evk2TzTriggerEvent::disable(uint32_t channel) {
    bool valid = is_valid_id(channel);

    if (valid) {
        (*register_map_)[prefix_ + "SYSTEM_MONITOR/EXT_TRIGGERS/ENABLE"]["TRIGGER_" + std::to_string(channel)]
            .write_value(0);
    }
    return valid;
}

bool Evk2TzTriggerEvent::is_enabled(uint32_t channel) {
    bool valid = is_valid_id(channel);
    long value = 0;

    if (valid) {
        value = (*register_map_)[prefix_ + "SYSTEM_MONITOR/EXT_TRIGGERS/ENABLE"]["TRIGGER_" + std::to_string(channel)]
                    .read_value();
    }
    return valid && (value == 1);
}

} // namespace Metavision
