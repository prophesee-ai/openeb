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

#include "devices/gen41/gen41_tz_trigger_event.h"
#include "utils/register_map.h"

using vfield = std::map<std::string, uint32_t>;

namespace Metavision {

Gen41TzTriggerEvent::Gen41TzTriggerEvent(const std::shared_ptr<RegisterMap> &register_map, const std::string &prefix,
                                         const std::shared_ptr<TzDevice> tzDev) :
    register_map_(register_map), prefix_(prefix), tzDev_(tzDev) {
    for (const auto &id : chan_ids_) {
        disable(static_cast<int>(id));
    }
}

bool Gen41TzTriggerEvent::is_valid_id(uint32_t channel) {
    for (const auto &id : chan_ids_) {
        if (static_cast<uint32_t>(id) == channel) {
            return true;
        }
    }
    return false;
}

bool Gen41TzTriggerEvent::enable(uint32_t channel) {
    bool valid = is_valid_id(channel);

    if (valid) {
        (*register_map_)[prefix_ + "dig_pad2_ctrl"]["Reserved_15_12"].write_value(0b1111);
        (*register_map_)[prefix_ + "edf/Reserved_7004"]["Reserved_10"].write_value(1);
    }
    return valid;
}

bool Gen41TzTriggerEvent::disable(uint32_t channel) {
    bool valid = is_valid_id(channel);

    if (valid) {
        (*register_map_)[prefix_ + "edf/Reserved_7004"]["Reserved_10"].write_value(0);
    }
    return valid;
}

bool Gen41TzTriggerEvent::is_enabled(uint32_t channel) {
    bool valid  = is_valid_id(channel);
    long value  = 0;
    long value2 = 0;

    if (valid) {
        value  = (*register_map_)[prefix_ + "dig_pad2_ctrl"]["Reserved_15_12"].read_value();
        value2 = (*register_map_)[prefix_ + "edf/Reserved_7004"]["Reserved_10"].read_value();
    }
    return valid && (value == 0xF) && (value2 == 1);
}

} // namespace Metavision
