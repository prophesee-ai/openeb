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

#include "metavision/psee_hw_layer/devices/gen41/gen41_tz_trigger_event.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

using vfield = std::map<std::string, uint32_t>;

namespace Metavision {

Gen41TzTriggerEvent::Gen41TzTriggerEvent(const std::shared_ptr<RegisterMap> &register_map, const std::string &prefix,
                                         const std::shared_ptr<TzDevice> tzDev) :
    register_map_(register_map), prefix_(prefix), tzDev_(tzDev), chan_map_({{Channel::Main, 0}}) {
    disable(Channel::Main);
}

bool Gen41TzTriggerEvent::enable(const Channel &channel) {
    auto it = chan_map_.find(channel);
    if (it == chan_map_.end()) {
        return false;
    }
    (*register_map_)[prefix_ + "dig_pad2_ctrl"]["Reserved_15_12"].write_value(0b1111);
    (*register_map_)[prefix_ + "edf/Reserved_7004"]["Reserved_10"].write_value(1);
    return true;
}

bool Gen41TzTriggerEvent::disable(const Channel &channel) {
    auto it = chan_map_.find(channel);
    if (it == chan_map_.end()) {
        return false;
    }
    (*register_map_)[prefix_ + "edf/Reserved_7004"]["Reserved_10"].write_value(0);
    return true;
}

bool Gen41TzTriggerEvent::is_enabled(const Channel &channel) const {
    auto it = chan_map_.find(channel);
    if (it == chan_map_.end()) {
        return false;
    }

    long value  = 0;
    long value2 = 0;
    value       = (*register_map_)[prefix_ + "dig_pad2_ctrl"]["Reserved_15_12"].read_value();
    value2      = (*register_map_)[prefix_ + "edf/Reserved_7004"]["Reserved_10"].read_value();
    return (value == 0xF) && (value2 == 1);
}

std::map<I_TriggerIn::Channel, short> Gen41TzTriggerEvent::get_available_channels() const {
    return chan_map_;
}
} // namespace Metavision
