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

#include "metavision/psee_hw_layer/devices/common/evk2_tz_trigger_event.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

namespace Metavision {

Evk2TzTriggerEvent::Evk2TzTriggerEvent(const std::shared_ptr<RegisterMap> &register_map, const std::string &prefix,
                                       const std::shared_ptr<TzDevice> tzDev) :
    register_map_(register_map), prefix_(prefix), tzDev_(tzDev), chan_map_{{Channel::Main, 1}, {Channel::Loopback, 3}} {
    for (const auto &p : chan_map_) {
        disable(p.first);
    }
}

bool Evk2TzTriggerEvent::enable(const Channel &channel) {
    auto it = chan_map_.find(channel);
    if (it == chan_map_.end()) {
        return false;
    }
    (*register_map_)[prefix_ + "SYSTEM_MONITOR/EXT_TRIGGERS/ENABLE"]["TRIGGER_" + std::to_string(it->second)]
        .write_value(1);
    return true;
}

bool Evk2TzTriggerEvent::disable(const Channel &channel) {
    auto it = chan_map_.find(channel);
    if (it == chan_map_.end()) {
        return false;
    }
    (*register_map_)[prefix_ + "SYSTEM_MONITOR/EXT_TRIGGERS/ENABLE"]["TRIGGER_" + std::to_string(it->second)]
        .write_value(0);
    return true;
}

bool Evk2TzTriggerEvent::is_enabled(const Channel &channel) const {
    auto it = chan_map_.find(channel);
    if (it == chan_map_.end()) {
        return false;
    }
    long value =
        (*register_map_)[prefix_ + "SYSTEM_MONITOR/EXT_TRIGGERS/ENABLE"]["TRIGGER_" + std::to_string(it->second)]
            .read_value();
    return (value == 1);
}

std::map<I_TriggerIn::Channel, short> Evk2TzTriggerEvent::get_available_channels() const {
    return chan_map_;
}

} // namespace Metavision
