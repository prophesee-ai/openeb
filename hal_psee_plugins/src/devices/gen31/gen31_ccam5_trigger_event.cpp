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

#include "devices/gen31/gen31_ccam5_trigger_event.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

namespace Metavision {

Gen31Ccam5TriggerEvent::Gen31Ccam5TriggerEvent(const std::shared_ptr<RegisterMap> &register_map,
                                               const std::shared_ptr<TzDevice> &device) :
    tzDev_(device), register_map_(register_map), chan_map_{{Channel::Main, 0}, {Channel::Loopback, 6}} {
    for (const auto &p : chan_map_) {
        disable(p.first);
    }
}

bool Gen31Ccam5TriggerEvent::enable(const Channel &channel) {
    auto it = chan_map_.find(channel);
    if (it == chan_map_.end()) {
        return false;
    }
    (*register_map_)["SYSTEM_MONITOR/EXT_TRIGGERS/ENABLE"]["TRIGGER_" + std::to_string(it->second)] = true;
    return true;
}

bool Gen31Ccam5TriggerEvent::disable(const Channel &channel) {
    auto it = chan_map_.find(channel);
    if (it == chan_map_.end()) {
        return false;
    }
    (*register_map_)["SYSTEM_MONITOR/EXT_TRIGGERS/ENABLE"]["TRIGGER_" + std::to_string(it->second)] = false;
    return true;
}

bool Gen31Ccam5TriggerEvent::is_enabled(const Channel &channel) const {
    auto it = chan_map_.find(channel);
    if (it == chan_map_.end()) {
        return false;
    }
    return (*register_map_)["SYSTEM_MONITOR/EXT_TRIGGERS/ENABLE"]["TRIGGER_" + std::to_string(it->second)].read_value();
}

std::map<I_TriggerIn::Channel, short> Gen31Ccam5TriggerEvent::get_available_channels() const {
    return chan_map_;
}
} // namespace Metavision
