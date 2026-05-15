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

#include "metavision/psee_hw_layer/devices/genx320/genx320_tz_trigger_event.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

namespace Metavision {

GenX320TzTriggerEvent::GenX320TzTriggerEvent(const std::shared_ptr<RegisterMap> &register_map,
                                             const std::string &prefix) :
    register_map_(register_map), prefix_(prefix), chan_map_({{Channel::Main, 0}}) {}

bool GenX320TzTriggerEvent::enable(const Channel &channel) {
    auto it = chan_map_.find(channel);
    if (it == chan_map_.end()) {
        return false;
    }
    (*register_map_)["io_ctrl2"].write_value(
        {{"exttrig_en", 1}, {"exttrig_enzi", 1}}); // Force to 1 in silicon, to be removed ?
    (*register_map_)["edf/event_type_en"]["en_ext_trigger"].write_value(1);
    return true;
}

bool GenX320TzTriggerEvent::disable(const Channel &channel) {
    auto it = chan_map_.find(channel);
    if (it == chan_map_.end()) {
        return false;
    }
    (*register_map_)["io_ctrl2"]["exttrig_enzi"].write_value(0);
    (*register_map_)["edf/event_type_en"]["en_ext_trigger"].write_value(1);
    return true;
}

bool GenX320TzTriggerEvent::is_enabled(const Channel &channel) const {
    auto it = chan_map_.find(channel);
    if (it == chan_map_.end()) {
        return false;
    }
    auto value  = (*register_map_)["io_ctrl2"]["exttrig_en"].read_value();
    auto value2 = (*register_map_)["io_ctrl2"]["exttrig_enzi"].read_value();
    return (value == 1) && (value2 == 1);
}

std::map<I_TriggerIn::Channel, short> GenX320TzTriggerEvent::get_available_channels() const {
    return chan_map_;
}

} // namespace Metavision
