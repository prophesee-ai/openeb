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

#include "metavision/psee_hw_layer/devices/imx636/imx636_tz_trigger_event.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

using vfield = std::map<std::string, uint32_t>;

namespace Metavision {

Imx636TzTriggerEvent::Imx636TzTriggerEvent(const std::shared_ptr<RegisterMap> &register_map, const std::string &prefix,
                                           const std::shared_ptr<TzDevice> tzDev) :
    Gen41TzTriggerEvent(register_map, prefix, tzDev), chan_map_({{Channel::Main, 0}}) {}

bool Imx636TzTriggerEvent::enable(const Channel &channel) {
    auto it = chan_map_.find(channel);
    if (it == chan_map_.end()) {
        return false;
    }
    (*register_map_)[prefix_ + "edf/Reserved_7004"]["Reserved_10"].write_value(1);
    return true;
}

bool Imx636TzTriggerEvent::is_enabled(const Channel &channel) const {
    auto it = chan_map_.find(channel);
    if (it == chan_map_.end()) {
        return false;
    }
    auto value = (*register_map_)[prefix_ + "edf/Reserved_7004"]["Reserved_10"].read_value();
    return (value == 1);
}

std::map<I_TriggerIn::Channel, short> Imx636TzTriggerEvent::get_available_channels() const {
    return chan_map_;
}

} // namespace Metavision
