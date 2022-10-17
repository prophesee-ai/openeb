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

#include "devices/imx636/imx636_tz_trigger_event.h"
#include "utils/register_map.h"

using vfield = std::map<std::string, uint32_t>;

namespace Metavision {

Imx636TzTriggerEvent::Imx636TzTriggerEvent(const std::shared_ptr<RegisterMap> &register_map, const std::string &prefix,
                                           const std::shared_ptr<TzDevice> tzDev) :
    Gen41TzTriggerEvent(register_map, prefix, tzDev) {}

bool Imx636TzTriggerEvent::enable(uint32_t channel) {
    bool valid    = is_valid_id(channel);
    long read_val = 0;
    long value    = 0;

    if (valid) {
        (*register_map_)[prefix_ + "edf/Reserved_7004"]["Reserved_10"].write_value(1);
    }
    return valid;
}

bool Imx636TzTriggerEvent::is_enabled(uint32_t channel) {
    bool valid = is_valid_id(channel);
    long value = 0;

    if (valid) {
        value = (*register_map_)[prefix_ + "edf/Reserved_7004"]["Reserved_10"].read_value();
    }
    return valid && (value == 1);
}

} // namespace Metavision
