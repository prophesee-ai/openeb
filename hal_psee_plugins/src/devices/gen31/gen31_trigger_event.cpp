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

#include "devices/gen31/gen31_trigger_event.h"
#include "facilities/psee_device_control.h"
#include "utils/register_map.h"

namespace Metavision {

Gen31TriggerEvent::Gen31TriggerEvent(const std::shared_ptr<RegisterMap> &register_map,
                                     const std::shared_ptr<PseeDeviceControl> &device_control) :
    PseeTriggerIn(device_control), register_map_(register_map) {
    for (uint32_t i = 0; i < 8; ++i) {
        disable(i);
    }
}

bool Gen31TriggerEvent::enable(uint32_t channel) {
    if (channel != 0 && channel != 6 && channel != 7) {
        return false;
    }

    if (channel == 7 && get_device_control()->get_mode() == I_DeviceControl::SyncMode::SLAVE) {
        return false;
    }

    (*register_map_)["SYSTEM_MONITOR/EXT_TRIGGERS/ENABLE"]["TRIGGER_" + std::to_string(channel)] = true;
    return true;
}

bool Gen31TriggerEvent::disable(uint32_t channel) {
    if (channel != 0 && channel != 6 && channel != 7) {
        return false;
    }

    (*register_map_)["SYSTEM_MONITOR/EXT_TRIGGERS/ENABLE"]["TRIGGER_" + std::to_string(channel)] = false;
    return true;
}

bool Gen31TriggerEvent::is_enabled(uint32_t index) {
    if (index != 0 && index != 6 && index != 7) {
        return false;
    }

    return (*register_map_)["SYSTEM_MONITOR/EXT_TRIGGERS/ENABLE"]["TRIGGER_" + std::to_string(index)].read_value();
}

} // namespace Metavision
