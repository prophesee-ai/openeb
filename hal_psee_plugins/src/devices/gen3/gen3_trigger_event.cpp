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

#include <memory>

#include "facilities/psee_device_control.h"
#include "boards/utils/psee_libusb_board_command.h"
#include "devices/gen3/legacy_regmap_headers/ccam3_single_gen3.h"
#include "devices/gen3/gen3_trigger_event.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"

namespace Metavision {

Gen3TriggerEvent::Gen3TriggerEvent(const std::shared_ptr<PseeLibUSBBoardCommand> &board_cmd,
                                   const std::shared_ptr<PseeDeviceControl> &device_control) :
    PseeTriggerIn(device_control), icmd_(board_cmd), base_address_(CCAM3_SYS_REG_BASE_ADDR) {
    if (!icmd_) {
        throw(HalException(PseeHalPluginErrorCode::BoardCommandNotFound, "Board command is null."));
    }
    for (uint32_t i = 0; i < 8; ++i) {
        disable(i);
    }
}

bool Gen3TriggerEvent::enable(uint32_t channel) {
    if (channel != 0 && channel != 6 && channel != 7) {
        return false;
    }

    if (channel == 7 && get_device_control()->get_mode() == I_DeviceControl::SyncMode::SLAVE) {
        return false;
    }

    icmd_->send_register_bit(base_address_ + CCAM3_SYSTEM_MONITOR_EXT_TRIGGERS_ENABLE_ADDR, channel, true);
    return true;
}

bool Gen3TriggerEvent::disable(uint32_t channel) {
    if (channel != 0 && channel != 6 && channel != 7) {
        return false;
    }

    icmd_->send_register_bit(base_address_ + CCAM3_SYSTEM_MONITOR_EXT_TRIGGERS_ENABLE_ADDR, channel, false);
    return true;
}

bool Gen3TriggerEvent::is_enabled(uint32_t index) {
    if (index != 0 && index != 6 && index != 7) {
        return false;
    }
    return icmd_->read_register_bit(base_address_ + CCAM3_SYSTEM_MONITOR_EXT_TRIGGERS_ENABLE_ADDR, index);
}
} // namespace Metavision
