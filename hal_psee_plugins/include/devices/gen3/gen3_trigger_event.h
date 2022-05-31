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

#ifndef METAVISION_HAL_GEN3_TRIGGER_EVENT_H
#define METAVISION_HAL_GEN3_TRIGGER_EVENT_H

#include <cstdint>

#include "facilities/psee_trigger_in.h"

namespace Metavision {

class PseeDeviceControl;
class PseeLibUSBBoardCommand;

class Gen3TriggerEvent : public PseeTriggerIn {
public:
    /// @brief Constructor
    Gen3TriggerEvent(const std::shared_ptr<PseeLibUSBBoardCommand> &board_cmd,
                     const std::shared_ptr<PseeDeviceControl> &device_control);

    /// @brief Enables external trigger monitoring
    ///
    /// Available channels:
    /// 0: main trigger in
    /// 6: loopback trigger out (Test purpose)
    /// 7: Auxiliary trigger in
    /// @param channel External trigger's channel
    virtual bool enable(uint32_t channel) override;

    /// @brief Disables external trigger monitoring
    /// @param channel External trigger's channel
    virtual bool disable(uint32_t channel) override;

    /// @brief Checks if trigger in index is enabled
    /// @param channel External trigger's channel
    /// @return true if trigger in index is enabled, False otherwise
    virtual bool is_enabled(uint32_t channel) override;

private:
    std::shared_ptr<PseeLibUSBBoardCommand> icmd_;
    uint32_t base_address_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN3_TRIGGER_EVENT_H
