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

#ifndef METAVISION_HAL_IMX636_TRIGGER_EVENT_H
#define METAVISION_HAL_IMX636_TRIGGER_EVENT_H

#include <cstdint>
#include <vector>
#include <string>

#include "metavision/psee_hw_layer/devices/gen41/gen41_tz_trigger_event.h"

namespace Metavision {

class RegisterMap;
class TzDevice;

class Imx636TzTriggerEvent : public Gen41TzTriggerEvent {
public:
    /// @brief Constructor
    Imx636TzTriggerEvent(const std::shared_ptr<RegisterMap> &register_map, const std::string &prefix,
                         const std::shared_ptr<TzDevice> tzDev);

    /// @brief Enables external trigger monitoring
    /// @param channel external trigger channel to enable
    bool enable(const Channel &channel) override;

    /// @brief Checks if trigger in index is enabled
    /// @param channel external trigger channel to check
    /// @return true if trigger in index is enabled, False otherwise
    bool is_enabled(const Channel &channel) const override;

    /// @brief Returns the set of available channels
    /// @return a map of available channels
    std::map<Channel, short> get_available_channels() const override;

private:
    const std::map<Channel, short> chan_map_;
};

} // namespace Metavision

#endif // METAVISION_HAL_IMX636_TRIGGER_EVENT_H
