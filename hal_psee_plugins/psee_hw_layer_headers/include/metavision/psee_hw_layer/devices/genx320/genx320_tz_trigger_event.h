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

#ifndef METAVISION_HAL_GENX320_TZ_TRIGGER_EVENT_H
#define METAVISION_HAL_GENX320_TZ_TRIGGER_EVENT_H

#include <cstdint>
#include <vector>
#include <string>

#include "metavision/hal/facilities/i_trigger_in.h"

namespace Metavision {

class RegisterMap;
class TzDevice;

class GenX320TzTriggerEvent : public I_TriggerIn {
public:
    /// @brief Constructor
    GenX320TzTriggerEvent(const std::shared_ptr<RegisterMap> &register_map, const std::string &prefix,
                          const std::shared_ptr<TzDevice> tzDev);

    /// @brief Enables external trigger monitoring
    /// @param channel external trigger channel to enable
    bool enable(const Channel &channel) override;

    /// @brief Disables external trigger monitoring
    /// @param channel external trigger channel to disable
    bool disable(const Channel &channel) override;

    /// @brief Checks if trigger in index is enabled
    /// @param channel external trigger channel to check
    /// @return true if trigger in index is enabled, False otherwise
    bool is_enabled(const Channel &channel) const override;

    /// @brief Returns the set of available channels
    /// @return set of available channels
    std::map<Channel, short> get_available_channels() const override;

protected:
    std::shared_ptr<RegisterMap> register_map_;
    std::string prefix_;

private:
    std::shared_ptr<TzDevice> tzDev_;

    const std::map<Channel, short> chan_map_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GENX320_TZ_TRIGGER_EVENT_H
