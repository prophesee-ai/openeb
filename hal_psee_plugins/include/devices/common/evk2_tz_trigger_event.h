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

#ifndef METAVISION_HAL_EVK2_TRIGGER_EVENT_H
#define METAVISION_HAL_EVK2_TRIGGER_EVENT_H

#include <cstdint>
#include <vector>
#include <string>

#include "metavision/hal/facilities/i_trigger_in.h"

namespace Metavision {

class RegisterMap;
class TzDevice;

class Evk2TzTriggerEvent : public I_TriggerIn {
public:
    /// @brief Constructor
    Evk2TzTriggerEvent(const std::shared_ptr<RegisterMap> &register_map, const std::string &prefix,
                       const std::shared_ptr<TzDevice> tzDev);

    enum class Channels { TRIG_IN = 1, TRIG_OUT = 3 };

    /// @brief Enables external trigger monitoring
    /// @param channel External trigger's channel
    bool enable(uint32_t channel);

    /// @brief Disables external trigger monitoring
    /// @param channel External trigger's channel
    bool disable(uint32_t channel);

    /// @brief Checks if trigger in index is enabled
    /// @param channel External trigger's channel
    /// @return true if trigger in index is enabled, False otherwise
    bool is_enabled(uint32_t channel);

private:
    bool is_valid_id(uint32_t channel);

    std::shared_ptr<RegisterMap> register_map_;
    std::shared_ptr<TzDevice> tzDev_;
    std::string prefix_;
    const std::vector<Channels> chan_ids_{Channels::TRIG_IN, Channels::TRIG_OUT};
};

} // namespace Metavision

#endif // METAVISION_HAL_EVK2_TRIGGER_EVENT_H
