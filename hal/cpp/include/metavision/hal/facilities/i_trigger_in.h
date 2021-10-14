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

#ifndef METAVISION_HAL_I_TRIGGER_IN_H
#define METAVISION_HAL_I_TRIGGER_IN_H

#include <cstdint>

#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Interface to handle external trigger signals
class I_TriggerIn : public I_RegistrableFacility<I_TriggerIn> {
public:
    /// @brief Enables external trigger monitoring
    /// @param channel External trigger's address (0 for Gen3/Gen3.1 sensors, 1 for Gen4/Gen4.1 sensors)
    /// @return true if trigger was successfully enabled, false otherwise
    /// @warning Trigger monitoring is disabled by default on camera start.
    /// So you need to call ``enable()`` to start detecting signal.
    virtual bool enable(uint32_t channel) = 0;

    /// @brief Disables external trigger monitoring
    /// @param channel External trigger's address (0 for Gen3/Gen3.1 sensors, 1 for Gen4/Gen4.1 sensors)
    /// @return true if trigger was successfully disabled, false otherwise
    virtual bool disable(uint32_t channel) = 0;

    /// @brief Checks if the trigger in index is enabled
    /// @return true if the trigger in index is enabled, False otherwise
    virtual bool is_enabled(uint32_t index) = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_TRIGGER_IN_H
