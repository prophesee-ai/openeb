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
#include <map>

#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Interface to handle external trigger signals
class I_TriggerIn : public I_RegistrableFacility<I_TriggerIn> {
public:
    /// @brief External trigger channel
    ///
    /// On most systems, only one (main) channel can be enabled. On some systems, an additional
    /// auxiliary and/or loopback channel may be available.
    enum class Channel { Main, Aux, Loopback };

    /// @brief Enables external trigger monitoring
    /// @param channel external trigger channel to enable
    /// @return true if external trigger monitoring was successfully enabled, false otherwise
    /// @warning External trigger monitoring is disabled by default on camera start
    virtual bool enable(const Channel &channel) = 0;

    /// @brief Disables external trigger monitoring
    /// @param channel external trigger channel to disable
    /// @return true if external trigger monitoring was successfully disabled, false otherwise
    virtual bool disable(const Channel &channel) = 0;

    /// @brief Checks if external trigger monitoring is enabled
    /// @param channel external trigger channel to check
    /// @return true if external trigger monitoring is enabled, False otherwise
    virtual bool is_enabled(const Channel &channel) const = 0;

    /// @brief Returns the map of available channels
    ///
    /// The returned map lists the available channels and gives the mapping between the
    /// Channel enum and the numeric value that can be found in the corresponding event id field
    /// @sa @ref Metavision::EventExtTrigger
    /// @return a map of available channels
    virtual std::map<Channel, short> get_available_channels() const = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_TRIGGER_IN_H
