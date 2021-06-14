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

#ifndef METAVISION_HAL_I_NOISE_FILTER_MODULE_H
#define METAVISION_HAL_I_NOISE_FILTER_MODULE_H

#include <cstdint>

#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Noise filter module
/// @note This feature is available only on Gen4.1 sensors
class I_NoiseFilterModule : public I_RegistrableFacility<I_NoiseFilterModule> {
public:
    /// @brief Type of implemented filter
    enum class Type { STC, TRAIL };

    /// @brief Enables the NoiseFilterModule in the mode STC or Trail with the corresponding threshold
    /// @param type Defines the type of the filter
    /// @param threshold Delay (in microseconds) between two bursts of events
    /// @note STC keeps the second event within a burst of events with the same polarity.\n
    /// Trail keeps the first event within a burst of events with the same polarity
    virtual void enable(Type type, uint32_t threshold) = 0;

    /// @brief Disables filtering
    ///
    /// No events are removed by this filter anymore.
    virtual void disable() = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_NOISE_FILTER_MODULE_H
