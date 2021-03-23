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

#ifndef METAVISION_SDK_DRIVER_NOISE_FILTER_MODULE_H
#define METAVISION_SDK_DRIVER_NOISE_FILTER_MODULE_H

#include <cstdint>

#include "metavision/hal/facilities/i_noise_filter_module.h"

namespace Metavision {

/// @brief Facility class to handle noise filter module configuration on the hardware side
///
/// The types of noise filter are Spatio-Temporal Contrast (STC) and Trail.
class NoiseFilterModule {
public:
    /// @brief Constructor
    NoiseFilterModule(I_NoiseFilterModule *noise_filter);

    /// @brief Destructor
    ~NoiseFilterModule();

    /// @brief Enables the NoiseFilterModule in the mode STC or Trail with the corresponding threshold
    /// @param type Defines the type of the filter
    /// @param threshold Delay in microsecond between two bursts of events
    /// @note STC keeps the second event within a burst of events with the same polarity
    /// Trail keeps the first event within a burst of events with the same polarity
    void enable(I_NoiseFilterModule::Type type, uint32_t threshold);

    /// @brief Disables filtering
    ///
    /// No events are removed by this filter anymore.
    void disable();

    /// @brief Gets corresponding facility in HAL library
    I_NoiseFilterModule *get_facility() const;

private:
    I_NoiseFilterModule *pimpl_;
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_NOISE_FILTER_MODULE_H
