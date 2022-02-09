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

#ifndef METAVISION_SDK_DRIVER_ANTIFLICKER_MODULE_H
#define METAVISION_SDK_DRIVER_ANTIFLICKER_MODULE_H

#include <cstdint>

#include "metavision/hal/facilities/i_antiflicker_module.h"

namespace Metavision {

/// @brief Facility class to handle anti-flicker configuration on the hardware side
class AntiFlickerModule {
public:
    /// @brief Constructor
    AntiFlickerModule(I_AntiFlickerModule *afk);

    /// @brief Destructor
    ~AntiFlickerModule();

    /// @brief Enables the anti-flicker filter
    void enable();

    /// @brief Disables the anti-flicker filter
    void disable();

    /// @brief Sets anti-flicker parameters.
    ///
    /// Defines the frequency band to be kept or removed :
    /// [frequency_center - bandwidth/2, frequency_center + bandwidth/2]
    /// This frequency range should be in the range [50 - 500] Hz
    ///
    /// @param frequency_center Center of the frequency band (in Hz)
    /// @param bandwidth Range of frequencies around the frequency_center (in Hz)
    /// @param stop If true, band-stop (by default); if false, band-pass
    ///
    /// @note band-stop removes all frequencies between min and max\n
    ///       band-pass removes all events outside of the band sequence defined
    /// @throw exception if frequency band is not in the range [50 - 500] Hz
    void set_frequency(uint32_t frequency_center, uint32_t bandwidth, bool stop = true);

    /// @brief Sets anti-flicker parameters.
    ///
    /// Defines the frequency band to be kept or removed in the range [50 - 500] Hz
    ///
    /// @param min_freq Lower frequency of the band (in Hz)
    /// @param max_freq Higher frequency of the band (in Hz)
    /// @param stop If true, band-stop; if false, band-pass
    ///
    /// @note band-stop removes all frequencies between min and max\n
    ///       band-pass removes all events outside of the band sequence defined
    /// @throw exception if frequencies are outside of the range [50 - 500] Hz
    void set_frequency_band(uint32_t min_freq, uint32_t max_freq, bool stop = true);

    /// @brief Gets corresponding facility in HAL library
    I_AntiFlickerModule *get_facility() const;

private:
    I_AntiFlickerModule *pimpl_;
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_ANTIFLICKER_MODULE_H
