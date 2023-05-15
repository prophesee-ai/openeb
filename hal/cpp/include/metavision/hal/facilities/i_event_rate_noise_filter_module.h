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

#ifndef METAVISION_HAL_I_EVENT_RATE_NOISE_FILTER_MODULE_H
#define METAVISION_HAL_I_EVENT_RATE_NOISE_FILTER_MODULE_H

#include <string>

#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Interface for accessing the sensor level event rate based on noise filtering of a sensor.
///
/// This sensor level noise filter is based on the event rate only. If enabled, the sensor will transfer data if and
/// only if the event rate is above a given event rate. It avoids streaming background noise information without
/// relevant activity information.
class I_EventRateNoiseFilterModule : public I_RegistrableFacility<I_EventRateNoiseFilterModule> {
public:
    /// @brief Enables/disables the noise filter
    /// @param enable_filter Whether to enable the noise filtering
    virtual bool enable(bool enable_filter) = 0;

    /// @brief Sets the event rate threshold. Below this threshold, no events are streamed.
    /// @param threshold_Kev_s Event rate threshold in Kevt/s
    /// @return true if the input value was correctly set (i.e. it falls in the range of acceptable values for the
    /// sensor)
    virtual bool set_event_rate_threshold(uint32_t threshold_Kev_s) = 0;

    /// @brief Gets the event rate threshold in Kevt/s below which no events are streamed
    /// @return Event rate threshold in Kevt/s
    virtual uint32_t get_event_rate_threshold() = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_EVENT_RATE_NOISE_FILTER_MODULE_H
