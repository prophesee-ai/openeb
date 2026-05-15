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

#ifndef METAVISION_HAL_I_EVENT_RATE_ACTIVITY_FILTER_MODULE_H
#define METAVISION_HAL_I_EVENT_RATE_ACTIVITY_FILTER_MODULE_H

#include <string>

#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Interface for accessing the sensor event rate activity filter.
///
/// This sensor level filter is based on the event rate only. If enabled, the sensor will transfer data if and
/// only if the event rate is above a given event rate and below a given event rate. It avoids streaming background
/// noise information without relevant activity and streaming high event rate flashing scenes, saving bandwidth and
/// power.
class I_EventRateActivityFilterModule : public I_RegistrableFacility<I_EventRateActivityFilterModule> {
public:
    /// @brief Band pass filter hysteresis thresholds for event rate activity filter
    /// @note Not all thresholds are supported given the sensor generation

    /// @param lower_bound_start Event rate threshold for the band pass filter to start filtering incoming events.
    /// Below this threshold, no events are streamed.
    /// @param lower_bound_stop Event rate threshold for the band pass filter to stop filtering and resume streaming
    /// of incoming events.
    /// Above this threshold, events are streamed again after the filter was actively dropping
    /// events using the @ref lower_bound_start threshold condition.
    /// @param upper_bound_start Event rate threshold for the band pass filter to start filtering incoming events.
    /// Above this threshold, no events are streamed.
    /// @param upper_bound_stop Event rate threshold for the band pass filter to stop filtering and resume streaming
    /// of incoming events.
    /// Below this threshold, events are streamed again after the filter was actively dropping events using the @ref
    /// upper_bound_start threshold condition.
    struct thresholds {
        uint32_t lower_bound_start;
        uint32_t lower_bound_stop;
        uint32_t upper_bound_start;
        uint32_t upper_bound_stop;
    };

    /// @brief Enables/disables the event rate activity filter
    /// @param enable_filter Whether to enable the filter
    virtual bool enable(bool enable_filter) = 0;

    /// @brief Returns the event rate activity filter state
    /// @return the filter state
    virtual bool is_enabled() const = 0;

    /// @brief Gets band pass filter hysteresis thresholds supported by the sensor.
    /// @return @ref thresholds thresholds structure with values 0/1 depending if corresponding threshold is supported.
    virtual thresholds is_thresholds_supported() const = 0;

    /// @brief Sets band pass filter hysteresis thresholds.
    /// @param thresholds_ev_s Event rate thresholds data structure in evt/s
    /// @warning Partially initialize structure \p thresholds_ev_s could lead to random threshold values being set.
    /// Use @ref is_thresholds_supported function to know which threshold values could be omitted.
    /// @return true if the input value was correctly set (i.e. it falls in the range of acceptable values for the
    /// sensor)
    virtual bool set_thresholds(const thresholds &thresholds_ev_s) = 0;

    /// @brief Gets band pass filter hysteresis thresholds.
    /// @return @ref thresholds event rate thresholds structure.
    virtual thresholds get_thresholds() const = 0;

    /// @brief Gets band pass filter hysteresis minimum thresholds configuration.
    /// @return @ref thresholds event rate thresholds structure.
    virtual thresholds get_min_supported_thresholds() const = 0;

    /// @brief Gets band pass filter hysteresis maximum thresholds configuration.
    /// @return @ref thresholds event rate thresholds structure.
    virtual thresholds get_max_supported_thresholds() const = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_EVENT_RATE_ACTIVITY_FILTER_MODULE_H
