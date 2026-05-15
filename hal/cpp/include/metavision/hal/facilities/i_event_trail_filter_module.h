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

#ifndef METAVISION_HAL_I_EVENT_TRAIL_FILTER_MODULE_H
#define METAVISION_HAL_I_EVENT_TRAIL_FILTER_MODULE_H

#include <cstdint>
#include <set>

#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Noise filter module
class I_EventTrailFilterModule : public I_RegistrableFacility<I_EventTrailFilterModule> {
public:
    /// @brief Type of implemented filter
    ///
    /// TRAIL: filters out trailing events from event bursts, keeping only the first event.
    /// STC_CUT_TRAIL: after a polarity change, filters out the first event of a burst as well as trailing events after
    ///                the second event, keeping only the second event from the burst. Single events are filtered out.
    /// STC_KEEP_TRAIL: after a polarity change, filters out the first event of burst, keeping all the trailing events
    ///                 from the burst. Single events are filtered out.
    ///
    /// A burst is defined as events of the same polarity occurring within a threshold period.
    ///
    /// @note Some filter types are not available for some devices
    enum class Type { TRAIL, STC_CUT_TRAIL, STC_KEEP_TRAIL };

    /// @brief Returns the set of available types of filters
    /// @return set of available types of filters
    virtual std::set<Type> get_available_types() const = 0;

    /// @brief Enables the EventTrailFilterModule with previously configured settings. Filtering type and threshold
    ///        should be set before enabling.
    /// @param state If true, enables the module. If false, disables it
    /// @return true on success
    virtual bool enable(bool state) = 0;

    /// @brief Returns EventTrailFilterModule activation state
    /// @return The EventTrailFilterModule state
    virtual bool is_enabled() const = 0;

    /// @brief Sets the event trail filtering type
    /// @note Facility might be reset if parameter is changed while enabled
    /// @return true on success
    virtual bool set_type(Type type) = 0;

    /// @brief Gets the event trail filtering type
    /// @return The event trail filtering type
    virtual Type get_type() const = 0;

    /// @brief Sets the event trail filtering threshold delay
    /// @param threshold Delay (in microseconds) between two bursts of events
    /// @note Facility might be reset if parameter is changed while enabled
    /// @return true on success
    virtual bool set_threshold(uint32_t threshold) = 0;

    /// @brief Gets the event trail filtering threshold delay (in microseconds)
    /// @return The event trail filtering threshold
    virtual uint32_t get_threshold() const = 0;

    /// @brief Gets the maximum supported value for event trail filtering threshold delay (in microseconds)
    /// @return The maximum supported threshold value
    virtual uint32_t get_max_supported_threshold() const = 0;

    /// @brief Gets the minimum supported value for event trail filtering threshold delay (in microseconds)
    /// @return The minimum supported threshold value
    virtual uint32_t get_min_supported_threshold() const = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_EVENT_TRAIL_FILTER_MODULE_H
