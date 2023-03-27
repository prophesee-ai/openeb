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

#ifndef METAVISION_SDK_DRIVER_EVENT_TRAIL_FILTER_MODULE_H
#define METAVISION_SDK_DRIVER_EVENT_TRAIL_FILTER_MODULE_H

#include <cstdint>
#include <set>

#include "metavision/hal/facilities/i_event_trail_filter_module.h"

namespace Metavision {

/// @brief Facility class to handle event trail filter module configuration on the hardware side
///
/// The types of event trail filter are Spatio-Temporal Contrast (STC) and Trail.
class EventTrailFilterModule {
public:
    /// @brief Constructor
    EventTrailFilterModule(I_EventTrailFilterModule *noise_filter);

    /// @brief Destructor
    ~EventTrailFilterModule();

    /// @brief Returns the set of available types of filters
    /// @return set of available types of filters
    std::set<I_EventTrailFilterModule::Type> get_available_types() const;

    /// @brief Enables the EventTrailFilterModule. Filtering type and threshold should be set before hand
    /// @param state If true, enables the module. If false, disables it
    /// @return true on success
    bool enable(bool state);

    /// @brief Gets corresponding facility in HAL library
    I_EventTrailFilterModule *get_facility() const;

private:
    I_EventTrailFilterModule *pimpl_;
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_EVENT_TRAIL_FILTER_MODULE_H
