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

#ifndef METAVISION_SDK_BASE_EVENT_TEMPERATURE_H
#define METAVISION_SDK_BASE_EVENT_TEMPERATURE_H

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Class representing a Temperature event
/// @note This class is deprecated since version 2.1.0 and will be removed in next releases
struct EventTemperature {
    /// @brief Default constructor
    EventTemperature() = default;

    /// @brief Constructor
    /// @param t Timestamp of the event
    /// @param s Source sensor of the event, see @ref source
    /// @param v Value of the temperature
    EventTemperature(timestamp t, short s, float v) : t(t), source(s), value(v) {}

    /// @brief Timestamp of the event
    timestamp t;

    /// @brief Source sensor of the event. Each device can have
    /// multiple temperature sensors. The exact ID corresponding
    /// to each sensor changes with differing devices.
    short source;

    /// @brief Temperature in degrees Celsius
    float value;
};

} // namespace Metavision

#endif // METAVISION_SDK_BASE_EVENT_TEMPERATURE_H
