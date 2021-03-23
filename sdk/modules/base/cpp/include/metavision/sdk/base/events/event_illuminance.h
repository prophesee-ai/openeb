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

#ifndef METAVISION_SDK_BASE_EVENT_ILLUMINANCE_H
#define METAVISION_SDK_BASE_EVENT_ILLUMINANCE_H

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Class representing an Illuminance event
/// @note This class is deprecated since version 2.1.0 and will be removed in next releases
struct EventIlluminance {
    /// @brief Default constructor
    EventIlluminance() = default;

    /// @brief Constructor
    /// @param time Timestamp at which the event happened (in us)
    /// @param illum Illuminance value (in lux)
    EventIlluminance(timestamp time, float illum) : t(time), illuminance(illum) {}

    /// @brief Timestamp at which the event happened (in us)
    timestamp t;

    /// @brief Illuminance value (in lux)
    float illuminance;
};

} // namespace Metavision

#endif // METAVISION_SDK_BASE_EVENT_ILLUMINANCE_H
