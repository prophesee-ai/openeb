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

#ifndef METAVISION_SDK_DRIVER_ILLUMINANCE_H
#define METAVISION_SDK_DRIVER_ILLUMINANCE_H

#include <memory>
#include <functional>

// Metavision SDK Base Illuminance event
#include "metavision/sdk/base/events/event_illuminance.h"

// Definition of CallbackId
#include "metavision/sdk/base/utils/callback_id.h"

namespace Metavision {

/// @note This alias is deprecated since version 2.1.0 and will be removed in next releases
/// @brief Callback type alias for @ref EventIlluminance
///
/// @param begin @ref EventIlluminance pointer to the beginning of the buffer.
/// @param end @ref EventIlluminance pointer to the end of the buffer.
using EventsIlluminanceCallback = std::function<void(const EventIlluminance *begin, const EventIlluminance *end)>;

/// @brief Facility class to handle illuminance events
class Illuminance {
public:
    /// @brief Destructor
    ///
    /// Deletes a Illuminance class instance.
    virtual ~Illuminance();

    /// @brief Subscribes to Illuminance events
    ///
    /// Registers a callback that will be called each time a buffer of EventIlluminance has been decoded.
    /// @param cb Callback to call each time a buffer of EventIlluminance has been decoded
    /// @sa @ref EventsIlluminanceCallback
    /// @return ID of the added callback
    CallbackId add_callback(const EventsIlluminanceCallback &cb);

    /// @brief Removes a previously registered callback
    ///
    /// @param callback_id Callback ID
    /// @return true if the callback has been unregistered correctly, false otherwise
    /// @sa @ref add_callback
    bool remove_callback(CallbackId callback_id);
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_ILLUMINANCE_H
