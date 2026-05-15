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

#ifndef METAVISION_SDK_STREAM_MONITORING_H
#define METAVISION_SDK_STREAM_MONITORING_H

#include <memory>
#include <functional>

// Metavision SDK Base monitoring event
#include "metavision/sdk/base/events/event_monitoring.h"

// Definition of CallbackId
#include "metavision/sdk/base/utils/callback_id.h"

namespace Metavision {

/// @brief Type alias for a callback on a buffer of @ref EventMonitoring
using EventsMonitoringCallback = std::function<void(const EventMonitoring *begin, const EventMonitoring *end)>;

/// @brief Facility class to handle monitoring events
class Monitoring {
public:
    /// @brief Destructor
    ///
    /// Deletes a Monitoring class instance.
    virtual ~Monitoring();

    /// @brief Subscribes to monitoring events
    ///
    /// Registers a callback that will be called each time a buffer of EventMonitoring has been decoded.
    ///
    /// @param cb Callback to call each time a buffer of EventMonitoring has been decoded
    /// @sa @ref EventsMonitoringCallback
    /// @return ID of the added callback
    CallbackId add_callback(const EventsMonitoringCallback &cb);

    /// @brief Removes a previously registered callback
    /// @param callback_id Callback ID
    /// @return true if the callback has been unregistered correctly, false otherwise.
    /// @sa @ref add_callback
    bool remove_callback(CallbackId callback_id);

    /// @brief For internal use
    class Private;
    /// @brief For internal use
    Private &get_pimpl();

private:
    Monitoring(Private *);
    std::unique_ptr<Private> pimpl_;
};

} // namespace Metavision

#endif // METAVISION_SDK_STREAM_MONITORING_H
