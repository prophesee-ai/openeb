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

#ifndef METAVISION_SDK_STREAM_ERC_COUNTER_H
#define METAVISION_SDK_STREAM_ERC_COUNTER_H

#include <memory>
#include <functional>

// Metavision SDK Base CDCount event
#include "metavision/sdk/base/events/event_erc_counter.h"

// Definition of CallbackId
#include "metavision/sdk/base/utils/callback_id.h"

namespace Metavision {

/// @brief Type alias for a callback on a buffer of @ref EventERCCounter
using EventsERCCounterCallback = std::function<void(const EventERCCounter *begin, const EventERCCounter *end)>;

/// @brief Facility class to handle ERCCounter events
class ERCCounter {
public:
    /// @brief Destructor
    ///
    /// Deletes a ERCCounter class instance.
    virtual ~ERCCounter();

    /// @brief Subscribes to CDCount events
    ///
    /// Registers a callback that will be called each time a buffer of EventERCCounter has been decoded.
    ///
    /// @param cb Callback to call each time a buffer of EventERCCounter has been decoded
    /// @sa @ref EventsERCCounterCallback
    /// @return ID of the added callback
    CallbackId add_callback(const EventsERCCounterCallback &cb);

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
    ERCCounter(Private *);
    std::unique_ptr<Private> pimpl_;
};

} // namespace Metavision

#endif // METAVISION_SDK_STREAM_ERC_COUNTER_H
