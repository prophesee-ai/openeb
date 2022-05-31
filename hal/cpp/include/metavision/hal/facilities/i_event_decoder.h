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

#ifndef METAVISION_HAL_I_EVENT_DECODER_H
#define METAVISION_HAL_I_EVENT_DECODER_H

#include <functional>
#include <map>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Class for decoding a specific type of event
/// @tparam Event type of event decoded by the instance
template<typename Event>
class I_EventDecoder : public I_RegistrableFacility<I_EventDecoder<Event>> {
public:
    using EventIterator_t       = const Event *;
    using EventBufferCallback_t = std::function<void(EventIterator_t begin, EventIterator_t end)>;
    using Event_t               = Event;

    /// @brief Sets the functions to call to each batch of decoded events
    /// @param cb Callback to add
    /// @return ID of the added callback
    /// @note This method is not thread safe. You should add/remove the various callback before starting the streaming
    /// @note It's not allowed to add/remove a callback from the callback itself
    size_t add_event_buffer_callback(const EventBufferCallback_t &cb);

    /// @brief Removes a previously registered callback
    /// @param callback_id Callback ID
    /// @return true if the callback has been unregistered correctly, false otherwise.
    /// @sa @ref add_event_buffer_callback
    bool remove_callback(size_t callback_id);

    /// @cond DEV
    void add_event_buffer(EventIterator_t begin, EventIterator_t end);
    /// @endcond

private:
    std::map<size_t, EventBufferCallback_t> cbs_map_;
    size_t next_cb_idx_{0};
};

} // namespace Metavision

#include "detail/i_event_decoder_impl.h"

#endif // METAVISION_HAL_I_EVENT_DECODER_H
