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

#ifndef METAVISION_HAL_I_EVENT_DECODER_IMPL_H
#define METAVISION_HAL_I_EVENT_DECODER_IMPL_H

#include <memory>

#include "metavision/hal/utils/hal_log.h"
#include "metavision/hal/utils/hal_error_code.h"
#include "metavision/hal/utils/hal_exception.h"

namespace Metavision {

template<typename Event>
size_t I_EventDecoder<Event>::add_event_buffer_callback(const EventBufferCallback_t &cb) {
    cbs_map_[next_cb_idx_] = cb;
    return next_cb_idx_++;
}

template<typename Event>
bool I_EventDecoder<Event>::remove_callback(size_t callback_id) {
    auto it = cbs_map_.find(callback_id);
    if (it != cbs_map_.end()) {
        cbs_map_.erase(it);
        return true;
    }
    return false;
}

/// @cond DEV
template<typename Event>
void I_EventDecoder<Event>::add_event_buffer(EventIterator_t begin, EventIterator_t end) {
    for (auto it = cbs_map_.begin(), it_end = cbs_map_.end(); it != it_end; ++it) {
        it->second(begin, end);
    }
}
/// @endcond

template<typename Event>
void I_EventDecoder<Event>::set_add_decoded_event_callback(AddEventCallback_t cb, bool add) {
    static bool warning_already_logged = false;
    if (!warning_already_logged) {
        MV_HAL_LOG_WARNING() << "I_EventDecoder<Event>::set_add_decoded_event_callback(...) is deprecated since "
                                "version 2.2.0 and will be removed in later releases.";
        MV_HAL_LOG_WARNING() << "Please use I_Decoder<Event>::add_event_buffer_callback(...) instead." << std::endl;
        warning_already_logged = true;
    }
    if (!add) {
        for (size_t i = 0; i < next_cb_idx_; ++i)
            remove_callback(i);
    }
    add_event_buffer_callback([cb](EventIterator_t begin, EventIterator_t end) {
        for (auto it = begin; it != end; ++it) {
            cb(*it);
        }
    });
}

template<typename Event>
void I_EventDecoder<Event>::set_add_decoded_vevent_callback(AddVEventCallback_t cb, bool add) {
    static bool warning_already_logged = false;
    if (!warning_already_logged) {
        MV_HAL_LOG_WARNING() << "I_EventDecoder<Event>::set_add_decoded_vevent_callback(...) is deprecated since "
                                "version 2.2.0 and will be removed in later releases.";
        MV_HAL_LOG_WARNING() << "Please use I_Decoder<Event>::add_event_buffer_callback(...) instead." << std::endl;
        warning_already_logged = true;
    }
    if (!add) {
        for (size_t i = 0; i < next_cb_idx_; ++i)
            remove_callback(i);
    }
    add_event_buffer_callback([cb](EventIterator_t begin, EventIterator_t end) { cb(begin, end); });
}

template<typename Event>
void I_EventDecoder<Event>::set_end_decode_callback(EndDecodeCallback_t, bool) {
    throw HalException(HalErrorCode::DeprecatedFunctionCalled,
                       "I_EventDecoder<Event>::set_end_decode_callback(...) is deprecated since "
                       "version 2.2.0 and will be removed in later releases.");
}

} // namespace Metavision

#endif // METAVISION_HAL_I_EVENT_DECODER_IMPL_H
