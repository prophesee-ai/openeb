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

} // namespace Metavision

#endif // METAVISION_HAL_I_EVENT_DECODER_IMPL_H
