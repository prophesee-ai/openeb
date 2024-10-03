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

#ifndef METAVISION_SDK_CORE_DETAIL_EVENT_RESCALER_ALGORITHM_IMPL_H
#define METAVISION_SDK_CORE_DETAIL_EVENT_RESCALER_ALGORITHM_IMPL_H

#include "metavision/sdk/core/algorithms/event_rescaler_algorithm.h"

namespace Metavision {

template<typename InputIt, typename OutputIt>
OutputIt EventRescalerAlgorithm::process_events(InputIt begin, InputIt end, OutputIt out_begin) const {
    for (auto it = begin; it != end; ++it) {
        auto ev    = *it;
        ev.x       = static_cast<int>(ev.x * scale_width_ + offset_width_);
        ev.y       = static_cast<int>(ev.y * scale_height_ + offset_height_);
        *out_begin = ev;
        ++out_begin;
    }
    return out_begin;
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_DETAIL_EVENT_RESCALER_ALGORITHM_IMPL_H
