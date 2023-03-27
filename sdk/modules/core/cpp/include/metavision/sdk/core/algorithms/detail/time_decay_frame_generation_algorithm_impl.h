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

#ifndef METAVISION_SDK_CORE_TIME_DECAY_FRAME_GENERATION_ALGORITHM_IMPL_H
#define METAVISION_SDK_CORE_TIME_DECAY_FRAME_GENERATION_ALGORITHM_IMPL_H

namespace Metavision {

template<typename EventIt>
inline void TimeDecayFrameGenerationAlgorithm::process_events(EventIt it_begin, EventIt it_end) {
    for (auto it = it_begin; it != it_end; ++it) {
        time_surface_.at(it->y, it->x, it->p) = it->t;
    }
    if (it_begin != it_end)
        last_ts_ = std::prev(it_end)->t;
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_TIME_DECAY_FRAME_GENERATION_ALGORITHM_IMPL_H
