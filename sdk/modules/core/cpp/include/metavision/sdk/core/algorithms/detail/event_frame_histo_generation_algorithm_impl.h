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

#ifndef METAVISION_SDK_CORE_EVENT_FRAME_HISTO_GENERATION_ALGORITHM_IMPL_H
#define METAVISION_SDK_CORE_EVENT_FRAME_HISTO_GENERATION_ALGORITHM_IMPL_H

namespace Metavision {

template<typename InputIt>
void EventFrameHistoGenerationAlgorithm::process_events(InputIt it_begin, InputIt it_end) {
    const auto &cfg = frame_unpacked_.get_config();
    auto &histo     = frame_unpacked_.get_data();
    for (auto it = it_begin; it != it_end; ++it) {
        const unsigned int idx = it->p + (it->x + it->y * cfg.width) * 2;
        const uint8_t sum_max  = (it->p == 0 ? sum_max_neg_ : sum_max_pos_);
        uint8_t &sum_events    = histo[idx];
        if (sum_events <= sum_max - 1) {
            ++sum_events;
        } else { // sum_events is saturated
        }
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_EVENT_FRAME_HISTO_GENERATION_ALGORITHM_IMPL_H
