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

#ifndef METAVISION_SDK_CORE_EVENT_FRAME_DIFF_GENERATION_ALGORITHM_IMPL_H
#define METAVISION_SDK_CORE_EVENT_FRAME_DIFF_GENERATION_ALGORITHM_IMPL_H

namespace Metavision {

template<typename InputIt>
void EventFrameDiffGenerationAlgorithm::process_events(InputIt it_begin, InputIt it_end) {
    auto &diff = frame_.get_data();
    for (auto it = it_begin; it != it_end; ++it) {
        const unsigned int idx = it->x + it->y * frame_.get_config().width;
        int8_t &sum_polarities = diff[idx];
        const bool should_rollover =
            (sum_polarities == min_val_ && it->p == 0) || (sum_polarities == max_val_ && it->p == 1);
        if (!should_rollover) {
            sum_polarities += (it->p == 0 ? -1 : 1);
        } else if (allow_rollover_) {
            sum_polarities = (it->p == 0 ? max_val_ : min_val_);
        } else { // sum_polarities is saturated
        }
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_EVENT_FRAME_DIFF_GENERATION_ALGORITHM_IMPL_H
