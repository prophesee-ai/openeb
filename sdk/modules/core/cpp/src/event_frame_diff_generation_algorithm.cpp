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

#include "metavision/sdk/core/algorithms/event_frame_diff_generation_algorithm.h"

namespace Metavision {

EventFrameDiffGenerationAlgorithm::EventFrameDiffGenerationAlgorithm(unsigned int width, unsigned int height,
                                                                     unsigned int bit_size, bool allow_rollover,
                                                                     timestamp min_generation_period_us) :
    allow_rollover_(allow_rollover),
    min_val_(-(1 << (bit_size - 1))),
    max_val_((1 << (bit_size - 1)) - 1),
    min_generation_period_us_(min_generation_period_us),
    frame_(height, width, bit_size) {}

void EventFrameDiffGenerationAlgorithm::generate(RawEventFrameDiff &frame) {
    const auto &cfg = frame_.get_config();
    frame.reset(cfg.height, cfg.width, cfg.bit_size); // Prepare next accumulating frame
    frame.swap(frame_);                               // Swap internal event frame with provided one
}

bool EventFrameDiffGenerationAlgorithm::generate(timestamp ts_event_frame, RawEventFrameDiff &event_frame) {
    if (is_ts_prev_set_ && ts_event_frame - ts_prev_ < min_generation_period_us_)
        return false;
    is_ts_prev_set_ = true;
    ts_prev_        = ts_event_frame;
    generate(event_frame);
    return true;
}

void EventFrameDiffGenerationAlgorithm::reset() {
    frame_.reset();
    is_ts_prev_set_ = false;
}

} // namespace Metavision
