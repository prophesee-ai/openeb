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

#include "metavision/sdk/core/algorithms/event_frame_diff_generation_algorithm.h"

namespace Metavision {

template<typename InputIt>
EventFrameDiffGenerationAlgorithm<InputIt>::EventFrameDiffGenerationAlgorithm(unsigned int width, unsigned int height,
                                                                              unsigned int bit_size,
                                                                              bool allow_rollover,
                                                                              timestamp min_generation_period_us) :
    allow_rollover_(allow_rollover),
    min_generation_period_us_(min_generation_period_us),
    frame_(height, width, bit_size),
    processor_(width, height, -(1 << (bit_size - 1)), (1 << (bit_size - 1)) - 1, allow_rollover) {
    reset_wrapper();
}

template<typename InputIt>
void EventFrameDiffGenerationAlgorithm<InputIt>::process_events(InputIt it_begin, InputIt it_end) {
    processor_.process_events(0, it_begin, it_end, diff_wrapper_);
}

template<typename InputIt>
void EventFrameDiffGenerationAlgorithm<InputIt>::generate(RawEventFrameDiff &frame) {
    const auto &cfg = frame_.get_config();
    frame.reset(cfg.height, cfg.width, cfg.bit_size); // Prepare next accumulating frame
    frame.swap(frame_);                               // Swap internal event frame with provided one
    reset_wrapper();
}

template<typename InputIt>
bool EventFrameDiffGenerationAlgorithm<InputIt>::generate(timestamp ts_event_frame, RawEventFrameDiff &event_frame) {
    if (is_ts_prev_set_ && ts_event_frame - ts_prev_ < min_generation_period_us_)
        return false;
    is_ts_prev_set_ = true;
    ts_prev_        = ts_event_frame;
    generate(event_frame);
    return true;
}

template<typename InputIt>
void EventFrameDiffGenerationAlgorithm<InputIt>::reset() {
    frame_.reset();
    reset_wrapper();
    is_ts_prev_set_ = false;
}

template<typename InputIt>
void EventFrameDiffGenerationAlgorithm<InputIt>::reset_wrapper() {
    diff_wrapper_.create(processor_.get_output_shape(), processor_.get_output_type(),
                         reinterpret_cast<std::byte *>(frame_.get_data().data()), false);
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_EVENT_FRAME_DIFF_GENERATION_ALGORITHM_IMPL_H
