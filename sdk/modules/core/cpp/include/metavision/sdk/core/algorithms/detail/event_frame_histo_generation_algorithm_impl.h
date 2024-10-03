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

#include "metavision/sdk/core/algorithms/event_frame_histo_generation_algorithm.h"

namespace Metavision {

template<typename InputIt>
EventFrameHistoGenerationAlgorithm<InputIt>::EventFrameHistoGenerationAlgorithm(unsigned width, unsigned height,
                                                                                unsigned channel_bit_neg,
                                                                                unsigned channel_bit_pos, bool packed,
                                                                                timestamp min_generation_period_us) :
    cfg_({width, height, {channel_bit_neg, channel_bit_pos}, packed}),
    min_generation_period_us_(min_generation_period_us),
    frame_unpacked_(height, width, channel_bit_neg, channel_bit_pos, false),
    processor_(width, height, (1 << channel_bit_neg) - 1, (1 << channel_bit_pos) - 1) {
    if (channel_bit_neg <= 0 || channel_bit_pos <= 0 || channel_bit_neg + channel_bit_pos > 8)
        throw std::runtime_error("Incorrect input channel_bit_neg and channel_bit_pos.");
}

template<typename InputIt>
void EventFrameHistoGenerationAlgorithm<InputIt>::process_events(InputIt it_begin, InputIt it_end) {
    processor_.process_events(it_begin, it_end, frame_unpacked_);
}

template<typename InputIt>
void EventFrameHistoGenerationAlgorithm<InputIt>::generate(RawEventFrameHisto &frame) {
    if (cfg_.packed) {
        frame.reset(cfg_.height, cfg_.width, cfg_.channel_bit_size[0], cfg_.channel_bit_size[1],
                    cfg_.packed); // Prepare target frame
        auto &histo_out      = frame.get_data();
        auto &histo_unpacked = frame_unpacked_.get_data();
        for (unsigned int npixels = cfg_.width * cfg_.height, idx_px = 0; idx_px < npixels; ++idx_px) {
            const uint8_t bitval_neg = histo_unpacked[2 * idx_px];
            const uint8_t bitval_pos = histo_unpacked[2 * idx_px + 1];
            histo_out[idx_px]        = (bitval_pos << cfg_.channel_bit_size[0]) | (bitval_neg);
        }
        frame_unpacked_.reset(); // Prepare next accumulating frame
    } else {
        frame.reset(cfg_.height, cfg_.width, cfg_.channel_bit_size[0], cfg_.channel_bit_size[1],
                    cfg_.packed);    // Prepare next accumulating frame
        frame.swap(frame_unpacked_); // Swap internal event frame with provided one
    }
}

template<typename InputIt>
bool EventFrameHistoGenerationAlgorithm<InputIt>::generate(timestamp ts_event_frame, RawEventFrameHisto &event_frame) {
    if (is_ts_prev_set_ && ts_event_frame - ts_prev_ < min_generation_period_us_)
        return false;
    is_ts_prev_set_ = true;
    ts_prev_        = ts_event_frame;
    generate(event_frame);
    return true;
}

template<typename InputIt>
void EventFrameHistoGenerationAlgorithm<InputIt>::reset() {
    frame_unpacked_.reset();
    is_ts_prev_set_ = false;
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_EVENT_FRAME_HISTO_GENERATION_ALGORITHM_IMPL_H
