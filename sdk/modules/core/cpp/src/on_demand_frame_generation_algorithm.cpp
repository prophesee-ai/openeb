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

#include <sstream>

#include "metavision/sdk/core/algorithms/on_demand_frame_generation_algorithm.h"

namespace Metavision {

OnDemandFrameGenerationAlgorithm::OnDemandFrameGenerationAlgorithm(int width, int height, uint32_t accumulation_time_us,
                                                                   const Metavision::ColorPalette &palette) :
    BaseFrameGenerationAlgorithm(width, height, palette), accumulation_time_us_(accumulation_time_us) {
    reset();
}

void OnDemandFrameGenerationAlgorithm::generate(timestamp ts, cv::Mat &frame, bool allocate) {
    if (allocate) {
        if (flags_ & Parameters::GRAY) {
            frame.create(height_, width_, CV_8U);
        } else if (flags_ & Parameters::RGB || flags_ & Parameters::BGR) {
            frame.create(height_, width_, CV_8UC3);
        } else {
            frame.create(height_, width_, CV_8UC4);
        }
    }

    if (ts < last_frame_ts_us_) {
        std::ostringstream ss;
        ss << "Call the method reset() before generating frames in the past. Last frame was generated at "
           << last_frame_ts_us_ << " and current frame generation is at " << ts << ".";
        throw std::invalid_argument(ss.str());
    }

    if (frame.rows != height_ || frame.cols != width_) {
        std::ostringstream ss;
        ss << "Incompatible matrix size. Must be (" << height_ << ", " << width_ << ").";
        throw std::invalid_argument(ss.str());
    }

    const timestamp ts_min = accumulation_time_us_ == 0 ? last_frame_ts_us_ + 1 : ts - accumulation_time_us_ + 1;
    const auto begin       = std::lower_bound(events_queue_.begin(), events_queue_.end(), ts_min,
                                        [](const auto &ev, timestamp t) { return ev.t < t; });
    const auto end =
        std::upper_bound(begin, events_queue_.end(), ts, [](timestamp t, const auto &ev) { return t < ev.t; });

    // Generate frame using events from the queue
    generate_frame_from_events(begin, end, frame, bg_color_, off_on_colors_, flags_);
    // Remove events older than ts - accumulation_time,
    // Or remove all the processed events if the accumulation time is null
    events_queue_.erase(events_queue_.begin(), (accumulation_time_us_ == 0 ? end : begin));

    last_frame_ts_us_ = ts;
}

void OnDemandFrameGenerationAlgorithm::set_accumulation_time_us(uint32_t accumulation_time_us) {
    if (accumulation_time_us == 0)
        throw std::invalid_argument("Accumulation time must be strictly positive.");

    accumulation_time_us_ = accumulation_time_us;
}

uint32_t OnDemandFrameGenerationAlgorithm::get_accumulation_time_us() const {
    return accumulation_time_us_;
}

void OnDemandFrameGenerationAlgorithm::reset() {
    events_queue_.clear();
    last_frame_ts_us_ = 0;
}

} // namespace Metavision