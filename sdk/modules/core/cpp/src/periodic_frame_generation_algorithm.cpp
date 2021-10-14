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

#include <stdexcept>
#include "metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h"

namespace Metavision {

PeriodicFrameGenerationAlgorithm::PeriodicFrameGenerationAlgorithm(int sensor_width, int sensor_height,
                                                                   uint32_t accumulation_time_us, double fps,
                                                                   const Metavision::ColorPalette &palette) :
    BaseFrameGenerationAlgorithm(sensor_width, sensor_height, palette),
    output_cb_([](auto, auto) {}),
    force_next_frame_(false) {
    set_accumulation_time_us(accumulation_time_us);
    set_fps(fps);
    set_processing_n_us(frame_period_us_);
    reset();
}

void PeriodicFrameGenerationAlgorithm::set_output_callback(const OutputCb &output_cb) {
    output_cb_ = output_cb;
}

void PeriodicFrameGenerationAlgorithm::force_generate() {
    force_next_frame_ = true;
    AsyncAlgorithm<PeriodicFrameGenerationAlgorithm>::flush();
    force_next_frame_ = false;
}

void PeriodicFrameGenerationAlgorithm::set_fps(double fps) {
    if (fps < 0)
        throw std::invalid_argument("Frame rate must be positive.");

    // Computes the frame period from the input fps. If the input fps is 0, then the frame period
    // is the accumulation time.
    if (fps == 0)
        frame_period_us_ = accumulation_time_us_;
    else
        frame_period_us_ = static_cast<uint32_t>(std::round(1000000. / fps));

    set_processing_n_us(frame_period_us_);
}

double PeriodicFrameGenerationAlgorithm::get_fps() {
    return 1000000. / frame_period_us_;
}

void PeriodicFrameGenerationAlgorithm::set_accumulation_time_us(uint32_t accumulation_time_us) {
    if (accumulation_time_us <= 0)
        throw std::invalid_argument("Accumulation time must be strictly positive.");

    accumulation_time_us_   = accumulation_time_us;
    min_event_ts_us_to_use_ = next_frame_ts_us_ - accumulation_time_us_;
}

uint32_t PeriodicFrameGenerationAlgorithm::get_accumulation_time_us() {
    return accumulation_time_us_;
}

void PeriodicFrameGenerationAlgorithm::reset() {
    force_generate();
    AsyncAlgorithm<PeriodicFrameGenerationAlgorithm>::reset();

    reset_time_surface();
    next_frame_ts_us_ = 0;

    min_event_ts_us_to_use_ = 0;
}

void PeriodicFrameGenerationAlgorithm::process_async(const timestamp processing_ts, const size_t n_processed_events) {
    if (processing_ts < next_frame_ts_us_ && !force_next_frame_)
        return;

    // Generate Frame using the time surface
    frame_.create(height_, width_, colored_ ? CV_8UC3 : CV_8U);

    // Compute the time threshold below which events are not to be displayed
    // N.B. min_event_ts_us_to_use_ might be wrong at the initialization.
    //      Let's subtract the accumulation time to the current processing timestamp
    const int32_t min_display_event_ts = static_cast<int32_t>((processing_ts - accumulation_time_us_) - ts_offset_);

    // Fill the frame from the time surface
    const size_t num_pixels = time_surface_.size();
    if (colored_) {
        auto img_it = frame_.begin<cv::Vec3b>();
        for (size_t i = 0; i < num_pixels; ++i, ++img_it) {
            const auto &last_pix_data = time_surface_[i];
            *img_it = last_pix_data.first < min_display_event_ts ? bg_color_ : off_on_colors_[last_pix_data.second];
        }
    } else {
        auto img_it = frame_.begin<uint8_t>();
        for (size_t i = 0; i < num_pixels; ++i, ++img_it) {
            const auto &last_pix_data = time_surface_[i];
            *img_it =
                last_pix_data.first < min_display_event_ts ? bg_color_[0] : off_on_colors_[last_pix_data.second][0];
        }
    }

    // Return generate frame through the output callback
    output_cb_(processing_ts, frame_);

    // Increment internal variables
    next_frame_ts_us_       = processing_ts + frame_period_us_;
    min_event_ts_us_to_use_ = next_frame_ts_us_ - accumulation_time_us_;
}

void PeriodicFrameGenerationAlgorithm::skip_frames_up_to(timestamp ts) {
    next_frame_ts_us_ =
        std::max(next_frame_ts_us_, static_cast<timestamp>(frame_period_us_) *
                                        static_cast<timestamp>(ts / static_cast<double>(frame_period_us_)));
    min_event_ts_us_to_use_ = next_frame_ts_us_ - accumulation_time_us_;
}

void PeriodicFrameGenerationAlgorithm::reset_time_surface() {
    time_surface_.resize(width_ * height_);
    std::fill(time_surface_.begin(), time_surface_.end(), std::make_pair(std::numeric_limits<int32_t>::min(), false));
    ts_offset_ = 0;
}

} // namespace Metavision