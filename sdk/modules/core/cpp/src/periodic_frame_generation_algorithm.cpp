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
    reslicer_([&](EventBufferReslicerAlgorithm::ConditionStatus slicing_status, timestamp processing_ts,
                  std::size_t n_processed_events) {
        this->process_new_slice(slicing_status, processing_ts, n_processed_events);
    }),
    force_next_frame_(false) {
    set_accumulation_time_us(accumulation_time_us);
    set_fps(fps);
    reset();
}

void PeriodicFrameGenerationAlgorithm::set_output_callback(const OutputCb &output_cb) {
    output_cb_ = output_cb;
}

void PeriodicFrameGenerationAlgorithm::notify_elapsed_time(timestamp ts) {
    reslicer_.notify_elapsed_time(ts);
}

void PeriodicFrameGenerationAlgorithm::force_generate() {
    force_next_frame_ = true;
    reslicer_.flush();
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

    reslicer_.set_slicing_condition(EventBufferReslicerAlgorithm::Condition::make_n_us(frame_period_us_));
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
    reslicer_.reset();

    reset_time_surface();
    next_frame_ts_us_ = 0;

    min_event_ts_us_to_use_ = 0;
}

void PeriodicFrameGenerationAlgorithm::process_new_slice(EventBufferReslicerAlgorithm::ConditionStatus slicing_status,
                                                         timestamp processing_ts, size_t n_processed_events) {
    if (processing_ts < next_frame_ts_us_ && !force_next_frame_)
        return;

    // Generate Frame using the time surface
    if (flags_ & Parameters::GRAY) {
        frame_.create(height_, width_, CV_8U);
    } else if (flags_ & Parameters::RGB || flags_ & Parameters::BGR) {
        frame_.create(height_, width_, CV_8UC3);
    } else {
        frame_.create(height_, width_, CV_8UC4);
    }

    // Compute the time threshold below which events are not to be displayed
    // N.B. min_event_ts_us_to_use_ might be wrong at the initialization.
    //      Let's subtract the accumulation time to the current processing timestamp
    const int32_t min_display_event_ts = static_cast<int32_t>((processing_ts - accumulation_time_us_) - ts_offset_);

    cv::Vec3b _bg_color3;
    cv::Vec4b _bg_color4;
    std::array<cv::Vec3b, 2> _off_on_colors3;
    std::array<cv::Vec4b, 2> _off_on_colors4;
    if (flags_ & Parameters::BGR || flags_ & Parameters::BGRA) {
        _bg_color3      = detail::bgr(bg_color_);
        _off_on_colors3 = {detail::bgr(off_on_colors_[0]), detail::bgr(off_on_colors_[1])};
        _bg_color4      = bg_color_;
        _off_on_colors4 = off_on_colors_;
    } else {
        _bg_color3      = detail::rgb(bg_color_);
        _off_on_colors3 = {detail::rgb(off_on_colors_[0]), detail::rgb(off_on_colors_[1])};
        _bg_color4      = detail::rgba(bg_color_);
        _off_on_colors4 = {detail::rgba(off_on_colors_[0]), detail::rgba(off_on_colors_[1])};
    }

    // Fill the frame from the time surface
    const size_t height = static_cast<size_t>(frame_.rows);
    const size_t width  = static_cast<size_t>(frame_.cols);
    // Matrices allocated with the create() method are always continuous in memory
    if (flags_ & Parameters::GRAY) {
        for (size_t y = 0; y < height; ++y) {
            const auto last_pix_data_ptr = &time_surface_[y * width];
            auto img_ptr                 = frame_.ptr<uint8_t>(flags_ & Parameters::FLIP_Y ? height - 1 - y : y);
            for (size_t x = 0; x < width; ++x) {
                img_ptr[x] = last_pix_data_ptr[x].first < min_display_event_ts ?
                                 bg_color_[0] :
                                 off_on_colors_[last_pix_data_ptr[x].second][0];
            }
        }
    } else if (flags_ & Parameters::RGB || flags_ & Parameters::BGR) {
        for (size_t y = 0; y < height; ++y) {
            const auto last_pix_data_ptr = &time_surface_[y * width];
            auto img_ptr                 = frame_.ptr<cv::Vec3b>(flags_ & Parameters::FLIP_Y ? height - 1 - y : y);
            for (size_t x = 0; x < width; ++x) {
                img_ptr[x] = last_pix_data_ptr[x].first < min_display_event_ts ?
                                 _bg_color3 :
                                 _off_on_colors3[last_pix_data_ptr[x].second];
            }
        }
    } else {
        for (size_t y = 0; y < height; ++y) {
            const auto last_pix_data_ptr = &time_surface_[y * width];
            auto img_ptr                 = frame_.ptr<cv::Vec4b>(flags_ & Parameters::FLIP_Y ? height - 1 - y : y);
            for (size_t x = 0; x < width; ++x) {
                img_ptr[x] = last_pix_data_ptr[x].first < min_display_event_ts ?
                                 _bg_color4 :
                                 _off_on_colors4[last_pix_data_ptr[x].second];
            }
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