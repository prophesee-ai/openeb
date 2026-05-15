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

#include "metavision/sdk/core/algorithms/event_count_frame_generation_algorithm.h"
#include <algorithm>

namespace Metavision {

EventCountFrameGenerationAlgorithm::EventCountFrameGenerationAlgorithm(int sensor_width, int sensor_height,
                                                                       uint32_t events_per_frame,
                                                                       const Metavision::ColorPalette &palette) :
    BaseFrameGenerationAlgorithm(sensor_width, sensor_height, palette),
    events_per_frame_(events_per_frame),
    accumulated_events_count_(0),
    sliding_mode_(false),
    hop_events_(events_per_frame),
    unprocessed_since_last_frame_(0) {
    
    // Always initialize with 3-channel color frame (CV_8UC3) to match CDFrameGenerator expectations
    frame_ = cv::Mat(sensor_height, sensor_width, CV_8UC3);
    
    // Initialize time surface (memory efficient: O(width*height) instead of O(events))
    time_surface_.resize(sensor_width * sensor_height, {0, 0});
    reset();
}

void EventCountFrameGenerationAlgorithm::set_sliding_mode(bool sliding) {
    sliding_mode_ = sliding;
}

void EventCountFrameGenerationAlgorithm::set_hop_events(uint32_t hop_events) {
    hop_events_ = hop_events == 0 ? events_per_frame_ : hop_events;
}

uint32_t EventCountFrameGenerationAlgorithm::get_hop_events() const {
    return hop_events_;
}

void EventCountFrameGenerationAlgorithm::set_output_callback(const OutputCb &output_cb) {
    output_cb_ = output_cb;
}

void EventCountFrameGenerationAlgorithm::set_events_per_frame(uint32_t events_per_frame) {
    events_per_frame_ = events_per_frame;
}

uint32_t EventCountFrameGenerationAlgorithm::get_events_per_frame() const {
    return events_per_frame_;
}

uint32_t EventCountFrameGenerationAlgorithm::get_accumulated_events_count() const {
    return accumulated_events_count_;
}

void EventCountFrameGenerationAlgorithm::force_generate(timestamp ts) {
    if (accumulated_events_count_ > 0) {
        generate_frame(ts);
        accumulated_events_count_ = 0;
    }
}

void EventCountFrameGenerationAlgorithm::reset() {
    accumulated_events_count_ = 0;
    std::fill(time_surface_.begin(), time_surface_.end(), std::make_pair(0, 0));
    ts_offset_ = 0;
    event_window_.clear();
    unprocessed_since_last_frame_ = 0;
}

void EventCountFrameGenerationAlgorithm::update_time_surface(const EventCD &event) {
    // Handle time overflow
    while (event.t > ts_offset_ + std::numeric_limits<int32_t>::max()) {
        ts_offset_ += std::numeric_limits<int32_t>::max();
        for (auto &pix_data : time_surface_) {
            pix_data.first =
                pix_data.first >= std::numeric_limits<int32_t>::min() + std::numeric_limits<int32_t>::max() ?
                    pix_data.first - std::numeric_limits<int32_t>::max() :
                    std::numeric_limits<int32_t>::min();
        }
    }
    
    const int32_t it_t = static_cast<int32_t>(event.t - ts_offset_);
    time_surface_[event.y * width_ + event.x] = {it_t, event.p};
}

void EventCountFrameGenerationAlgorithm::generate_frame(timestamp ts) {
    if (output_cb_) {
        // Ensure frame is properly allocated with correct type and dimensions
        if (frame_.empty() || frame_.type() != CV_8UC3 || 
            frame_.rows != height_ || frame_.cols != width_) {
            frame_ = cv::Mat(height_, width_, CV_8UC3);
        }
        
        // Fill frame with background color (use only first 3 channels)
        cv::Vec3b bg_color_3ch(bg_color_[0], bg_color_[1], bg_color_[2]);
        frame_.setTo(bg_color_3ch);
        
        // Render time surface to frame
        render_time_surface_to_frame();
        
        output_cb_(ts, frame_);
        
        // Reset time surface for next frame
        std::fill(time_surface_.begin(), time_surface_.end(), std::make_pair(0, 0));
    }
}

void EventCountFrameGenerationAlgorithm::render_time_surface_to_frame() {
    // Render each pixel in the time surface to the frame
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const auto &pix_data = time_surface_[y * width_ + x];
            const bool polarity = pix_data.second;
            
            if (pix_data.first != 0) {  // If pixel has been updated
                // Extract BGR values from the 4-channel color
                frame_.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    off_on_colors_[polarity][0],
                    off_on_colors_[polarity][1],
                    off_on_colors_[polarity][2]
                );
            }
        }
    }
}

} // namespace Metavision
