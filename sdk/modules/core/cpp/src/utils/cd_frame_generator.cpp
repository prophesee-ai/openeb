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

#include "metavision/sdk/core/utils/cd_frame_generator.h"
#include "metavision/sdk/core/utils/colors.h"

namespace Metavision {

CDFrameGenerator::CDFrameGenerator(long width, long height, bool process_all_frames) :
    process_all_frames_(process_all_frames) {
    // Builds algo
    frame_generation_algo_.reset(new PeriodicFrameGenerationAlgorithm(width, height));

    // Update accumulation time
    set_display_accumulation_time_us(frame_generation_algo_->get_accumulation_time_us());
    next_notify_us_ = notify_slice_us_;

    // Update display colors
    for (int i = 0; i < 3; ++i) {
        background_color_[i] = PeriodicFrameGenerationAlgorithm::bg_color_default()[i];
        on_color_[i]         = PeriodicFrameGenerationAlgorithm::on_color_default()[i];
        off_color_[i]        = PeriodicFrameGenerationAlgorithm::off_color_default()[i];
    }

    set_colors(background_color_, on_color_, off_color_, true);
}

CDFrameGenerator::~CDFrameGenerator() {
    stop();
}

void CDFrameGenerator::set_colors(const cv::Scalar &background_color, const cv::Scalar &on_color,
                                  const cv::Scalar &off_color, bool colored) {
    std::lock_guard<std::mutex> lock(processing_mutex_);
    off_color_        = off_color;
    on_color_         = on_color;
    background_color_ = background_color;
    colored_          = colored;
}

void CDFrameGenerator::set_color_palette(const Metavision::ColorPalette &palette) {
    std::lock_guard<std::mutex> lock(processing_mutex_);
    off_color_        = get_bgr_color(palette, ColorType::Negative);
    on_color_         = get_bgr_color(palette, ColorType::Positive);
    background_color_ = get_bgr_color(palette, ColorType::Background);
    colored_          = palette != ColorPalette::Gray;
}

void CDFrameGenerator::add_events(const EventCD *begin, const EventCD *end) {
    if (begin == end) {
        return;
    }

    if (stop_ && !process_all_frames_) {
        // if the generator is not yet started and we don't have to process all frames, we don't need to keep the events
        return;
    }

    std::lock_guard<std::mutex> lock(processing_mutex_);
    // Note: one could call frame_generation_algorithm->process_events directly but it may have a high overhead
    // depending on the inputs.and decreases the performance. Better ensure that bigger chunks of data are processed
    events_back_.insert(events_back_.end(), begin, end);
    if (std::prev(end)->t > next_notify_us_) {
        events_available_ = true;
        next_notify_us_   = notify_slice_us_ * (1 + begin->t / notify_slice_us_);
        events_available_cond_.notify_all();
    }
}

void CDFrameGenerator::set_display_accumulation_time_us(timestamp display_accumulation_time_us) {
    std::lock_guard<std::mutex> lock(processing_mutex_);
    accumulation_time_us_ = display_accumulation_time_us;
    notify_slice_us_      = std::max(timestamp(100), display_accumulation_time_us / 3);
}

bool CDFrameGenerator::generate() {
    {
        std::unique_lock<std::mutex> lock(processing_mutex_);
        events_available_cond_.wait(lock, [this]() { return events_available_ || stop_; });
        events_front_.clear();
        events_front_.swap(events_back_);
        events_available_ = false;

        frame_generation_algo_->set_accumulation_time_us(accumulation_time_us_);
        frame_generation_algo_->set_colors(background_color_, on_color_, off_color_, colored_);
    }

    if (!process_all_frames_ && !events_front_.empty()) {
        // Generates only the last possible frame
        frame_generation_algo_->skip_frames_up_to(events_front_.back().t);
    }

    frame_generation_algo_->process_events(events_front_.cbegin(), events_front_.cend());
    if (stop_) {
        frame_generation_algo_->force_generate();
    }

    for (size_t i = 0; i < frames_count_; ++i) {
        // If process_all_frames is false, frames_count is 1 at most
        frame_cb_(frames_[i].ts_us_, frames_[i].frame_);
    }
    frames_count_ = 0;

    return !stop_;
}

bool CDFrameGenerator::start(std::uint16_t fps, const PeriodicFrameGenerationAlgorithm::OutputCb &cb) {
    auto ret = processing_thread_.start();
    if (!ret) {
        return false;
    }

    // Init algos and state variables
    frame_generation_algo_->reset();
    frame_generation_algo_->set_fps(fps);
    frame_cb_ = cb ? cb : [](auto, auto) {};

    frame_generation_algo_->set_output_callback([this](timestamp frame_ts_us, cv::Mat &mat) {
        if (frames_count_ == frames_.size()) {
            frames_.resize(frames_.size() + 1);
        }
        std::swap(frames_[frames_count_].frame_, mat);
        frames_[frames_count_].ts_us_ = frame_ts_us;
        ++frames_count_;
    });

    stop_ = false;

    processing_thread_.add_repeating_task(std::bind(&CDFrameGenerator::generate, this));

    return true;
}

bool CDFrameGenerator::stop() {
    stop_ = true;
    events_available_cond_.notify_all();
    if (process_all_frames_) {
        processing_thread_.stop();
    } else {
        processing_thread_.abort();
    }

    frames_.clear();
    frames_.shrink_to_fit();
    frames_count_ = 0;

    return true;
}

void CDFrameGenerator::reset() {
    std::lock_guard<std::mutex> lock(processing_mutex_);
    frame_generation_algo_->reset();
    events_back_.clear();
    next_notify_us_ = notify_slice_us_;
}

} // namespace Metavision
