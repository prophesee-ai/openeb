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

#ifndef METAVISION_SDK_CORE_EVENT_COUNT_FRAME_GENERATION_ALGORITHM_H
#define METAVISION_SDK_CORE_EVENT_COUNT_FRAME_GENERATION_ALGORITHM_H

#include <functional>
#include <limits>
#include <deque>
#include <algorithm>
#include "metavision/sdk/core/algorithms/base_frame_generation_algorithm.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/base/events/event_cd.h"

namespace Metavision {

/// @brief Algorithm that generates frames from a fixed number of accumulated events
/// 
/// This algorithm generates a new frame every time a specified number of events have been accumulated.
/// Unlike PeriodicFrameGenerationAlgorithm which generates frames at a fixed time rate, this algorithm
/// generates frames based on event count, making it ideal for event-driven processing.
class EventCountFrameGenerationAlgorithm : public BaseFrameGenerationAlgorithm {
public:
    /// @brief Alias for frame generated callback
    using OutputCb = std::function<void(timestamp, cv::Mat &)>;

    /// @brief Constructor
    /// @param sensor_width Sensor's width (in pixels)
    /// @param sensor_height Sensor's height (in pixels)
    /// @param events_per_frame Number of events to accumulate before generating a frame
    /// @param palette The Prophesee's color palette to use
    EventCountFrameGenerationAlgorithm(int sensor_width, int sensor_height, uint32_t events_per_frame = 10000,
                                       const Metavision::ColorPalette &palette = default_palette());

    /// @brief Sets the callback to call when an image has been generated
    ///
    /// @warning For efficiency purpose, the frame passed in the callback is a non const reference. If it is to be
    /// used outside the scope of the callback, the user must ensure to swap or copy it to another object
    void set_output_callback(const OutputCb &output_cb);

    /// @brief Processes a buffer of events to update the internal time surface for the frame generation
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to the first input event
    /// @param it_end Iterator to the past-the-end event
    template<typename EventIt>
    inline void process_events(EventIt it_begin, EventIt it_end);

    /// @brief Sets the number of events to accumulate before generating a frame
    /// @param events_per_frame Number of events per frame
    void set_events_per_frame(uint32_t events_per_frame);

    /// @brief Enable or disable sliding (overlapping) window mode
    /// @param sliding If true, generate frames using a sliding window with hop size set by set_hop_events
    void set_sliding_mode(bool sliding);

    /// @brief Sets the hop (stride) in events between successive frames when sliding is enabled
    /// @param hop_events Number of events between successive frames
    void set_hop_events(uint32_t hop_events);

    /// @brief Returns the currently configured hop events
    uint32_t get_hop_events() const;

    /// @brief Returns the current number of events per frame
    uint32_t get_events_per_frame() const;

    /// @brief Returns the number of events accumulated since the last frame
    uint32_t get_accumulated_events_count() const;

    /// @brief Forces the generation of a frame with the currently accumulated events
    /// @param ts Timestamp of the frame
    void force_generate(timestamp ts);

    /// @brief Resets the internal states
    void reset();

private:
    void generate_frame(timestamp ts);
    void update_time_surface(const EventCD &event);
    void render_time_surface_to_frame();

    // Sliding window members
    bool sliding_mode_{false};
    uint32_t hop_events_{0};
    std::deque<EventCD> event_window_;            ///< Ring-like window storing recent events when sliding
    size_t unprocessed_since_last_frame_{0};

    OutputCb output_cb_;                ///< The callback to call when a frame is generated
    cv::Mat frame_;                     ///< Internal image that is filled and output
    uint32_t events_per_frame_;         ///< Number of events to accumulate before generating a frame
    uint32_t accumulated_events_count_; ///< Current count of accumulated events
    
    // Time surface for efficient frame generation (memory efficient)
    std::vector<std::pair<int32_t, bool>> time_surface_; ///< Pixels' history (timestamp and polarity)
    timestamp ts_offset_{0}; ///< State variable to handle time overflow
};

template<typename EventIt>
void EventCountFrameGenerationAlgorithm::process_events(EventIt it_begin, EventIt it_end) {
    const auto n = static_cast<size_t>(std::distance(it_begin, it_end));
    if (n == 0)
        return;

    if (!sliding_mode_) {
        // Non-sliding behavior: accumulate events and generate every events_per_frame_
        for (auto it = it_begin; it != it_end; ++it) {
            update_time_surface(*it);
            ++accumulated_events_count_;

            if (accumulated_events_count_ >= events_per_frame_) {
                generate_frame(it->t);
                accumulated_events_count_ = 0;
            }
        }
        return;
    }

    // Sliding mode: keep a deque window of recent events and generate frames every hop_events_
    // Append incoming events to the window
    for (auto it = it_begin; it != it_end; ++it) {
        event_window_.push_back(*it);
    }

    unprocessed_since_last_frame_ += n;

    // Cap window size to avoid unbounded memory growth: keep at most events_per_frame_ + hop_events_
    const size_t max_window = static_cast<size_t>(events_per_frame_) + static_cast<size_t>(hop_events_);
    while (event_window_.size() > max_window) {
        event_window_.pop_front();
    }

    // Generate frames while enough new events have arrived (at least hop_events_) and we have enough windowed events
    while (unprocessed_since_last_frame_ >= hop_events_ && event_window_.size() >= events_per_frame_) {
        // Rebuild time surface from the last events_per_frame_ events
        std::fill(time_surface_.begin(), time_surface_.end(), std::make_pair(0, 0));

        auto start_it = event_window_.end() - static_cast<std::ptrdiff_t>(events_per_frame_);
        for (auto it = start_it; it != event_window_.end(); ++it) {
            update_time_surface(*it);
        }

        // Use timestamp of the last event as frame timestamp
        const timestamp frame_ts = event_window_.empty() ? 0 : event_window_.back().t;
        generate_frame(frame_ts);

        // Slide the window by popping hop_events_ events from front
        for (uint32_t i = 0; i < hop_events_ && !event_window_.empty(); ++i) {
            event_window_.pop_front();
        }

        if (unprocessed_since_last_frame_ >= hop_events_)
            unprocessed_since_last_frame_ -= hop_events_;
        else
            unprocessed_since_last_frame_ = 0;
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_EVENT_COUNT_FRAME_GENERATION_ALGORITHM_H
