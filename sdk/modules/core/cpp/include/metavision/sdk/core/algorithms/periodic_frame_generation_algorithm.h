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

#ifndef METAVISION_SDK_CORE_PERIODIC_FRAME_GENERATION_ALGORITHM_H
#define METAVISION_SDK_CORE_PERIODIC_FRAME_GENERATION_ALGORITHM_H

#include <functional>

#include "metavision/sdk/core/algorithms/base_frame_generation_algorithm.h"
#include "metavision/sdk/core/algorithms/event_buffer_reslicer_algorithm.h"
#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

class PeriodicFrameGenerationAlgorithm;

/// @brief Algorithm that generates frames from events at a fixed rate (fps). The reference clock used is the one of
/// the input events.
///
/// As an asynchronous algorithm, this class processes any stream of events and triggers the frames generation by
/// itself to output frames regularly spaced in time through its output callback. Note that the implementation of this
/// class uses EventBufferReslicerAlgorithm in place of AsyncAlgorithm, with the same overall behavior.
///
/// The elapsed time between two frame generations (frame period = 1 / fps) and the accumulation time can be updated
/// on the fly.
///
/// This class should be used when the user prefers to register to an output callback than having to manually ask the
/// algorithm to generate the frames for increasing multiples of the frame period.
/// However, this class shouldn't be used in case the user wants to generate a frame inside the callback of an
/// algorithm (See @ref OnDemandFrameGenerationAlgorithm)
class PeriodicFrameGenerationAlgorithm : public BaseFrameGenerationAlgorithm {
public:
    /// @brief Alias for frame generated callback
    using OutputCb = std::function<void(timestamp, cv::Mat &)>;

    /// @brief Constructor
    /// @param sensor_width Sensor's width (in pixels)
    /// @param sensor_height Sensor's height (in pixels)
    /// @param accumulation_time_us Time range of events to update the frame with (in us)
    /// @param fps The fps at which to generate the frames. The time reference used is the one from the input events. If
    /// the fps is 0, the accumulation time is used to compute it (@ref set_fps).
    /// @param palette The Prophesee's color palette to use (@ref set_color_palette)
    /// @throw std::invalid_argument If the input fps is negative
    PeriodicFrameGenerationAlgorithm(int sensor_width, int sensor_height, uint32_t accumulation_time_us = 10000,
                                     double fps = 0., const Metavision::ColorPalette &palette = default_palette());

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

    /// @brief Notify the frame generator that time has elapsed without new events, which may trigger several calls
    /// to the image generated callback depending on the configured slicing condition.
    /// @param ts current timestamp
    void notify_elapsed_time(timestamp ts);

    /// @brief Forces the generation of a frame for the current period with the input events that have been processed
    ///
    /// This is intended to be used at the end of a process if one wants to generate frames with the remaining events
    /// This effectively calls the output_cb and updates the next timestamp at which a frame is to be generated
    void force_generate();

    /// @brief Sets the fps at which to generate frames and thus the frequency of the asynchronous calls
    ///
    /// The time reference used is the one from the input events
    ///
    /// @param fps The fps to use. If the fps is 0, the current accumulation time is used to compute it
    /// @throw std::invalid_argument If the input fps is negative
    void set_fps(double fps);

    /// @brief Returns the current fps at which frames are generated
    double get_fps();

    /// @brief Sets the accumulation time (in us) to use to generate a frame
    ///
    /// Frame generated will only hold events in the interval [t - dt, t[ where t is the timestamp at
    /// which the frame is generated, and dt the accumulation time
    ///
    /// @param accumulation_time_us Time range of events to update the frame with (in us)
    void set_accumulation_time_us(uint32_t accumulation_time_us);

    /// @brief Returns the current accumulation time (in us).
    uint32_t get_accumulation_time_us();

    /// @brief Skips the generation of frames up to the timestamp @p ts
    /// @param ts Timestamp up to which only one image will be generated, i.e. the closest full timeslice
    /// before this timestamp
    void skip_frames_up_to(timestamp ts);

    /// @brief Resets the internal states
    /// @warning the user is responsible for explicitly calling @ref force_generate if needed to retrieve the frame for
    /// the last processed events.
    void reset();

private:
    template<typename EventIt>
    inline void process_event_buffer(EventIt it_begin, EventIt it_end);

    /// @brief Generates a frame from the events history and accumulation time
    ///
    /// This method is called at the input fps frequency
    void process_new_slice(EventBufferReslicerAlgorithm::ConditionStatus slicing_status, timestamp processing_ts,
                           size_t n_processed_events);

    /// @brief Resets the time surface
    void reset_time_surface();

    OutputCb output_cb_; ///< The callback to call when a frame is generated

    EventBufferReslicerAlgorithm reslicer_; ///< Event buffer reslicer algorithm.

    cv::Mat frame_;            ///< Internal image that is filled when asynchronous condition is met
    uint32_t frame_period_us_; ///< Period (in us) between two frames generation (i.e. period between two calls to
                               ///< asynchronous processing calls). The period is measured with the input events'
                               ///< timestamp

    uint32_t accumulation_time_us_;    ///< Accumulation time of the events to generate the frame
    timestamp next_frame_ts_us_;       ///< The next frame generated timestamp
    timestamp min_event_ts_us_to_use_; ///< The lower bound of time interval of events to process

    bool force_next_frame_; ///< Flag indicating if the next frame must be generated no matter the timestamp of the
                            /// processed time slice

    // Time surface
    std::vector<std::pair<int32_t, bool>> time_surface_; ///< Pixels' history (time surface). This object stores the
                                                         ///< last events data that occurred at a given pixel
    timestamp ts_offset_{0}; ///< State variable to handle time overflow in the time surface. This is to minimize the
                             ///< memory footprint of the time surface access
};

template<typename EventIt>
void PeriodicFrameGenerationAlgorithm::process_events(EventIt it_begin, EventIt it_end) {
    reslicer_.process_events(it_begin, it_end,
                             [&](EventIt it_begin, EventIt it_end) { this->process_event_buffer(it_begin, it_end); });
}

template<typename EventIt>
void PeriodicFrameGenerationAlgorithm::process_event_buffer(EventIt it_begin, EventIt it_end) {
    if (std::distance(it_begin, it_end) == 0)
        return;

    if (std::prev(it_end)->t < min_event_ts_us_to_use_)
        return; // No event to process

    if (it_begin->t < min_event_ts_us_to_use_) {
        // Time slice begins is in the middle of the input events
        it_begin = std::lower_bound(it_begin, it_end, min_event_ts_us_to_use_,

                                    [](const auto &ev, const timestamp ts) { return ev.t < ts; });
    }

    // Add events in the time surface

    // Checks time overflow. If one occurs, updates the time offset to apply and the timesurface accordingly.
    while (std::prev(it_end)->t > ts_offset_ + std::numeric_limits<int32_t>::max()) {
        ts_offset_ += std::numeric_limits<int32_t>::max();
        for (auto &pix_data : time_surface_) {
            pix_data.first =
                pix_data.first >= std::numeric_limits<int32_t>::min() + std::numeric_limits<int32_t>::max() ?
                    pix_data.first - std::numeric_limits<int32_t>::max() :
                    std::numeric_limits<int32_t>::min();
        }
    }

    // Refresh the time-surface using the event buffer
    for (auto it = it_begin; it != it_end; ++it) {
        const int32_t it_t                    = static_cast<int32_t>(it->t - ts_offset_);
        time_surface_[it->y * width_ + it->x] = {it_t, it->p};
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_PERIODIC_FRAME_GENERATION_ALGORITHM_H