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

#ifndef METAVISION_SDK_CORE_ON_DEMAND_FRAME_GENERATION_ALGORITHM_H
#define METAVISION_SDK_CORE_ON_DEMAND_FRAME_GENERATION_ALGORITHM_H

#include <assert.h>
#include <deque>

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/core/algorithms/base_frame_generation_algorithm.h"

namespace Metavision {

/// @brief Algorithm that generates a CD frame on demand
///
/// After providing events to the class through the @ref process_events method, the user can request a frame generation
/// at any timestamp using the @ref generate method. Note that @ref generate is expected to be called with timestamps
/// increasing monotonically.
///
/// The class allows managing the generation of overlapping frames, i.e. the time between two consecutive frame
/// generations can be shorter than the accumulation time.
///
/// @note It's possible to generate a frame at any timestamp even though more recent events have already been provided.
/// It allows the user not to worry about the event buffers he or she is sending.
///
/// @warning This class shouldn't be used in case the user prefers to register to an output callback rather than having
/// to manually ask the algorithm to generate the frames (See @ref PeriodicFrameGenerationAlgorithm).
class OnDemandFrameGenerationAlgorithm : public BaseFrameGenerationAlgorithm {
public:
    /// @brief Constructor
    /// @param width Sensor's width (in pixels)
    /// @param height Sensor's height (in pixels)
    /// @param accumulation_time_us Time range of events to update the frame with (in us)
    /// (See @ref set_accumulation_time_us)
    /// @param palette The Prophesee's color palette to use
    OnDemandFrameGenerationAlgorithm(int width, int height, uint32_t accumulation_time_us = 0,
                                     const Metavision::ColorPalette &palette = default_palette());

    /// @brief Processes a buffer of events
    /// @warning Call @ref reset before starting processing events from a timestamp in the past
    /// @tparam EventIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to the first input event
    /// @param it_end Iterator to the past-the-end event
    /// @warning This method is expected to be called with timestamps increasing monotonically and events from the past
    template<typename EventIt>
    void process_events(EventIt it_begin, EventIt it_end);

    /// @brief Generates a frame
    /// @param ts Timestamp at which to generate the frame
    /// @param frame Frame that will be filled with CD events
    /// @param allocate Allocates the frame if true. Otherwise, the user must ensure the validity of the input frame.
    /// This is to be used when the data ptr must not change (external allocation, ROI over another cv::Mat, ...)
    /// @warning This method is expected to be called with timestamps increasing monotonically.
    /// @throw invalid_argument if @p ts is older than the last frame generation and @ref reset method hasn't
    /// been called in the meantime
    /// @throw invalid_argument if the frame doesn't have the expected type and geometry
    void generate(timestamp ts, cv::Mat &frame, bool allocate = true);

    /// @brief Sets the accumulation time (in us) to use to generate a frame
    ///
    /// Frame generated will only hold events in the interval [t - dt, t[ where t is the timestamp at
    /// which the frame is generated, and dt the accumulation time.
    /// However, if @p accumulation_time_us is set to 0, all events since the last generated frame are used
    ///
    /// @param accumulation_time_us Time range of events to update the frame with (in us)
    void set_accumulation_time_us(uint32_t accumulation_time_us);

    /// @brief Returns the current accumulation time (in us).
    uint32_t get_accumulation_time_us() const;

    /// @brief Resets the internal states
    ///
    /// The method @ref generate must be called with timestamps increasing monotonically. However there are no
    /// constraints on the timestamp of the first generation following the reset call. It allows the user to restart the
    /// algo from any timestamp
    void reset();

private:
    uint32_t accumulation_time_us_; ///< Accumulation time of the events to generate the frame
    timestamp last_frame_ts_us_;    ///< Timestamp of the last generated frame
    std::deque<EventCD> events_queue_;
};

template<typename EventIt>
inline void OnDemandFrameGenerationAlgorithm::process_events(EventIt it_begin, EventIt it_end) {
    events_queue_.insert(events_queue_.end(), it_begin, it_end);
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_ON_DEMAND_FRAME_GENERATION_ALGORITHM_H