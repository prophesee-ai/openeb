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

#ifndef METAVISION_SDK_CORE_TIME_DECAY_FRAME_GENERATION_ALGORITHM_H
#define METAVISION_SDK_CORE_TIME_DECAY_FRAME_GENERATION_ALGORITHM_H

#include <assert.h>
#include <deque>
#include <opencv2/core/core.hpp>

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/core/utils/mostrecent_timestamp_buffer.h"
#include "metavision/sdk/core/utils/colors.h"

namespace Metavision {

/// @brief Algorithm that generates a time decay visualization of input CD events on demand
///
/// After processing events through the `process_events` method, the user can request the generation of a time decay
/// visualization at the current timestamp using the `generate` method.
class TimeDecayFrameGenerationAlgorithm {
public:
    /// @brief Constructor
    /// @param width Sensor's width (in pixels)
    /// @param height Sensor's height (in pixels)
    /// @param exponential_decay_time_us Characteristic time for the exponential decay (in us)
    /// @param palette The color palette to use for the visualization
    TimeDecayFrameGenerationAlgorithm(int width, int height, timestamp exponential_decay_time_us,
                                      Metavision::ColorPalette palette);

    /// @brief Processes a buffer of events
    /// @warning Call @ref reset before starting processing events from a timestamp in the past after later events have
    /// been processed.
    /// @tparam EventIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to the first input event
    /// @param it_end Iterator to the past-the-end event
    template<typename EventIt>
    void process_events(EventIt it_begin, EventIt it_end);

    /// @brief Generates a frame
    /// @param frame Frame that will be filled with CD events
    /// @param allocate Allocates the frame if true. Otherwise, the user must ensure the validity of the input frame.
    /// This is to be used when the data ptr must not change (external allocation, ROI over another cv::Mat, ...)
    /// @throw invalid_argument if the frame doesn't have the expected type and geometry
    void generate(cv::Mat &frame, bool allocate = true);

    /// @brief Sets the characteristic time of the exponential decay to use
    /// @param exponential_decay_time_us Characteristic time for the exponential decay (in us)
    void set_exponential_decay_time_us(timestamp exponential_decay_time_us);

    /// @brief Returns the current characteristic time of the exponential decay (in us).
    timestamp get_exponential_decay_time_us() const;

    /// @brief Sets the color palette used to generate the frame
    /// @param palette The color palette to use for the visualization
    void set_color_palette(Metavision::ColorPalette palette);

    /// @brief Resets the internal states
    /// @note This method needs to be called before processing events from a timestamp in the past after later events
    /// have been processed.
    void reset();

private:
    const std::vector<float> exp_decay_lut_;
    timestamp exponential_decay_time_us_;
    bool colored_;
    std::vector<cv::Vec3b> colormap_;
    MostRecentTimestampBuffer time_surface_;
    timestamp last_ts_;
};

} // namespace Metavision

#include "metavision/sdk/core/algorithms/detail/time_decay_frame_generation_algorithm_impl.h"

#endif // METAVISION_SDK_CORE_TIME_DECAY_FRAME_GENERATION_ALGORITHM_H