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

#ifndef METAVISION_SDK_CORE_EVENT_FRAME_DIFF_GENERATION_ALGORITHM_H
#define METAVISION_SDK_CORE_EVENT_FRAME_DIFF_GENERATION_ALGORITHM_H

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/raw_event_frame_diff.h"
#include "metavision/sdk/core/preprocessors/hardware_diff_processor.h"

namespace Metavision {

/// @brief Produces diff event frames from a stream of events.
///
/// This class implements the algorithm for producing diff event frames from a stream of events. This algorithm
/// computes, at each pixel, the sum of polarities of all processed events.
/// Generated event frames are stored using row-major convention.
/// @tparam InputIt The type of the input iterator for the range of events.
template<typename InputIt>
class EventFrameDiffGenerationAlgorithm {
public:
    /// @brief Constructor for the EventFrameDiffGenerationAlgorithm class.
    /// @param width The width of the event stream.
    /// @param height The height of the event stream.
    /// @param bit_size Number of bits used to represent the sum of events. The supported range is [2;8].
    /// @param allow_rollover Flag indicating whether to allow overflow / underflow when summing polarities in the 8-bit
    /// signed counters (default: true).
    /// @param min_generation_period_us minimum duration between two successive calls to the generate function,
    /// to optionally simulate a limited transfer bandwidth (default: 1000 us).
    /// @throws invalid_argument if the bit size is outside the supported range of [2;8].
    EventFrameDiffGenerationAlgorithm(unsigned int width, unsigned int height, unsigned int bit_size = 8,
                                      bool allow_rollover = true, timestamp min_generation_period_us = 1000);

    /// @brief Getter for the allow_rollover setting.
    inline bool is_rollover_allowed() const {
        return allow_rollover_;
    }

    /// @brief Getter for the event frame configuration.
    inline const RawEventFrameDiffConfig &get_config() const {
        return frame_.get_config();
    }

    /// @brief Processes a range of events and updates the sum of polarities at each pixel.
    /// @param it_begin An iterator pointing to the beginning of the events range.
    /// @param it_end An iterator pointing to the end of the events range.
    void process_events(InputIt it_begin, InputIt it_end);

    /// @brief Retrieves the diff event frame aggregating the events processed so far, and resets the
    /// internal counters for upcoming calls to @p process_events.
    /// @note This version of the function does not simulate the limited transfer bandwidth typically met on hardware
    /// implementations and hence ignores the practical lower-bound on event frame generation frequency.
    /// @param event_frame diff event frame.
    void generate(RawEventFrameDiff &event_frame);

    /// @brief Retrieves the diff event frame aggregating the events processed so far, and resets the
    /// aggregation for upcoming calls to @p process_events. This version of the function simulates the limited transfer
    /// bandwidth typically met on hardware implementations and hence may fail to retrieve the event frame. Internal
    /// counters are only reset if the event frame retrieval is successful.
    /// @param ts_event_frame diff event frame.
    /// @param event_frame diff event frame.
    /// @return False if the time since the last call to generate is below the lower-bound generation period, true
    /// otherwise.
    bool generate(timestamp ts_event_frame, RawEventFrameDiff &event_frame);

    /// @brief Forces a reset of the internal counters.
    void reset();

private:
    void reset_wrapper();

    const bool allow_rollover_;
    const timestamp min_generation_period_us_;
    bool is_ts_prev_set_ = false;
    timestamp ts_prev_   = -1;
    HardwareDiffProcessor<InputIt> processor_;
    RawEventFrameDiff frame_;
    Tensor diff_wrapper_;
};

} // namespace Metavision

#include "metavision/sdk/core/algorithms/detail/event_frame_diff_generation_algorithm_impl.h"

#endif // METAVISION_SDK_CORE_EVENT_FRAME_DIFF_GENERATION_ALGORITHM_H
