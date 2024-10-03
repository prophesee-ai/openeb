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

#ifndef METAVISION_SDK_CORE_HARDWARE_DIFF_PROCESSOR_H
#define METAVISION_SDK_CORE_HARDWARE_DIFF_PROCESSOR_H

#include "metavision/sdk/base/events/raw_event_frame_diff.h"
#include "metavision/sdk/core/preprocessors/event_preprocessor.h"

namespace Metavision {

/// @brief Updates the data from a diff event frame in the form of a tensor with an input stream of events.
///
/// This class implements the algorithm for producing diff event frames from a stream of events. This algorithm
/// computes, at each pixel, the sum of polarities of all processed events.
/// Generated event frames are stored using row-major convention.
/// @tparam InputIt The type of the input iterator for the range of events to process.
template<typename InputIt>
class HardwareDiffProcessor : public EventPreprocessor<InputIt> {
public:
    using EventPreprocessor<InputIt>::process_events;

    /// @brief Constructor
    /// @param width Width of the event stream
    /// @param height Height of the event stream
    /// @param min_val Lower representable value
    /// @param max_val Higher representable value
    /// @param allow_rollover If true, a roll-over will be realized when reaching minimal or maximal value. Else,
    /// the pixel value will be saturated.
    HardwareDiffProcessor(int width, int height, int8_t min_val, int8_t max_val, bool allow_rollover = true);

    /// @brief Updates the provided diff frame with the input events
    /// @param[in] begin Iterator pointing to the beginning of the events buffer
    /// @param[in] end Iterator pointing to the end of the events buffer
    /// @param[out] diff Difference frame to update
    void process_events(InputIt begin, InputIt end, RawEventFrameDiff &diff) const;

private:
    void compute(const timestamp cur_frame_start_ts, InputIt begin, InputIt end, Tensor &tensor) const override;

    const bool allow_rollover_;
    const int8_t min_val_, max_val_;
    const int width_;
};

} // namespace Metavision

#include "metavision/sdk/core/preprocessors/detail/hardware_diff_processor_impl.h"

#endif // METAVISION_SDK_CORE_HARDWARE_DIFF_PROCESSOR_H
