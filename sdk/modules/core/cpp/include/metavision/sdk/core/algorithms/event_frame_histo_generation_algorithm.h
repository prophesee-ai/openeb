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

#ifndef METAVISION_SDK_CORE_EVENT_FRAME_HISTO_GENERATION_ALGORITHM_H
#define METAVISION_SDK_CORE_EVENT_FRAME_HISTO_GENERATION_ALGORITHM_H

#include "metavision/sdk/base/events/raw_event_frame_histo.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/core/preprocessors/hardware_histo_processor.h"

namespace Metavision {

/// @brief Produces histo event frames from a stream of events.
///
/// This class implements the algorithm for producing histo event frames from a stream of events. This algorithm
/// computes, separately at each pixel, the sum of positive and negative events received between two calls to the
/// 'generate' function. Generated event frames are stored using row-major convention, either splitting or interleaving
/// polarities. The histogram values will saturate if more events than the maximum representable integers are received
/// at a given pixel for a given polarity. Generated event frames are stored using row-major convention with interleaved
/// channels for each polarity.
/// @tparam InputIt The type of the input iterator for the range of events to process.
template<typename InputIt>
class EventFrameHistoGenerationAlgorithm {
public:
    /// @brief Constructor for the EventFrameHistoGenerationAlgorithm class.
    /// @param width The width of the event stream.
    /// @param height The height of the event stream.
    /// @param channel_bit_neg Number of bits used to represent the sum of negative events.
    /// This should be strictly positive and the sum of negative and positive channels bit sizes should be less than 8.
    /// @param channel_bit_pos Number of bits used to represent the sum of positive events.
    /// This should be strictly positive and the sum of negative and positive channels bit sizes should be less than 8.
    /// @param packed Flag indicating whether sum counter are stored in an aligned way in memory, by padding with zeros,
    /// or in a packed way.
    /// @param min_generation_period_us minimum duration between two successive calls to the generate function,
    /// to optionally simulate a limited transfer bandwidth (default: 1000 us).
    /// @throws invalid_argument if the sum of negative and positive channels bit sizes is more than 8 or either one is
    /// zero.
    EventFrameHistoGenerationAlgorithm(unsigned int width, unsigned int height, unsigned int channel_bit_neg = 4,
                                       unsigned int channel_bit_pos = 4, bool packed = false,
                                       timestamp min_generation_period_us = 1000);

    /// @brief Getter for the event frame configuration.
    inline const RawEventFrameHistoConfig &get_config() const {
        return cfg_;
    }

    /// @brief Processes a range of events and updates the sum of events at each pixel & polarity.
    /// @param it_begin An iterator pointing to the beginning of the events range.
    /// @param it_end An iterator pointing to the end of the events range.
    void process_events(InputIt it_begin, InputIt it_end);

    /// @brief Retrieves the histo event frame aggregating events since the last call to @p generate, and resets the
    /// aggregation for upcoming calls to @p process_events.
    /// @param event_frame histo event frame.
    /// @note in packed mode, a packed copy of the accumulation data will be returned, incurring a slight performance
    /// cost compared to unpacked mode.
    /// @note This version of the function does not simulate the limited transfer bandwidth typically met on hardware
    /// implementations and hence ignores the practical lower-bound on event frame generation frequency.
    void generate(RawEventFrameHisto &event_frame);

    /// @brief Retrieves the diff event frame aggregating the events processed so far, and resets the
    /// aggregation for upcoming calls to @p process_events. This version of the function simulates the limited transfer
    /// bandwidth typically met on hardware implementations and hence may fail to retrieve the event frame. Internal
    /// counters are only reset if the event frame retrieval is successful.
    /// @param ts_event_frame diff event frame.
    /// @param event_frame diff event frame.
    /// @return False if the time since the last call to generate is below the lower-bound generation period, true
    /// otherwise.
    bool generate(timestamp ts_event_frame, RawEventFrameHisto &event_frame);

    /// @brief Forces a reset of the internal counters.
    void reset();

private:
    HardwareHistoProcessor<InputIt> processor_;
    const RawEventFrameHistoConfig cfg_;
    const timestamp min_generation_period_us_;
    bool is_ts_prev_set_ = false;
    timestamp ts_prev_;
    RawEventFrameHisto frame_unpacked_;
};

} // namespace Metavision

#include "metavision/sdk/core/algorithms/detail/event_frame_histo_generation_algorithm_impl.h"

#endif // METAVISION_SDK_CORE_EVENT_FRAME_HISTO_GENERATION_ALGORITHM_H
