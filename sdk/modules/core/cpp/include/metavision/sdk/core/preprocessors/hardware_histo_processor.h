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

#ifndef METAVISION_SDK_CORE_HARDWARE_HISTO_PROCESSOR_H
#define METAVISION_SDK_CORE_HARDWARE_HISTO_PROCESSOR_H

#include "metavision/sdk/base/events/raw_event_frame_histo.h"
#include "metavision/sdk/core/preprocessors/event_preprocessor.h"

namespace Metavision {

/// @brief Updates an input histogram with an input stream of events.
///
/// This algorithm updates, separately at each pixel, the sum of positive and negative events from the event stream.
/// The histogram values will saturate if more events than the maximum representable integers are received
/// at a given pixel for a given polarity. This algorithm expects a pre-allocated tensor using row-major convention with
/// interleaved channels for each polarity.
/// @tparam InputIt The type of the input iterator for the range of events to process.
template<typename InputIt>
class HardwareHistoProcessor : public EventPreprocessor<InputIt> {
public:
    using EventPreprocessor<InputIt>::process_events;

    /// @brief Constructor
    /// @param width Width of the event stream
    /// @param height Height of the event stream
    /// @param neg_saturation Maximum value for the count of negative events in the histogram at each pixel
    /// @param pos_saturation Maximum value for the count of positive events in the histogram at each pixel
    HardwareHistoProcessor(int width, int height, uint8_t neg_saturation = 255, uint8_t pos_saturation = 255);

    /// @brief Updates the provided histogram with the input events
    /// @param[in] begin Iterator pointing to the beginning of the events buffer
    /// @param[in] end Iterator pointing to the end of the events buffer
    /// @param[out] histo Histogram to update
    void process_events(InputIt begin, InputIt end, RawEventFrameHisto &histo) const;

private:
    void compute(const timestamp cur_frame_start_ts, InputIt begin, InputIt end, Tensor &tensor) const override;

    const uint8_t sum_max_neg_, sum_max_pos_;
    const int width_;
};

} // namespace Metavision

#include "metavision/sdk/core/preprocessors/detail/hardware_histo_processor_impl.h"

#endif // METAVISION_SDK_CORE_HARDWARE_HISTO_PROCESSOR_H
