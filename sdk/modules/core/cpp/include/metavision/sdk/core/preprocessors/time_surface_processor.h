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

#ifndef METAVISION_SDK_CORE_TIME_SURFACE_PROCESSOR_H
#define METAVISION_SDK_CORE_TIME_SURFACE_PROCESSOR_H

#include <functional>
#include <type_traits>

#include "metavision/sdk/core/preprocessors/event_preprocessor.h"
#include "metavision/sdk/core/utils/mostrecent_timestamp_buffer.h"

namespace Metavision {

/// @brief Class that produces a @ref MostRecentTimestampBuffer (a.k.a. time surface) from events
/// @tparam InputIt The type of the input iterator for the range of events to process
/// @tparam CHANNELS Number of channels to use for producing the time surface. Only two values are possible for now: 1
/// or 2. When a 1-channel time surface is used, events with different polarities are stored all together while they are
/// stored separately when using a 2-channels time surface.
template<typename InputIt, int CHANNELS = 1>
class TimeSurfaceProcessor : public EventPreprocessor<InputIt> {
public:
    static_assert(CHANNELS == 1 || CHANNELS == 2, "The timesurface producer is only compatible with 1 or 2 channels");

    using EventPreprocessor<InputIt>::process_events;

    /// @brief Constructs a new time surface producer
    /// @param width Sensor's width
    /// @param height Sensor's height
    TimeSurfaceProcessor(int width, int height);

    /// @brief Updates the provided time surface with the input events
    /// @param[in] begin Iterator pointing to the beginning of the events buffer
    /// @param[in] end Iterator pointing to the end of the events buffer
    /// @param[out] time_surface Time surface to update
    void process_events(InputIt begin, InputIt end, MostRecentTimestampBuffer &time_surface) const;

private:
    /// @brief Updates the input time surface with the provided events
    /// @param ts starting timestamps of the current frame (not used)
    /// @param it_begin Iterator pointing to the beginning of the events buffer
    /// @param it_end Iterator pointing to the end of the events buffer
    /// @param tensor The tensor to update with the provided events
    void compute(const timestamp ts, InputIt it_begin, InputIt it_end, Tensor &tensor) const override;

    const int width_;
};

template<typename InputIt>
using TimeSurfaceProcessorMergePolarities = TimeSurfaceProcessor<InputIt, 1>;

template<typename InputIt>
using TimeSurfaceProcessorSplitPolarities = TimeSurfaceProcessor<InputIt, 2>;

} // namespace Metavision

#include "metavision/sdk/core/preprocessors/detail/time_surface_processor_impl.h"

#endif // METAVISION_SDK_CORE_TIME_SURFACE_PROCESSOR_H
