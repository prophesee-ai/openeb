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

#ifndef METAVISION_SDK_CORE_EVENT_RESCALER_ALGORITHM_H
#define METAVISION_SDK_CORE_EVENT_RESCALER_ALGORITHM_H

#include <memory>
#include <stdexcept>

namespace Metavision {

/// @brief Base class to operate a rescaling of events locations in both horizontal and vertical directions
class EventRescalerAlgorithm {
public:
    /// @brief Constructor
    /// @param scale_width The horizontal scale for events
    /// @param scale_height The vertical scale for events
    EventRescalerAlgorithm(float scale_width, float scale_height) :
        scale_width_(scale_width),
        scale_height_(scale_height),
        offset_width_((scale_width < 1.f) ? 0.f : 0.5f),
        offset_height_((scale_height < 1.f) ? 0.f : 0.5f) {
        if (scale_width <= 0.f || scale_height <= 0)
            throw std::runtime_error("Provided W/H scales should be > 0. Got " + std::to_string(scale_width) + " and " +
                                     std::to_string(scale_height));
    }

    /// @brief Processes the input events and fills the output iterator with rescaled events
    /// @tparam InputIt The input event iterator type
    /// @tparam OutputIt The output event iterator type (typically, a vector inserter)
    /// @param begin Iterator pointing to the first event in the stream
    /// @param end Iterator pointing to the past-the-end element in the stream
    /// @param out_begin Iterator to the first rescaled event
    /// @return The iterator to the past-the-last rescaled event
    template<typename InputIt, typename OutputIt>
    OutputIt process_events(InputIt begin, InputIt end, OutputIt out_begin) const;

protected:
    const float scale_width_, scale_height_;
    const float offset_width_, offset_height_;
};

} // namespace Metavision

#include "metavision/sdk/core/algorithms/detail/event_rescaler_algorithm_impl.h"

#endif // METAVISION_SDK_CORE_EVENT_RESCALER_ALGORITHM_H
