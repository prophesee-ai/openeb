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

#ifndef METAVISION_SDK_CORE_FLIP_X_ALGORITHM_H
#define METAVISION_SDK_CORE_FLIP_X_ALGORITHM_H

#include <memory>
#include <sstream>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/core/algorithms/detail/internal_algorithms.h"
#include "metavision/sdk/base/events/event2d.h"

namespace Metavision {

/// @brief Class that allows to mirror the X axis of an event stream.
///
/// The transfer function of this filter impacts only the X coordinates of the Event2d by:\n
/// x = width_minus_one - x
class FlipXAlgorithm {
public:
    /// @brief Builds a new FlipXAlgorithm object with the given width
    /// @param width_minus_one Maximum X coordinate of the events (width-1)
    inline explicit FlipXAlgorithm(std::int16_t width_minus_one);

    /// Default destructor
    ~FlipXAlgorithm() = default;

    /// @brief Applies the Flip X filter to the given input buffer storing the result in the output buffer
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @tparam OutputIt Read-Write output event iterator type. Works for iterators over containers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to first input event
    /// @param it_end Iterator to the past-the-end event
    /// @param inserter Output iterator or back inserter
    template<class InputIt, class OutputIt>
    inline void process_events(InputIt it_begin, InputIt it_end, OutputIt inserter) {
        detail::transform(it_begin, it_end, inserter, std::ref(*this));
    }

    /// @brief Returns the maximum X coordinate of the events
    /// @return Maximum X coordinate of the events
    inline std::int16_t width_minus_one() const;

    /// @brief Sets the maximum X coordinate of the events
    /// @param width_minus_one Maximum X coordinate of the events
    inline void set_width_minus_one(std::int16_t width_minus_one);

    /// @brief Applies the Flip X filter to the given input buffer storing the result in the output buffer.
    /// @param ev Event2d to be updated
    inline void operator()(Event2d &ev) const;

private:
    std::int16_t width_minus_one_{0};
};

} // namespace Metavision

#include "detail/flip_x_algorithm_impl.h"

#endif // METAVISION_SDK_CORE_FLIP_X_ALGORITHM_H
