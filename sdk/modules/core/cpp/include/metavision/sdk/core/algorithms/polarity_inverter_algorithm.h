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

#ifndef METAVISION_SDK_CORE_POLARITY_INVERTER_ALGORITHM_H
#define METAVISION_SDK_CORE_POLARITY_INVERTER_ALGORITHM_H

#include <memory>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/core/algorithms/detail/internal_algorithms.h"
#include "metavision/sdk/base/events/event2d.h"

namespace Metavision {

/// @brief Class that implements a Polarity Inverter filter.
///
/// The filter changes the polarity of all the filtered events.
class PolarityInverterAlgorithm {
public:
    /// @brief Builds a new PolarityInverterAlgorithm object
    PolarityInverterAlgorithm() = default;

    // @brief Default destructor
    ~PolarityInverterAlgorithm() = default;

    /// @brief Processes a buffer of events and outputs filtered events
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @tparam OutputIt Read-Write output event iterator type. Works for iterators over containers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to first input event
    /// @param it_end Iterator to the past-the-end event
    /// @param inserter Output iterator or back inserter
    template<class InputIt, class OutputIt>
    inline void process_events(InputIt it_begin, InputIt it_end, OutputIt inserter) {
        Metavision::detail::transform(it_begin, it_end, inserter, std::ref(*this));
    }

    /// @brief Changes the polarity of an events
    /// @param ev Event2D that want to be changed
    inline void operator()(Event2d &ev) const;
};

inline void PolarityInverterAlgorithm::operator()(Event2d &ev) const {
    ev.p = (ev.p > 0) ? 0 : 1;
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_POLARITY_INVERTER_ALGORITHM_H
