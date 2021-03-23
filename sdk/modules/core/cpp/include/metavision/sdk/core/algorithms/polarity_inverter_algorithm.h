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

    /// @brief Applies the Polarity Inverter filter to the given input buffer storing the result in the output buffer.
    /// @param first Beginning of the range of the input elements
    /// @param last End of the range of the input elements
    /// @param d_first Beginning of the destination range
    template<class InputIt, class OutputIt>
    inline void process_events(InputIt first, InputIt last, OutputIt d_first) {
        Metavision::detail::transform(first, last, d_first, std::ref(*this));
    }

    /// @note process(...) is deprecated since version 2.2.0 and will be removed in later releases.
    ///       Please use process_events(...) instead
    template<class InputIt, class OutputIt>
    // clang-format off
    [[deprecated("process(...) is deprecated since version 2.2.0 and will be removed in later releases. "
                 "Please use process_events(...) instead")]]
    inline void process(InputIt first, InputIt last, OutputIt d_first)
    // clang-format on
    {
        static bool warning_already_logged = false;
        if (!warning_already_logged) {
            std::ostringstream oss;
            oss << "PolarityInverterAlgorithm::process(...) is deprecated since version 2.2.0 ";
            oss << "and will be removed in later releases. ";
            oss << "Please use PolarityInverterAlgorithm::process_events(...) instead" << std::endl;
            MV_SDK_LOG_WARNING() << oss.str();
            warning_already_logged = true;
        }
        process_events(first, last, d_first);
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
