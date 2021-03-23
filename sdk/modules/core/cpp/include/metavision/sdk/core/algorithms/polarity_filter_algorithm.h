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

#ifndef METAVISION_SDK_CORE_POLARITY_FILTER_ALGORITHM_H
#define METAVISION_SDK_CORE_POLARITY_FILTER_ALGORITHM_H

#include <memory>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/core/algorithms/detail/internal_algorithms.h"
#include "metavision/sdk/base/events/event2d.h"

namespace Metavision {

/// @brief Class filter that only propagates events of a certain polarity
class PolarityFilterAlgorithm {
public:
    /// @brief Creates a PolarityFilterAlgorithm class with the given polarity
    /// @param polarity Polarity to keep
    inline explicit PolarityFilterAlgorithm(std::int16_t polarity);

    /// @brief Default destructor
    ~PolarityFilterAlgorithm() = default;

    /// @brief Applies the Polarity filter to the given input buffer storing the result in the output buffer.
    /// @param first Beginning of the range of the input elements
    /// @param last End of the range of the input elements
    /// @param d_first Beginning of the destination range
    /// @return Iterator pointing to the last + 1 event added in the output
    template<class InputIt, class OutputIt>
    inline OutputIt process_events(InputIt first, InputIt last, OutputIt d_first) {
        return Metavision::detail::insert_if(first, last, d_first, std::ref(*this));
    }

    /// @note process(...) is deprecated since version 2.2.0 and will be removed in later releases.
    ///       Please use process_events(...) instead
    template<class InputIt, class OutputIt>
    // clang-format off
    [[deprecated("process(...) is deprecated since version 2.2.0 and will be removed in later releases. "
                 "Please use process_events(...) instead")]]
    inline OutputIt process(InputIt first, InputIt last, OutputIt d_first)
    // clang-format on
    {
        static bool warning_already_logged = false;
        if (!warning_already_logged) {
            std::ostringstream oss;
            oss << "PolarityFilterAlgorithm::process(...) is deprecated since version 2.2.0 ";
            oss << "and will be removed in later releases. ";
            oss << "Please use PolarityFilterAlgorithm::process_events(...) instead" << std::endl;
            MV_SDK_LOG_WARNING() << oss.str();
            warning_already_logged = true;
        }
        return process_events(first, last, d_first);
    }

    /// @brief Basic operator to check if an event is accepted
    /// @param ev Event2D to be tested
    inline bool operator()(const Event2d &ev) const;

    /// @brief Sets the polarity of the filter
    /// @param polarity Polarity to be used in the filtering process
    inline void set_polarity(std::int16_t polarity);

    /// @brief Returns the polarity used to filter the events
    /// @return Current polarity used in the filtering process
    inline std::int16_t polarity() const;

private:
    std::int16_t pol_{0};
};

inline PolarityFilterAlgorithm::PolarityFilterAlgorithm(std::int16_t polarity) : pol_(polarity) {}

inline void PolarityFilterAlgorithm::set_polarity(int16_t polarity) {
    pol_ = polarity;
}

inline int16_t PolarityFilterAlgorithm::polarity() const {
    return pol_;
}

inline bool PolarityFilterAlgorithm::operator()(const Event2d &ev) const {
    return (ev.p == pol_);
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_POLARITY_FILTER_ALGORITHM_H
