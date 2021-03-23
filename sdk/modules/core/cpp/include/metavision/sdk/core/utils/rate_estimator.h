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

#ifndef METAVISION_SDK_CORE_RATE_ESTIMATOR_H
#define METAVISION_SDK_CORE_RATE_ESTIMATOR_H

#include <functional>
#include <deque>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Simple estimator class that can estimate average and peak rate online from sample counts
class RateEstimator {
public:
    /// @brief Callback type to be used in the constructor
    using Callback = std::function<void(timestamp, double, double)>;

    /// @brief Constructor
    /// @param cb Callback that will be called every @p step_time with the current timestamp, estimated average and
    ///           peak rates over the counts added in the @p window time span
    /// @param step_time Minimum period between two successive callbacks
    /// @param window_time Time window used to compute the average and peak rates
    RateEstimator(const Callback &cb = Callback(), timestamp step_time = 100000, timestamp window_time = 1000000);

    /// @brief Adds a sample @p count at t = @p time
    /// @param time Time of the sample
    /// @param count Count of the sample
    ///
    /// If the sample count added has a time @p time > next_time (next_time = last_callback_time + step_time),
    /// then the callback will be called with estimated rates from the available sample counts
    /// in the window timespan.
    /// @note this function must be called with the same time used at last call (the count will then be added to the
    /// previous one) or with a @p time greater than the previous one
    void add_data(timestamp time, size_t count);

    /// @brief Get the minimum time between two callbacks
    /// @return The step time
    timestamp step_time() const;

    /// @brief Get the window duration on which average and peak rates are computed
    /// @return The window time
    timestamp window_time() const;

private:
    Callback cb_;
    timestamp window_time_, step_time_, next_time_;
    std::deque<std::pair<timestamp, size_t>> counts_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_RATE_ESTIMATOR_H
