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

#include <mutex>
#include <functional>
#include <deque>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Simple estimator class that can estimate average and peak rate online from sample counts
///
/// This class estimates @e average and <em>peak rate</em> according to the added samples using the @e peak, @e step and
/// <em>window time durations</em> as follows :
///  - the <em>average rate</em> is the mean event rate computed over the samples added during last <em>window
///  duration</em>
///  - the <em>peak rate</em> is the maximum event rate of the estimated rates computed over subwindows of <em>peak
///  duration</em> during last <em>window duration</em> (thus, peak duration must always be less or equal to window
///  duration)
///
/// The average and peak rate are then output via a callback which is called :
/// - every <em>step duration</em> of system time, if @c system_time_flag is true
/// - everytime a sample with a timestamp > (last callback time + <em>step duration</em>) is added,
/// if @c system_time_flag is false
///
/// @note Every duration must be expressed in microseconds.
///
/// @warning This class is not protected against concurrent accesses and does not provide thread safe functions
class RateEstimator {
public:
    /// @brief Callback type to be used in the constructor
    using Callback = std::function<void(timestamp, double, double)>;

    /// @brief Constructor
    ///
    /// This constructor sets <tt>peak_time = step_time</tt>
    ///
    /// @param cb Callback that will be called with the current timestamp, estimated average and
    ///           peak rates over the counts added in the @p window_time span
    /// @param step_time Minimum period between two successive callbacks
    /// @param window_time Time window used to compute the average and peak rates
    /// @param system_time_flag If true, the callback will be called when the sytem time becomes higher than current
    ///                         multiple of @p step_time, otherwise the sample time is used
    RateEstimator(const Callback &cb = Callback(), timestamp step_time = 100000, timestamp window_time = 1000000,
                  bool system_time_flag = false);

    /// @brief Constructor
    /// @param step_time Minimum period between two successive callbacks
    /// @param window_time Time window used to compute the average and peak rates
    /// @param peak_time Time window used to estimate peak rates
    /// @param cb Callback that will be called with the current timestamp, estimated average and
    ///           peak rates over the counts added in the @p window_time span
    /// @param system_time_flag If true, the callback will be called when the sytem time becomes higher than current
    ///                         multiple of @p step_time, otherwise the sample time is used
    RateEstimator(timestamp step_time, timestamp window_time, timestamp peak_time, const Callback &cb,
                  bool system_time_flag = false);

    /// @brief Adds a sample @p count at t = @p time
    /// @param time Time of the sample
    /// @param count Count of the sample
    ///
    /// The callback will be called with estimated rates from the available sample counts
    /// in the window timespan, if :
    /// - <tt>system_time_flag = true</tt> and the sample count added has a time @p time > next_time (next_time =
    /// last_callback_sample_time + step_time)
    /// - <tt>system_time_flag = false</tt> and the function is called with a system time > next_time (next_time =
    /// last_callback_sytem_time + step_time)
    ///
    /// @note this function must be called with the same time used at last call (the count will then be added to the
    /// previous one) or with a @p time greater than the previous one
    void add_data(timestamp time, size_t count);

    /// @brief Reset all sample counts
    void reset_data();

    /// @brief Get the minimum time between two callbacks
    /// @return The step time
    timestamp step_time() const;

    /// @brief Get the window duration on which average and peak rates are computed
    /// @return The window time
    timestamp window_time() const;

    /// @brief Get the peak duration on which peak rates are computed
    /// @return The peak time
    timestamp peak_time() const;

private:
    Callback cb_;
    timestamp window_time_, step_time_, peak_time_, next_time_;
    std::deque<std::pair<timestamp, size_t>> counts_;
    bool system_time_flag_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CORE_RATE_ESTIMATOR_H
