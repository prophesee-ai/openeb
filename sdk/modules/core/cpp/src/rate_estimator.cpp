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

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include "metavision/sdk/core/utils/rate_estimator.h"

namespace Metavision {

RateEstimator::RateEstimator(const Callback &cb, timestamp step_time, timestamp window_time, bool system_time_flag) {
    cb_               = cb;
    step_time_        = step_time;
    window_time_      = window_time;
    peak_time_        = step_time;
    next_time_        = step_time;
    system_time_flag_ = system_time_flag;
}

RateEstimator::RateEstimator(timestamp step_time, timestamp window_time, timestamp peak_time, const Callback &cb,
                             bool system_time_flag) {
    cb_          = cb;
    step_time_   = step_time;
    window_time_ = window_time;
    if (peak_time < window_time) {
        peak_time_ = peak_time;
    } else {
        throw std::runtime_error("Peak time must be <= window time");
    }
    next_time_        = step_time;
    system_time_flag_ = system_time_flag;
}

void RateEstimator::add_data(timestamp time, size_t count) {
    long long current_time = time;
    if (!counts_.empty() && counts_.back().first == time) {
        counts_.back().second += count;
    } else {
        counts_.emplace_back(time, count);
    }

    if (system_time_flag_) {
        current_time = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::steady_clock::now())
                           .time_since_epoch()
                           .count();
    }

    if (current_time > next_time_) {
        // handle calls to reset_data where next_time has been reset to a "too far" value
        // this set next_time_ so that time - step_time_ < next_time_ < time
        while (current_time > next_time_ + step_time_) {
            next_time_ += step_time_;
        }
        if (cb_) {
            timestamp next_peak_time = peak_time_;
            timestamp callback_time  = (system_time_flag_ ? time : next_time_);
            double cur_peak_rate = 0., peak_rate = 0., avg_rate = 0.;

            // find the first count corresponding to the next callback timestamp minus the time window
            auto begin_it = std::lower_bound(counts_.begin(), counts_.end(), callback_time - window_time_ + 1,
                                             [](const auto &p, const timestamp &t) { return p.first < t; }),
                 end_it   = std::lower_bound(counts_.begin(), counts_.end(), callback_time + 1,
                                           [](const auto &p, const timestamp &t) { return p.first < t; });

            // update the average and peak rate from counts in the window timespan
            for (auto it = begin_it; it != end_it; ++it) {
                avg_rate += it->second;
                cur_peak_rate += it->second;
                if (it->first >= next_peak_time) {
                    cur_peak_rate /= peak_time_;
                    peak_rate      = std::max(peak_rate, cur_peak_rate);
                    cur_peak_rate  = 0;
                    next_peak_time = static_cast<timestamp>(1 + it->first / peak_time_) * peak_time_;
                }
            }
            avg_rate /= std::min(callback_time, window_time_);
            cb_(callback_time, avg_rate * 1.e6, peak_rate * 1.e6);

            // remove older countime from the map, they won't be needed anymore
            counts_.erase(counts_.begin(), begin_it);
        }
        next_time_ += step_time_;
    }
}

void RateEstimator::reset_data() {
    counts_.clear();
    next_time_ = step_time_;
}

timestamp RateEstimator::step_time() const {
    return step_time_;
}

timestamp RateEstimator::window_time() const {
    return window_time_;
}

timestamp RateEstimator::peak_time() const {
    return peak_time_;
}

} // namespace Metavision
