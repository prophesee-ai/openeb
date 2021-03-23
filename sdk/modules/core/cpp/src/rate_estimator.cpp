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
#include "metavision/sdk/core/utils/rate_estimator.h"

namespace Metavision {

RateEstimator::RateEstimator(const Callback &cb, timestamp step_time, timestamp window_time) {
    cb_          = cb;
    step_time_   = step_time;
    window_time_ = window_time;
    next_time_   = step_time;
}

void RateEstimator::add_data(timestamp time, size_t count) {
    if (!counts_.empty() && counts_.back().first == time) {
        counts_.back().second += count;
    } else {
        counts_.emplace_back(time, count);
    }
    if (time > next_time_) {
        if (cb_) {
            timestamp last_time = 0;
            int count           = 0;
            double peak_rate = 0., avg_rate = 0.;

            // find the first count corresponding to the next callback timestamp minus the time window
            auto begin_it = std::lower_bound(counts_.begin(), counts_.end(), next_time_ - window_time_ + 1,
                                             [](const auto &p, const timestamp &t) { return p.first < t; }),
                 end_it   = std::lower_bound(counts_.begin(), counts_.end(), next_time_ + 1,
                                           [](const auto &p, const timestamp &t) { return p.first < t; });
            if (begin_it != counts_.begin()) {
                last_time = std::prev(begin_it)->first;
            }

            // update the average and peak rate from countime in the window timespan
            for (auto it = begin_it; it != end_it; ++it) {
                double rate = it->second / static_cast<double>(it->first - last_time);
                avg_rate += rate;
                peak_rate = std::max(peak_rate, rate);
                last_time = it->first;
                ++count;
            }
            avg_rate /= count;
            cb_(next_time_, avg_rate * 1.e6, peak_rate * 1.e6);

            // remove older countime from the map, they won't be needed anymore
            counts_.erase(counts_.begin(), begin_it);
        }
        next_time_ += step_time_;
    }
}

timestamp RateEstimator::step_time() const {
    return step_time_;
}

timestamp RateEstimator::window_time() const {
    return window_time_;
}

} // namespace Metavision
