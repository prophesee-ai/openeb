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

#include "activity_monitoring.h"

ActivityMonitor::ActivityMonitor(const Config &conf, int sensor_width) :
    conf_(conf), bin_width_(sensor_width / conf.n_bins) {
    histogram_.resize(conf.n_bins);
}

void ActivityMonitor::reset() {
    std::fill(histogram_.begin(), histogram_.end(), 0);
}

void ActivityMonitor::get_ev_rate_per_bin(std::vector<float> &ev_rate_per_bin) const {
    ev_rate_per_bin.clear();
    ev_rate_per_bin.reserve(histogram_.size());
    for (uint32_t i = 0; i < histogram_.size(); ++i) {
        const float ev_rate_bin = histogram_[i] / (conf_.accumulation_time * 1e-6f);
        ev_rate_per_bin.emplace_back(ev_rate_bin);
    }
}
