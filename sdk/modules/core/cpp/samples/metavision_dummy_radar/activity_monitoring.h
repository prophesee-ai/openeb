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

#ifndef ACTIVITY_MONITORING_H
#define ACTIVITY_MONITORING_H

#include <stdint.h>
#include <vector>

#include <metavision/sdk/base/utils/timestamp.h>

/// @brief Class to monitor the event rate along the X dimension of the sensor
/// It counts the events falling in each bin and finally computes the event rate
/// from the accumulation time
class ActivityMonitor {
public:
    struct Config {
        uint8_t n_bins                          = 10;    ///< Number of bins to compute event rate for
        Metavision::timestamp accumulation_time = 50000; ///< Accumulation time to measure event rate
    };

    /// @brief Constructor
    /// @param conf Monitoring parameters
    /// @param sensor_width Sensor width (in pixels)
    ActivityMonitor(const Config &conf, int sensor_width);

    /// @brief Destructor
    ~ActivityMonitor() {}

    /// @brief Increments the bins associated to each event
    /// @tparam InputIt Input event iterator type, works for iterators over
    /// containers of @ref EventCD
    /// @param[in] begin Iterator pointing to the first event in the stream
    /// @param[in] end Iterator pointing to the past-the-end element in the stream
    template<typename InputIt>
    void process_events(InputIt begin, InputIt end);

    /// @brief Resets the bins counters to 0
    void reset();

    /// @brief Retrieves the event rate of each bin
    /// @param[out] ev_rate_per_bin The event rate for each bin
    void get_ev_rate_per_bin(std::vector<float> &ev_rate_per_bin) const;

private:
    const Config conf_;
    const int bin_width_;

    std::vector<uint32_t> histogram_;
};

template<typename InputIt>
void ActivityMonitor::process_events(InputIt begin, InputIt end) {
    if (begin == end) {
        return;
    }

    for (auto it = begin; it != end; ++it) {
        const uint32_t id_bin = it->x / bin_width_;
        if (id_bin < conf_.n_bins)
            histogram_[id_bin]++;
    }
}

#endif // ACTIVITY_MONITORING_H
