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

#ifndef METAVISION_SDK_BASE_GET_TIME_H
#define METAVISION_SDK_BASE_GET_TIME_H

#include <chrono>
#include <stdint.h>
#include <iomanip>
#include <cmath>
#include <sstream>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Gets current time point
/// @return Current time point
inline std::chrono::system_clock::time_point get_system_time_point() {
    return std::chrono::system_clock::now();
}

/// @brief Gets time point corresponding to given input time
/// @param time_since_epoch_us Time since epoch in microseconds
/// @return Time point of the input time
inline std::chrono::system_clock::time_point
    get_system_time_point(const std::chrono::microseconds &time_since_epoch_us) {
    return std::chrono::system_clock::time_point(time_since_epoch_us);
}

/// @brief Gets current system time
/// @return Current system time (computed from epoch) in microseconds
inline uint64_t get_system_time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(get_system_time_point().time_since_epoch()).count();
}

/// @brief Gets system time corresponding to given input time
/// @param time_point Time point to transform in microseconds
/// @return System time (computed from epoch) of given input time, in microseconds
constexpr uint64_t get_system_time_us(const std::chrono::system_clock::time_point &time_point) {
    return std::chrono::duration_cast<std::chrono::microseconds>(time_point.time_since_epoch()).count();
}

/// @brief Converts timestamp to string describing UTC time
/// @param ts Timestamp to convert
/// @param UTC_offset_in_seconds Offset (in seconds) to apply to the result
/// @param include_date If true, includes also the date in the output string
/// @return A string containing the UTC time corresponding to input timestamp
inline std::string timestamp_to_utc_string(timestamp ts, double UTC_offset_in_seconds, bool include_date) {
    std::ostringstream timestamp_sstr;

    // add offset to timestamp
    timestamp input_ts = static_cast<timestamp>(std::round(ts + UTC_offset_in_seconds * 1000000));

    std::chrono::microseconds us(input_ts);

    auto secs = std::chrono::duration_cast<std::chrono::seconds>(us);
    us -= std::chrono::duration_cast<std::chrono::microseconds>(secs);

    auto mins = std::chrono::duration_cast<std::chrono::minutes>(secs);
    secs -= std::chrono::duration_cast<std::chrono::seconds>(mins);

    auto hour = std::chrono::duration_cast<std::chrono::hours>(mins);
    mins -= std::chrono::duration_cast<std::chrono::minutes>(hour);

    struct tm timeinfo;
    timeinfo.tm_sec = static_cast<int>(secs.count());
    timeinfo.tm_min = static_cast<int>(mins.count());

    if (UTC_offset_in_seconds && include_date) {
        uint64_t ts_days = hour.count() / 24ll;
        timeinfo.tm_hour = hour.count() % 24ll;
        const std::chrono::system_clock::time_point tp(std::chrono::hours(24 * ts_days));
        const std::time_t utc_date_t = std::chrono::system_clock::to_time_t(tp);
        struct tm tm_buf;
#ifdef _WIN32
        gmtime_s(&tm_buf, &utc_date_t);
#else
        gmtime_r(&utc_date_t, &tm_buf);
#endif
        timestamp_sstr << std::put_time(&tm_buf, "%a %b %d %Y ");
    } else {
        timeinfo.tm_hour = hour.count();
    }

    char buffer[80];
    strftime(buffer, 80, "%H:%M:%S", &timeinfo);
    timestamp_sstr << buffer << "." << us.count();

    return timestamp_sstr.str();
}

} // namespace Metavision

#endif // METAVISION_SDK_BASE_GET_TIME_H
