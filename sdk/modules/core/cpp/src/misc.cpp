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

#include <array>
#include <iomanip>
#include <sstream>
#include <opencv2/core/core.hpp>
#include "metavision/sdk/core/utils/misc.h"

namespace Metavision {

std::string getHumanReadableRate(double rate) {
    std::ostringstream oss;
    if (rate < 1000) {
        oss << std::setprecision(0) << std::fixed << rate << " ev/s";
    } else if (rate < 1000 * 1000) {
        oss << std::setprecision(1) << std::fixed << (rate / 1000) << " Kev/s";
    } else if (rate < 1000 * 1000 * 1000) {
        oss << std::setprecision(1) << std::fixed << (rate / (1000 * 1000)) << " Mev/s";
    } else {
        oss << std::setprecision(1) << std::fixed << (rate / (1000 * 1000 * 1000)) << " Gev/s";
    }
    return oss.str();
}

std::string getHumanReadableTime(Metavision::timestamp t) {
    std::ostringstream oss;
    std::array<std::string, 4> ls{":", ":", ".", ""};
    std::array<std::string, 4> vs;
    std::array<int, 4> ts;
    ts[3] = t % 1000000;
    vs[3] = cv::format("%06d", int(ts[3]));
    t /= 1000000; // s
    ts[2] = t % 60;
    vs[2] = cv::format("%02d", int(ts[2]));
    t /= 60; // m
    ts[1] = t % 60;
    vs[1] = cv::format("%02d", int(ts[1]));
    t /= 60; // h
    ts[0] = t;
    vs[0] = cv::format("%02d", int(ts[0]));

    size_t i = 0;
    // skip hour and minutes if t is not high enough, but keep s and us
    for (; i < 2; ++i) {
        if (ts[i] != 0) {
            break;
        }
    }
    for (; i < 4; ++i) {
        oss << vs[i] << ls[i];
    }
    return oss.str();
}

} // namespace Metavision
