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

#include <cstdlib>
#include <cstring>
#include <unordered_map>

#include "utils/psee_hal_utils.h"

namespace Metavision {

const std::string &get_bias_description(const std::string &bias) {
    static const std::unordered_map<std::string, std::string> bias_descriptions = {
        {"bias_diff", "reference value for comparison with bias_diff_on and bias_diff_off"},
        {"bias_diff_on", "controls the light sensitivity for ON events"},
        {"bias_diff_off", "controls the light sensitivity for OFF events"},
        {"bias_fo", "controls the pixel low-pass cut-off frequency"},
        {"bias_fo_p", "controls the pixel low-pass cut-off frequency"},
        {"bias_fo_n", "controls the pixel low-pass cut-off frequency"},
        {"bias_hpf", "controls the pixel high-pass cut-off frequency"},
        {"bias_pr", "controls the photoreceptor bandwidth"},
        {"bias_refr", "controls the refractory period during which the change detector is switched off after "
                      "generating an event"}};
    static const std::string empty_string;
    auto it = bias_descriptions.find(bias);
    if (it != bias_descriptions.end()) {
        return it->second;
    }
    return empty_string;
}

const std::string &get_bias_category(const std::string &bias) {
    static const std::unordered_map<std::string, std::string> bias_category = {
        {"bias_diff", "Contrast"}, {"bias_diff_on", "Contrast"}, {"bias_diff_off", "Contrast"},
        {"bias_fo", "Bandwidth"},  {"bias_fo_p", "Bandwidth"},   {"bias_fo_n", "Bandwidth"},
        {"bias_hpf", "Bandwidth"}, {"bias_pr", "Advanced"},      {"bias_refr", "Advanced"}};
    static const std::string empty_string = "";
    auto it                               = bias_category.find(bias);
    if (it != bias_category.end()) {
        return it->second;
    }
    return empty_string;
}

} // namespace Metavision
