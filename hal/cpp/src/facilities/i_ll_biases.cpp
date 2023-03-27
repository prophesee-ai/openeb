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

#include <memory>

#include "metavision/hal/facilities/i_ll_biases.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/utils/hal_log.h"

namespace Metavision {

I_LL_Biases::I_LL_Biases(const DeviceConfig &device_config) : device_config_(device_config) {}

bool I_LL_Biases::set(const std::string &bias_name, int bias_value) {
    LL_Bias_Info bias_info;
    if (!get_bias_info(bias_name, bias_info)) {
        throw HalException(HalErrorCode::NonExistingValue, "Unavailable bias: \"" + bias_name + "\".");
        return false;
    }
    if (!bias_info.is_modifiable()) {
        throw HalException(HalErrorCode::OperationNotPermitted, "Bias \"" + bias_name + "\" cannot be modified.");
        return false;
    }
    if (!device_config_.biases_range_check_bypass()) {
        auto range = bias_info.get_bias_range();
        if (bias_value > range.second || bias_value < range.first) {
            std::stringstream ss;
            ss << "Invalid value " << bias_value << " for bias \"" << bias_name << "\". Value should be within range ["
               << range.first << ", " << range.second << "].";
            throw HalException(HalErrorCode::ValueOutOfRange, ss.str());
            return false;
        }
    }
    return set_impl(bias_name, bias_value);
}

int I_LL_Biases::get(const std::string &bias_name) {
    LL_Bias_Info bias_info;
    if (!get_bias_info(bias_name, bias_info)) {
        throw HalException(HalErrorCode::NonExistingValue, "Unavailable bias: \"" + bias_name + "\".");
        return false;
    }
    return get_impl(bias_name);
}

bool I_LL_Biases::get_bias_info(const std::string &bias_name, LL_Bias_Info &bias_info) const {
    if (!get_bias_info_impl(bias_name, bias_info)) {
        throw HalException(HalErrorCode::NonExistingValue, "Unavailable bias: \"" + bias_name + "\".");
        return false;
    }
    if (device_config_.biases_range_check_bypass()) {
        bias_info.disable_recommended_range();
    }
    return true;
}

LL_Bias_Info::LL_Bias_Info(int min_value, int max_value, const std::string &description, bool modifiable,
                           const std::string &category) :
    description_(description),
    category_(category),
    modifiable_(modifiable),
    use_recommended_range_(true),
    bias_allowed_range_(std::make_pair(min_value, max_value)),
    bias_recommended_range_(std::make_pair(min_value, max_value)) {}

LL_Bias_Info::LL_Bias_Info(int min_allowed_value, int max_allowed_value, int min_recommended_value,
                           int max_recommended_value, const std::string &description, bool modifiable,
                           const std::string &category) :
    description_(description),
    category_(category),
    modifiable_(modifiable),
    use_recommended_range_(true),
    bias_allowed_range_(std::make_pair(min_allowed_value, max_allowed_value)),
    bias_recommended_range_(std::make_pair(min_recommended_value, max_recommended_value)) {}

const std::string &LL_Bias_Info::get_description() const {
    return description_;
}

const std::string &LL_Bias_Info::get_category() const {
    return category_;
}

std::pair<int, int> LL_Bias_Info::get_bias_range() const {
    return use_recommended_range_ ? bias_recommended_range_ : bias_allowed_range_;
}

std::pair<int, int> LL_Bias_Info::get_bias_allowed_range() const {
    return bias_allowed_range_;
}

std::pair<int, int> LL_Bias_Info::get_bias_recommended_range() const {
    return bias_recommended_range_;
}

bool LL_Bias_Info::is_modifiable() const {
    return modifiable_;
}

void LL_Bias_Info::disable_recommended_range() {
    use_recommended_range_ = false;
}

} // namespace Metavision
