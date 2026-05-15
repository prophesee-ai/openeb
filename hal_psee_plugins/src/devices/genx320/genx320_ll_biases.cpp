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

#include <cassert>
#include <iomanip>
#include "metavision/psee_hw_layer/utils/register_map.h"
#include "devices/genx320/genx320_bias_settings.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_ll_biases.h"
#include "metavision/hal/facilities/i_hw_register.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"
#include "utils/psee_hal_utils.h"

namespace Metavision {

GenX320LLBiases::GenX320Bias::GenX320Bias(std::string register_name, const uint8_t &default_val,
                                          const bias_settings &bias_conf, const std::string &description,
                                          const std::string &category) :
    LL_Bias_Info(bias_conf.min_allowed_offset, bias_conf.max_allowed_offset, bias_conf.min_recommended_offset,
                 bias_conf.max_recommended_offset, description, bias_conf.modifiable, category) {
    register_name_ = register_name;
    reg_max_       = (1 << 7) - 1;
    default_value_ = default_val;
}

GenX320LLBiases::GenX320Bias::~GenX320Bias() {}

const std::string &GenX320LLBiases::GenX320Bias::get_register_name() const {
    return register_name_;
}

const uint32_t &GenX320LLBiases::GenX320Bias::get_register_max_value() const {
    return reg_max_;
}

void GenX320LLBiases::GenX320Bias::display_bias() const {
    MV_HAL_LOG_INFO() << "register name:" << register_name_;
    MV_HAL_LOG_INFO() << "default      :" << std::dec << std::setw(3) << unsigned(default_value_);
}

GenX320LLBiases::GenX320LLBiases(const std::shared_ptr<RegisterMap> &register_map, const DeviceConfig &device_config) :
    I_LL_Biases(device_config), register_map_(register_map) {
    std::string reg_name;

    for (auto &bias_setting : genx320_biases_settings) {
        if (bias_setting.name == "bias_pr" || bias_setting.name == "bias_fo") {
            reg_name = "bias/" + bias_setting.name + "_hv0";
        } else {
            reg_name = "bias/" + bias_setting.name + "_lv0";
        }
        auto init_value = (*register_map_)[reg_name]["bias_ctl"].read_value();
        GenX320Bias bias(reg_name, (init_value & 0xFF), bias_setting, get_bias_description(bias_setting.name),
                         get_bias_category(bias_setting.name));
        biases_map_.insert({bias_setting.name, bias});
        (*register_map_)[reg_name]["bias_en"].write_value(1);
    }
}

bool GenX320LLBiases::set_impl(const std::string &bias_name, int bias_value) {
    auto it = biases_map_.find(bias_name);
    assert(it != biases_map_.end());
    auto &bias_info = it->second;

    if (0 <= bias_value && (uint32_t)bias_value <= bias_info.get_register_max_value()) {
        (*register_map_)[bias_info.get_register_name()]["bias_ctl"].write_value(bias_value);
        (*register_map_)[bias_info.get_register_name()]["single"].write_value(1);
        return true;
    } else {
        std::stringstream ss;
        ss << "Invalid value " << bias_value << " for bias \"" << bias_name << "\". Value should be within range [" << 0
           << ", " << bias_info.get_register_max_value() << "].";
        throw HalException(HalErrorCode::ValueOutOfRange, ss.str());
        return false;
    }
}

int GenX320LLBiases::get_impl(const std::string &bias_name) const {
    auto it = biases_map_.find(bias_name);
    assert(it != biases_map_.end());
    auto &bias_info = it->second;

    auto r = (*register_map_)[bias_info.get_register_name()]["bias_ctl"].read_value();

    return r;
}

bool GenX320LLBiases::get_bias_info_impl(const std::string &bias_name, LL_Bias_Info &bias_info) const {
    auto it = biases_map_.find(bias_name);
    if (it == biases_map_.end()) {
        return false;
    }
    bias_info = it->second;
    return true;
}

std::map<std::string, int> GenX320LLBiases::get_all_biases() const {
    std::map<std::string, int> ret;
    for (auto &b : biases_map_) {
        ret[b.first] = get(b.first);
    }
    return ret;
}

} // namespace Metavision
