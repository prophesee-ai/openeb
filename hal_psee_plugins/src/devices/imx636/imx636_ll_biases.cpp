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
#include <vector>
#include "metavision/psee_hw_layer/devices/imx636/imx636_ll_biases.h"
#include "metavision/hal/facilities/i_hw_register.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"
#include "utils/psee_hal_utils.h"

namespace Metavision {

#include "devices/imx636/imx636_bias_settings.h"
#include "devices/imx636/imx636_bias_settings_iterator.h"

Imx636_LL_Biases::Imx636_LL_Biases(const DeviceConfig &device_config,
                                   const std::shared_ptr<I_HW_Register> &i_hw_register,
                                   const std::string &sensor_prefix) :
    Imx636_LL_Biases(device_config, i_hw_register, sensor_prefix, bias_settings /* from bias_settings_iterator.h */) {}

Imx636_LL_Biases::Imx636_LL_Biases(const DeviceConfig &device_config,
                                   const std::shared_ptr<I_HW_Register> &i_hw_register,
                                   const std::string &sensor_prefix, std::vector<imx636_bias_setting> &bias_settings) :
    I_LL_Biases(device_config), bypass_range_check_(device_config.biases_range_check_bypass()) {
    std::string BIAS_PATH = "bias/";

    for (auto &bias_setting : bias_settings) {
        Imx636LLBias bias(bias_setting.name, sensor_prefix + BIAS_PATH, i_hw_register, bias_setting.min_allowed_offset,
                          bias_setting.max_allowed_offset, bias_setting.min_recommended_offset,
                          bias_setting.max_recommended_offset, get_bias_description(bias_setting.name),
                          bias_setting.modifiable, get_bias_category(bias_setting.name));
        biases_map_.insert({bias_setting.name, bias});
    }
}

/**
 * @brief Attempts to adjust the bias by some offset
 *
 * @param bias_name Bias to adjust
 * @param bias_value Offset to adjust by
 * @return true if successful, false otherwise
 */
bool Imx636_LL_Biases::set_impl(const std::string &bias_name, int bias_value) {
    auto it = biases_map_.find(bias_name);
    if (it == biases_map_.end()) {
        return false;
    }

    // Update the value
    it->second.set_offset(bias_value);
    return true;
}

/**
 * @brief get the current bias offset
 *
 * @param bias_name name of the desired bias
 * @return int the offset
 */
int Imx636_LL_Biases::get_impl(const std::string &bias_name) const {
    auto it = biases_map_.find(bias_name);
    assert(it != biases_map_.end());
    auto &bias_info = it->second;

    return bias_info.current_offset();
}

bool Imx636_LL_Biases::get_bias_info_impl(const std::string &bias_name, LL_Bias_Info &bias_info) const {
    auto it = biases_map_.find(bias_name);
    if (it == biases_map_.end()) {
        return false;
    }
    bias_info = it->second;
    return true;
}

std::map<std::string, int> Imx636_LL_Biases::get_all_biases() const {
    std::map<std::string, int> ret;
    for (auto &b : biases_map_) {
        ret[b.first] = get(b.first);
    }
    return ret;
}

Imx636_LL_Biases::Imx636LLBias::Imx636LLBias(std::string register_name, std::string bias_path,
                                             std::shared_ptr<I_HW_Register> hw_register, int min_allowed_offset,
                                             int max_allowed_offset, int min_recommended_offset,
                                             int max_recommended_offset, const std::string &description,
                                             bool modifiable, const std::string &category) :
    LL_Bias_Info(min_allowed_offset, max_allowed_offset, min_recommended_offset, max_recommended_offset, description,
                 modifiable, category),
    register_name_(register_name),
    bias_path_(bias_path),
    i_hw_register_(hw_register) {
    factory_default_ = 0xff & (hw_register->read_register(bias_path + register_name));
    current_value_   = factory_default_;

    display_bias();
}
int Imx636_LL_Biases::Imx636LLBias::current_offset() const {
    display_bias();
    return current_value_ - factory_default_;
}
void Imx636_LL_Biases::Imx636LLBias::set_offset(const int val) {
    display_bias();
    current_value_ = factory_default_ + val;
    i_hw_register_->write_register(bias_path_ + register_name_, get_encoding());
    display_bias();
}

void Imx636_LL_Biases::Imx636LLBias::display_bias() const {
    MV_HAL_LOG_TRACE() << "register name:" << register_name_ << ", factory default:" << factory_default_
                       << ", current value:" << current_value_ << ", diff:" << current_value_ - factory_default_
                       << ", value range: [" << get_bias_range().first << ", " << get_bias_range().second << "]";
}

static constexpr uint32_t BIAS_CONF = 0x11A10000;

uint32_t Imx636_LL_Biases::Imx636LLBias::get_encoding() {
    if (current_value_ < 0) {
        current_value_ = 0;
    }
    if (current_value_ > 255) {
        current_value_ = 255;
    }
    return (uint32_t)current_value_ | BIAS_CONF;
}

} // namespace Metavision
