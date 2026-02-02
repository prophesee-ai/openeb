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
#include "metavision/psee_hw_layer/devices/gen41/gen41_ll_biases.h"
#include "metavision/hal/facilities/i_hw_register.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"
#include "utils/psee_hal_utils.h"

namespace Metavision {

static constexpr uint32_t BIAS_CONF = 0x11A10000;

uint32_t get_gen41_bias_encoding(const Gen41_LL_Biases::Gen41LLBias &bias, int bias_value, bool saturate_value) {
    if (saturate_value) {
        if (bias_value < 0) {
            bias_value = 0;
        }
        if (bias_value > 255) {
            bias_value = 255;
        }
    }
    return (uint32_t)bias_value | BIAS_CONF;
}

} // namespace Metavision

namespace Metavision {

Gen41_LL_Biases::Gen41_LL_Biases(const DeviceConfig &device_config, const std::shared_ptr<I_HW_Register> &i_hw_register,
                                 const std::string &sensor_prefix) :
    I_LL_Biases(device_config), i_hw_register_(i_hw_register), base_name_(sensor_prefix) {
    if (!i_hw_register_) {
        throw(HalException(PseeHalPluginErrorCode::HWRegisterNotFound, "HW Register facility is null."));
    }
    // Init map with the values in the registers
    auto &gen41_biases_map = get_gen41_biases_map();
    gen41_biases_map.clear();
    gen41_biases_map.insert({"bias_fo", Gen41LLBias(0x2D, 0x6E, true, "bias/bias_fo", get_bias_description("bias_fo"),
                                                    get_bias_category("bias_fo"))});
    gen41_biases_map.insert({"bias_hpf", Gen41LLBias(0x00, 0x78, true, "bias/bias_hpf",
                                                     get_bias_description("bias_hpf"), get_bias_category("bias_hpf"))});
    gen41_biases_map.insert(
        {"bias_diff_on", Gen41LLBias(0x00, 0x8C, true, "bias/bias_diff_on", get_bias_description("bias_diff_on"),
                                     get_bias_category("bias_diff_on"))});
    gen41_biases_map.insert(
        {"bias_diff", Gen41LLBias(0x34, 0x64, true, "bias/bias_diff", get_bias_description("bias_diff"),
                                  get_bias_category("bias_diff"))});
    gen41_biases_map.insert(
        {"bias_diff_off", Gen41LLBias(0x19, 0xFF, true, "bias/bias_diff_off", get_bias_description("bias_diff_off"),
                                      get_bias_category("bias_diff_off"))});
    gen41_biases_map.insert(
        {"bias_refr", Gen41LLBias(0x14, 0x64, true, "bias/bias_refr", get_bias_description("bias_refr"),
                                  get_bias_category("bias_refr"))});
}

bool Gen41_LL_Biases::set_impl(const std::string &bias_name, int bias_value) {
    bool bypass_range_check = device_config_.biases_range_check_bypass();
    if (!bypass_range_check) {
        if (bias_name == "bias_diff_on") {
            auto b                     = get("bias_diff");
            int min_bias_diff_on_value = b + 15;
            if (bias_value < min_bias_diff_on_value) {
                MV_HAL_LOG_WARNING() << "Current bias_diff_on minimal value is" << min_bias_diff_on_value;
                return false;
            }
        }
        if (bias_name == "bias_diff_off") {
            auto b                      = get("bias_diff");
            int max_bias_diff_off_value = b - 15;
            if (bias_value > max_bias_diff_off_value) {
                bias_value = max_bias_diff_off_value;
                MV_HAL_LOG_WARNING() << "Current bias_diff_off maximal value is" << max_bias_diff_off_value;
                return false;
            }
        }
    }

    auto it = get_gen41_biases_map().find(bias_name);
    assert(it != get_gen41_biases_map().end());
    auto &bias_info = it->second;

    auto reg = get_gen41_bias_encoding(bias_info, bias_value, !bypass_range_check);
    get_hw_register()->write_register(base_name_ + bias_info.get_register_name(), reg);
    return true;
}

int Gen41_LL_Biases::get_impl(const std::string &bias_name) const {
    auto it = get_gen41_biases_map().find(bias_name);
    assert(it != get_gen41_biases_map().end());
    auto &bias_info = it->second;

    auto r = get_hw_register()->read_register(base_name_ + bias_info.get_register_name());
    if (r == uint32_t(-1))
        return -1;
    r = r & 0xFF;
    if (r > 255) {
        return -1;
    }
    return r;
}

bool Gen41_LL_Biases::get_bias_info_impl(const std::string &bias_name, LL_Bias_Info &bias_info) const {
    auto it = get_gen41_biases_map().find(bias_name);
    if (it == get_gen41_biases_map().end()) {
        return false;
    }
    bias_info = it->second;
    return true;
}

std::map<std::string, int> Gen41_LL_Biases::get_all_biases() const {
    std::map<std::string, int> ret;
    for (auto &b : get_gen41_biases_map()) {
        ret[b.first] = get(b.first);
    }
    return ret;
}

const std::shared_ptr<I_HW_Register> &Gen41_LL_Biases::get_hw_register() const {
    return i_hw_register_;
}

std::map<std::string, Gen41_LL_Biases::Gen41LLBias> &Gen41_LL_Biases::get_gen41_biases_map() {
    return biases_map_;
}

const std::map<std::string, Gen41_LL_Biases::Gen41LLBias> &Gen41_LL_Biases::get_gen41_biases_map() const {
    return biases_map_;
}

} // namespace Metavision
