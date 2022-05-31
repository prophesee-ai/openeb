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

#include "devices/imx636/imx636_ll_biases.h"
#include "devices/imx636/imx636_bias_settings.h"
#include "metavision/hal/facilities/i_hw_register.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"
#include "utils/psee_hal_utils.h"

namespace Metavision {

static constexpr uint32_t BIAS_CONF = 0x11A10000;

class Imx636LLBias {
public:
    Imx636LLBias(bool modifiable, std::string register_name, int sensor_offset, int current_value, int factory_default,
                 int min_offset, int max_offset) {
        register_name_   = register_name;
        modifiable_      = modifiable;
        current_offset_  = sensor_offset;
        factory_default_ = factory_default;
        current_value_   = factory_default;
        min_offset_      = min_offset;
        max_offset_      = max_offset;

        display_bias();
    }

    ~Imx636LLBias() {}
    bool is_modifiable() const {
        return modifiable_;
    }
    const std::string &get_register_name() const {
        return register_name_;
    }
    int get_current_offset() {
        return current_offset_;
    }
    void set_current_offset(const int val) {
        current_offset_ = val;
    }
    int get_current_value() {
        return current_value_;
    }
    void set_current_value(const int val) {
        current_value_ = val;
    }
    int get_factory_default_value() {
        return factory_default_;
    }
    void set_factory_default_value(const int val) {
        factory_default_ = val;
    }
    int get_min_offset() {
        return min_offset_;
    }
    int get_max_offset() {
        return max_offset_;
    }
    void display_bias() {
        MV_HAL_LOG_TRACE() << "register name:" << register_name_ << ", factory default:" << factory_default_
                           << ", current value:" << current_value_ << ", current offset:" << current_offset_
                           << ", min offset:" << min_offset_ << ", max offset:" << max_offset_;
    }

private:
    std::string register_name_;
    bool modifiable_;
    int current_value_;
    int current_offset_;
    int factory_default_;
    int min_offset_;
    int max_offset_;
};

std::map<std::string, Imx636LLBias> &get_imx636_biases_map() {
    static std::map<std::string, Imx636LLBias> biases_map_;
    return biases_map_;
}

uint32_t get_imx636_bias_encoding(const Imx636LLBias &bias, int bias_value) {
    if (!Metavision::is_expert_mode_enabled()) {
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

Imx636_LL_Biases::Imx636_LL_Biases(const std::shared_ptr<I_HW_Register> &i_hw_register,
                                   const std::string &sensor_prefix) :
    i_hw_register_(i_hw_register), base_name_(sensor_prefix) {
    if (!i_hw_register_) {
        throw(HalException(PseeHalPluginErrorCode::HWRegisterNotFound, "HW Register facility is null."));
    }

    std::string BIAS_PATH = "bias/";

    std::string bias_fo_name       = "bias_fo";
    std::string bias_hpf_name      = "bias_hpf";
    std::string bias_diff_on_name  = "bias_diff_on";
    std::string bias_diff_name     = "bias_diff";
    std::string bias_diff_off_name = "bias_diff_off";
    std::string bias_refr_name     = "bias_refr";

    int bias_fo_sensor_current_offset       = 0;
    int bias_hpf_sensor_current_offset      = 0;
    int bias_diff_on_sensor_current_offset  = 0;
    int bias_diff_sensor_current_offset     = 0;
    int bias_diff_off_sensor_current_offset = 0;
    int bias_refr_sensor_current_offset     = 0;

    int bias_fo_factory_default  = 0xff & (get_hw_register()->read_register(base_name_ + BIAS_PATH + bias_fo_name));
    int bias_hpf_factory_default = 0xff & (get_hw_register()->read_register(base_name_ + BIAS_PATH + bias_hpf_name));
    int bias_diff_on_factory_default =
        0xff & (get_hw_register()->read_register(base_name_ + BIAS_PATH + bias_diff_on_name));
    int bias_diff_factory_default = 0xff & (get_hw_register()->read_register(base_name_ + BIAS_PATH + bias_diff_name));
    int bias_diff_off_factory_default =
        0xff & (get_hw_register()->read_register(base_name_ + BIAS_PATH + bias_diff_off_name));
    int bias_refr_factory_default = 0xff & (get_hw_register()->read_register(base_name_ + BIAS_PATH + bias_refr_name));

    int bias_fo_current_value       = bias_fo_factory_default;
    int bias_hpf_current_value      = bias_hpf_factory_default;
    int bias_diff_on_current_value  = bias_diff_on_factory_default;
    int bias_diff_current_value     = bias_diff_factory_default;
    int bias_diff_off_current_value = bias_diff_off_factory_default;
    int bias_refr_current_value     = bias_refr_factory_default;

    int bias_fo_min_offset       = BIAS_FO_MIN_OFFSET;
    int bias_fo_max_offset       = BIAS_FO_MAX_OFFSET;
    int bias_hpf_min_offset      = BIAS_HPF_MIN_OFFSET;
    int bias_hpf_max_offset      = BIAS_HPF_MAX_OFFSET;
    int bias_diff_on_min_offset  = BIAS_DIFF_ON_MIN_OFFSET;
    int bias_diff_on_max_offset  = BIAS_DIFF_ON_MAX_OFFSET;
    int bias_diff_min_offset     = BIAS_DIFF_MIN_OFFSET;
    int bias_diff_max_offset     = BIAS_DIFF_MAX_OFFSET;
    int bias_diff_off_min_offset = BIAS_DIFF_OFF_MIN_OFFSET;
    int bias_diff_off_max_offset = BIAS_DIFF_OFF_MAX_OFFSET;
    int bias_refr_min_offset     = BIAS_REFR_MIN_OFFSET;
    int bias_refr_max_offset     = BIAS_REFR_MAX_OFFSET;

    bool bias_fo_modifiable       = true;
    bool bias_hpf_modifiable      = true;
    bool bias_diff_on_modifiable  = true;
    bool bias_diff_modifiable     = false;
    bool bias_diff_off_modifiable = true;
    bool bias_refr_modifiable     = true;

    Imx636LLBias fo(bias_fo_modifiable, BIAS_PATH + bias_fo_name, bias_fo_sensor_current_offset, bias_fo_current_value,
                    bias_fo_factory_default, bias_fo_min_offset, bias_fo_max_offset);
    Imx636LLBias hpf(bias_hpf_modifiable, BIAS_PATH + bias_hpf_name, bias_hpf_sensor_current_offset,
                     bias_hpf_current_value, bias_hpf_factory_default, bias_hpf_min_offset, bias_hpf_max_offset);
    Imx636LLBias diff_on(bias_diff_on_modifiable, BIAS_PATH + bias_diff_on_name, bias_diff_on_sensor_current_offset,
                         bias_diff_on_current_value, bias_diff_on_factory_default, bias_diff_on_min_offset,
                         bias_diff_on_max_offset);
    Imx636LLBias diff(bias_diff_modifiable, BIAS_PATH + bias_diff_name, bias_diff_sensor_current_offset,
                      bias_diff_current_value, bias_diff_factory_default, bias_diff_min_offset, bias_diff_max_offset);
    Imx636LLBias diff_off(bias_diff_off_modifiable, BIAS_PATH + bias_diff_off_name, bias_diff_off_sensor_current_offset,
                          bias_diff_off_current_value, bias_diff_off_factory_default, bias_diff_off_min_offset,
                          bias_diff_off_max_offset);
    Imx636LLBias refr(bias_refr_modifiable, BIAS_PATH + bias_refr_name, bias_refr_sensor_current_offset,
                      bias_refr_current_value, bias_refr_factory_default, bias_refr_min_offset, bias_refr_max_offset);

    // Init map with the values in the registers
    auto &imx636_biases_map = get_imx636_biases_map();
    imx636_biases_map.clear();
    imx636_biases_map.insert({bias_fo_name, fo});
    imx636_biases_map.insert({bias_hpf_name, hpf});
    imx636_biases_map.insert({bias_diff_on_name, diff_on});
    imx636_biases_map.insert({bias_diff_name, diff});
    imx636_biases_map.insert({bias_diff_off_name, diff_off});
    imx636_biases_map.insert({bias_refr_name, refr});
}

/**
 * @brief Attempts to adjust the bias by some offset
 *
 * @param bias_name Bias to adjust
 * @param bias_value Offset to adjust by
 * @return true if successful, false otherwise
 */
bool Imx636_LL_Biases::set(const std::string &bias_name, int bias_value) {
    auto it = get_imx636_biases_map().find(bias_name);
    if (it == get_imx636_biases_map().end()) {
        return false;
    }
    if (it->second.is_modifiable() == false) {
        return false;
    }

    // Display old bias settings
    it->second.display_bias();

    // What the new value will be if all checks pass
    auto potential_update_value = it->second.get_factory_default_value() + bias_value;
    // Check bounds
    if (!Metavision::is_expert_mode_enabled()) {
        if (bias_value < it->second.get_min_offset()) {
            MV_HAL_LOG_WARNING() << "Attempted to set" << bias_name << "offset lower than min offset of"
                                 << it->second.get_min_offset();
            return false;
        } else if (bias_value > it->second.get_max_offset()) {
            MV_HAL_LOG_WARNING() << "Attempted to set" << bias_name << "offset greater than max offset of"
                                 << it->second.get_max_offset();
            return false;
        }
    }

    // Update the value
    it->second.set_current_value(potential_update_value);
    // Tracking the total offset for later
    it->second.set_current_offset(bias_value);
    it->second.display_bias();

    // Update the hardware
    auto reg = get_imx636_bias_encoding(it->second, it->second.get_current_value());
    get_hw_register()->write_register(base_name_ + it->second.get_register_name(), reg);
    return true;
}

/**
 * @brief get the current bias offset
 *
 * @param bias_name name of the desired bias
 * @return int the offset
 */
int Imx636_LL_Biases::get(const std::string &bias_name) {
    auto it = get_imx636_biases_map().find(bias_name);
    if (it == get_imx636_biases_map().end()) {
        MV_HAL_LOG_WARNING() << "Failed Bias Name check";
        return -1;
    }
    it->second.display_bias();
    return it->second.get_current_offset();
}

std::map<std::string, int> Imx636_LL_Biases::get_all_biases() {
    std::map<std::string, int> ret;
    for (auto &b : get_imx636_biases_map()) {
        ret[b.first] = get(b.first);
    }
    return ret;
}

const std::shared_ptr<I_HW_Register> &Imx636_LL_Biases::get_hw_register() const {
    return i_hw_register_;
}

} // namespace Metavision
