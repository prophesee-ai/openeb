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

#include "devices/gen41/gen41_ll_biases.h"
#include "metavision/hal/facilities/i_hw_register.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"
#include "utils/psee_hal_utils.h"

namespace Metavision {

static constexpr uint32_t BIAS_CONF = 0x11A10000;

class Gen41LLBias {
public:
    Gen41LLBias(int min_value, int max_value, bool modifiable, std::string register_name) {
        min_value_     = min_value;
        max_value_     = max_value;
        register_name_ = register_name;
        modifiable_    = modifiable;
    }

    ~Gen41LLBias() {}
    bool is_modifiable() const {
        return modifiable_;
    }
    const std::string &get_register_name() const {
        return register_name_;
    }
    bool is_in_range(const int value) {
        return (min_value_ <= value && value <= max_value_);
    }

private:
    int min_value_;
    int max_value_;
    std::string register_name_;
    bool modifiable_;
};

std::map<std::string, Gen41LLBias> &get_gen41_biases_map() {
    static std::map<std::string, Gen41LLBias> biases_map_;
    return biases_map_;
}

uint32_t get_gen41_bias_encoding(const Gen41LLBias &bias, int bias_value) {
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

Gen41_LL_Biases::Gen41_LL_Biases(const std::shared_ptr<I_HW_Register> &i_hw_register,
                                 const std::string &sensor_prefix) :
    i_hw_register_(i_hw_register), base_name_(sensor_prefix) {
    if (!i_hw_register_) {
        throw(HalException(PseeHalPluginErrorCode::HWRegisterNotFound, "HW Register facility is null."));
    }
    // Init map with the values in the registers
    auto &gen41_biases_map = get_gen41_biases_map();
    gen41_biases_map.clear();
    gen41_biases_map.insert({"bias_fo", Gen41LLBias(0x2D, 0x6E, true, "bias/bias_fo")});
    gen41_biases_map.insert({"bias_hpf", Gen41LLBias(0x00, 0x78, true, "bias/bias_hpf")});
    gen41_biases_map.insert({"bias_diff_on", Gen41LLBias(0x00, 0x8C, true, "bias/bias_diff_on")});
    gen41_biases_map.insert({"bias_diff", Gen41LLBias(0x34, 0x64, false, "bias/bias_diff")});
    gen41_biases_map.insert({"bias_diff_off", Gen41LLBias(0x19, 0xFF, true, "bias/bias_diff_off")});
    gen41_biases_map.insert({"bias_refr", Gen41LLBias(0x14, 0x64, true, "bias/bias_refr")});
}

bool Gen41_LL_Biases::set(const std::string &bias_name, int bias_value) {
    auto it = get_gen41_biases_map().find(bias_name);
    if (it == get_gen41_biases_map().end()) {
        return false;
    }
    if (it->second.is_modifiable() == false) {
        return false;
    }
    if (!it->second.is_in_range(bias_value)) {
        MV_HAL_LOG_WARNING() << bias_value << "is not in acceptable range for" << bias_name;
        return false;
    }

    if (bias_name == "bias_diff_on") {
        auto b = get("bias_diff");
        if (bias_value < b + 15) {
            bias_value = b + 15;
            MV_HAL_LOG_WARNING() << "Current bias_diff_on minimal value is" << bias_value;
            return false;
        }
    }
    if (bias_name == "bias_diff_off") {
        auto b = get("bias_diff");
        if (bias_value > b - 15) {
            bias_value = b - 15;
            MV_HAL_LOG_WARNING() << "Current bias_diff_off maximal value is" << bias_value;
            return false;
        }
    }
    auto reg = get_gen41_bias_encoding(it->second, bias_value);
    get_hw_register()->write_register(base_name_ + it->second.get_register_name(), reg);
    return true;
}

int Gen41_LL_Biases::get(const std::string &bias_name) {
    auto it = get_gen41_biases_map().find(bias_name);
    if (it == get_gen41_biases_map().end()) {
        return -1;
    }

    auto r = get_hw_register()->read_register(base_name_ + it->second.get_register_name());
    if (r == uint32_t(-1))
        return -1;
    r = r & 0xFF;
    if (r > 255) {
        return -1;
    }
    return r;
}

std::map<std::string, int> Gen41_LL_Biases::get_all_biases() {
    std::map<std::string, int> ret;
    for (auto &b : get_gen41_biases_map()) {
        ret[b.first] = get(b.first);
    }
    return ret;
}

const std::shared_ptr<I_HW_Register> &Gen41_LL_Biases::get_hw_register() const {
    return i_hw_register_;
}

} // namespace Metavision
