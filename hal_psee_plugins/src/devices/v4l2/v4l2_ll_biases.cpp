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
#include "metavision/hal/utils/detail/hal_log_impl.h"
#include "metavision/psee_hw_layer/devices/v4l2/v4l2_ll_biases.h"
#include "metavision/psee_hw_layer/boards/v4l2/v4l2_controls.h"

namespace Metavision {

V4L2LLBiases::V4L2LLBiases(const DeviceConfig &device_config, std::shared_ptr<V4L2Controls> controls, bool relative) :
    I_LL_Biases(device_config), controls_(controls), relative_(relative) {
    // reset all biases to default values
    controls_->foreach ([&](V4L2Controls::V4L2Control &ctrl) {
        auto name = std::string(ctrl.query_.name);
        // skip non bias controls
        if (name.find("bias_") != 0) {
            return 0;
        }

        ctrl.reset();
        return 0;
    });
}

bool V4L2LLBiases::set_impl(const std::string &bias_name, int bias_value) {
    auto ctrl = controls_->get(bias_name);
    int ret;

    if (relative_) {
        int current_val = get_impl(bias_name);
        bias_value += ctrl.query_.default_value;
    }

    ret = ctrl.set_int(bias_value);
    if (ret != 0) {
        MV_HAL_LOG_ERROR() << "Failed to set" << bias_name << "Control value to" << bias_value;
        return false;
    }

    MV_HAL_LOG_INFO() << "Success setting" << bias_name << "Control value to" << bias_value;
    return true;
}

int V4L2LLBiases::get_impl(const std::string &bias_name) const {
    auto ctrl      = controls_->get(bias_name);
    auto maybe_val = ctrl.get_int();
    if (!maybe_val.has_value())
        throw std::runtime_error("could not get control value");

    MV_HAL_LOG_INFO() << bias_name << "Control value:" << *maybe_val;
    return relative_ ? *maybe_val - ctrl.query_.default_value : *maybe_val;
}

bool V4L2LLBiases::get_bias_info_impl(const std::string &bias_name, LL_Bias_Info &bias_info) const {
    auto ctrl = controls_->get(bias_name);
    bias_info = LL_Bias_Info(0, 127, ctrl.query_.minimum, ctrl.query_.maximum, std::string("todo::description"), true,
                             std::string("todo::category"));

    return true;
}

std::map<std::string, int> V4L2LLBiases::get_all_biases() const {
    std::map<std::string, int> biases;

    controls_->foreach ([&biases](V4L2Controls::V4L2Control &ctrl) {
        auto name = std::string(ctrl.query_.name);
        // skip non bias controls
        if (name.find("bias_") != 0) {
            return 0;
        }

        biases[ctrl.query_.name] = ctrl.get_int().value_or(0xFFFFFFFF);
        return 0;
    });

    return biases;
}

} // namespace Metavision
