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

#ifndef METAVISION_HAL_V4L2_LL_BIASES_H
#define METAVISION_HAL_V4L2_LL_BIASES_H

#include <string>
#include <map>
#include <vector>

#include "metavision/hal/facilities/i_ll_biases.h"
#include "metavision/psee_hw_layer/devices/common/bias_settings.h"
#include "metavision/psee_hw_layer/boards/v4l2/v4l2_controls.h"

namespace Metavision {

class RegisterMap;
class V4L2Controls;

class V4L2LLBiases : public I_LL_Biases {
public:
    V4L2LLBiases(const DeviceConfig &device_config, std::shared_ptr<V4L2Controls> controls, bool relative = false);

    virtual std::map<std::string, int> get_all_biases() const override;

protected:
    class V4L2Bias : public LL_Bias_Info {
    public:
        V4L2Bias(std::string register_name, const uint8_t &default_val, const bias_settings &bias_conf,
                 const std::string &description, const std::string &category);

        ~V4L2Bias() = default;

        const std::string &get_bias_name() const;
        const uint32_t &get_bias_max_value() const;
        void display_bias() const;

    private:
        std::string bias_name_;
        uint8_t default_value_;
    };

    std::map<std::string, V4L2Bias> biases_map_;
    std::shared_ptr<V4L2Controls> controls_;

private:
    virtual bool set_impl(const std::string &bias_name, int bias_value) override;
    virtual int get_impl(const std::string &bias_name) const override;
    virtual bool get_bias_info_impl(const std::string &bias_name, LL_Bias_Info &info) const override;

    std::shared_ptr<RegisterMap> register_map_;
    int fd_;
    bool relative_;
};

} // namespace Metavision

#endif // METAVISION_HAL_V4L2_LL_BIASES_H
