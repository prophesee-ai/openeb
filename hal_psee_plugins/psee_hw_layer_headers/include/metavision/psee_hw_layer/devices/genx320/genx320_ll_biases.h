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

#ifndef METAVISION_HAL_GENX320_LL_BIASES_H
#define METAVISION_HAL_GENX320_LL_BIASES_H

#include <string>
#include <map>
#include <vector>

#include "metavision/hal/facilities/i_ll_biases.h"
#include "metavision/psee_hw_layer/devices/common/bias_settings.h"

namespace Metavision {

class RegisterMap;

class GenX320LLBiases : public I_LL_Biases {
public:
    GenX320LLBiases(const std::shared_ptr<RegisterMap> &register_map, const DeviceConfig &device_config);

    virtual std::map<std::string, int> get_all_biases() override;

protected:
    class GenX320Bias : public LL_Bias_Info {
    public:
        GenX320Bias(std::string register_name, const uint8_t &default_val, const bias_settings &bias_conf,
                    const std::string &description, const std::string &category);

        ~GenX320Bias();

        const std::string &get_register_name() const;
        const uint32_t &get_register_max_value() const;
        void display_bias() const;

    private:
        std::string register_name_;
        uint8_t default_value_;
        uint32_t reg_max_;
    };

    std::map<std::string, GenX320Bias> biases_map_;

private:
    virtual bool set_impl(const std::string &bias_name, int bias_value) override;
    virtual int get_impl(const std::string &bias_name) override;
    virtual bool get_bias_info_impl(const std::string &bias_name, LL_Bias_Info &info) const override;

    std::shared_ptr<RegisterMap> register_map_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GENX320_LL_BIASES_H
