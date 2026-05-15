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

#ifndef METAVISION_HAL_GEN41_LL_BIASES_H
#define METAVISION_HAL_GEN41_LL_BIASES_H

#include <string>
#include <map>

#include "metavision/hal/facilities/i_ll_biases.h"

namespace Metavision {

class I_HW_Register;

class Gen41_LL_Biases : public I_LL_Biases {
public:
    class Gen41LLBias : public LL_Bias_Info {
    public:
        Gen41LLBias(int min_recommended_value, int max_recommended_value, bool modifiable, std::string register_name,
                    const std::string &description, const std::string &category) :
            LL_Bias_Info(0x00, 0xFF, min_recommended_value, max_recommended_value, description, modifiable, category) {
            register_name_ = register_name;
        }

        ~Gen41LLBias() {}
        const std::string &get_register_name() const {
            return register_name_;
        }

    private:
        std::string register_name_;
    };

    Gen41_LL_Biases(const DeviceConfig &device_config, const std::shared_ptr<I_HW_Register> &i_hw_register,
                    const std::string &sensor_prefix);

    virtual std::map<std::string, int> get_all_biases() const override;

protected:
    const std::shared_ptr<I_HW_Register> &get_hw_register() const;
    std::map<std::string, Gen41LLBias> &get_gen41_biases_map();
    const std::map<std::string, Gen41LLBias> &get_gen41_biases_map() const;

private:
    virtual bool set_impl(const std::string &bias_name, int bias_value) override;
    virtual int get_impl(const std::string &bias_name) const override;
    virtual bool get_bias_info_impl(const std::string &bias_name, LL_Bias_Info &info) const override;

    std::shared_ptr<I_HW_Register> i_hw_register_;
    std::string base_name_;
    std::map<std::string, Gen41_LL_Biases::Gen41LLBias> biases_map_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN41_LL_BIASES_H
