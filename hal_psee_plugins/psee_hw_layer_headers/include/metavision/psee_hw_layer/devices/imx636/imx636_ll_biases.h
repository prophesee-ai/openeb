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

#ifndef METAVISION_HAL_IMX636_LL_BIASES_H
#define METAVISION_HAL_IMX636_LL_BIASES_H

#include <string>
#include <map>
#include <vector>

#include "metavision/hal/facilities/i_ll_biases.h"

namespace Metavision {

class I_HW_Register;

struct imx636_bias_setting {
    std::string name;
    int min_allowed_offset;
    int max_allowed_offset;
    int min_recommended_offset;
    int max_recommended_offset;
    bool modifiable;
};

class Imx636_LL_Biases : public I_LL_Biases {
public:
    Imx636_LL_Biases(const DeviceConfig &device_config, const std::shared_ptr<I_HW_Register> &i_hw_register,
                     const std::string &sensor_prefix);
    Imx636_LL_Biases(const DeviceConfig &device_config, const std::shared_ptr<I_HW_Register> &i_hw_register,
                     const std::string &sensor_prefix, std::vector<imx636_bias_setting> &bias_settings);

    virtual std::map<std::string, int> get_all_biases() override;

protected:
    class Imx636LLBias : public LL_Bias_Info {
    public:
        Imx636LLBias(std::string register_name, std::string bias_path, std::shared_ptr<I_HW_Register> hw_register,
                     int min_allowed_offset, int max_allowed_offset, int min_recommended_offset,
                     int max_recommended_offset, const std::string &description, bool modifiable,
                     const std::string &category);

        ~Imx636LLBias() {}
        int current_offset() const;
        void set_offset(const int val);

    private:
        void display_bias() const;
        uint32_t get_encoding();
        std::shared_ptr<I_HW_Register> i_hw_register_;
        std::string register_name_;
        std::string bias_path_;
        int current_value_;
        int factory_default_;
    };
    std::map<std::string, Imx636LLBias> biases_map_;

private:
    virtual bool set_impl(const std::string &bias_name, int bias_value) override;
    virtual int get_impl(const std::string &bias_name) override;
    virtual bool get_bias_info_impl(const std::string &bias_name, LL_Bias_Info &info) const override;

    bool bypass_range_check_;
};

} // namespace Metavision

#endif // METAVISION_HAL_IMX636_LL_BIASES_H
