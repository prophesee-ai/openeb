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

#include "metavision/hal/facilities/i_ll_biases.h"
#include "devices/imx636/imx636_bias.h"

namespace Metavision {

class I_HW_Register;

class Imx636_LL_Biases : public I_LL_Biases {
public:
    Imx636_LL_Biases(const std::shared_ptr<I_HW_Register> &i_hw_register, const std::string &sensor_prefix);

    virtual bool set(const std::string &bias_name, int bias_value) override;
    virtual int get(const std::string &bias_name) override;
    virtual std::map<std::string, int> get_all_biases() override;

protected:
    const std::shared_ptr<I_HW_Register> &get_hw_register() const;

    std::shared_ptr<I_HW_Register> i_hw_register_;
    std::string base_name_;

    std::map<std::string, Imx636LLBias> biases_map_;

    std::string BIAS_PATH          = "bias/";
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

    bool bias_fo_modifiable       = true;
    bool bias_hpf_modifiable      = true;
    bool bias_diff_on_modifiable  = true;
    bool bias_diff_modifiable     = false;
    bool bias_diff_off_modifiable = true;
    bool bias_refr_modifiable     = true;
};

} // namespace Metavision

#endif // METAVISION_HAL_IMX636_LL_BIASES_H
