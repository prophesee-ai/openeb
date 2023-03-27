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
    Gen41_LL_Biases(const DeviceConfig &device_config, const std::shared_ptr<I_HW_Register> &i_hw_register,
                    const std::string &sensor_prefix);

    virtual std::map<std::string, int> get_all_biases() override;

protected:
    const std::shared_ptr<I_HW_Register> &get_hw_register() const;

private:
    virtual bool set_impl(const std::string &bias_name, int bias_value) override;
    virtual int get_impl(const std::string &bias_name) override;
    virtual bool get_bias_info_impl(const std::string &bias_name, LL_Bias_Info &info) const override;

    std::shared_ptr<I_HW_Register> i_hw_register_;
    std::string base_name_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN41_LL_BIASES_H
