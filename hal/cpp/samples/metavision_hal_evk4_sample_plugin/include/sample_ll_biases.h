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

#ifndef METAVISION_HAL_SAMPLE_LL_BIASES_H
#define METAVISION_HAL_SAMPLE_LL_BIASES_H

#include <memory>
#include <string>

#include <metavision/hal/facilities/i_ll_biases.h>

class SampleUSBConnection;

/// @brief Class to access low level biases on the sensor
///
/// This class is the implementation of HAL's facility @ref Metavision::I_LL_Biases
class SampleLLBiases : public Metavision::I_LL_Biases {
public:
    SampleLLBiases(const Metavision::DeviceConfig &device_config, std::shared_ptr<SampleUSBConnection> usb_connection);
    ~SampleLLBiases();
    virtual std::map<std::string, int> get_all_biases() const override;

private:
    virtual bool set_impl(const std::string &bias_name, int bias_value) override;
    virtual int get_impl(const std::string &bias_name)  const override;
    virtual bool get_bias_info_impl(const std::string &bias_name, Metavision::LL_Bias_Info &bias_info) const override;

    std::map<std::string, Metavision::LL_Bias_Info> biases_map_;
    std::shared_ptr<SampleUSBConnection> usb_connection_;
};

#endif // METAVISION_HAL_SAMPLE_LL_BIASES_H
