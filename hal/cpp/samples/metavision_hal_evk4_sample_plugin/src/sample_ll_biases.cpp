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

#include <iostream>

#include "sample_ll_biases.h"
#include "internal/sample_register_access.h"


SampleLLBiases::SampleLLBiases(const Metavision::DeviceConfig &device_config,
                               std::shared_ptr<SampleUSBConnection> usb_connection) :
    I_LL_Biases(device_config), usb_connection_(usb_connection) {}

SampleLLBiases::~SampleLLBiases() = default;


std::map<std::string, int> SampleLLBiases::get_all_biases() const {
    std::map<std::string, int> Biases;
    std::vector<std::string> bias_names = {"bias_diff_on", "bias_diff_off", "bias_fo", "bias_hpf", "bias_refr"};
    for (const auto& name : bias_names) {
        Biases[name] = get_impl(name);
    }
    return Biases;
}

bool SampleLLBiases::set_impl(const std::string &bias_name, int bias_value) {
    return false;
}

int SampleLLBiases::get_impl(const std::string &bias_name) const {
    if (bias_name == "bias_diff_on") {
        return read_register(*usb_connection_, 0x00001010) & 0xFF;
    } else if (bias_name == "bias_diff_off") {
        return read_register(*usb_connection_, 0x00001018) & 0xFF;
    } else if (bias_name == "bias_fo") {
        return read_register(*usb_connection_, 0x00001004) & 0xFF;
    } else if (bias_name == "bias_hpf") {
        return read_register(*usb_connection_, 0x0000100C) & 0xFF;
    } else if (bias_name == "bias_refr") {
        return read_register(*usb_connection_, 0x00001020) & 0xFF;
    } else return -1;
}

bool SampleLLBiases::get_bias_info_impl(const std::string &bias_name, Metavision::LL_Bias_Info &bias_info) const {
    // Non-modifiable bias with no information
    bias_info = Metavision::LL_Bias_Info();
    return true;
}