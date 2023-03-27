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

#include <iomanip>
#include <sstream>

#include "sample_hw_identification.h"

constexpr const char *const SampleHWIdentification::SAMPLE_SERIAL;
constexpr long SampleHWIdentification::SAMPLE_SYSTEM_ID;
constexpr const char *const SampleHWIdentification::SAMPLE_INTEGRATOR;
constexpr const char *const SampleHWIdentification::SAMPLE_FORMAT;

SampleHWIdentification::SampleHWIdentification(const std::shared_ptr<Metavision::I_PluginSoftwareInfo> &plugin_sw_info,
                                               const std::string &connection_type) :
    Metavision::I_HW_Identification(plugin_sw_info), connection_type_(connection_type) {}

std::string SampleHWIdentification::get_serial() const {
    return SAMPLE_SERIAL;
}
long SampleHWIdentification::get_system_id() const {
    return SAMPLE_SYSTEM_ID;
}

SampleHWIdentification::SensorInfo SampleHWIdentification::get_sensor_info() const {
    return SensorInfo({1, 0, "Gen1.0"});
}

std::vector<std::string> SampleHWIdentification::get_available_data_encoding_formats() const {
    return {SAMPLE_FORMAT};
}

std::string SampleHWIdentification::get_current_data_encoding_format() const {
    return SAMPLE_FORMAT;
}

std::string SampleHWIdentification::get_integrator() const {
    return SAMPLE_INTEGRATOR;
}

std::string SampleHWIdentification::get_connection_type() const {
    return connection_type_;
}

Metavision::DeviceConfigOptionMap SampleHWIdentification::get_device_config_options_impl() const {
    return {};
}
