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

#include "boards/v4l2/v4l2_hardware_identification.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"
#include "metavision/psee_hw_layer/boards/rawfile/psee_raw_file_header.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"

namespace Metavision {

V4l2HwIdentification::V4l2HwIdentification(const V4l2Capability cap,
                                           const std::shared_ptr<I_PluginSoftwareInfo> &plugin_sw_info,
                                           const SensorDescriptor &sensor_descriptor) :
    I_HW_Identification(plugin_sw_info), cap_(cap), sensor_descriptor_(sensor_descriptor) {}

long V4l2HwIdentification::get_system_id() const {
    // @TODO Retrieve those info through V4L2
    return 1234;
}
I_HW_Identification::SensorInfo V4l2HwIdentification::get_sensor_info() const {
    // @TODO Retrieve those info through V4L2
    return sensor_descriptor_.info;
}
std::vector<std::string> V4l2HwIdentification::get_available_data_encoding_formats() const {
    // @TODO Retrieve those info through V4L2
    auto format = get_current_data_encoding_format();
    auto pos    = format.find(";");
    if (pos != std::string::npos) {
        auto evt_type = format.substr(0, pos);
        return {evt_type};
    }
    return {};
}
std::string V4l2HwIdentification::get_current_data_encoding_format() const {
    // @TODO Retrieve those info through V4L2
    return sensor_descriptor_.encoding_format;
}
std::string V4l2HwIdentification::get_serial() const {
    std::stringstream ss;
    ss << cap_.card;
    return ss.str();
}
std::string V4l2HwIdentification::get_integrator() const {
    std::stringstream ss;
    ss << cap_.driver;
    return ss.str();
}
std::string V4l2HwIdentification::get_connection_type() const {
    std::stringstream ss;
    ss << cap_.bus_info;
    return ss.str();
}
DeviceConfigOptionMap V4l2HwIdentification::get_device_config_options_impl() const {
    return {};
}
RawFileHeader V4l2HwIdentification::get_header_impl() const {
    return PseeRawFileHeader{*this, {sensor_descriptor_.encoding_format}};
}
} // namespace Metavision
