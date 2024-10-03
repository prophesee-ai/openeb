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

#include <numeric>
#include <memory>

#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"
#include "metavision/hal/utils/hal_error_code.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/utils/hal_log.h"

namespace Metavision {

I_HW_Identification::SensorInfo::SensorInfo(uint16_t major_version, uint16_t minor_version, const std::string& name) :
    major_version_(major_version), minor_version_(minor_version), name_(name) {
}

I_HW_Identification::SensorInfo::SensorInfo(const std::string& name) :
    major_version_(0), minor_version_(0), name_(name) {
}

I_HW_Identification::I_HW_Identification(const std::shared_ptr<I_PluginSoftwareInfo> &plugin_sw_info) :
    plugin_sw_info_(plugin_sw_info) {
    if (!plugin_sw_info_) {
        throw(HalException(HalErrorCode::InternalInitializationError, "Plugin software info facility not set."));
    }
}

I_HW_Identification::SystemInfo I_HW_Identification::get_system_info() const {
    SystemInfo infos;
    infos.insert({"Serial", get_serial()});
    infos.insert({"Integrator", get_integrator()});
    infos.insert({"Sensor Name", get_sensor_info().name_});
    auto formats     = get_available_data_encoding_formats();
    auto str_formats = std::accumulate(
        formats.begin(), formats.end(), std::string(),
        [](const std::string &a, const std::string &b) -> std::string { return a + (a.length() > 0 ? "," : "") + b; });
    infos.insert({"Available Data Encoding Formats", str_formats});
    infos.insert({"Current Data Encoding Format", get_current_data_encoding_format()});
    infos.insert({"Connection", get_connection_type()});

    return infos;
}

RawFileHeader I_HW_Identification::get_header() const {
    RawFileHeader header = get_header_impl();

    const auto camera_integrator_name        = get_integrator();
    const auto header_camera_integrator_name = header.get_camera_integrator_name();
    if (!header_camera_integrator_name.empty() && header_camera_integrator_name != camera_integrator_name) {
        MV_HAL_LOG_TRACE() << "The integrator name found in the header:" << header_camera_integrator_name
                           << "is invalid. Replacing it with:" << camera_integrator_name;
    }
    header.set_camera_integrator_name(camera_integrator_name);

    const auto plugin_integrator_name        = plugin_sw_info_->get_plugin_integrator_name();
    const auto header_plugin_integrator_name = header.get_plugin_integrator_name();
    if (!header_plugin_integrator_name.empty() && header_plugin_integrator_name != plugin_integrator_name) {
        MV_HAL_LOG_TRACE() << "The plugin integrator found in the header:" << header_plugin_integrator_name
                           << "is invalid. Replacing it with:" << plugin_integrator_name;
    }
    header.set_plugin_integrator_name(plugin_integrator_name);

    const auto plugin_name        = plugin_sw_info_->get_plugin_name();
    const auto header_plugin_name = header.get_plugin_name();
    if (!header_plugin_name.empty() && header_plugin_name != plugin_name) {
        MV_HAL_LOG_TRACE() << "The plugin name found in the header:" << header_plugin_name
                           << "is invalid. Replacing it with:" << plugin_name;
    }
    header.set_plugin_name(plugin_name);

    return header;
}

RawFileHeader I_HW_Identification::get_header_impl() const {
    return RawFileHeader();
}

const std::shared_ptr<I_PluginSoftwareInfo> &I_HW_Identification::get_plugin_software_info() const {
    return plugin_sw_info_;
}

DeviceConfigOptionMap I_HW_Identification::get_device_config_options() const {
    DeviceConfigOptionMap res = hal_device_config_options_;

    for (const auto &p : get_device_config_options_impl()) {
        res[p.first] = p.second;
    }

    return res;
}

void I_HW_Identification::add_hal_device_config_option(const std::string &key, const DeviceConfigOption &option) {
    hal_device_config_options_[key] = option;
}

} // namespace Metavision
