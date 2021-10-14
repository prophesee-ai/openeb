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

#include <metavision/hal/facilities/i_plugin_software_info.h>

#include "common/raw_hw_identification.h"
#include "common/raw_constants.h"

namespace Metavision {

RawHWIdentification::RawHWIdentification(const std::shared_ptr<Metavision::I_PluginSoftwareInfo> &plugin_sw_info,
                                         const std::string serial,
                                         const Metavision::I_HW_Identification::SensorInfo &sensor_info,
                                         const std::string &evt_version) :
    Metavision::I_HW_Identification(plugin_sw_info),
    serial_(serial),
    sensor_info_(sensor_info),
    evt_format_(evt_version) {}

std::string RawHWIdentification::get_serial() const {
    return serial_;
}

long RawHWIdentification::get_system_id() const {
    return 0;
}

Metavision::I_HW_Identification::SensorInfo RawHWIdentification::get_sensor_info() const {
    return sensor_info_;
}

long RawHWIdentification::get_system_version() const {
    return 0;
}

std::vector<std::string> RawHWIdentification::get_available_raw_format() const {
    std::vector<std::string> formats;
    formats.push_back(evt_format_);
    return formats;
}

std::string RawHWIdentification::get_integrator() const {
    return raw_default_integrator;
}

std::string RawHWIdentification::get_connection_type() const {
    return raw_default_connection_type;
}

Metavision::RawFileHeader RawHWIdentification::get_header_impl() const {
    Metavision::RawFileHeader header;
    header.set_plugin_name(get_plugin_software_info()->get_plugin_name());
    header.set_integrator_name(raw_default_integrator);
    header.add_date();
    header.set_field(raw_key_evt, evt_format_);
    header.set_field(raw_key_serial_number, serial_);
    if (sensor_info_.major_version_ == 3) {
        header.set_field(raw_key_width, "640");
        header.set_field(raw_key_height, "480");
    }
    if (sensor_info_.major_version_ == 4) {
        header.set_field(raw_key_width, "1280");
        header.set_field(raw_key_height, "720");
    }

    return header;
}

} // namespace Metavision
