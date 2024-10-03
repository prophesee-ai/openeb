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

#include <cstdlib>
#include <exception>
#include <map>
#include <utility>

#include "metavision/hal/utils/hal_log.h"
#include "metavision/hal/utils/hal_error_code.h"
#include "metavision/hal/utils/hal_exception.h"
#include "devices/utils/device_system_id.h"
#include "metavision/psee_hw_layer/boards/rawfile/file_hw_identification.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"

namespace Metavision {

FileHWIdentification::FileHWIdentification(const std::shared_ptr<I_PluginSoftwareInfo> &plugin_sw_info,
                                           const PseeRawFileHeader &raw_header) :
    I_HW_Identification(plugin_sw_info), raw_header_(raw_header) {}

std::string FileHWIdentification::get_serial() const {
    return raw_header_.get_serial();
}

I_HW_Identification::SensorInfo FileHWIdentification::get_sensor_info() const {
    return raw_header_.get_sensor_info();
}

std::vector<std::string> FileHWIdentification::get_available_data_encoding_formats() const {
    return {raw_header_.get_format().name()};
}

std::string FileHWIdentification::get_current_data_encoding_format() const {
    return raw_header_.get_format().name();
}

std::string FileHWIdentification::get_integrator() const {
    return raw_header_.get_camera_integrator_name();
}

std::string FileHWIdentification::get_connection_type() const {
    return "File";
}

RawFileHeader FileHWIdentification::get_header_impl() const {
    return raw_header_;
}

DeviceConfigOptionMap FileHWIdentification::get_device_config_options_impl() const {
    return {};
}

} // namespace Metavision
