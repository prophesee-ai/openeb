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

#ifndef METAVISION_HAL_FILE_HW_IDENTIFICATION_H
#define METAVISION_HAL_FILE_HW_IDENTIFICATION_H

#include <cstdint>
#include <fstream>
#include <string>

#include "metavision/psee_hw_layer/boards/rawfile/psee_raw_file_header.h"
#include "metavision/hal/facilities/i_hw_identification.h"

namespace Metavision {

class FileHWIdentification : public I_HW_Identification {
public:
    FileHWIdentification(const std::shared_ptr<I_PluginSoftwareInfo> &plugin_sw_info,
                         const PseeRawFileHeader &raw_header);

    std::string get_serial() const override final;
    long get_system_id() const override final;
    std::vector<std::string> get_available_data_encoding_formats() const override final;
    std::string get_current_data_encoding_format() const override final;
    std::string get_integrator() const override final;
    std::string get_connection_type() const override final;
    SensorInfo get_sensor_info() const override final;
    DeviceConfigOptionMap get_device_config_options_impl() const override final;

protected:
    RawFileHeader get_header_impl() const override;

private:
    PseeRawFileHeader raw_header_;
};

} // namespace Metavision

#endif // METAVISION_HAL_FILE_HW_IDENTIFICATION_H
