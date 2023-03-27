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

#ifndef METAVISION_HAL_PSEE_RAW_FILE_HEADER_H
#define METAVISION_HAL_PSEE_RAW_FILE_HEADER_H

#include <memory>
#include <cstdint>
#include <fstream>
#include <string>

#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_geometry.h"
#include "metavision/hal/utils/raw_file_header.h"

namespace Metavision {

class StreamFormat;

/// @brief Convenient class to handle Prophesee RAW files header
class PseeRawFileHeader : public RawFileHeader {
public:
    PseeRawFileHeader(const I_HW_Identification &, const StreamFormat &);
    PseeRawFileHeader(std::istream &);
    PseeRawFileHeader(const HeaderMap &);
    PseeRawFileHeader(const RawFileHeader &);

    std::string get_serial() const;

    long get_system_id() const;

    // This isn't part of standard I_HW_Identification
    void set_sub_system_id(long);
    long get_sub_system_id() const;

    I_HW_Identification::SensorInfo get_sensor_info() const;

    StreamFormat get_format() const;

private:
    void check_header();
    void set_serial(std::string);
    void set_system_id(long system_id);
    void set_sensor_info(const I_HW_Identification::SensorInfo &);
    void set_system_version(long);
    void set_format(const StreamFormat &);
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_RAW_FILE_HEADER_H
