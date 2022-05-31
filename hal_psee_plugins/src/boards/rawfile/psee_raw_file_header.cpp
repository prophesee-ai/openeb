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
#include <sstream>

#include "metavision/hal/utils/hal_log.h"
#include "devices/utils/device_system_id.h"
#include "boards/rawfile/psee_raw_file_header.h"
#include "metavision/sdk/base/utils/string.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"

namespace Metavision {

namespace {

static const std::string format_key           = "format";
static const std::string geometry_key         = "geometry";
static const std::string sensor_gen_key       = "sensor_generation";
static const std::string system_id_key        = "system_ID";
static const std::string subsystem_id_key     = "subsystem_ID";
static const std::string firmware_version_key = "firmware_version";
static const std::string serial_number_key    = "serial_number";

static const std::string subsystem_id_key_legacy = "sub_system_ID";
static const std::string legacy_events_type_key  = "evt";
static const std::string legacy_evt3_value       = "3.0";
static const std::string legacy_evt2_value       = "2.0";

} // anonymous namespace

class Geometry : public Metavision::I_Geometry {
private:
    int width_;
    int height_;

public:
    Geometry(const int width, const int height) : width_(width), height_(height) {}
    int get_width() const override final {
        return width_;
    }
    int get_height() const override final {
        return height_;
    }
};

PseeRawFileHeader::PseeRawFileHeader(const I_HW_Identification &hw, const I_Geometry &geometry) {
    set_serial(hw.get_serial());
    set_system_id(hw.get_system_id());
    set_sensor_info(hw.get_sensor_info());
    set_system_version(hw.get_system_version());
    set_geometry(geometry);
    auto available_format = hw.get_available_raw_format();
    if (available_format.empty()) {
        throw(HalException(PseeHalPluginErrorCode::UnknownFormat, "System advertises no output format"));
    }
    set_format(available_format[0]); // caller shall fix format if it's not the 1st one
}

PseeRawFileHeader::PseeRawFileHeader(std::istream &stream) : RawFileHeader(stream) {
    check_header();
}
PseeRawFileHeader::PseeRawFileHeader(const HeaderMap &header) : RawFileHeader(header) {
    check_header();
}
PseeRawFileHeader::PseeRawFileHeader(const RawFileHeader &raw_header) :
    PseeRawFileHeader(raw_header.get_header_map()) {}

void PseeRawFileHeader::set_serial(std::string serial) {
    set_field(serial_number_key, serial);
}

std::string PseeRawFileHeader::get_serial() const {
    return get_field(serial_number_key);
}

void PseeRawFileHeader::set_system_id(long system_id) {
    set_field(system_id_key, std::to_string(system_id));
}

long PseeRawFileHeader::get_system_id() const {
    const auto &str = get_field(system_id_key);
    unsigned long result;
    if (!unsigned_long_from_str(str, result)) {
        return -1;
    }
    return result;
}

void PseeRawFileHeader::set_sub_system_id(long subsytem_id) {
    set_field(subsystem_id_key, std::to_string(subsytem_id));
}

long PseeRawFileHeader::get_sub_system_id() const {
    std::string system_sub_id_as_string = get_field(subsystem_id_key_legacy);
    unsigned long result;
    if (system_sub_id_as_string.empty()) {
        system_sub_id_as_string = get_field(subsystem_id_key);
    }
    if (!unsigned_long_from_str(system_sub_id_as_string, result)) {
        return -1;
    }
    return result;
}

void PseeRawFileHeader::set_sensor_info(const I_HW_Identification::SensorInfo &sensor) {
    set_field(sensor_gen_key, sensor.as_string());
}

I_HW_Identification::SensorInfo PseeRawFileHeader::get_sensor_info() const {
    std::string sensor_str                      = get_field(sensor_gen_key);
    I_HW_Identification::SensorInfo sensor_info = {0, 0};
    try {
        std::string str;
        std::istringstream sensor(sensor_str);
        std::getline(sensor, str, '.');
        sensor_info.major_version_ = std::stoi(str);
        std::getline(sensor, str, '.');
        sensor_info.minor_version_ = std::stoi(str);
    } catch (std::exception &e) {}
    return sensor_info;
}

void PseeRawFileHeader::set_system_version(long system_version) {
    const std::string system_version_str = std::to_string(((system_version >> 16) & 0xFF)) + "." +
                                           std::to_string(((system_version >> 8) & 0xFF)) + "." +
                                           std::to_string(((system_version >> 0) & 0xFF));
    set_field(firmware_version_key, system_version_str);
}

long PseeRawFileHeader::get_system_version() const {
    auto value = get_field(firmware_version_key);
    long ret   = -1;
    try {
        ret = 0;
        for (int i = 0; i < 3; i++) {
            ret <<= 8;
            size_t nb_number = 0;
            ret += std::stol(value, &nb_number, 10);
            value.erase(0, nb_number + 1);
        }
    } catch (const std::exception &) { ret = -1; }
    return ret;
}

void PseeRawFileHeader::set_geometry(const Metavision::I_Geometry &geometry) {
    set_field(geometry_key, std::to_string(geometry.get_width()) + "x" + std::to_string(geometry.get_height()));
}

std::unique_ptr<Metavision::I_Geometry> PseeRawFileHeader::get_geometry() const {
    std::string geometry_str = get_field(geometry_key);
    try {
        int width = 0, height = 0;
        std::string str;
        std::istringstream geometry(geometry_str);
        // Expected format: 1080x720
        // This code isn't guaranteed to detect malformed headers
        std::getline(geometry, str, 'x');
        width = std::stoi(str);
        std::getline(geometry, str, 'x');
        height = std::stoi(str);
        if (!width || !height) {
            throw std::invalid_argument("Sensor surface can't be null");
        }
        return std::make_unique<Geometry>(width, height);
    } catch (...) {
        /* Use catch-all handler on purpose, as `stoi(¨¨)` may throw exceptions that
         * are of other type than std::exception on some platforms (eg. Android...)
         */
        return nullptr;
    }
}

void PseeRawFileHeader::set_format(const std::string &format) {
    set_field(format_key, format);
    // Keep previous field version for old readers
    if (format == "EVT2") {
        set_field(legacy_events_type_key, legacy_evt2_value);
    } else if (format == "EVT3") {
        set_field(legacy_events_type_key, legacy_evt3_value);
    }
}

std::string PseeRawFileHeader::get_format() const {
    return get_field(format_key);
}

struct SystemConfig {
    Metavision::I_HW_Identification::SensorInfo sensor_info_;
    Geometry geometry_;
    std::string format_;
};

void PseeRawFileHeader::check_header() {
    // Here, we check if we can handle the file, possibly by infering HW information from the system ID
    std::string integrator_name = get_integrator_name();
    if (integrator_name.empty()) {
        MV_HAL_LOG_INFO() << "Invalid RAW : no integrator found in header, assuming it is a RAW "
                             "written with previous versions of Prophesee's software";
        integrator_name = get_psee_plugin_integrator_name();
        set_integrator_name(integrator_name);
    }

    static const std::map<long, SystemConfig> system_id_map = {
        {SYSTEM_CCAM3_GEN3, {{3, 0}, Geometry(640, 480), "EVT2"}},
        {SYSTEM_CCAM3_GEN31, {{3, 1}, Geometry(640, 480), "EVT2"}},
        {SYSTEM_CCAM3_GEN4, {{4, 0}, Geometry(1280, 720), "EVT2"}}, // Evt2 was default with old headers
        {SYSTEM_CCAM4_GEN3, {{3, 0}, Geometry(640, 480), "EVT2"}},
        {SYSTEM_CCAM4_GEN3_EVK, {{3, 0}, Geometry(640, 480), "EVT2"}},
        {SYSTEM_CCAM4_GEN3_REV_B, {{3, 0}, Geometry(640, 480), "EVT2"}},
        {SYSTEM_CCAM4_GEN3_REV_B_EVK, {{3, 0}, Geometry(640, 480), "EVT2"}},
        {SYSTEM_CCAM4_GEN3_REV_B_EVK_BRIDGE, {{3, 0}, Geometry(640, 480), "EVT2"}},
        {SYSTEM_CCAM5_GEN31, {{3, 1}, Geometry(640, 480), "EVT2"}},
        {SYSTEM_CCAM5_GEN4, {{4, 0}, Geometry(1280, 720), "EVT3"}},
        {SYSTEM_CCAM5_GEN4_FIXED_FRAME, {{4, 0}, Geometry(1280, 720), "EVT3"}},
        {SYSTEM_VISIONCAM_GEN3, {{3, 0}, Geometry(640, 480), "EVT2"}},
        {SYSTEM_VISIONCAM_GEN3_EVK, {{3, 0}, Geometry(640, 480), "EVT2"}},
        {SYSTEM_VISIONCAM_GEN31, {{3, 1}, Geometry(640, 480), "EVT2"}},
        {SYSTEM_VISIONCAM_GEN31_EVK, {{3, 1}, Geometry(640, 480), "EVT2"}},
        {SYSTEM_CX3_CCAM5_GEN4, {{4, 0}, Geometry(1280, 720), "EVT3"}},
        {SYSTEM_EVK2_GEN31, {{3, 1}, Geometry(640, 480), "EVT2"}},
        {SYSTEM_EVK2_GEN4, {{4, 0}, Geometry(1280, 720), "EVT2"}},  // Evt2 was default with old headers
        {SYSTEM_EVK2_GEN41, {{4, 1}, Geometry(1280, 720), "EVT2"}}, // Evt2 was default with old headers
        {SYSTEM_EVK3_GEN31_EVT2, {{3, 1}, Geometry(640, 480), "EVT2"}},
        {SYSTEM_EVK3_GEN31_EVT3, {{3, 1}, Geometry(640, 480), "EVT3"}},
        {SYSTEM_EVK3_GEN41, {{4, 1}, Geometry(1280, 720), "EVT3"}},
    };
    long system_id = get_system_id();
    auto info      = system_id_map.find(system_id);

    // If there is no serial, don't invent one
    // Same for system id
    // We don't need sensor info to decode the file, but we may infer it from systemID
    if (get_field(sensor_gen_key).empty()) {
        if (info != system_id_map.end()) {
            set_sensor_info(info->second.sensor_info_);
        }
    }

    if (!get_geometry()) {
        if (info != system_id_map.end()) {
            set_geometry(info->second.geometry_);
        } else {
            throw(HalException(PseeHalPluginErrorCode::BoardCommandNotFound, "Could not get Raw file sensor geometry"));
        }
    }

    if (get_field(format_key).empty()) {
        if (get_field(legacy_events_type_key) == legacy_evt2_value) {
            set_format("EVT2");
        } else if (get_field(legacy_events_type_key) == legacy_evt3_value) {
            set_format("EVT3");
        } else if (info != system_id_map.end()) {
            set_format(info->second.format_);
        } else {
            throw(HalException(PseeHalPluginErrorCode::BoardCommandNotFound, "Could not get Raw file event format"));
        }
    }
}

} // namespace Metavision
