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
#include <list>
#include <sstream>

#include "metavision/hal/utils/hal_log.h"
#include "devices/utils/device_system_id.h"
#include "metavision/psee_hw_layer/boards/rawfile/psee_raw_file_header.h"
#include "metavision/sdk/base/utils/string.h"
#include "metavision/hal/utils/hal_exception.h"
#include "plugin/psee_plugin.h"
#include "utils/psee_hal_plugin_error_code.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"
#include "utils/psee_geometry.h"

namespace Metavision {

namespace {

static const std::string format_key           = "format";
static const std::string geometry_key         = "geometry";
static const std::string sensor_gen_key       = "sensor_generation";
static const std::string sensor_name_key      = "sensor_name";
static const std::string system_id_key        = "system_ID";
static const std::string subsystem_id_key     = "subsystem_ID";
static const std::string firmware_version_key = "firmware_version";
static const std::string serial_number_key    = "serial_number";
static const std::string endianness_key       = "endianness";

static const std::string subsystem_id_key_legacy = "sub_system_ID";
static const std::string legacy_events_type_key  = "evt";
static const std::string legacy_evt3_value       = "3.0";
static const std::string legacy_evt2_value       = "2.0";
static const std::string legacy_evt21_value      = "2.1";

} // anonymous namespace

PseeRawFileHeader::PseeRawFileHeader(const I_HW_Identification &hw, const StreamFormat &format) {
    set_serial(hw.get_serial());
    set_sensor_info(hw.get_sensor_info());
    set_format(format);
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
    std::stringstream ss;
    ss << sensor.major_version_ << "." << sensor.minor_version_;
    set_field(sensor_gen_key, ss.str());
    // Latest sensors don't fit in the major.minor pattern, store directly their name
    set_field(sensor_name_key, sensor.name_);
}

I_HW_Identification::SensorInfo PseeRawFileHeader::get_sensor_info() const {
    std::string generation_str = get_field(sensor_gen_key);
    I_HW_Identification::SensorInfo sensor_info("");

    try {
        std::string str;
        std::istringstream sensor(generation_str);
        std::getline(sensor, str, '.');
        sensor_info.major_version_ = std::stoi(str);
        std::getline(sensor, str, '.');
        sensor_info.minor_version_ = std::stoi(str);
    } catch (const std::exception &) {}

    sensor_info.name_ = get_field(sensor_name_key);
    if (sensor_info.name_.empty()) {
        // If the name was not in the header, build one from the sensor generation
        if (sensor_info.major_version_ == 4 && sensor_info.minor_version_ == 2) {
            // Gen 4.2 was used to describe IMX636
            sensor_info.name_ = "IMX636";
        } else {
            // Call the sensor Gen x.y (0.0 if we couldn't parse the generation)
            std::stringstream ss;
            ss << "Gen" << sensor_info.major_version_ << "." << sensor_info.minor_version_;
            sensor_info.name_ = ss.str();
        }
    }
    return sensor_info;
}

void PseeRawFileHeader::set_format(const StreamFormat &format) {
    set_field(format_key, format.to_string());
    // Keep previous field version for old readers
    // Event type
    if (format.name() == "EVT2") {
        set_field(legacy_events_type_key, legacy_evt2_value);
    } else if (format.name() == "EVT3") {
        set_field(legacy_events_type_key, legacy_evt3_value);
    } else if (format.name() == "EVT21") {
        if (format.contains("endianness")) {
            set_field(endianness_key, format["endianness"]);
        } else {
            set_field(endianness_key, "little");
        }
    }
    // Geometry
    if (format.contains("width") && format.contains("height")) {
        set_field(geometry_key, format["width"] + "x" + format["height"]);
    }
}

StreamFormat PseeRawFileHeader::get_format() const {
    return StreamFormat(get_field(format_key));
}

struct SystemConfig {
    Metavision::I_HW_Identification::SensorInfo sensor_info_;
    StreamFormat format_;
};

void PseeRawFileHeader::check_header() {
    // Pre-Metavision 4.0 headers used a single integrator_name for both the camera and the plugin. This duplicates
    // this information in the two current fields.
    // This is redundant with DeviceDiscovery::open_stream, but plugins tests dont always use DeviceDiscovery
    // and may instanciate directly PseeRawFileHeader.
    if (get_plugin_integrator_name().empty() && get_camera_integrator_name().empty()) {
        MV_HAL_LOG_TRACE() << "Invalid RAW : no plugin/camera integrator found in header, using generic one";
        set_camera_integrator_name(get_field("integrator_name"));
        set_plugin_integrator_name(get_field("integrator_name"));
    }
    // Here, we check if we can handle the file, possibly by infering HW information from the system ID
    if (get_plugin_integrator_name().empty() && get_camera_integrator_name().empty()) {
        MV_HAL_LOG_INFO() << "Invalid RAW : no integrator found in header, assuming it is a RAW "
                             "written with previous versions of Prophesee's software";
        std::string integrator_name = get_psee_plugin_integrator_name();
        set_plugin_integrator_name(integrator_name);
        set_camera_integrator_name(integrator_name);
    }

    static const std::list<std::string> formats_needing_geometry{
        "EVT2", "EVT21", "EVT3", "HISTO3D", "DIFF3D", "AER-8b", "AER-4b",
    };

    static const std::map<long, SystemConfig> system_id_map = {
        {SYSTEM_CCAM3_GEN3, {{3, 0, "Gen3.0"}, StreamFormat("EVT2;height=480;width=640")}},
        {SYSTEM_CCAM3_GEN31, {{3, 1, "Gen3.1"}, StreamFormat("EVT2;height=480;width=640")}},
        {SYSTEM_CCAM3_GEN4, {{4, 0, "Gen4.0"}, StreamFormat("EVT2;height=720;width=1280")}},
        {SYSTEM_CCAM4_GEN3, {{3, 0, "Gen3.0"}, StreamFormat("EVT2;height=480;width=640")}},
        {SYSTEM_CCAM4_GEN3_EVK, {{3, 0, "Gen3.0"}, StreamFormat("EVT2;height=480;width=640")}},
        {SYSTEM_CCAM4_GEN3_REV_B, {{3, 0, "Gen3.0"}, StreamFormat("EVT2;height=480;width=640")}},
        {SYSTEM_CCAM4_GEN3_REV_B_EVK, {{3, 0, "Gen3.0"}, StreamFormat("EVT2;height=480;width=640")}},
        {SYSTEM_CCAM4_GEN3_REV_B_EVK_BRIDGE, {{3, 0, "Gen3.0"}, StreamFormat("EVT2;height=480;width=640")}},
        {SYSTEM_CCAM5_GEN31, {{3, 1, "Gen3.1"}, StreamFormat("EVT2;height=480;width=640")}},
        {SYSTEM_CCAM5_GEN4, {{4, 0, "Gen4.0"}, StreamFormat("EVT3;height=720;width=1280")}},
        {SYSTEM_CCAM5_GEN4_FIXED_FRAME, {{4, 0, "Gen4.0"}, StreamFormat("EVT3;height=720;width=1280")}},
        {SYSTEM_VISIONCAM_GEN3, {{3, 0, "Gen3.0"}, StreamFormat("EVT2;height=480;width=640")}},
        {SYSTEM_VISIONCAM_GEN3_EVK, {{3, 0, "Gen3.0"}, StreamFormat("EVT2;height=480;width=640")}},
        {SYSTEM_VISIONCAM_GEN31, {{3, 1, "Gen3.1"}, StreamFormat("EVT2;height=480;width=640")}},
        {SYSTEM_VISIONCAM_GEN31_EVK, {{3, 1, "Gen3.1"}, StreamFormat("EVT2;height=480;width=640")}},
        {SYSTEM_EVK2_GEN31, {{3, 1, "Gen3.1"}, StreamFormat("EVT2;height=480;width=640")}},
        {SYSTEM_EVK2_GEN4, {{4, 0, "Gen4.0"}, StreamFormat("EVT2;height=720;width=1280")}},
        {SYSTEM_EVK2_GEN41, {{4, 1, "Gen4.1"}, StreamFormat("EVT2;height=720;width=1280")}},
        {SYSTEM_EVK3_GEN31_EVT3, {{3, 1, "Gen3.1"}, StreamFormat("EVT3;height=480;width=640")}},
        {SYSTEM_EVK3_GEN41, {{4, 1, "Gen4.1"}, StreamFormat("EVT3;height=720;width=1280")}},
        {SYSTEM_EVK2_IMX636, {{4, 2, "IMX636"}, StreamFormat("EVT3;height=720;width=1280")}},
        {SYSTEM_EVK3_IMX636, {{4, 2, "IMX636"}, StreamFormat("EVT3;height=720;width=1280")}},
    };
    long system_id = get_system_id();
    auto info      = system_id_map.find(system_id);

    // If the header has an integrator which is not Prophesee, don't try to interpret the system id
    if ((get_plugin_integrator_name() != get_psee_plugin_integrator_name()) ||
        (get_camera_integrator_name() != get_psee_plugin_integrator_name())) {
        info = system_id_map.end();
    }

    // We don't need sensor info to decode the file, but we may infer it from systemID
    if (get_field(sensor_gen_key).empty()) {
        if (info != system_id_map.end()) {
            set_sensor_info(info->second.sensor_info_);
        }
    }

    // In old software, the "format" key didn't exist. Create it, if possible
    if (get_field(format_key).empty()) {
        StreamFormat format("");
        if (info == system_id_map.end()) {
            // Format was introduced along with geometry, if we don't have the format string, we need the system ID
            // to infer the geometry of the sensor
            throw(HalException(PseeHalPluginErrorCode::BoardCommandNotFound, "Could not get Raw file event format"));
        } else if (get_field(legacy_events_type_key) == legacy_evt2_value) {
            format = StreamFormat("EVT2");
        } else if (get_field(legacy_events_type_key) == legacy_evt3_value) {
            format = StreamFormat("EVT3");
        } else if (get_field(legacy_events_type_key) == legacy_evt21_value) {
            format = StreamFormat("EVT21");
        } else {
            format = StreamFormat(info->second.format_.name());
        }
        format["width"]  = info->second.format_["width"];
        format["height"] = info->second.format_["height"];
        set_field(format_key, format.to_string());
    }

    // Now we are sure we have a key, but we need to set the options to make it usable
    StreamFormat format(get_field(format_key));

    // Geometry was once handled in a dedicated field
    if (!get_field(geometry_key).empty()) {
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
            // If the geometry is present in both the format string and the geometry key, it shall be the same
            // In the format comes from the system_id_map above, and the header has a geometry key, then it was
            // forged by hand, and the user is able to provide a correct format string
            if (format.contains("width") && (strtol(format["width"].c_str(), NULL, 0) != width)) {
                throw std::invalid_argument("Width mismatch between format and geometry keys");
            }
            format["width"] = std::to_string(width);
            if (format.contains("height") && (strtol(format["height"].c_str(), NULL, 0) != height)) {
                throw std::invalid_argument("Height mismatch between format and geometry keys");
            }
            format["height"] = std::to_string(height);
        } catch (const std::exception &e) {
            MV_HAL_LOG_TRACE() << "Invalid RAW : wrong geometry key:" << e.what();
            throw HalException(PseeHalPluginErrorCode::BoardCommandNotFound, e.what());
        }
    }

    // Some headers were edited, removing the Geometry. Try to infer it from the systemID
    // Some format may not need geometry, only add it on whitelist
    for (auto f : formats_needing_geometry) {
        if (format.name() == f) {
            try {
                format.geometry();
            } catch (const std::exception &e) {
                if (info == system_id_map.end()) {
                    // Format was introduced along with geometry, if we don't have the format string, we need the system
                    // ID to infer the geometry of the sensor
                    throw(
                        HalException(PseeHalPluginErrorCode::BoardCommandNotFound, "Could not get Raw file geometry"));
                }
                format["width"]  = info->second.format_["width"];
                format["height"] = info->second.format_["height"];
                MV_HAL_LOG_TRACE() << "Invalid RAW : wrong geometry key:" << e.what();
            }
        }
    }

    // Fix Evt2.1 decoding
    if (format.name() == "EVT21" && !format.contains("endianness")) {
        std::string endianness = get_field(endianness_key);
        if (!endianness.empty()) {
            format["endianness"] = endianness;
        } else if ((get_plugin_integrator_name() == get_psee_plugin_integrator_name()) &&
                   (get_camera_integrator_name() == get_psee_plugin_integrator_name()) && system_id &&
                   (system_id < SYSTEM_EVK2_SAPHIR)) {
            // On our systems older that SYSTEM_EVK2_SAPHIR (TB_SAPHIR was used with sensorlib, system_id is not set)
            format["endianness"] = "legacy";
        }
    }

    // Copy other format-related fields from the header to the format
    std::list<std::string> fields = {"pixellayout", "pixelbytes"};
    for (auto &field : fields) {
        if (!get_field(field).empty()) {
            if (format.contains(field) && (format[field] != get_field(field))) {
                throw HalException(PseeHalPluginErrorCode::BoardCommandNotFound, "Mismatched definition of " + field);
            }
            format[field] = get_field(field);
        }
    }

    // Then save everything in the header
    set_format(format);
}

} // namespace Metavision
