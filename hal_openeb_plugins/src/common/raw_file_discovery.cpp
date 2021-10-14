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

#include <metavision/hal/utils/raw_file_header.h>
#include <metavision/hal/facilities/i_hw_identification.h>
#include <metavision/hal/utils/device_builder.h>
#include <metavision/hal/utils/hal_log.h>
#include <metavision/hal/facilities/i_decoder.h>
#include <metavision/hal/facilities/i_event_decoder.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/events/event_ext_trigger.h>
#include <metavision/hal/facilities/i_events_stream.h>
#include <metavision/hal/utils/file_data_transfer.h>

#include "common/raw_constants.h"
#include "common/raw_evt2_decoder.h"
#include "common/raw_evt3_decoder.h"
#include "common/raw_file_discovery.h"
#include "common/raw_geometry.h"
#include "common/raw_hw_identification.h"
#include "metavision/sdk/base/utils/string.h"

namespace Metavision {

namespace {

/// @brief Provides a plugin configuration matching plugin and system id.
///
/// @param plugin_name Filename of the plugin.
/// @param integrator_name Name of the integrator.
/// @param system_id ID of the system that generated the file.
/// @return Configuration matching the plugin name. If the name of integrator is empty, system_id is matched instead.
PluginConfig get_plugin_config(const std::string &plugin_name, const std::string &integrator_name,
                               const std::string &system_id) {
    std::string temp_plugin_name = plugin_name;
    unsigned long system_id_long;
    bool is_system_id_correct = unsigned_long_from_str(system_id, system_id_long);
    if (plugin_name.empty() || integrator_name.empty()) {
        if (!is_system_id_correct) {
            MV_HAL_LOG_ERROR() << "Invalid RAW : missing both plugin indentification and system ID";
            throw std::invalid_argument("Raw file not handled");
        }
        MV_HAL_LOG_INFO() << "Invalid RAW : no integrator and/or plugin found in header, assuming it is a RAW "
                             "written with previous versions of Prophesee's software";
        auto id_map_iter = system_id_map.find(static_cast<SystemId>(system_id_long));
        if (id_map_iter == system_id_map.end()) {
            MV_HAL_LOG_ERROR() << "Invalid RAW : incompatible system ID";
            throw std::invalid_argument("Raw file not handled");
        }
        temp_plugin_name = id_map_iter->second;
    }
    auto plugin_config = plugins_map.find(temp_plugin_name);
    if (plugin_config == plugins_map.end()) {
        // No insight on plugin version, Legacy raw... Try to guess EVT format & resolution

        // For now, don't handle
        MV_HAL_LOG_ERROR() << "Invalid RAW : incompatible plugin";
        throw std::invalid_argument("Raw file not handled");
    }
    return plugin_config->second;
}

} // anonymous namespace

bool RawFileDiscovery::discover(Metavision::DeviceBuilder &device_builder, std::unique_ptr<std::istream> &stream,
                                const Metavision::RawFileHeader &header, const Metavision::RawFileConfig &config) {
    const std::string integrator_name = header.get_integrator_name();
    std::string plugin_name           = header.get_plugin_name();
    std::string system_id             = header.get_field(raw_key_system_id);
    PluginConfig plugin_config;

    if (ends_with(plugin_name, std::string("_raw"))) {
        plugin_name.resize(plugin_name.size() - 4);
    }

    try {
        plugin_config = get_plugin_config(plugin_name, integrator_name, system_id);
    } catch (const std::exception &e) {
        MV_HAL_LOG_ERROR() << "Raw file not handled! Was generated with plugin" << plugin_name;
        return false;
    }

    std::string evt_version = raw_evt_version_2;
    if ((plugin_config.encodings_.size() == 1 &&
         plugin_config.encodings_.find(ENCODING_EVT3) != plugin_config.encodings_.end()) ||
        header.get_field("evt") == "3.0") {
        evt_version = raw_evt_version_3;
    }

    // Here we know:
    // * Sensor generation for the HW identification
    // * Geometry
    // * EVT format for the decoder
    // ==> We can proceed with device creation
    auto i_hw_identification = device_builder.add_facility(std::make_unique<RawHWIdentification>(
        device_builder.get_plugin_software_info(), header.get_field(raw_key_serial_number), plugin_config.sensor_info_,
        evt_version));
    auto i_geometry          = device_builder.add_facility(
        std::make_unique<RawGeometry>(plugin_config.sensor_width_, plugin_config.sensor_height_));
    auto cd_event_decoder =
        device_builder.add_facility(std::make_unique<Metavision::I_EventDecoder<Metavision::EventCD>>());
    auto ext_trigger_event_decoder =
        device_builder.add_facility(std::make_unique<Metavision::I_EventDecoder<Metavision::EventExtTrigger>>());

    uint8_t raw_ev_size = 0;
    if (evt_version == raw_evt_version_2) {
        auto decoder = device_builder.add_facility(
            std::make_unique<EVT2::Decoder>(config.do_time_shifting_, cd_event_decoder, ext_trigger_event_decoder));
        raw_ev_size = decoder->get_raw_event_size_bytes();
    } else {
        auto decoder = device_builder.add_facility(std::make_unique<EVT3::Decoder>(
            config.do_time_shifting_, plugin_config.sensor_height_, cd_event_decoder, ext_trigger_event_decoder));
        raw_ev_size  = decoder->get_raw_event_size_bytes();
    }

    auto i_events_stream = device_builder.add_facility(std::make_unique<Metavision::I_EventsStream>(
        std::make_unique<Metavision::FileDataTransfer>(std::move(stream), raw_ev_size, config), i_hw_identification));

    return true;
}

} // namespace Metavision
