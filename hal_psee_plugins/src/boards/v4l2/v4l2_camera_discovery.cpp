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

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <sys/ioctl.h>

#include "boards/v4l2/v4l2_device.h"
#include "boards/v4l2/v4l2_camera_discovery.h"
#include "boards/v4l2/v4l2_data_transfer.h"

#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/facilities/i_camera_synchronization.h"
#include "metavision/hal/facilities/i_decoder.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"
#include "metavision/hal/plugin/plugin_entrypoint.h"
#include "metavision/hal/utils/camera_discovery.h"
#include "metavision/hal/utils/device_builder.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"
#include "metavision/psee_hw_layer/boards/v4l2/v4l2_board_command.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_regmap_device.h"

#include "metavision/psee_hw_layer/devices/genx320/genx320_erc.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_ll_roi.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_ll_biases.h"
#include "metavision/psee_hw_layer/devices/common/antiflicker_filter.h"
#include "metavision/psee_hw_layer/devices/common/event_trail_filter.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_tz_trigger_event.h"

#include "devices/genx320/register_maps/genx320es_registermap.h"

#include "utils/make_decoder.h"

namespace Metavision {

using RegmapData = RegisterMap::RegmapData;
std::string ROOT_PREFIX   = "PSEE/GENX320/";
bool V4l2CameraDiscovery::is_for_local_camera() const {
    return true;
}

CameraDiscovery::SerialList V4l2CameraDiscovery::list() {
    return {"V4l2_device_serial_0"};
}

CameraDiscovery::SystemList V4l2CameraDiscovery::list_available_sources() {
    return {{"V4l2_device_serial_0", ConnectionType::MIPI_LINK, 000420}};
}

bool V4l2CameraDiscovery::discover(DeviceBuilder &device_builder, const std::string &serial,
                                   const DeviceConfig &config) {
    MV_HAL_LOG_TRACE() << "V4l2Discovery - Discovering...";

    std::vector<std::string> device_names = {
        // We might want to iterate over all /dev/mediaX files
        "/dev/media0", "/dev/media1", "/dev/media2", "/dev/media3", "/dev/media4",
    };

    std::vector<std::shared_ptr<V4L2BoardCommand>> devices;
    for (auto device_name : device_names) {
        try {
            devices.emplace_back(std::make_shared<V4L2BoardCommand>(device_name));
        } catch (std::exception &e) {
            MV_HAL_LOG_TRACE() << "Cannot open V4L2 device '" << device_name << "' (err: " << e.what();
        }
    }

    if (devices.empty()) {
        return false;
    }


    auto &main_device  = devices[0];
    auto software_info = device_builder.get_plugin_software_info();
    // TODO: Request sensor/board info and select which regmap to generate
    // hardcoded to Genx320 for now
    auto regmap_data = RegisterMap::RegmapData(1 ,std::make_tuple(GenX320ESRegisterMap, GenX320ESRegisterMapSize, ROOT_PREFIX, 0));
    auto register_map = std::make_shared<RegisterMap>(regmap_data);

    // TODO: use shadow values in otder to avoid too many i2c accesses.
    // build a V4l2DeviceWithRegmap structure for this.
    register_map->set_read_cb([this, &main_device](uint32_t address) {
        return main_device->read_device_register(0, address, 1)[0];
    });
    register_map->set_write_cb([this, &main_device](uint32_t address, uint32_t v) { main_device->write_device_register(0, address, {v}); });

    try {
        auto hw_id = device_builder.add_facility(
            std::make_unique<V4l2HwIdentification>(main_device->get_device()->get_capability(), software_info));

        auto encoding_format = StreamFormat(hw_id->get_current_data_encoding_format());
        size_t raw_size_bytes;
        auto decoder = make_decoder(device_builder, encoding_format, raw_size_bytes, false);

        device_builder.add_facility(std::make_unique<Metavision::I_EventsStream>(
            main_device->build_data_transfer(raw_size_bytes), hw_id, decoder, main_device->get_device()));

    // FIXME: make_shared called on a reference
        device_builder.add_facility(std::make_unique<AntiFlickerFilter>(
            register_map, hw_id->get_sensor_info(), ""));

        device_builder.add_facility(std::make_unique<EventTrailFilter>(
            register_map, hw_id->get_sensor_info(), ""));

        device_builder.add_facility(std::make_unique<V4l2Synchronization>());
        device_builder.add_facility(std::make_unique<GenX320Erc>(register_map));
        device_builder.add_facility(std::make_unique<GenX320LowLevelRoi>(config, register_map, ""));
        device_builder.add_facility(std::make_unique<GenX320LLBiases>(register_map, config));
        device_builder.add_facility(
            std::make_unique<GenX320TzTriggerEvent>(register_map, ""));
    } catch (std::exception &e) { MV_HAL_LOG_ERROR() << "Failed to build streaming facilities :" << e.what(); }

    MV_HAL_LOG_INFO() << "V4l2 Discovery with great success +1";
    return true;
}

} // namespace Metavision
