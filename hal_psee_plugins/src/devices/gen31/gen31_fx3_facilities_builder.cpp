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

#ifdef _MSC_VER
#define NOMINMAX
#endif

#include "boards/fx3/fx3_camera_discovery.h"
#include "boards/fx3/fx3_hw_identification.h"
#include "boards/fx3/fx3_libusb_board_command.h"
#include "boards/utils/psee_libusb_data_transfer.h"
#include "decoders/evt2/evt2_decoder.h"
#include "devices/gen31/gen31_event_rate_noise_filter_module.h"
#include "devices/gen31/gen31_fx3_device_control.h"
#include "devices/gen31/gen31_fx3_facilities_builder.h"
#include "devices/gen31/gen31_ll_biases.h"
#include "devices/gen31/gen31_monitoring.h"
#include "devices/gen31/gen31_pattern_generator.h"
#include "devices/gen31/gen31_roi_command.h"
#include "devices/gen31/gen31_trigger_event.h"
#include "devices/common/ccam_trigger_out.h"
#include "devices/gen31/register_maps/gen31_evk1_device.h"
#include "facilities/psee_hw_register.h"
#include "geometries/vga_geometry.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/utils/device_builder.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_ext_trigger.h"
#include "plugin/psee_plugin.h"

namespace Metavision {

bool build_gen31_fx3_device(DeviceBuilder &device_builder, const DeviceBuilderParameters &device_builder_params,
                            const DeviceConfig &device_config) {
    Fx3CameraDiscovery::DeviceBuilderParameters params =
        static_cast<const Fx3CameraDiscovery::DeviceBuilderParameters &>(device_builder_params);

    auto board_cmd    = params.board_cmd;
    auto register_map = std::make_shared<RegisterMap>();
    build_gen31_register_map(*register_map);
    register_map->set_read_cb([board_cmd](uint32_t address) {
        board_cmd->load_register(address);
        return board_cmd->read_register(address);
    });
    register_map->set_write_cb([board_cmd](uint32_t address, uint32_t v) { board_cmd->write_register(address, v); });

    auto hw_identification = device_builder.add_facility(std::make_unique<Fx3HWIdentification>(
        device_builder.get_plugin_software_info(), board_cmd, false,
        (long)Gen31Fx3DeviceControl::get_sensor_id(*register_map), get_psee_plugin_integrator_name()));

    auto geometry                  = device_builder.add_facility(std::make_unique<VGAGeometry>());
    auto cd_event_decoder          = device_builder.add_facility(std::make_unique<I_EventDecoder<EventCD>>());
    auto ext_trigger_event_decoder = device_builder.add_facility(std::make_unique<I_EventDecoder<EventExtTrigger>>());
    auto decoder =
        device_builder.add_facility(std::make_unique<EVT2Decoder>(false, cd_event_decoder, ext_trigger_event_decoder));
    auto events_stream = device_builder.add_facility(std::make_unique<I_EventsStream>(
        std::make_unique<PseeLibUSBDataTransfer>(board_cmd, decoder->get_raw_event_size_bytes()), hw_identification));

    auto gen31_device_control = device_builder.add_facility(std::make_unique<Gen31Fx3DeviceControl>(register_map));

    device_builder.add_facility(std::make_unique<Gen31TriggerEvent>(register_map, gen31_device_control));
    if (board_cmd->get_system_version() >= 0x30000) {
        device_builder.add_facility(std::make_unique<CCamTriggerOut>(register_map, gen31_device_control, ""));
    }

    auto sensor_prefix = gen31_device_control->get_sensor_prefix();
    auto hw_register   = device_builder.add_facility(std::make_unique<PseeHWRegister>(register_map));
    device_builder.add_facility(std::make_unique<Gen31_LL_Biases>(hw_register, sensor_prefix));
    device_builder.add_facility(std::make_unique<Gen31_EventRateNoiseFilterModule>(hw_register, sensor_prefix));

    device_builder.add_facility(std::make_unique<Gen31Monitoring>(hw_register));

    device_builder.add_facility(
        std::make_unique<Gen31ROICommand>(geometry->get_width(), geometry->get_height(), register_map, sensor_prefix));

    // those facilities are not exposed in the public API yet
    // device_builder.add_facility(std::make_unique<Gen31PatternGenerator>(register_map));

    return true;
}

} // namespace Metavision
