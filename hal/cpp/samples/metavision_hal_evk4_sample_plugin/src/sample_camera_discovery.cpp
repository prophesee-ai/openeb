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

#include <memory>

#include <metavision/hal/device/device.h>
#include <metavision/hal/facilities/i_events_stream.h>
#include <metavision/hal/utils/data_transfer.h>
#include <metavision/hal/utils/device_builder.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/hal/decoders/evt3/evt3_decoder.h>

#include "sample_antiflicker.h"
#include "sample_camera_discovery.h"
#include "sample_camera_synchronization.h"
#include "sample_data_transfer.h"
#include "sample_digital_crop.h"
#include "sample_digital_event_mask.h"
#include "sample_device_control.h"
#include "sample_erc.h"
#include "sample_event_trail_filter.h"
#include "sample_geometry.h"
#include "sample_hw_identification.h"
#include "sample_ll_biases.h"
#include "internal/sample_register_access.h"
#include "internal/sample_usb_connection.h"

// Function prototype declaration
bool initialize_usb_connection_with_device(libusb_context* ctx);


Metavision::CameraDiscovery::SerialList SampleCameraDiscovery::list() {
    SerialList ret;
    ret.push_back(SampleHWIdentification::SAMPLE_SERIAL);
    return ret;
}

Metavision::CameraDiscovery::SystemList SampleCameraDiscovery::list_available_sources() {
    SystemList systems;

    Metavision::PluginCameraDescription description;
    description.serial_     = SampleHWIdentification::SAMPLE_SERIAL;
    description.connection_ = Metavision::USB_LINK;

    systems.push_back(description);
    return systems;
}

bool SampleCameraDiscovery::discover(Metavision::DeviceBuilder &device_builder, const std::string &serial,
                                     const Metavision::DeviceConfig &config) {

    if (!(serial.empty() || serial == SampleHWIdentification::SAMPLE_SERIAL)) {
        return false;
    }

    std::shared_ptr<SampleUSBConnection> connection;
    try {
        connection = std::make_shared<SampleUSBConnection>(0x04b4, 0x00f5, kEvk4Interface);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return false;
    }

    // Add facilities to the device builder
    auto hw_identification = device_builder.add_facility(
        std::make_unique<SampleHWIdentification>(device_builder.get_plugin_software_info(), "USB"));
    device_builder.add_facility(std::make_unique<SampleGeometry>());
    device_builder.add_facility(std::make_unique<SampleLLBiases>(config, connection));
    device_builder.add_facility(std::make_unique<SampleCameraSynchronization>(connection));

    auto cd_event_decoder = device_builder.add_facility(std::make_unique<Metavision::I_EventDecoder<Metavision::EventCD>>());
    auto ext_trig_decoder     = device_builder.add_facility(std::make_unique<Metavision::I_EventDecoder<Metavision::EventExtTrigger>>());
    auto erc_count_ev_decoder = device_builder.add_facility(std::make_unique<Metavision::I_EventDecoder<Metavision::EventERCCounter>>());
    auto decoder = device_builder.add_facility(make_evt3_decoder(false, 1280, 720, cd_event_decoder, ext_trig_decoder, erc_count_ev_decoder));

    device_builder.add_facility(std::make_unique<Metavision::I_EventsStream>(
                                    std::make_unique<SampleDataTransfer>(decoder->get_raw_event_size_bytes(),
                                                                         connection),
                                    hw_identification, decoder,
                                    std::make_shared<SampleDeviceControl>(connection)));

    device_builder.add_facility(std::make_unique<SampleAntiFlicker>());
    device_builder.add_facility(std::make_unique<SampleDigitalCrop>(connection));
    device_builder.add_facility(std::make_unique<SampleDigitalEventMask>());
    device_builder.add_facility(std::make_unique<SampleErc>(connection));
    device_builder.add_facility(std::make_unique<SampleEventTrailFilter>(connection));

    return true;
}

bool SampleCameraDiscovery::is_for_local_camera() const {
    return true;
}

bool initialize_usb_connection_with_device(const SampleUSBConnection &connection) {
    // Do the INIT sequence with Register Accesses
    write_register(connection, 0x0000001C, 0x00000001);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    write_register(connection, 0x00400004, 0x00000001);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    write_register(connection, 0x00400004, 0x00000000);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    write_register(connection, 0x0000B000, 0x00000158);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    write_register(connection, 0x0000B044, 0x00000000);
    write_register(connection, 0x0000B004, 0x0000000A);
    write_register(connection, 0x0000B040, 0x00000000);
    write_register(connection, 0x0000B0C8, 0x00000000);
    write_register(connection, 0x0000B040, 0x00000000);
    write_register(connection, 0x0000B040, 0x00000000);
    write_register(connection, 0x00000000, 0x4F006442);
    write_register(connection, 0x00000000, 0x0F006442);
    write_register(connection, 0x000000B8, 0x00000400);
    write_register(connection, 0x000000B8, 0x00000400);
    write_register(connection, 0x0000B07C, 0x00000000);
    write_register(connection, 0x0000B074, 0x00000002);
    write_register(connection, 0x0000B078, 0x000000A0);
    write_register(connection, 0x000000C0, 0x00000110);
    write_register(connection, 0x000000C0, 0x00000210);
    write_register(connection, 0x0000B120, 0x00000001);
    write_register(connection, 0x0000E120, 0x00000000);
    write_register(connection, 0x0000B068, 0x00000004);
    write_register(connection, 0x0000B07C, 0x00000001);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    write_register(connection, 0x0000B07C, 0x00000003);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    write_register(connection, 0x000000B8, 0x00000401);
    write_register(connection, 0x000000B8, 0x00000409);
    write_register(connection, 0x00000000, 0x4F006442);
    write_register(connection, 0x00000000, 0x4F00644A);
    write_register(connection, 0x0000B080, 0x00000077);
    write_register(connection, 0x0000B084, 0x0000000F);
    write_register(connection, 0x0000B088, 0x00000037);
    write_register(connection, 0x0000B08C, 0x00000037);
    write_register(connection, 0x0000B090, 0x000000DF);
    write_register(connection, 0x0000B094, 0x00000057);
    write_register(connection, 0x0000B098, 0x00000037);
    write_register(connection, 0x0000B09C, 0x00000067);
    write_register(connection, 0x0000B0A0, 0x00000037);
    write_register(connection, 0x0000B0A4, 0x0000002F);
    write_register(connection, 0x0000B0AC, 0x00000028);
    write_register(connection, 0x0000B0CC, 0x00000001);
    write_register(connection, 0x0000B000, 0x000002F8);
    write_register(connection, 0x0000B004, 0x0000008A);
    write_register(connection, 0x0000B01C, 0x00000030);
    write_register(connection, 0x0000B020, 0x00002000);
    write_register(connection, 0x0000B02C, 0x000000FF);
    write_register(connection, 0x0000B030, 0x00003E80);
    write_register(connection, 0x0000B028, 0x00000FA0);
    write_register(connection, 0x0000A000, 0x000B0501);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    write_register(connection, 0x0000A008, 0x00002405);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    write_register(connection, 0x0000A004, 0x000B0501);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    write_register(connection, 0x0000A020, 0x00000150);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    write_register(connection, 0x0000B040, 0x00000007);
    write_register(connection, 0x0000B064, 0x00000006);
    write_register(connection, 0x0000B040, 0x0000000F);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    write_register(connection, 0x0000B004, 0x0000008A);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    write_register(connection, 0x0000B0C8, 0x00000003);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    write_register(connection, 0x0000B044, 0x00000001);
    write_register(connection, 0x0000B000, 0x000002F9);
    write_register(connection, 0x00007008, 0x00000001);
    write_register(connection, 0x00007000, 0x00070001);
    write_register(connection, 0x00008000, 0x0001E085);
    write_register(connection, 0x00009008, 0x00000644);
    write_register(connection, 0x00000004, 0xF0005042);
    write_register(connection, 0x00000018, 0x00000200);
    write_register(connection, 0x00001014, 0x11A1504D);
    write_register(connection, 0x00009004, 0x00000000);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    write_register(connection, 0x00009000, 0x00000200);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    return true;
}
